"""
pipeline.py
PHASE: Physiology-guided Hybrid Adaptive Signal Extraction

Dual-path fetal IC selection:
  Path A -- ICA1 Direct: best non-maternal IC from ICA1
  Path B -- WSVD + ICA2: best IC from ICA2 on WSVD residual

CHANGES FROM ORIGINAL:
  [FIX-1] Path B ICA2: exclude ICA2 components correlated with maternal IC
          (|corr| > MATERNAL_ICA2_CORR_THRESH). Previously exclude_idx=-1
          meant no exclusion, allowing residual-maternal ICs to be selected.
  [FIX-2] min_usable_peaks is now recording-length-adaptive (was hardcoded 100).
  [FIX-3] ECHO has_reference passed explicitly; None passed for NIFECGDB so
          morphology score is disabled rather than self-referential.

DISSERTATION MODIFICATIONS (unified, dataset-adaptive):
  [MOD-1] _best_ic(): unified three-factor scoring replaces n_peaks × hr_score.
          final_score = base_score × maternal_penalty × (1 + morphology_score)
          score_fetal_ic() is now called inside _best_ic() as the base_score.
          maternal_penalty uses corr(IC, maternal_IC) and corr(IC, maternal
          harmonic at 2×mhr). Path B uses half-weight penalty (WSVD already
          suppressed maternal content). morphology_score = mean pairwise
          correlation of beat windows, gated on min peak count.
  [MOD-2] determine_n_components(): PCA-adaptive n_components before each ICA
  [FIX-2] PCA floor raised from 2 to 3: minimum 3 components ensures ICA
          can separate maternal/fetal/noise as three distinct sources.
  [FIX-1] Stability-gated ensemble selection: winning IC must appear as
          top scorer in >= 2 seeds. Falls back to seed=42 if no stable IC.
          call. Counts components explaining ≥5% variance, clipped to [2,4].
          Applied independently to ICA1 (on abd_proc) and ICA2 (on residual).
  [MOD-3] ICA ensemble in _best_ic(): N_ENSEMBLE=5 runs with fixed seeds 0–4.
          All N×k IC candidates scored with [MOD-1] formula; global winner
          selected. Not applied to maternal IC selection (already reliable).
          Stability bonus logged for ECHO (cross-run correlation of top ICs).
  [MOD-4] adaptive_windowed_wsvd() now receives duration_sec; window length
          is derived from min_windows=15 floor, capped at WSVD_WINDOW_SEC and
          floored at 1.5 s. Same code — adaptive behaviour across lengths.

PATH SELECTION FIXES (bimodal F1 diagnosis — CinC2013):
  [FIX-PATH-1] Path selection now uses unified scores rather than peak-count
               comparison. The old logic (a_n >= b_n * PATH_A_PREFERENCE)
               never fired on CinC2013 because both paths produce near-equal
               beat counts on 60-second recordings, causing Path B to win
               100% of the time even when Path A scored higher. The new logic
               compares a_score vs b_score directly, using PATH_A_PREFERENCE
               as a score multiplier (default 1.0 from cinc2013.yaml).

  [FIX-PATH-2] Confidence gate: when the chosen path's score falls below
               CONFIDENCE_GATE_THRESHOLD (set in cinc2013.yaml), the result
               is flagged as low-confidence in metadata. This surfaces the
               Mode 1 degenerate cases (a07, a09, a60) where score = 0.

  [FIX-PATH-3] Harmonic confusion guard in _is_fetal_hr(): candidate HR is
               rejected if it is within HR_SEP_MIN_BPM of EITHER the maternal
               HR OR half the maternal HR. Targets Mode 2 failures where high
               maternal HR (~115 bpm) causes the half-harmonic (~58 bpm
               → doubled detection at ~115 bpm) to pass the fetal HR filter.

  [FIX-PATH-4] Annotation anomaly guard in run(): recordings where
               n_reference < duration_sec * FETAL_HR_MIN / 60 * 0.5 are
               flagged as having a sparse annotation in metadata. This catches
               a54 (37 reference beats in 60 s) without special-casing it.
"""

import sys
import numpy as np
from pathlib import Path
from scipy.stats import kurtosis as scipy_kurtosis

sys.path.insert(0, str(Path(__file__).parent))

from config_loader import get_config
from preprocessing.filters import preprocess_multichannel, preprocess_channel
from preprocessing.qrs_detector import (
    detect_maternal_qrs, detect_fetal_qrs,
    detect_reference_fetal_qrs, compute_hr_stats, pan_tompkins,
    load_adfecgdb_annotation
)
from separation.ica import (
    run_ica, select_maternal_ic, select_fetal_ic, get_ic_as_signal,
    score_fetal_ic,
)
from separation.wsvd import (
    gaussian_weight_matrix, adaptive_windowed_wsvd,
    subtract_maternal, svd_explained_variance
)
from separation.ekf import FetalECGKalmanFilter
from evaluation.metrics import evaluate
from xai.echo import ECHOExplainer
from preprocessing.qrs_detector import load_wfdb_annotation

# ── Ensemble hyperparameter ──────────────────────────────────────────────────
N_ENSEMBLE = 5          # number of ICA runs per ensemble call (seeds 0 … N-1)
ENSEMBLE_SEEDS = list(range(N_ENSEMBLE))   # [0, 1, 2, 3, 4] — fixed for reproducibility
PCA_VARIANCE_THRESHOLD = 0.05   # components explaining ≥ 5% variance are kept
PCA_N_MIN, PCA_N_MAX  = 3, 4   # clip range for adaptive n_components
                                # Floor raised 2→3: with only 2 components
                                # ICA cannot separate maternal/fetal/noise
                                # as three distinct sources. 3 is the minimum
                                # meaningful decomposition for this problem.
MORPHOLOGY_MIN_PEAKS  = 5       # minimum peaks to compute morphology score
MORPHOLOGY_WIN_SEC    = 0.3     # ± seconds around each peak for beat window
STABILITY_LOG_THRESH  = 0.7     # cross-run corr above this is logged as "stable"

# [FIX-PATH-2] Default confidence gate — overridden by cinc2013.yaml
_DEFAULT_CONFIDENCE_GATE = 0.05


def _min_usable_peaks(duration_sec: float, cfg, dataset: str = "ADFECGDB") -> int:
    if dataset == "NIFECGDB":
        expected = duration_sec * cfg.FETAL_HR_MIN / 60.0
        return max(20, int(expected * 0.15))
    else:
        expected = duration_sec * cfg.FETAL_HR_LOW / 60.0
        return max(30, int(expected * 0.5))


def _norm(sig):
    sig = sig - np.mean(sig)
    return sig / (np.std(sig) + 1e-10)


def _candidate_hr(sig, fs, cfg):
    peaks   = detect_fetal_qrs(sig, fs, cfg=cfg)
    stats   = compute_hr_stats(peaks, fs)
    mean_hr = stats["mean_hr"] if len(peaks) >= 2 else np.nan
    return peaks, mean_hr


def _is_fetal_hr(mean_hr: float, maternal_hr: float, cfg) -> bool:
    """
    Check if a candidate HR is in the fetal range and sufficiently
    separated from maternal HR.

    [FIX-PATH-3] Harmonic confusion guard:
    In addition to the existing HR_SEP_MIN_BPM separation check against
    the maternal HR, we now also reject candidates within HR_SEP_MIN_BPM
    of HALF the maternal HR. This prevents Mode 2 failures where maternal
    HR ~115 bpm causes the detection of a sub-harmonic at ~115 bpm (via
    every-other-beat detection) to pass the fetal HR filter — the primary
    cause of the high-MAE identity confusion failures (a06, a16, a18, etc.).

    The relaxed separation threshold (× 0.7) when maternal HR > 85 bpm is
    preserved from the original, applied to BOTH the maternal HR check and
    the half-harmonic check.
    """
    print("HR_SEP_MIN_BPM", cfg.HR_SEP_MIN_BPM)
    if np.isnan(mean_hr):
        return False
    in_range = cfg.FETAL_HR_LOW <= mean_hr <= cfg.FETAL_HR_HIGH

    sep_threshold = (cfg.HR_SEP_MIN_BPM * 0.7
                     if (not np.isnan(maternal_hr) and maternal_hr > 85)
                     else cfg.HR_SEP_MIN_BPM)

    # Original: separation from maternal HR
    sep_from_maternal = (np.isnan(maternal_hr) or
                         abs(mean_hr - maternal_hr) >= sep_threshold)

    # [FIX-PATH-3]: separation from half the maternal HR (sub-harmonic guard)
    half_maternal_hr = maternal_hr / 2.0 if not np.isnan(maternal_hr) else np.nan
    sep_from_half = (np.isnan(half_maternal_hr) or
                     abs(mean_hr - half_maternal_hr) >= sep_threshold)

    return in_range and sep_from_maternal and sep_from_half


def _hr_score(mean_hr, cfg, expected_hr=None):
    centre = expected_hr if expected_hr is not None else cfg.FETAL_HR_CENTRE
    if np.isnan(mean_hr):
        return 0.0
    return 1.0 / (1.0 + abs(mean_hr - centre) / 30.0)


def _find_maternal_residual_idx(ICs, maternal_ic, cfg):
    """
    [FIX-1] Find ICA2 component most correlated with maternal IC.
    Returns the index to exclude, or -1 if none exceed threshold.
    """
    best_idx  = -1
    best_corr = cfg.MATERNAL_ICA2_CORR_THRESH
    for i, ic in enumerate(ICs):
        if np.var(ic) < 1e-10:
            continue
        try:
            corr = abs(float(np.corrcoef(ic, maternal_ic)[0, 1]))
        except Exception:
            continue
        if corr > best_corr:
            best_corr = corr
            best_idx  = i
    if best_idx >= 0:
        print(f"[PHASE] Path B: excluding IC{best_idx+1} "
              f"(|corr| with maternal IC = {best_corr:.3f} "
              f"> threshold {cfg.MATERNAL_ICA2_CORR_THRESH})")
    return best_idx


# ── [MOD-2] PCA-adaptive n_components ────────────────────────────────────────

def determine_n_components(signals: np.ndarray,
                            variance_threshold: float = PCA_VARIANCE_THRESHOLD,
                            n_min: int = PCA_N_MIN,
                            n_max: int = PCA_N_MAX,
                            label: str = "") -> int:
    """
    [MOD-2] Count PCA components that each explain ≥ variance_threshold of
    total variance, then clip to [n_min, n_max].

    This replaces the fixed cfg.ICA_N_COMPONENTS for both ICA1 and ICA2 calls,
    making n_components data-driven rather than dataset-specific.

    On clean 5-minute ADFECGDB recordings all four components carry signal
    (≥ 5% each) → n_components = 4, no behaviour change.
    On noisy 60-second CinC2013 recordings only 2–3 components clear the
    threshold → n_components = 2 or 3, reducing the ICA solution space.
    """
    try:
        _, S, _ = np.linalg.svd(signals, full_matrices=False)
        var_ratio = (S ** 2) / (np.sum(S ** 2) + 1e-12)
        n_above   = int(np.sum(var_ratio >= variance_threshold))
        n_comp    = int(np.clip(n_above, n_min, n_max))
    except Exception:
        n_comp = n_max

    tag = f" [{label}]" if label else ""
    print(f"[PCA-ADAPT]{tag} n_components = {n_comp} "
          f"(components ≥ {variance_threshold*100:.0f}% variance: {n_above if 'n_above' in dir() else '?'})")
    return n_comp


# ── [MOD-1] Unified three-factor IC scoring ───────────────────────────────────

def _morphology_score(sig: np.ndarray, peaks: np.ndarray,
                      fs: int, win_sec: float = MORPHOLOGY_WIN_SEC) -> float:
    """
    [MOD-1] Mean pairwise correlation of beat windows centred on detected peaks.

    Returns 0.0 if fewer than MORPHOLOGY_MIN_PEAKS peaks are found (to avoid
    degenerate mean from two beats). Higher values indicate consistent QRS
    morphology across beats — a hallmark of true fetal ECG vs noise ICs.
    """
    if len(peaks) < MORPHOLOGY_MIN_PEAKS:
        return 0.0
    hw = int(win_sec * fs)
    N  = len(sig)
    windows = []
    for p in peaks:
        lo, hi = p - hw, p + hw
        if lo >= 0 and hi <= N:
            w = sig[lo:hi]
            w = w / (np.std(w) + 1e-10)
            windows.append(w)
    if len(windows) < 2:
        return 0.0
    # Sample up to 20 pairs to keep runtime bounded
    rng   = np.random.default_rng(42)
    pairs = min(len(windows) * (len(windows) - 1) // 2, 20)
    corrs = []
    idx   = list(range(len(windows)))
    for _ in range(pairs):
        i, j = rng.choice(idx, size=2, replace=False)
        n = min(len(windows[i]), len(windows[j]))
        if n < 4:
            continue
        c = float(np.corrcoef(windows[i][:n], windows[j][:n])[0, 1])
        if np.isfinite(c):
            corrs.append(abs(c))
    return float(np.mean(corrs)) if corrs else 0.0


def _maternal_penalty(ic: np.ndarray,
                      maternal_ic: np.ndarray,
                      maternal_hr: float,
                      fs: int,
                      path_b_half_weight: bool = False) -> float:
    """
    [MOD-1] Maternal leakage penalty factor.

    penalty = 1 − max(|corr(IC, maternal_IC)|,
                      |corr(IC, synthetic_maternal_harmonic)|)

    The synthetic harmonic is a sinusoid at 2 × maternal_HR / 60 Hz (i.e. the
    second harmonic of the maternal R-peak rate), constructed at the same length
    as the IC. Using np.corrcoef against a synthetic signal is precise and
    fully reproducible.

    On Path B, WSVD has already suppressed maternal content, so a residual
    small correlation should not unfairly penalise a valid fetal IC. The penalty
    is therefore applied at half-weight (×0.5) on Path B, as documented here
    and in the dissertation. The effective penalty term becomes:

        path_a: penalty = 1 − raw_penalty
        path_b: penalty = 1 − 0.5 × raw_penalty
    """
    N = len(ic)

    # Correlation with maternal IC
    try:
        corr_mat = abs(float(np.corrcoef(ic, maternal_ic)[0, 1]))
    except Exception:
        corr_mat = 0.0
    if not np.isfinite(corr_mat):
        corr_mat = 0.0

    # Correlation with synthetic second harmonic of maternal rate
    corr_harm = 0.0
    if np.isfinite(maternal_hr) and maternal_hr > 0:
        t = np.arange(N) / fs
        harmonic = np.sin(2 * np.pi * (2 * maternal_hr / 60.0) * t)
        try:
            corr_harm = abs(float(np.corrcoef(ic, harmonic)[0, 1]))
        except Exception:
            corr_harm = 0.0
        if not np.isfinite(corr_harm):
            corr_harm = 0.0

    raw_penalty = max(corr_mat, corr_harm)
    weight      = 0.5 if path_b_half_weight else 1.0
    return float(1.0 - weight * raw_penalty)


def _score_ic_unified(ic: np.ndarray,
                      peaks: np.ndarray,
                      maternal_peaks: np.ndarray,
                      maternal_ic: np.ndarray,
                      maternal_hr: float,
                      fs: int,
                      path_b: bool = False) -> float:
    """
    [MOD-1] Unified three-factor fetal IC scoring formula:

        final_score = base_score × maternal_penalty × (1 + morphology_score)

    base_score      = score_fetal_ic() — regularity × completeness × (1 + kurtosis)
    maternal_penalty = 1 − max(|corr(IC, maternal_IC)|, |corr(IC, harmonic)|)
    morphology_score = mean pairwise beat-window correlation

    Path B flag halves the maternal penalty weight (WSVD already suppressed
    maternal content; full penalty would unfairly downgrade valid fetal ICs
    with small residual maternal correlation).
    """
    base      = score_fetal_ic(ic, maternal_peaks, fs)
    mat_pen   = _maternal_penalty(ic, maternal_ic, maternal_hr, fs,
                                  path_b_half_weight=path_b)
    morph     = _morphology_score(ic, peaks, fs)
    return float(base * mat_pen * (1.0 + morph))


# ── [MOD-3] Ensemble _best_ic ─────────────────────────────────────────────────

def _best_ic(ICs_or_signals, exclude_idx, maternal_hr, fs, cfg,
             label="", expected_hr=None, min_peaks=100,
             maternal_ic=None, maternal_peaks=None,
             path_b=False,
             n_components=None):
    """
    [MOD-1 + MOD-3] Select the best fetal IC using the unified three-factor
    score, with ICA ensemble for robustness.

    For backward compatibility with run_with_ablation() which passes
    pre-computed ICs, this function scores those ICs directly with the
    three-factor formula (no ensemble re-run).
    """
    centre     = expected_hr if expected_hr is not None else cfg.FETAL_HR_CENTRE
    ICs        = ICs_or_signals
    candidates = []

    _maternal_ic     = maternal_ic if maternal_ic is not None else np.zeros(ICs.shape[1])
    _maternal_peaks  = maternal_peaks if maternal_peaks is not None else np.array([])
    _maternal_hr_val = maternal_hr if not np.isnan(maternal_hr) else 75.0

    for i, ic in enumerate(ICs):
        if i == exclude_idx:
            continue
        if np.var(ic) < 1e-10:
            if label:
                print(f"[PHASE]   {label} IC{i+1}: skipped (zero-variance pad)")
            continue

        sig_norm       = _norm(ic)
        peaks, mean_hr = _candidate_hr(sig_norm, fs, cfg)
        n_peaks        = len(peaks)
        passes_hr      = _is_fetal_hr(mean_hr, maternal_hr, cfg)
        hr_sc          = _hr_score(mean_hr, cfg, expected_hr)

        # [MOD-1] unified three-factor score
        unified = _score_ic_unified(
            sig_norm, peaks, _maternal_peaks, _maternal_ic,
            _maternal_hr_val, fs, path_b=path_b)

        candidates.append({
            "idx"      : i,
            "sig"      : sig_norm,
            "peaks"    : peaks,
            "n_peaks"  : n_peaks,
            "mean_hr"  : mean_hr,
            "passes_hr": passes_hr,
            "hr_score" : hr_sc,
            "unified"  : unified,
        })

        if label:
            ann_note = f" [ann~{centre:.0f}]" if expected_hr is not None else ""
            print(f"[PHASE]   {label} IC{i+1}: {n_peaks} peaks, "
                  f"HR={mean_hr:.1f} BPM, "
                  f"fetal_hr={'YES' if passes_hr else 'NO'}, "
                  f"unified_score={unified:.4f}{ann_note}")

    if not candidates:
        raise ValueError(f"{label}: no usable IC candidates found")

    valid = [c for c in candidates
             if c["passes_hr"] and c["n_peaks"] >= min_peaks]
    if valid:
        best = max(valid, key=lambda c: c["unified"])
        return (best["sig"], best["idx"], best["peaks"],
                best["mean_hr"], best["unified"])

    if label:
        print(f"[PHASE]   {label}: no candidate passed HR filter "
              f"-- using closest to {centre:.0f} BPM (by unified score)")
    best = max(candidates, key=lambda c: c["unified"])
    return (best["sig"], best["idx"], best["peaks"],
            best["mean_hr"], best["unified"])


def _best_ic_ensemble(mixed_signals, exclude_idx, maternal_hr, fs, cfg,
                      label="", expected_hr=None, min_peaks=100,
                      maternal_ic=None, maternal_peaks=None,
                      path_b=False, n_components=None):
    """
    [MOD-3] ICA ensemble wrapper for _best_ic.

    Runs FastICA N_ENSEMBLE times with seeds ENSEMBLE_SEEDS (0–4 by default).
    All N×k IC candidates are scored with the three-factor formula from [MOD-1].
    The global winner across all runs is returned.

    Stability bonus (ECHO dimension): After selection, cross-run correlation of
    top-scoring ICs from different seeds is computed. If mean stability ≥
    STABILITY_LOG_THRESH it is logged; this value can be fed into ECHO as a
    fourth attribution dimension (the "stability" dimension).
    """
    if n_components is None:
        n_components = PCA_N_MAX

    centre          = expected_hr if expected_hr is not None else cfg.FETAL_HR_CENTRE
    _maternal_ic    = maternal_ic if maternal_ic is not None else np.zeros(mixed_signals.shape[1])
    _maternal_peaks = maternal_peaks if maternal_peaks is not None else np.array([])
    _mat_hr         = maternal_hr if (maternal_hr is not None and not np.isnan(maternal_hr)) else 75.0

    all_candidates = []

    for seed in ENSEMBLE_SEEDS:
        try:
            from sklearn.decomposition import FastICA
            from configs import BaseConfig
            _cfg_base = BaseConfig()
            ica = FastICA(
                n_components=n_components,
                max_iter=_cfg_base.ICA_MAX_ITER,
                random_state=seed,
                tol=_cfg_base.ICA_TOL,
                whiten='arbitrary-variance',
            )
            variances   = np.var(mixed_signals, axis=1)
            active_mask = variances > 1e-10
            active_sigs = mixed_signals[active_mask]
            n_active    = active_sigs.shape[0]
            n_comp_act  = min(n_components, n_active)
            if n_active == 0:
                continue

            ica.set_params(n_components=n_comp_act)
            ICs_active = ica.fit_transform(active_sigs.T).T

            # Pad back to full n_components shape if needed
            if n_comp_act < n_components:
                N_sig = mixed_signals.shape[1]
                ICs   = np.zeros((n_components, N_sig), dtype=ICs_active.dtype)
                ICs[:n_comp_act] = ICs_active
            else:
                ICs = ICs_active

        except Exception as e:
            print(f"[ENSEMBLE] Seed {seed} ICA failed: {e}")
            continue

        for i, ic in enumerate(ICs):
            if i == exclude_idx:
                continue
            if np.var(ic) < 1e-10:
                continue

            sig_norm       = _norm(ic)
            peaks, mean_hr = _candidate_hr(sig_norm, fs, cfg)
            n_peaks        = len(peaks)
            passes_hr      = _is_fetal_hr(mean_hr, maternal_hr, cfg)
            hr_sc          = _hr_score(mean_hr, cfg, expected_hr)
            unified        = _score_ic_unified(
                sig_norm, peaks, _maternal_peaks, _maternal_ic,
                _mat_hr, fs, path_b=path_b)

            all_candidates.append({
                "seed"     : seed,
                "ic_idx"   : i,
                "sig"      : sig_norm,
                "peaks"    : peaks,
                "n_peaks"  : n_peaks,
                "mean_hr"  : mean_hr,
                "passes_hr": passes_hr,
                "hr_score" : hr_sc,
                "unified"  : unified,
            })

    if not all_candidates:
        raise ValueError(f"{label} ensemble: no usable IC candidates found across all seeds")

    if label:
        print(f"[ENSEMBLE] {label}: {len(all_candidates)} total candidates "
              f"from {len(ENSEMBLE_SEEDS)} seeds × {n_components} components")

    valid = [c for c in all_candidates
             if c["passes_hr"] and c["n_peaks"] >= min_peaks]
    pool  = valid if valid else all_candidates

    if not valid and label:
        print(f"[ENSEMBLE] {label}: no candidate passed HR filter "
              f"-- selecting by unified score across all candidates")

    # ── [FIX-1] Stability-gated selection ──────────────────────────────────────
    ENSEMBLE_MIN_WINS = 2

    top_by_seed = {}
    for c in pool:
        s = c["seed"]
        if s not in top_by_seed or c["unified"] > top_by_seed[s]["unified"]:
            top_by_seed[s] = c

    win_counts = {}
    for seed_winner in top_by_seed.values():
        key = seed_winner["ic_idx"]
        win_counts[key] = win_counts.get(key, 0) + 1

    stable_pool = [c for c in pool if win_counts.get(c["ic_idx"], 0) >= ENSEMBLE_MIN_WINS]

    if stable_pool:
        best = max(stable_pool, key=lambda c: c["unified"])
        if label:
            print(f"[ENSEMBLE-STABLE] {label}: ic_idx={best['ic_idx']} "
                  f"won in {win_counts[best['ic_idx']]}/{len(ENSEMBLE_SEEDS)} seeds")
    else:
        seed42_pool = [c for c in pool if c["seed"] == 42]
        if seed42_pool:
            best = max(seed42_pool, key=lambda c: c["unified"])
            if label:
                print(f"[ENSEMBLE-FALLBACK] {label}: no stable IC found "
                      f"-- using seed=42 result")
        else:
            best = max(pool, key=lambda c: c["unified"])
            if label:
                print(f"[ENSEMBLE-FALLBACK] {label}: no stable IC, no seed=42 "
                      f"-- using global max")

    if label:
        ann_note = f" [ann~{centre:.0f}]" if expected_hr is not None else ""
        print(f"[ENSEMBLE] {label} winner: seed={best['seed']}, "
              f"IC{best['ic_idx']+1}, {best['n_peaks']} peaks, "
              f"HR={best['mean_hr']:.1f} BPM, "
              f"unified={best['unified']:.4f}{ann_note}")

    # ── Stability bonus (ECHO fourth dimension, not used for selection) ──────
    stability_score = 0.0
    top_per_seed = {}
    for c in all_candidates:
        s = c["seed"]
        if s not in top_per_seed or c["unified"] > top_per_seed[s]["unified"]:
            top_per_seed[s] = c

    top_sigs = [v["sig"] for v in top_per_seed.values()]
    if len(top_sigs) >= 2:
        corrs = []
        for p in range(len(top_sigs)):
            for q in range(p + 1, len(top_sigs)):
                n = min(len(top_sigs[p]), len(top_sigs[q]))
                try:
                    c = float(np.corrcoef(top_sigs[p][:n], top_sigs[q][:n])[0, 1])
                    if np.isfinite(c):
                        corrs.append(abs(c))
                except Exception:
                    pass
        if corrs:
            stability_score = float(np.mean(corrs))
            if stability_score >= STABILITY_LOG_THRESH:
                print(f"[ENSEMBLE] {label} stability score = {stability_score:.3f} "
                      f"(≥ {STABILITY_LOG_THRESH} — high cross-run agreement; "
                      f"candidate for ECHO fourth dimension)")
            else:
                print(f"[ENSEMBLE] {label} stability score = {stability_score:.3f} "
                      f"(< {STABILITY_LOG_THRESH} — moderate cross-run variability)")

    return (best["sig"], best["ic_idx"], best["peaks"],
            best["mean_hr"], stability_score, best["unified"])


def _refine_peaks_on_smoothed(smoothed, rough_peaks, fs, search_radius_ms=40.0):
    radius  = int(search_radius_ms * fs / 1000)
    refined = []
    for p in rough_peaks:
        lo  = max(0, p - radius)
        hi  = min(len(smoothed), p + radius)
        window = smoothed[lo:hi]
        if smoothed[p] >= 0:
            local_max = lo + int(np.argmax(window))
        else:
            local_max = lo + int(np.argmin(window))
        refined.append(local_max)
    return np.array(refined, dtype=int)


def _apply_ekf(fetal_ic, fetal_peaks, fs, use_rts, cfg=None):
    if len(fetal_peaks) < 5:
        return fetal_ic
    hr_init = compute_hr_stats(fetal_peaks, fs)["mean_hr"]
    if np.isnan(hr_init):
        hr_init = 140.0
    ekf = FetalECGKalmanFilter(fs=fs, fetal_hr_init=hr_init)
    out = (ekf.smooth(fetal_ic, detected_peaks=fetal_peaks) if use_rts
           else ekf.filter(fetal_ic, detected_peaks=fetal_peaks)[0])
    peaks_post = detect_fetal_qrs(out, fs, cfg=cfg)
    if len(peaks_post) < max(10, len(fetal_peaks) * 0.3):
        return fetal_ic
    return out


class PHASEPipeline:
    def __init__(self, fs=None, use_rts=True, ekf_bypass=False, verbose=True,
                 dataset=None):
        self.cfg        = get_config(dataset)
        self.fs         = fs if fs is not None else self.cfg.FS
        self.use_rts    = use_rts
        self.ekf_bypass = ekf_bypass
        self.verbose    = verbose

    def _log(self, msg):
        if self.verbose:
            print(f"[PHASE] {msg}")

    def run(self, recording, save_figures=False, figures_dir="figures"):
        cfg      = self.cfg
        dataset  = recording.get("dataset", "ADFECGDB")
        rec_id   = recording["recording"]
        abd      = recording["abdomen"]
        direct   = recording.get("direct")
        fs       = recording["fs"]
        duration = recording.get("duration_sec", abd.shape[1] / fs)
        min_peaks = _min_usable_peaks(duration, cfg, dataset)

        # [FIX-PATH-2] Read confidence gate from config (default 0.05)
        confidence_gate = getattr(cfg, "CONFIDENCE_GATE_THRESHOLD",
                                  _DEFAULT_CONFIDENCE_GATE)

        self._log("=" * 55)
        self._log(f"Processing: {rec_id}  [{recording.get('dataset','?')}]")
        self._log(f"Duration: {duration:.1f}s  |  min_usable_peaks: {min_peaks}")
        self._log("=" * 55)

        # Step 1: Preprocess
        self._log("Step 1: Preprocessing...")
        abd_proc = preprocess_multichannel(abd, fs, cfg=cfg)
        dir_proc = preprocess_channel(direct, fs, cfg=cfg) if direct is not None else None

        # Step 2: ICA1 — [MOD-2] PCA-adaptive n_components
        self._log("Step 2: ICA1 (PCA-adaptive n_components)...")
        n_comp_ica1    = determine_n_components(abd_proc, label="ICA1/abd_proc")
        ICs1, _        = run_ica(abd_proc, n_components=n_comp_ica1)
        maternal_ic_idx, _ = select_maternal_ic(ICs1, fs, cfg=cfg)
        maternal_ic    = get_ic_as_signal(ICs1, maternal_ic_idx)

        # Step 3: Maternal QRS
        self._log("Step 3: Maternal QRS detection...")
        maternal_peaks = detect_maternal_qrs(maternal_ic, fs, cfg=cfg)
        mat_hr_stats   = compute_hr_stats(maternal_peaks, fs)
        maternal_hr    = mat_hr_stats["mean_hr"]
        self._log(f"  {len(maternal_peaks)} maternal peaks, HR = {maternal_hr:.1f} BPM")

        ann_path      = recording.get("annotation_path")
        ann_ext       = recording.get("annotation_ext", "qrs")
        ann_is_fetal  = recording.get("annotation_is_fetal", False)
        expected_fhr  = None

        if ann_path and ann_is_fetal:
            from preprocessing.qrs_detector import load_wfdb_annotation
            ann_peaks = load_wfdb_annotation(ann_path, ann_ext)
            if len(ann_peaks) >= 5:
                ann_stats    = compute_hr_stats(ann_peaks, fs)
                expected_fhr = ann_stats["mean_hr"]
                self._log(f"  Annotation prior: {len(ann_peaks)} peaks, "
                          f"expected fetal HR = {expected_fhr:.1f} BPM")
        elif ann_path and not ann_is_fetal:
            self._log("  Annotation skipped (not fetal ground truth)")

        # Step 4: Path A — [MOD-3] ensemble + [MOD-1] unified score
        self._log("Step 4: Path A -- ICA1 ensemble (HR-aware, three-factor score)...")
        a_sig, a_idx, a_peaks, a_hr, a_stability, a_score = _best_ic_ensemble(
            abd_proc, maternal_ic_idx, maternal_hr, fs, cfg,
            label="Path A", expected_hr=expected_fhr, min_peaks=min_peaks,
            maternal_ic=maternal_ic, maternal_peaks=maternal_peaks,
            path_b=False, n_components=n_comp_ica1)
        a_n     = len(a_peaks)
        a_valid = _is_fetal_hr(a_hr, maternal_hr, cfg)
        self._log(f"  Path A: IC{a_idx+1}, {a_n} peaks, "
                  f"HR={a_hr:.1f} BPM, valid={'YES' if a_valid else 'NO'}, "
                  f"stability={a_stability:.3f}, score={a_score:.4f}")

        # Step 5: Gaussian weights
        self._log("Step 5: Gaussian weight matrix...")
        weights = gaussian_weight_matrix(abd_proc.shape[1], maternal_peaks, fs)

        # Step 6: AW-WSVD — [MOD-4] adaptive window from recording duration
        self._log("Step 6: AW-WSVD maternal reconstruction (adaptive window)...")
        svd_explained_variance(abd_proc)
        channel_r2 = np.array([
            float(np.corrcoef(abd_proc[ch], maternal_ic)[0, 1] ** 2)
            for ch in range(abd_proc.shape[0])
        ])
        maternal_recon = adaptive_windowed_wsvd(
            abd_proc, weights, fs,
            mat_ic=maternal_ic, channel_r2=channel_r2,
            duration_sec=duration,
            cfg=cfg)    # [MOD-4]

        # Step 7: Maternal cancellation
        self._log("Step 7: Maternal cancellation...")
        residual = subtract_maternal(abd_proc, maternal_recon)

        # Step 8: Path B — [MOD-2] adaptive n_components, [MOD-3] ensemble,
        #                   [MOD-1] unified score with half-weight maternal penalty
        self._log("Step 8: Path B -- ICA2 ensemble on residual (adaptive n_comp, half-penalty)...")
        n_comp_ica2      = determine_n_components(residual, label="ICA2/residual")
        ICs2_ref, _      = run_ica(residual, n_components=n_comp_ica2)
        mat_residual_idx = _find_maternal_residual_idx(ICs2_ref, maternal_ic, cfg)

        b_sig, b_idx, b_peaks, b_hr, b_stability, b_score = _best_ic_ensemble(
            residual, mat_residual_idx, maternal_hr, fs, cfg,
            label="Path B", expected_hr=expected_fhr, min_peaks=min_peaks,
            maternal_ic=maternal_ic, maternal_peaks=maternal_peaks,
            path_b=True,             # [MOD-1] half-weight maternal penalty
            n_components=n_comp_ica2)
        b_n     = len(b_peaks)
        b_valid = _is_fetal_hr(b_hr, maternal_hr, cfg)
        self._log(f"  Path B: IC{b_idx+1}, {b_n} peaks, "
                  f"HR={b_hr:.1f} BPM, valid={'YES' if b_valid else 'NO'}, "
                  f"stability={b_stability:.3f}, score={b_score:.4f}")

        # ── Step 9: Select best path ──────────────────────────────────────────
        # [FIX-PATH-1] Score-based path selection replaces the peak-count
        # heuristic. The old logic (a_n >= b_n * PATH_A_PREFERENCE) never
        # fired on CinC2013 because both paths produce near-equal beat counts
        # on 60-second recordings — Path B won 100% of the time even when
        # Path A scored higher.
        #
        # New logic: when both paths are valid, select the one with the higher
        # unified score. PATH_A_PREFERENCE (from cinc2013.yaml = 1.0, from
        # base.py = 1.5) is now a score multiplier: Path A is chosen if
        #   a_score >= b_score * PATH_A_PREFERENCE
        # Setting PATH_A_PREFERENCE = 1.0 in cinc2013.yaml makes selection a
        # pure score comparison on this dataset, while ADFECGDB retains a 1.5×
        # preference for Path A (which tends to be slightly cleaner on long
        # recordings with good SNR).
        #
        # When only one path is valid, that path is used unconditionally.
        # When neither is valid, the higher-scoring path is used (same as
        # before but now score-based rather than HR-score-based).
        self._log("Step 9: Selecting best path (score-based)...")

        if a_valid and b_valid:
            # [FIX-PATH-1] Score comparison with PATH_A_PREFERENCE multiplier
            if a_score >= b_score * cfg.PATH_A_PREFERENCE:
                chosen_sig, chosen_peaks = a_sig, a_peaks
                chosen_path = f"A_ICA1_direct_IC{a_idx+1}_{a_hr:.0f}bpm"
                self._log(f"  Both valid — Path A score ({a_score:.4f}) >= "
                          f"Path B score ({b_score:.4f}) × {cfg.PATH_A_PREFERENCE} "
                          f"→ Path A selected")
            else:
                chosen_sig, chosen_peaks = b_sig, b_peaks
                chosen_path = f"B_WSVD_ICA2_IC{b_idx+1}_{b_hr:.0f}bpm"
                self._log(f"  Both valid — Path B score ({b_score:.4f}) > "
                          f"Path A score ({a_score:.4f}) / {cfg.PATH_A_PREFERENCE} "
                          f"→ Path B selected")
        elif a_valid:
            chosen_sig, chosen_peaks = a_sig, a_peaks
            chosen_path = f"A_ICA1_direct_IC{a_idx+1}_{a_hr:.0f}bpm"
            self._log(f"  Only Path A valid → Path A selected")
        elif b_valid:
            chosen_sig, chosen_peaks = b_sig, b_peaks
            chosen_path = f"B_WSVD_ICA2_IC{b_idx+1}_{b_hr:.0f}bpm"
            self._log(f"  Only Path B valid → Path B selected")
        else:
            # [FIX-PATH-1] Neither valid: use score comparison (was HR-score-based)
            if a_score >= b_score:
                chosen_sig, chosen_peaks = a_sig, a_peaks
                chosen_path = f"A_fallback_IC{a_idx+1}_{a_hr:.0f}bpm"
                self._log(f"  Neither valid — Path A fallback (score {a_score:.4f} "
                          f">= {b_score:.4f})")
            else:
                chosen_sig, chosen_peaks = b_sig, b_peaks
                chosen_path = f"B_fallback_IC{b_idx+1}_{b_hr:.0f}bpm"
                self._log(f"  Neither valid — Path B fallback (score {b_score:.4f} "
                          f"> {a_score:.4f})")

        chosen_score = a_score if "A_" in chosen_path else b_score

        # [FIX-PATH-2] Confidence gate: flag low-confidence outputs
        low_confidence = chosen_score < confidence_gate
        if low_confidence:
            self._log(f"  *** LOW CONFIDENCE *** chosen score {chosen_score:.4f} "
                      f"< gate {confidence_gate} — result flagged in metadata")

        self._log(f"  Selected: {chosen_path} ({len(chosen_peaks)} peaks), "
                  f"score={chosen_score:.4f}, low_confidence={low_confidence}")

        # Step 10: EKF-RTS
        self._log("Step 10: EKF-RTS morphological refinement...")
        fetal_ic_raw = chosen_sig
        if self.ekf_bypass:
            fetal_ecg = fetal_ic_raw
            self._log("  EKF bypassed")
        else:
            fetal_ecg = _apply_ekf(fetal_ic_raw, chosen_peaks, fs, self.use_rts, cfg=cfg)
            n_post = len(detect_fetal_qrs(fetal_ecg, fs, cfg=cfg))
            self._log(f"  EKF complete -- {n_post} peaks post-EKF (was {len(chosen_peaks)})")

        # Step 11: Final QRS
        self._log("Step 11: Final fetal QRS detection...")
        fetal_peaks = detect_fetal_qrs(fetal_ecg, fs, cfg=cfg)
        fet_hr = compute_hr_stats(fetal_peaks, fs)
        self._log(f"  {len(fetal_peaks)} peaks, HR = {fet_hr['mean_hr']:.1f} BPM")

        # Step 12: Evaluation
        self._log("Step 12: Evaluation...")
        if ann_path and ann_is_fetal:
            ref_peaks = load_wfdb_annotation(ann_path, ann_ext)
            self._log(f"  Reference: .{ann_ext} annotation — {len(ref_peaks)} peaks")
        elif dir_proc is not None:
            ref_peaks = detect_reference_fetal_qrs(dir_proc, fs)
            self._log(f"  Reference: Direct_1 detector — {len(ref_peaks)} peaks")
        else:
            ref_peaks = np.array([])
            self._log("  Reference: none available")

        # [FIX-PATH-4] Annotation anomaly guard: flag recordings where the
        # reference annotation is implausibly sparse (< 50% of expected minimum
        # beat count). Catches a54 (37 reference beats in 60 s) without
        # special-casing it. The result is still computed but flagged.
        sparse_annotation = False
        if len(ref_peaks) > 0:
            expected_min_beats = duration * cfg.FETAL_HR_MIN / 60.0 * 0.5
            if len(ref_peaks) < expected_min_beats:
                sparse_annotation = True
                self._log(f"  *** SPARSE ANNOTATION *** {len(ref_peaks)} reference "
                          f"peaks < expected minimum {expected_min_beats:.0f} "
                          f"— metrics for this recording may be unreliable")

        metrics = evaluate(
            fetal_ecg, dir_proc, fetal_peaks, ref_peaks, fs,
            label=f"PHASE ({rec_id})",
            tolerance_ms=cfg.EVAL_TOLERANCE_MS
        )

        # Step 13: ECHO XAI — [FIX-3] explicit has_reference flag
        self._log("Step 13: ECHO XAI...")
        has_ref  = dir_proc is not None
        echo_ref = dir_proc if has_ref else None
        echo = ECHOExplainer(
            fs=fs, maternal_peaks=maternal_peaks,
            fetal_peaks=fetal_peaks, fetal_signal=fetal_ecg,
            reference_signal=echo_ref, has_reference=has_ref)
        attribution = echo.compute_attributions()
        chosen_stability = a_stability if "A_" in chosen_path else b_stability

        metadata = {
            "ica1_pca_n_components"                  : int(n_comp_ica1),
            "ica1_maternal_ic_index"                 : int(maternal_ic_idx),
            "path_a_selected_ic_index"               : int(a_idx),
            "path_a_selected_ic_peak_count"          : int(a_n),
            "path_a_selected_ic_hr_bpm"              : float(a_hr),
            "path_a_selected_ic_is_valid"            : bool(a_valid),
            "path_a_selected_ic_stability"           : float(a_stability),
            "path_a_selected_ic_score"               : float(a_score),
            "ica2_pca_n_components"                  : int(n_comp_ica2),
            "ica2_excluded_maternal_residual_ic_index": int(mat_residual_idx),
            "path_b_selected_ic_index"               : int(b_idx),
            "path_b_selected_ic_peak_count"          : int(b_n),
            "path_b_selected_ic_hr_bpm"              : float(b_hr),
            "path_b_selected_ic_is_valid"            : bool(b_valid),
            "path_b_selected_ic_stability"           : float(b_stability),
            "path_b_selected_ic_score"               : float(b_score),
            "chosen_path_description"                : chosen_path,
            "chosen_ic_index"                        : int(a_idx if "A_" in chosen_path else b_idx),
            "chosen_ic_hr_bpm"                       : float(a_hr if "A_" in chosen_path else b_hr),
            "chosen_ic_peak_count"                   : int(len(chosen_peaks)),
            "chosen_ic_selection_score"              : float(chosen_score),
            "chosen_ic_stability"                    : float(chosen_stability),
            # [FIX-PATH-2] Confidence gate flag
            "low_confidence"                         : bool(low_confidence),
            "confidence_gate_threshold"              : float(confidence_gate),
            # [FIX-PATH-4] Annotation anomaly flag
            "sparse_annotation"                      : bool(sparse_annotation),
        }

        if isinstance(attribution, dict):
            attribution["ic_stability"] = float(chosen_stability)
        print(echo.generate_summary_stats(attribution))
        if attribution and attribution["n_beats"] > 0:
            print(echo.generate_clinical_report(0, attribution))

        if save_figures:
            self._save_figures(
                recording, abd_proc, maternal_recon, residual,
                fetal_ecg, fetal_ic_raw, dir_proc,
                fetal_peaks, ref_peaks, echo, figures_dir, rec_id)

        return {
            "recording"     : rec_id,
            "fetal_ecg"     : fetal_ecg,
            "fetal_ecg_pre" : fetal_ic_raw,
            "fetal_peaks"   : fetal_peaks,
            "maternal_peaks": maternal_peaks,
            "ref_peaks"     : ref_peaks,
            "maternal_recon": maternal_recon,
            "residual"      : residual,
            "abd_proc"      : abd_proc,
            "dir_proc"      : dir_proc,
            "weights"       : weights,
            "metrics"       : metrics,
            "echo"          : echo,
            "attribution"   : attribution,
            "chosen_path"   : chosen_path,
            "ic_stability"  : chosen_stability,
            "metadata"      : metadata,
        }

    def run_with_ablation(self, recording):
        self._log("Running ablation study...")
        fs       = recording["fs"]
        abd      = recording["abdomen"]
        direct   = recording["direct"]
        duration = recording.get("duration_sec", abd.shape[1] / fs)
        cfg      = self.cfg
        min_peaks = _min_usable_peaks(duration, cfg)

        abd_proc  = preprocess_multichannel(abd, fs, cfg=cfg)
        dir_proc  = preprocess_channel(direct, fs, cfg=cfg)
        ref_peaks = detect_reference_fetal_qrs(dir_proc, fs)
        results   = {}

        ICs1, _          = run_ica(abd_proc)
        mat_idx_blind, _ = select_maternal_ic(ICs1, fs, cfg=cfg)
        mat_ic_blind     = get_ic_as_signal(ICs1, mat_idx_blind)
        mat_peaks_blind  = detect_maternal_qrs(mat_ic_blind, fs, cfg=cfg)
        mat_hr_blind     = compute_hr_stats(mat_peaks_blind, fs)["mean_hr"]
        weights_gauss    = gaussian_weight_matrix(abd_proc.shape[1], mat_peaks_blind, fs)

        def _eval(sig, peaks, label):
            return evaluate(sig, dir_proc, peaks, ref_peaks, fs, label=label)

        def _select(ICs, excl, mat_hr, mat_ic=None, mat_peaks=None, p_b=False):
            sig, idx, peaks, hr, _ = _best_ic(
                ICs, excl, mat_hr, fs, cfg, min_peaks=min_peaks,
                maternal_ic=mat_ic, maternal_peaks=mat_peaks, path_b=p_b)
            return sig, peaks

        # Config 1: Baseline
        self._log("  Config 1: Baseline -- naive ICA + global binary WSVD...")
        mat_idx_naive   = int(np.argmax([np.var(ic) for ic in ICs1]))
        mat_ic_naive    = get_ic_as_signal(ICs1, mat_idx_naive)
        mat_peaks_naive = detect_maternal_qrs(mat_ic_naive, fs, cfg=self.cfg)
        weights_binary  = _binary_weight_matrix(abd_proc.shape[1], mat_peaks_naive, fs)
        mat_recon_1 = _global_wsvd(abd_proc, weights_binary)
        residual_1  = subtract_maternal(abd_proc, mat_recon_1)
        ICs2_1, _   = run_ica(residual_1)
        corrs       = [abs(np.corrcoef(ic, dir_proc)[0, 1]) for ic in ICs2_1]
        ic_base     = _norm(ICs2_1[int(np.argmax(corrs))])
        pks_base    = detect_fetal_qrs(ic_base, fs, cfg=self.cfg)
        results["1_Baseline_ICA_WSVD"] = _eval(ic_base, pks_base, "Baseline ICA+WSVD")

        # Config 2: + Blind IC selection
        self._log("  Config 2: + Blind IC selection...")
        mat_recon_2 = _global_wsvd(abd_proc, _binary_weight_matrix(abd_proc.shape[1], mat_peaks_blind, fs))
        residual_2  = subtract_maternal(abd_proc, mat_recon_2)
        ICs2_2, _   = run_ica(residual_2)
        excl_2      = _find_maternal_residual_idx(ICs2_2, mat_ic_blind, cfg)
        sig_2, pks_2 = _select(ICs2_2, excl_2, mat_hr_blind, mat_ic_blind, mat_peaks_blind)
        results["2_Blind_IC_Selection"] = _eval(sig_2, pks_2, "+Blind IC Selection")

        # Config 3: + Gaussian weights
        self._log("  Config 3: + Gaussian weights...")
        mat_recon_3 = _global_wsvd(abd_proc, weights_gauss)
        residual_3  = subtract_maternal(abd_proc, mat_recon_3)
        ICs2_3, _   = run_ica(residual_3)
        excl_3      = _find_maternal_residual_idx(ICs2_3, mat_ic_blind, cfg)
        sig_3, pks_3 = _select(ICs2_3, excl_3, mat_hr_blind, mat_ic_blind, mat_peaks_blind)
        results["3_Gaussian_Weights"] = _eval(sig_3, pks_3, "+Gaussian Weights")

        # Config 4: + Adaptive windowed WSVD  ([MOD-4] duration passed through)
        self._log("  Config 4: + Adaptive Windowed WSVD...")
        channel_r2  = np.array([float(np.corrcoef(abd_proc[ch], mat_ic_blind)[0, 1] ** 2)
                                 for ch in range(abd_proc.shape[0])])
        mat_recon_4 = adaptive_windowed_wsvd(abd_proc, weights_gauss, fs,
                                              mat_ic=mat_ic_blind, channel_r2=channel_r2,
                                              duration_sec=duration,
                                              cfg=cfg)   # [MOD-4]
        residual_4  = subtract_maternal(abd_proc, mat_recon_4)
        ICs2_4, _   = run_ica(residual_4)
        excl_4      = _find_maternal_residual_idx(ICs2_4, mat_ic_blind, cfg)
        sig_4, pks_4 = _select(ICs2_4, excl_4, mat_hr_blind, mat_ic_blind, mat_peaks_blind, p_b=True)
        results["4_Adaptive_WSVD"] = _eval(sig_4, pks_4, "+Adaptive WSVD")

        # Config 5: + EKF-RTS
        self._log("  Config 5: Full PHASE (+ EKF-RTS)...")
        fetal_ecg_5 = _apply_ekf(sig_4, pks_4, fs, use_rts=True, cfg=self.cfg)
        pks_5       = detect_fetal_qrs(fetal_ecg_5, fs, cfg=self.cfg)
        if len(pks_5) < max(10, len(pks_4) * 0.3):
            fetal_ecg_5, pks_5 = sig_4, pks_4
        results["5_PHASE_Full"] = _eval(fetal_ecg_5, pks_5, "PHASE Full")

        return results

    def _save_figures(self, recording, abd_proc, maternal_recon, residual,
                      fetal_ecg, fetal_ic_raw, dir_proc,
                      fetal_peaks, ref_peaks, echo, figures_dir, rec_id):
        from utils.visualization import (
            plot_preprocessing, plot_maternal_cancellation,
            plot_fetal_comparison, plot_ekf_refinement
        )
        fdir = Path(figures_dir)
        fdir.mkdir(parents=True, exist_ok=True)

        plot_preprocessing(
            recording["abdomen"][0], abd_proc[0], self.fs,
            save_path=str(fdir / f"{rec_id}_preprocessing.png"))

        plot_maternal_cancellation(
            abd_proc, maternal_recon, residual, self.fs,
            save_path=str(fdir / f"{rec_id}_maternal_cancellation.png"))

        has_direct = dir_proc is not None and hasattr(dir_proc, '__len__') and len(dir_proc) > 0
        plot_fetal_comparison(
            fetal_ecg,
            dir_proc if has_direct else None,
            fetal_peaks,
            ref_peaks if has_direct else None,
            self.fs,
            save_path=str(fdir / f"{rec_id}_fetal_comparison.png"))

        plot_ekf_refinement(
            fetal_ic_raw, fetal_ecg,
            dir_proc if has_direct else None,
            self.fs,
            save_path=str(fdir / f"{rec_id}_ekf_refinement.png"))

        echo.plot_attribution_heatmap(
            save_path=str(fdir / f"{rec_id}_echo_attribution.png"))
        self._log(f"Figures saved to {figures_dir}/")


# ── Ablation helpers ─────────────────────────────────────────────────────────

def _binary_weight_matrix(n_samples, qrs_peaks, fs, window_sec=0.1):
    weights = np.ones(n_samples) * 0.1
    hw = int(window_sec * fs)
    for peak in qrs_peaks:
        lo = max(0, peak - hw)
        hi = min(n_samples, peak + hw)
        weights[lo:hi] = 1.0
    return weights


def _global_wsvd(abd_signals, weights, n_components=2):
    weighted = abd_signals * weights[np.newaxis, :]
    U, S, VT = np.linalg.svd(weighted, full_matrices=False)
    recon = np.zeros_like(abd_signals)
    for i in range(min(n_components, len(S))):
        recon += np.outer(U[:, i], S[i] * VT[i, :])
    return recon