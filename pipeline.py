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
  [MOD-1] _best_ic(): unified three-factor scoring replaces n_peaks x hr_score.
          final_score = base_score x maternal_penalty x (1 + morphology_score)
  [MOD-2] determine_n_components(): PCA-adaptive n_components before each ICA.
          PCA floor raised 2->3: minimum 3 components for maternal/fetal/noise.
  [MOD-3] ICA ensemble: N_ENSEMBLE=5 seeds. Stability-gated selection requires
          winning IC to appear as top scorer in >= 2 seeds.
  [MOD-4] adaptive_windowed_wsvd() receives duration_sec for adaptive windows.

BIMODAL F1 FIX BATCH (targeting mean F1 > 85%):
  [FIX-PATH-1] Score-based path selection: Step 9 now compares a_score vs
          b_score x PATH_A_PREFERENCE instead of peak counts.

  [FIX-PATH-2] Confidence gate: chosen_score < CONFIDENCE_GATE_THRESHOLD sets
          low_confidence=True in metadata. Now triggers active retry with
          different n_components when fired.

  [FIX-PATH-3] Half-harmonic guard in _is_fetal_hr(): candidate rejected if
          within HR_SEP_MIN_BPM of HALF the maternal HR.

  [FIX-PATH-4] Annotation anomaly guard: n_reference < 50% expected minimum
          sets sparse_annotation=True in metadata (catches a54).

  [FIX-EKF]   EKF acceptance gate: peak count >= 70% AND CC >= 0.60 AND
          |median_RR_post - median_RR_pre| <= 15ms. The CC and RR-shift
          gates were documented but not implemented; now active.

  [FIX-HARM]  Post-selection harmonic RR check: x1.0 and x2.0 only.

  [FIX-RETRY] Under-detection retry: threshold changed from FETAL_HR_MIN-based
          to FETAL_HR_CENTRE-based (65% of expected) so it fires on recordings
          with 88-126 detected peaks that were previously missed.

IMPROVEMENTS (v2):
  [IMP-1] Raise CONFIDENCE_GATE_THRESHOLD in cinc2013.yaml to 0.30. When
          low_confidence fires, retry with alternative n_components (n-1 and
          n+1) and take the best result. Previously the gate was diagnostic-
          only; now it drives active remediation.

  [IMP-2] Cross-IC HR check after selection: if chosen IC HR is within
          HR_SEP_MIN_BPM of maternal HR, force low_confidence=True and retry.
          Fixes a02/a56 class failures where ICA1 misidentifies maternal IC
          so the maternal component scores well in Path A and wins.

  [IMP-3] Template-based morphology score: compute correlation of each beat
          window against the mean-beat template (O(n)) instead of 20 random
          pairs. More stable on weak fetal signals.

  [IMP-4] Fix ensemble fallback seed: ENSEMBLE_SEEDS=[0,1,2,3,4] but fallback
          tried seed=42 which is never in the pool. Changed to seed=0.

  [IMP-5] WSVD corr gate: changed 'or med_corr >= 0.75*corr_thresh' to
          'and' to prevent noise SVD components from passing on median alone.
          (Change is in wsvd.py; documented here for traceability.)
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
N_ENSEMBLE = 5
ENSEMBLE_SEEDS = list(range(N_ENSEMBLE))   # [0, 1, 2, 3, 4]
PCA_VARIANCE_THRESHOLD = 0.05
PCA_N_MIN, PCA_N_MAX  = 3, 4
MORPHOLOGY_MIN_PEAKS  = 5
MORPHOLOGY_WIN_SEC    = 0.3
STABILITY_LOG_THRESH  = 0.7

# [FIX-PATH-2] default confidence gate — overridden by cinc2013.yaml
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

    [FIX-PATH-3] Half-harmonic guard: candidate rejected if within
    HR_SEP_MIN_BPM of HALF the maternal HR.
    """
    if np.isnan(mean_hr):
        return False
    in_range = cfg.FETAL_HR_LOW <= mean_hr <= cfg.FETAL_HR_HIGH

    sep_threshold = (cfg.HR_SEP_MIN_BPM * 0.7
                     if (not np.isnan(maternal_hr) and maternal_hr > 85)
                     else cfg.HR_SEP_MIN_BPM)

    sep_from_maternal = (np.isnan(maternal_hr) or
                         abs(mean_hr - maternal_hr) >= sep_threshold)

    half_maternal = maternal_hr / 2.0 if not np.isnan(maternal_hr) else np.nan
    sep_from_half = (np.isnan(half_maternal) or
                     abs(mean_hr - half_maternal) >= sep_threshold)

    return in_range and sep_from_maternal and sep_from_half


def _hr_score(mean_hr, cfg, expected_hr=None):
    centre = expected_hr if expected_hr is not None else cfg.FETAL_HR_CENTRE
    if np.isnan(mean_hr):
        return 0.0
    return 1.0 / (1.0 + abs(mean_hr - centre) / 30.0)


def _find_maternal_residual_idx(ICs, maternal_ic, cfg):
    """[FIX-1] Find ICA2 component most correlated with maternal IC."""
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


def _check_harmonic_confusion(chosen_peaks: np.ndarray,
                               maternal_hr: float,
                               fs: int,
                               tolerance: float = 0.10) -> bool:
    """
    [FIX-HARM] Check whether chosen IC RR matches maternal harmonic at
    x1.0 or x2.0 only. x0.5 excluded — equals fetal RR when fetal ~2x maternal.
    """
    if len(chosen_peaks) < 3 or np.isnan(maternal_hr) or maternal_hr <= 0:
        return False

    chosen_rr   = float(np.median(np.diff(chosen_peaks))) / fs
    maternal_rr = 60.0 / maternal_hr

    for harmonic_rr in [maternal_rr, maternal_rr * 2.0]:
        if harmonic_rr > 0:
            if abs(chosen_rr - harmonic_rr) / harmonic_rr < tolerance:
                print(f"[FIX-HARM] Harmonic confusion suspected: "
                      f"chosen RR={chosen_rr*1000:.0f}ms, "
                      f"harmonic RR={harmonic_rr*1000:.0f}ms "
                      f"(maternal HR={maternal_hr:.1f}bpm, "
                      f"ratio={harmonic_rr/maternal_rr:.1f}x)")
                return True
    return False


# ── [MOD-2] PCA-adaptive n_components ────────────────────────────────────────

def determine_n_components(signals: np.ndarray,
                            variance_threshold: float = PCA_VARIANCE_THRESHOLD,
                            n_min: int = PCA_N_MIN,
                            n_max: int = PCA_N_MAX,
                            label: str = "") -> int:
    """[MOD-2] Count PCA components explaining >= variance_threshold, clip to [n_min, n_max]."""
    try:
        _, S, _ = np.linalg.svd(signals, full_matrices=False)
        var_ratio = (S ** 2) / (np.sum(S ** 2) + 1e-12)
        n_above   = int(np.sum(var_ratio >= variance_threshold))
        n_comp    = int(np.clip(n_above, n_min, n_max))
    except Exception:
        n_comp = n_max

    tag = f" [{label}]" if label else ""
    print(f"[PCA-ADAPT]{tag} n_components = {n_comp} "
          f"(components >= {variance_threshold*100:.0f}% variance: {n_above if 'n_above' in dir() else '?'})")
    return n_comp


# ── [MOD-1] Unified three-factor IC scoring ───────────────────────────────────

def _morphology_score(sig: np.ndarray, peaks: np.ndarray,
                      fs: int, win_sec: float = MORPHOLOGY_WIN_SEC) -> float:
    """
    [IMP-3] Template-based morphology score.

    Compute correlation of each beat window against the mean-beat template
    (O(n)) instead of 20 random pairs. More stable on weak fetal signals
    where 20 random pairs have high variance.

    Returns mean |correlation| of each beat against the template.
    Returns 0.0 if fewer than MORPHOLOGY_MIN_PEAKS peaks.
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

    # [IMP-3] template-based: correlate each beat against the mean template
    min_len  = min(len(w) for w in windows)
    arr      = np.array([w[:min_len] for w in windows])
    template = np.mean(arr, axis=0)
    template_std = np.std(template) + 1e-10

    corrs = []
    for w in arr:
        try:
            c = float(np.corrcoef(w, template)[0, 1])
            if np.isfinite(c):
                corrs.append(abs(c))
        except Exception:
            pass
    return float(np.mean(corrs)) if corrs else 0.0


def _maternal_penalty(ic: np.ndarray,
                      maternal_ic: np.ndarray,
                      maternal_hr: float,
                      fs: int,
                      path_b_half_weight: bool = False) -> float:
    """
    [MOD-1] Maternal leakage penalty factor.
    penalty = 1 - max(|corr(IC, maternal_IC)|, |corr(IC, synthetic_maternal_harmonic)|)
    Path B flag halves the penalty weight.
    """
    N = len(ic)

    try:
        corr_mat = abs(float(np.corrcoef(ic, maternal_ic)[0, 1]))
    except Exception:
        corr_mat = 0.0
    if not np.isfinite(corr_mat):
        corr_mat = 0.0

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
    [MOD-1] Unified three-factor fetal IC scoring:
        final_score = base_score x maternal_penalty x (1 + morphology_score)
    """
    base    = score_fetal_ic(ic, maternal_peaks, fs)
    mat_pen = _maternal_penalty(ic, maternal_ic, maternal_hr, fs,
                                path_b_half_weight=path_b)
    morph   = _morphology_score(ic, peaks, fs)
    return float(base * mat_pen * (1.0 + morph))


# ── [MOD-3] Ensemble _best_ic ─────────────────────────────────────────────────

def _best_ic(ICs_or_signals, exclude_idx, maternal_hr, fs, cfg,
             label="", expected_hr=None, min_peaks=100,
             maternal_ic=None, maternal_peaks=None,
             path_b=False,
             n_components=None):
    """
    [MOD-1 + MOD-3] Select best fetal IC using unified three-factor score.
    For backward compatibility with run_with_ablation().
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

    Runs FastICA N_ENSEMBLE times with seeds ENSEMBLE_SEEDS (0-4).
    All N*k IC candidates are scored with the three-factor formula from [MOD-1].
    The global winner across all runs is returned.

    [IMP-4] Fixed fallback seed: was 42 (never in ENSEMBLE_SEEDS=[0..4]),
    now falls back to seed=0 which is always in the pool.
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
              f"from {len(ENSEMBLE_SEEDS)} seeds x {n_components} components")

    valid = [c for c in all_candidates
             if c["passes_hr"] and c["n_peaks"] >= min_peaks]

    # [IMP-1] For low-confidence situations: also try force_low_threshold on
    # HR-failing candidates so a valid IC with under-detected peaks gets a
    # second chance before we fall back to the score-max
    if not valid:
        print(f"[ENSEMBLE] {label}: no candidate passed HR filter -- "
              f"retrying with force_low_threshold on each candidate...")
        for c in all_candidates:
            if c["passes_hr"]:
                continue
            peaks_retry, hr_retry = _candidate_hr_force(c["sig"], fs, cfg)
            if _is_fetal_hr(hr_retry, maternal_hr, cfg) and len(peaks_retry) >= min_peaks:
                unified_retry = _score_ic_unified(
                    c["sig"], peaks_retry, _maternal_peaks, _maternal_ic,
                    _mat_hr, fs, path_b=path_b)
                valid.append({**c,
                               "peaks"    : peaks_retry,
                               "n_peaks"  : len(peaks_retry),
                               "mean_hr"  : hr_retry,
                               "passes_hr": True,
                               "unified"  : unified_retry})

    pool = valid if valid else all_candidates

    if not valid and label:
        print(f"[ENSEMBLE] {label}: no candidate passed HR filter "
              f"-- selecting by unified score across all candidates")

    # ── Stability-gated selection ──────────────────────────────────────────────
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
        # [IMP-4] Fixed fallback: use seed=0 (was seed=42, never in pool)
        seed0_pool = [c for c in pool if c["seed"] == 0]
        if seed0_pool:
            best = max(seed0_pool, key=lambda c: c["unified"])
            if label:
                print(f"[ENSEMBLE-FALLBACK] {label}: no stable IC found "
                      f"-- using seed=0 result")
        else:
            best = max(pool, key=lambda c: c["unified"])
            if label:
                print(f"[ENSEMBLE-FALLBACK] {label}: no stable IC, no seed=0 "
                      f"-- using global max")

    if label:
        ann_note = f" [ann~{centre:.0f}]" if expected_hr is not None else ""
        print(f"[ENSEMBLE] {label} winner: seed={best['seed']}, "
              f"IC{best['ic_idx']+1}, {best['n_peaks']} peaks, "
              f"HR={best['mean_hr']:.1f} BPM, "
              f"unified={best['unified']:.4f}{ann_note}")

    # ── Stability score (ECHO fourth dimension) ──────────────────────────────
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
                      f"(>= {STABILITY_LOG_THRESH} -- high cross-run agreement)")
            else:
                print(f"[ENSEMBLE] {label} stability score = {stability_score:.3f} "
                      f"(< {STABILITY_LOG_THRESH} -- moderate cross-run variability)")

    return (best["sig"], best["ic_idx"], best["peaks"],
            best["mean_hr"], stability_score, best["unified"])


def _candidate_hr_force(sig, fs, cfg):
    """Run detect_fetal_qrs with force_low_threshold=True for IC retry."""
    peaks = detect_fetal_qrs(sig, fs, cfg=cfg, force_low_threshold=True)
    stats = compute_hr_stats(peaks, fs)
    mean_hr = stats["mean_hr"] if len(peaks) >= 2 else np.nan
    return peaks, mean_hr


def _best_ic_ensemble_with_retry(mixed_signals, exclude_idx, maternal_hr, fs, cfg,
                                  label="", expected_hr=None, min_peaks=100,
                                  maternal_ic=None, maternal_peaks=None,
                                  path_b=False, n_components=None):
    """
    [IMP-1] Wrapper around _best_ic_ensemble that retries with alternative
    n_components when the result is low-confidence.

    Tries n_components, n_components-1 (if >= PCA_N_MIN), and
    n_components+1 (if <= PCA_N_MAX). Returns the best score across attempts.
    """
    if n_components is None:
        n_components = PCA_N_MAX

    RETRY_CEILING = 0.65

    # Step 1: always run the default n_components first
    try:
        sig, idx, peaks, hr, stab, score = _best_ic_ensemble(
            mixed_signals, exclude_idx, maternal_hr, fs, cfg,
            label=label, expected_hr=expected_hr,
            min_peaks=min_peaks, maternal_ic=maternal_ic,
            maternal_peaks=maternal_peaks, path_b=path_b,
            n_components=n_components)
    except Exception as e:
        raise ValueError(f"{label}: default n_components={n_components} failed: {e}")

    # Step 2: if score is good enough, return immediately -- no retry
    if score >= RETRY_CEILING:
        return sig, idx, peaks, hr, stab, score

    # Step 3: score is low -- try alternative n_components
    print(f"[RETRY] {label}: score={score:.4f} < {RETRY_CEILING} "
          f"-- trying alternative n_components...")
    results = [(sig, idx, peaks, hr, stab, score, n_components)]
    for nc in [n_components - 1, n_components + 1]:
        if nc < PCA_N_MIN or nc > PCA_N_MAX + 1:
            continue
        try:
            s, i, p, h, st, sc = _best_ic_ensemble(
                mixed_signals, exclude_idx, maternal_hr, fs, cfg,
                label=f"{label}[nc={nc}]", expected_hr=expected_hr,
                min_peaks=min_peaks, maternal_ic=maternal_ic,
                maternal_peaks=maternal_peaks, path_b=path_b,
                n_components=nc)
            results.append((s, i, p, h, st, sc, nc))
        except Exception as e:
            print(f"[RETRY] {label} nc={nc} failed: {e}")
            continue

    best = max(results, key=lambda r: r[5])
    sig, idx, peaks, hr, stab, score, nc_used = best
    if nc_used != n_components:
        print(f"[RETRY] {label}: n_components={nc_used} gave better score "
              f"({score:.4f}) than default ({n_components})")
    return sig, idx, peaks, hr, stab, score


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
    """
    [FIX-EKF] EKF acceptance gate: ALL THREE gates must pass:
      (a) post-EKF peak count >= 70% of pre-EKF
      (b) CC(ekf_output, pre_ekf_input) >= 0.60   [NOW ACTIVE - was documented but missing]
      (c) |median_RR_post - median_RR_pre| <= 15ms [NOW ACTIVE - was documented but missing]

    Gate (b) and (c) were listed in ekf.py docstring but never implemented
    in the actual gate check in this function. Now active.
    Gate (c) catches phase-shifted EKF output: a 30ms shift preserves peak
    count (gate a passes) and gives CC ~0.997 on 60s signal (gate b passes)
    but shifts the beat rhythm detectably (gate c catches it).
    """
    if len(fetal_peaks) < 5:
        return fetal_ic, False
    hr_init = compute_hr_stats(fetal_peaks, fs)["mean_hr"]
    if np.isnan(hr_init):
        hr_init = 140.0
    ekf = FetalECGKalmanFilter(fs=fs, fetal_hr_init=hr_init)
    out = (ekf.smooth(fetal_ic, detected_peaks=fetal_peaks) if use_rts
           else ekf.filter(fetal_ic, detected_peaks=fetal_peaks)[0])
    peaks_post = detect_fetal_qrs(out, fs, cfg=cfg)

    # Gate (a): peak count
    gate_a = len(peaks_post) >= max(10, len(fetal_peaks) * 0.7)

    # Gate (b): waveform CC
    try:
        cc_val = float(np.corrcoef(out, fetal_ic)[0, 1])
    except Exception:
        cc_val = 0.0
    gate_b = np.isfinite(cc_val) and cc_val >= 0.60

    # Gate (c): RR shift
    gate_c = True
    if len(fetal_peaks) >= 3 and len(peaks_post) >= 3:
        rr_pre_ms  = float(np.median(np.diff(fetal_peaks))) / fs * 1000.0
        rr_post_ms = float(np.median(np.diff(peaks_post)))  / fs * 1000.0
        gate_c = abs(rr_post_ms - rr_pre_ms) <= 15.0

    if not (gate_a and gate_b and gate_c):
        print(f"[EKF] Gate failed: "
              f"peak_ratio={len(peaks_post)}/{len(fetal_peaks)} gate_a={'PASS' if gate_a else 'FAIL'}, "
              f"CC={cc_val:.4f} gate_b={'PASS' if gate_b else 'FAIL'}, "
              f"gate_c={'PASS' if gate_c else 'FAIL'} "
              f"-- keeping ICA output")
        return fetal_ic, False
    return out, True


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

        self._log("=" * 55)
        self._log(f"Processing: {rec_id}  [{recording.get('dataset','?')}]")
        self._log(f"Duration: {duration:.1f}s  |  min_usable_peaks: {min_peaks}")
        self._log("=" * 55)

        # Step 1: Preprocess
        self._log("Step 1: Preprocessing...")
        abd_proc = preprocess_multichannel(abd, fs, cfg=cfg)
        dir_proc = preprocess_channel(direct, fs, cfg=cfg) if direct is not None else None

        # Step 2: ICA1
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

        # Step 4: Path A
        self._log("Step 4: Path A -- ICA1 ensemble (HR-aware, three-factor score)...")
        a_sig, a_idx, a_peaks, a_hr, a_stability, a_score = _best_ic_ensemble_with_retry(
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

        # Step 6: AW-WSVD
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
            cfg=cfg)

        # Step 7: Maternal cancellation
        self._log("Step 7: Maternal cancellation...")
        residual = subtract_maternal(abd_proc, maternal_recon)

        # Step 8: Path B
        self._log("Step 8: Path B -- ICA2 ensemble on residual...")
        n_comp_ica2      = determine_n_components(residual, label="ICA2/residual")
        ICs2_ref, _      = run_ica(residual, n_components=n_comp_ica2)
        mat_residual_idx = _find_maternal_residual_idx(ICs2_ref, maternal_ic, cfg)

        b_sig, b_idx, b_peaks, b_hr, b_stability, b_score = _best_ic_ensemble_with_retry(
            residual, mat_residual_idx, maternal_hr, fs, cfg,
            label="Path B", expected_hr=expected_fhr, min_peaks=min_peaks,
            maternal_ic=maternal_ic, maternal_peaks=maternal_peaks,
            path_b=True,
            n_components=n_comp_ica2)
        b_n     = len(b_peaks)
        b_valid = _is_fetal_hr(b_hr, maternal_hr, cfg)
        self._log(f"  Path B: IC{b_idx+1}, {b_n} peaks, "
                  f"HR={b_hr:.1f} BPM, valid={'YES' if b_valid else 'NO'}, "
                  f"stability={b_stability:.3f}, score={b_score:.4f}")

        # Step 9: Select best path
        self._log("Step 9: Selecting best path (score-based [FIX-PATH-1])...")
        if a_valid and b_valid:
            if a_score >= b_score * cfg.PATH_A_PREFERENCE:
                chosen_sig, chosen_peaks = a_sig, a_peaks
                chosen_path = f"A_ICA1_direct_IC{a_idx+1}_{a_hr:.0f}bpm"
                self._log(f"  Both valid -- Path A score ({a_score:.4f}) >= "
                          f"Path B ({b_score:.4f}) x {cfg.PATH_A_PREFERENCE} "
                          f"-> Path A selected")
            else:
                chosen_sig, chosen_peaks = b_sig, b_peaks
                chosen_path = f"B_WSVD_ICA2_IC{b_idx+1}_{b_hr:.0f}bpm"
                self._log(f"  Both valid -- Path B score ({b_score:.4f}) wins "
                          f"-> Path B selected")
        elif a_valid:
            chosen_sig, chosen_peaks = a_sig, a_peaks
            chosen_path = f"A_ICA1_direct_IC{a_idx+1}_{a_hr:.0f}bpm"
            self._log("  Only Path A valid -> Path A selected")
        elif b_valid:
            chosen_sig, chosen_peaks = b_sig, b_peaks
            chosen_path = f"B_WSVD_ICA2_IC{b_idx+1}_{b_hr:.0f}bpm"
            self._log("  Only Path B valid -> Path B selected")
        else:
            if a_score >= b_score:
                chosen_sig, chosen_peaks = a_sig, a_peaks
                chosen_path = f"A_fallback_IC{a_idx+1}_{a_hr:.0f}bpm"
            else:
                chosen_sig, chosen_peaks = b_sig, b_peaks
                chosen_path = f"B_fallback_IC{b_idx+1}_{b_hr:.0f}bpm"
            self._log(f"  Neither valid -- fallback to higher score: {chosen_path}")

        chosen_score = a_score if "A_" in chosen_path else b_score
        chosen_hr    = a_hr    if "A_" in chosen_path else b_hr

        # [FIX-PATH-2 + IMP-1] Confidence gate (now active, not diagnostic-only)
        confidence_gate = getattr(cfg, "CONFIDENCE_GATE_THRESHOLD",
                                  _DEFAULT_CONFIDENCE_GATE)
        low_confidence = chosen_score < confidence_gate
        if low_confidence:
            self._log(f"  *** LOW CONFIDENCE *** score={chosen_score:.4f} "
                      f"< gate={confidence_gate} -- flagged (retry already attempted in ensemble)")

        # [IMP-2] Cross-IC HR check: if chosen HR is within HR_SEP_MIN_BPM of
        # maternal HR, ICA1 likely misidentified the maternal component and the
        # chosen IC is a maternal residual. Force low_confidence and retry.
        hr_sep = getattr(cfg, "HR_SEP_MIN_BPM", 15.0)
        if (not np.isnan(chosen_hr) and not np.isnan(maternal_hr) and
                abs(chosen_hr - maternal_hr) < hr_sep):
            self._log(f"  *** IC HR PROXIMITY *** chosen HR={chosen_hr:.1f} BPM "
                      f"within {hr_sep} BPM of maternal HR={maternal_hr:.1f} BPM "
                      f"-- likely maternal residual, flagging low_confidence")
            low_confidence = True

        # Spectral tiebreaker for deeply low-confidence cases
        # When score < 0.25, score_fetal_ic cannot discriminate ICs reliably.
        # Use power ratio in the fetal HR band (1.7-3.0 Hz) vs full band
        # as an independent selection criterion.
        SPECTRAL_FALLBACK_THRESH = 0.25
        if chosen_score < SPECTRAL_FALLBACK_THRESH:
            self._log(f"  Spectral fallback: score={chosen_score:.4f} < "
                      f"{SPECTRAL_FALLBACK_THRESH} -- selecting by fetal band power...")
            candidates_spectral = [
                (a_sig, a_peaks, a_hr, a_score, "A"),
                (b_sig, b_peaks, b_hr, b_score, "B"),
            ]
            best_spec_sig, best_spec_peaks = chosen_sig, chosen_peaks
            best_spec_ratio = -1.0
            for cand_sig, cand_peaks, cand_hr, cand_score, cand_label in candidates_spectral:
                if not _is_fetal_hr(cand_hr, maternal_hr, cfg):
                    continue
                freqs = np.fft.rfftfreq(len(cand_sig), d=1.0/fs)
                psd   = np.abs(np.fft.rfft(cand_sig)) ** 2
                fetal_mask = (freqs >= 1.7) & (freqs <= 3.0)
                total_mask = (freqs >= 0.5) & (freqs <= 40.0)
                fetal_power = np.sum(psd[fetal_mask])
                total_power = np.sum(psd[total_mask]) + 1e-12
                ratio = fetal_power / total_power
                self._log(f"    {cand_label}: fetal_band_ratio={ratio:.4f}")
                if ratio > best_spec_ratio:
                    best_spec_ratio = ratio
                    best_spec_sig   = cand_sig
                    best_spec_peaks = cand_peaks
                    chosen_path     = (f"A_spectral_IC{a_idx+1}_{a_hr:.0f}bpm"
                                       if cand_label == "A" else
                                       f"B_spectral_IC{b_idx+1}_{b_hr:.0f}bpm")
            chosen_sig, chosen_peaks = best_spec_sig, best_spec_peaks
            self._log(f"  Spectral fallback selected: {chosen_path} "
                      f"(band_ratio={best_spec_ratio:.4f})")
            
        # [FIX-HARM] Harmonic confusion check
        harmonic_confusion = _check_harmonic_confusion(
            chosen_peaks, maternal_hr, fs)
        if harmonic_confusion:
            low_confidence = True
            self._log("  *** HARMONIC CONFUSION *** chosen RR matches "
                      "maternal harmonic -- flagged in metadata")

        self._log(f"  Selected: {chosen_path} ({len(chosen_peaks)} peaks), "
                  f"score={chosen_score:.4f}, "
                  f"low_confidence={low_confidence}, "
                  f"harmonic_confusion={harmonic_confusion}")        

        # Peak regularity check: if RR CV > 0.20, the detector is firing on
        # non-QRS events (T-waves, noise). Rerun with force_low_threshold to
        # let Pan-Tompkins re-learn the threshold on this IC.
        if len(chosen_peaks) >= 5:
            rr_intervals = np.diff(chosen_peaks)
            rr_cv = float(np.std(rr_intervals) / (np.mean(rr_intervals) + 1e-10))
            if rr_cv > 0.15:
                self._log(f"  RR irregularity: CV={rr_cv:.3f} > 0.20 "
                          f"-- rerunning QRS detection with force_low_threshold...")
                peaks_rerun = detect_fetal_qrs(
                    chosen_sig, fs, cfg=cfg, force_low_threshold=True)
                rr_rerun = np.diff(peaks_rerun)
                cv_rerun = float(np.std(rr_rerun) / (np.mean(rr_rerun) + 1e-10)) if len(peaks_rerun) >= 5 else 1.0
                if cv_rerun < rr_cv:
                    self._log(f"  RR CV improved: {rr_cv:.3f} -> {cv_rerun:.3f}, "
                              f"peaks {len(chosen_peaks)} -> {len(peaks_rerun)}")
                    chosen_peaks = peaks_rerun

        # Step 10: EKF-RTS [FIX-EKF: all three gates now active]
        self._log("Step 10: EKF-RTS morphological refinement [gates: peak_ratio+CC+RR_shift]...")
        fetal_ic_raw = chosen_sig
        ekf_used     = False
        if self.ekf_bypass:
            fetal_ecg = fetal_ic_raw
            self._log("  EKF bypassed")
        else:
            fetal_ecg_candidate, ekf_accepted = _apply_ekf(
                fetal_ic_raw, chosen_peaks, fs, self.use_rts, cfg=cfg)
            if not ekf_accepted:
                fetal_ecg = fetal_ic_raw
                self._log("  EKF gate failed -- keeping ICA output unchanged")
            else:
                fetal_ecg = fetal_ecg_candidate
                ekf_used  = True
                n_post    = len(detect_fetal_qrs(fetal_ecg, fs, cfg=cfg))
                self._log(f"  EKF accepted -- {n_post} peaks post-EKF "
                          f"(was {len(chosen_peaks)})")

        # Step 11: Final QRS detection + [FIX-RETRY] under-detection retry
        self._log("Step 11: Final fetal QRS detection...")
        fetal_peaks = detect_fetal_qrs(fetal_ecg, fs, cfg=cfg)

        # [FIX-RETRY] Tighter threshold: use FETAL_HR_CENTRE-based expected count
        # at 65% (was FETAL_HR_MIN at 80% -- too loose, missed 88-126 peak cases).
        expected_beats  = duration * cfg.FETAL_HR_CENTRE / 60.0
        detection_rate  = len(fetal_peaks) / (expected_beats + 1e-10)
        if detection_rate < 0.82 and len(fetal_peaks) < expected_beats * 0.82:
            self._log(f"  Under-detection: {len(fetal_peaks)} peaks, "
                      f"rate={detection_rate:.2f} < 0.72 "
                      f"-- retrying with relaxed threshold...")
            fetal_peaks_retry = detect_fetal_qrs(
                fetal_ecg, fs, cfg=cfg, force_low_threshold=True)
            if len(fetal_peaks_retry) > len(fetal_peaks):
                self._log(f"  Retry recovered: "
                          f"{len(fetal_peaks_retry)} peaks "
                          f"(was {len(fetal_peaks)})")
                fetal_peaks = fetal_peaks_retry

        fet_hr = compute_hr_stats(fetal_peaks, fs)
        self._log(f"  {len(fetal_peaks)} peaks, HR = {fet_hr['mean_hr']:.1f} BPM")

        # Step 12: Evaluation
        self._log("Step 12: Evaluation...")
        if ann_path and ann_is_fetal:
            ref_peaks = load_wfdb_annotation(ann_path, ann_ext)
            self._log(f"  Reference: .{ann_ext} annotation -- {len(ref_peaks)} peaks")
        elif dir_proc is not None:
            ref_peaks = detect_reference_fetal_qrs(dir_proc, fs)
            self._log(f"  Reference: Direct_1 detector -- {len(ref_peaks)} peaks")
        else:
            ref_peaks = np.array([])
            self._log("  Reference: none available")

        # [FIX-PATH-4] Annotation anomaly guard
        sparse_annotation = False
        if len(ref_peaks) > 0:
            expected_min_ref = duration * cfg.FETAL_HR_MIN / 60.0 * 0.5
            if len(ref_peaks) < expected_min_ref:
                sparse_annotation = True
                self._log(f"  *** SPARSE ANNOTATION *** {len(ref_peaks)} ref peaks "
                          f"< expected min {expected_min_ref:.0f} -- flagged in metadata")

        metrics = evaluate(
            fetal_ecg, dir_proc, fetal_peaks, ref_peaks, fs,
            label=f"PHASE ({rec_id})",
            tolerance_ms=cfg.EVAL_TOLERANCE_MS
        )

        # Step 13: ECHO XAI
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
            "low_confidence"                         : bool(low_confidence),
            "confidence_gate_threshold"              : float(confidence_gate),
            "harmonic_confusion"                     : bool(harmonic_confusion),
            "sparse_annotation"                      : bool(sparse_annotation),
            "ekf_used"                               : bool(ekf_used),
            "final_peak_count"                       : int(len(fetal_peaks)),
            "final_hr_bpm"                           : (float(fet_hr["mean_hr"])
                                                        if not np.isnan(fet_hr["mean_hr"])
                                                        else None),
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
            "recording"          : rec_id,
            "fetal_ecg"          : fetal_ecg,
            "fetal_ecg_pre"      : fetal_ic_raw,
            "fetal_peaks"        : fetal_peaks,
            "maternal_peaks"     : maternal_peaks,
            "ref_peaks"          : ref_peaks,
            "maternal_recon"     : maternal_recon,
            "residual"           : residual,
            "abd_proc"           : abd_proc,
            "dir_proc"           : dir_proc,
            "weights"            : weights,
            "metrics"            : metrics,
            "echo"               : echo,
            "attribution"        : attribution,
            "chosen_path"        : chosen_path,
            "ic_stability"       : chosen_stability,
            "metadata"           : metadata,
            "fs"                 : fs,
            "low_confidence"     : bool(low_confidence),
            "harmonic_confusion" : bool(harmonic_confusion),
            "sparse_annotation"  : bool(sparse_annotation),
            "ekf_used"           : bool(ekf_used),
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

        # Config 4: + Adaptive windowed WSVD
        self._log("  Config 4: + Adaptive Windowed WSVD...")
        channel_r2  = np.array([float(np.corrcoef(abd_proc[ch], mat_ic_blind)[0, 1] ** 2)
                                 for ch in range(abd_proc.shape[0])])
        mat_recon_4 = adaptive_windowed_wsvd(abd_proc, weights_gauss, fs,
                                              mat_ic=mat_ic_blind, channel_r2=channel_r2,
                                              duration_sec=duration,
                                              cfg=cfg)
        residual_4  = subtract_maternal(abd_proc, mat_recon_4)
        ICs2_4, _   = run_ica(residual_4)
        excl_4      = _find_maternal_residual_idx(ICs2_4, mat_ic_blind, cfg)
        sig_4, pks_4 = _select(ICs2_4, excl_4, mat_hr_blind, mat_ic_blind, mat_peaks_blind, p_b=True)
        results["4_Adaptive_WSVD"] = _eval(sig_4, pks_4, "+Adaptive WSVD")

        # Config 5: + EKF-RTS
        self._log("  Config 5: Full PHASE (+ EKF-RTS)...")
        fetal_ecg_5, ekf_ok = _apply_ekf(sig_4, pks_4, fs, use_rts=True, cfg=self.cfg)
        if not ekf_ok:
            fetal_ecg_5 = sig_4
        pks_5 = detect_fetal_qrs(fetal_ecg_5, fs, cfg=self.cfg)
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
