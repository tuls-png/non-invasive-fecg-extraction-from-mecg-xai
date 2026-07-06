"""
pipeline.py
PHASE: Physiology-guided Hybrid Adaptive Signal Extraction

Dual-path fetal IC selection:
  Path A -- ICA1 Direct: best non-maternal IC from ICA1
  Path C -- Adaptive template subtraction + ICA3 [Rank 1]

Path A/C candidates are fused using an SQI-weighted trust score
[Rank 2] on top of the unified three-factor IC-selection score,
which includes a periodicity/autocorrelation bonus term
[Rank 3] computed inside separation/ica.py.

IMPLEMENTATION:
  This module implements the core PHASE pipeline with Path B removed.
  Available features:
    Rank 1 - Path C: beat-template subtraction (separation/template_subtraction.py)
    Rank 2 - SQI-weighted fusion across Path A/C (evaluation/sqi.py)
    Rank 3 - Periodicity-constrained IC scoring (separation/ica.py)
    Rank 5 - EKF forward-backward (RTS) smoothing (separation/ekf.py, already
             implemented as FetalECGKalmanFilter.smooth(); now also exposed
             as a dataset-tunable default via cfg.EKF_USE_RTS_DEFAULT)
    Rank 6 - Hyperparameters for every feature are configurable via
             configs/base.py defaults + per-dataset YAML overrides, and
             every feature has an *_ENABLED flag so it can be toggled off
             for ablation without touching this file.

  Path C is optional: it can override Path A only if it produces a higher
  (optionally SQI-fused) score. When PATH_C_ENABLED and SQI_FUSION_ENABLED
  are False and PERIODICITY_SCORE_ENABLED is False, this file behaves as
  a simplified Path A-only pipeline.
"""

import copy
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
from separation.template_subtraction import (
    adaptive_template_subtraction, verify_cancellation,
)
from separation.adaptive_filter import adaptive_residual_cleanup
from evaluation.metrics import evaluate
from evaluation.sqi import sqi_weighted_fusion, compute_candidate_sqi
from xai.echo import ECHOExplainer
from preprocessing.qrs_detector import load_wfdb_annotation
from preprocessing.qrs_detector import dump_peak_positions

# ── Ensemble hyperparameter ──────────────────────────────────────────────────
N_ENSEMBLE = 5
ENSEMBLE_SEEDS = list(range(N_ENSEMBLE))   # [0, 1, 2, 3, 4]
PCA_VARIANCE_THRESHOLD = 0.05
PCA_N_MIN, PCA_N_MAX  = 3, 4
MORPHOLOGY_MIN_PEAKS  = 5
MORPHOLOGY_WIN_SEC    = 0.3
STABILITY_LOG_THRESH  = 0.7


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

   Half-harmonic guard: candidate rejected if within
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


def _estimate_fhr_unsupervised(abd_proc: np.ndarray, fs: int, maternal_hr: float,
                               cfg, notch_bpm: float = 6.0):
    """
    [FIX-LEAKAGE] Unsupervised fetal-HR prior estimated from the abdominal
    signal's own power spectrum. Replaces reading expected_fhr from the
    ground-truth annotation file (the same file later used as ref_peaks for
    evaluation) -- that was ground-truth leakage into IC/path selection.
    Uses only abd_proc and maternal_hr, both already available at this point
    in the pipeline -- no reference annotation.

    Searches each channel's Welch PSD for the strongest peak inside
    [FETAL_HR_MIN, FETAL_HR_MAX] BPM, notching out the maternal fundamental
    and its 2nd harmonic first so the search doesn't just re-lock onto
    maternal energy leaking into the fetal band.

    Returns the estimated fetal HR in BPM, or None if no band peak survives
    the notch (caller falls back to the static clinical centre).
    """
    from scipy.signal import welch

    lo_hz = cfg.FETAL_HR_MIN / 60.0
    hi_hz = cfg.FETAL_HR_MAX / 60.0

    best_hr, best_power = None, -np.inf
    for ch in range(abd_proc.shape[0]):
        nperseg = min(8192, abd_proc.shape[1])
        freqs, psd = welch(abd_proc[ch], fs=fs, nperseg=nperseg)
        band = (freqs >= lo_hz) & (freqs <= hi_hz)
        if not np.any(band):
            continue
        band_freqs, band_psd = freqs[band], psd[band].copy()

        for mult in (1.0, 2.0):
            mat_hz = (maternal_hr * mult) / 60.0
            notch  = np.abs(band_freqs - mat_hz) <= (notch_bpm / 60.0)
            band_psd[notch] = 0.0

        if band_psd.max() <= 0:
            continue
        peak_hz    = band_freqs[np.argmax(band_psd)]
        peak_power = band_psd.max()
        if peak_power > best_power:
            best_power = peak_power
            best_hr    = float(peak_hz * 60.0)

    return best_hr


def _hr_score(mean_hr, cfg, expected_hr=None):
    if np.isnan(mean_hr):
        return 0.0
    if expected_hr is not None:
        centre    = expected_hr
        bandwidth = 15.0
    else:
        centre    = cfg.FETAL_HR_CENTRE
        bandwidth = 30.0
    return 1.0 / (1.0 + abs(mean_hr - centre) / bandwidth)


def _find_maternal_residual_idx(ICs, maternal_ic, cfg):
    """ Find ICA2 component most correlated with maternal IC."""
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
   Check whether chosen IC RR matches maternal harmonic at
    x1.0 or x2.0 only. x0.5 excluded — equals fetal RR when fetal ~2x maternal.
    """
    if len(chosen_peaks) < 3 or np.isnan(maternal_hr) or maternal_hr <= 0:
        return False

    chosen_rr   = float(np.median(np.diff(chosen_peaks))) / fs
    maternal_rr = 60.0 / maternal_hr

    for harmonic_rr in [maternal_rr, maternal_rr * 2.0]:
        if harmonic_rr > 0:
            if abs(chosen_rr - harmonic_rr) / harmonic_rr < tolerance:
                print(f"Harmonic confusion suspected: "
                      f"chosen RR={chosen_rr*1000:.0f}ms, "
                      f"harmonic RR={harmonic_rr*1000:.0f}ms "
                      f"(maternal HR={maternal_hr:.1f}bpm, "
                      f"ratio={harmonic_rr/maternal_rr:.1f}x)")
                return True
    return False



def determine_n_components(signals: np.ndarray,
                            variance_threshold: float = PCA_VARIANCE_THRESHOLD,
                            n_min: int = PCA_N_MIN,
                            n_max: int = PCA_N_MAX,
                            label: str = "") -> int:
    """Count PCA components explaining >= variance_threshold, clip to [n_min, n_max]."""
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



def _morphology_score(sig: np.ndarray, peaks: np.ndarray,
                      fs: int, win_sec: float = MORPHOLOGY_WIN_SEC) -> float:
    """
    

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
   Maternal leakage penalty factor.
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
    Unified three-factor fetal IC scoring:
        final_score = base_score x maternal_penalty x (1 + morphology_score)
    """
    base    = score_fetal_ic(ic, maternal_peaks, fs)
    mat_pen = _maternal_penalty(ic, maternal_ic, maternal_hr, fs,
                                path_b_half_weight=path_b)
    morph   = _morphology_score(ic, peaks, fs)
    return float(base * mat_pen * (1.0 + morph))


def _best_ic(ICs_or_signals, exclude_idx, maternal_hr, fs, cfg,
             label="", expected_hr=None, min_peaks=100,
             maternal_ic=None, maternal_peaks=None,
             path_b=False,
             n_components=None):
    """
    Select best fetal IC using unified three-factor score.
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
    ICA ensemble wrapper for _best_ic.

    Runs FastICA N_ENSEMBLE times with seeds ENSEMBLE_SEEDS (0-4).
    All N*k IC candidates are scored with the three-factor formula from .
    The global winner across all runs is returned.

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
   Wrapper around _best_ic_ensemble that retries with alternative
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
   EKF acceptance gate: ALL THREE gates must pass:
      (a) post-EKF peak count >= 70% of pre-EKF
      (b) CC(ekf_output, pre_ekf_input) >= 0.60  
      (c) |median_RR_post - median_RR_pre| <= 15ms

 
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
    def __init__(self, fs=None, use_rts=None, ekf_bypass=False, verbose=True,
                 dataset=None, stdout_log_path=None, config=None,
                 config_overrides=None, dump_peaks=True, peak_dump_dir="peak_dumps"):
        self.dump_peaks    = dump_peaks
        self.peak_dump_dir = peak_dump_dir
        self.cfg        = config if config is not None else get_config(dataset)
        if config_overrides:
            for key, value in config_overrides.items():
                setattr(self.cfg, key.upper(), value)
        self.fs         = fs if fs is not None else self.cfg.FS
        # [Rank 5/6] use_rts now defaults from cfg.EKF_USE_RTS_DEFAULT when not
        # explicitly given, making the EKF smoothing mode a dataset-tunable,
        # ablatable hyperparameter rather than a hardcoded constructor default.
        # Passing True/False explicitly still overrides the config value, so
        # all existing call sites (which pass use_rts=True explicitly) are
        # unaffected.
        self.use_rts    = (use_rts if use_rts is not None
                            else getattr(self.cfg, "EKF_USE_RTS_DEFAULT", True))
        self.ekf_bypass = ekf_bypass
        self.verbose    = verbose
        self._log_file  = None
        if stdout_log_path is not None:
            self._log_file = open(stdout_log_path, "a", encoding="utf-8", buffering=1)

    def __del__(self):
        if self._log_file is not None:
            try:
                self._log_file.close()
            except Exception:
                pass

    def _log(self, msg):
        line = f"[PHASE] {msg}"
        if self.verbose:
            print(line)
        if self._log_file is not None:
            try:
                self._log_file.write(line + "")
            except Exception:
                pass

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

        # [FIX-LEAKAGE] expected_fhr must never be derived from annotation_path --
        # that file is also used as ref_peaks for evaluation in Step 11/12.
        # Estimate it from the signal itself instead (unsupervised).
        expected_fhr = _estimate_fhr_unsupervised(abd_proc, fs, maternal_hr, cfg)
        if expected_fhr is not None:
            self._log(f"  Unsupervised HR prior (signal spectrum): "
                      f"expected fetal HR ~ {expected_fhr:.1f} BPM")
        else:
            self._log("  No confident spectral HR prior -- using static clinical centre")

        if ann_path and ann_is_fetal:
            self._log("  Annotation available -- reserved for evaluation only "
                      "(not used as a selection prior)")

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

        # Step 4.5: Path C -- Adaptive Windowed Weighted SVD, epoch domain
        # (beat-template subtraction) + ICA3  [NEW - Rank 1]
        # Runs on abd_proc directly (NOT on Path B's residual): Path A/B/C
        # are independent, parallel candidate-generation paths, unified
        # only at Step 9/9b scoring. See separation/template_subtraction.py
        # module docstring for the two-axis AW-WSVD framing.
        c_sig = c_idx = c_peaks = c_hr = c_stability = c_score = None
        c_valid = False
        ts_metrics = None
        if getattr(cfg, "PATH_C_ENABLED", False):
            _tmpl_estimator = getattr(cfg, "TEMPLATE_ESTIMATOR", "median")
            self._log(f"Step 4.5: Path C -- AW-WSVD (epoch domain, "
                      f"estimator={_tmpl_estimator}) + ICA3...")
            residual_c = adaptive_template_subtraction(
                abd_proc, maternal_peaks, fs,
                half_window_sec=cfg.TEMPLATE_HALF_WINDOW_SEC,
                update_every=cfg.TEMPLATE_UPDATE_EVERY_BEATS,
                context_beats=cfg.TEMPLATE_CONTEXT_BEATS,
                min_beats_for_template=cfg.TEMPLATE_MIN_BEATS,
                estimator=_tmpl_estimator,
                svd_n_components=getattr(cfg, "TEMPLATE_SVD_N_COMPONENTS", 1),
            )
            ts_metrics = verify_cancellation(
                abd_proc, residual_c, maternal_peaks, fs,
                half_window_sec=cfg.TEMPLATE_HALF_WINDOW_SEC)
            self._log(f"  Template subtraction: {ts_metrics['n_beats_checked']} beats, "
                      f"energy reduction at beats = "
                      f"{ts_metrics['energy_reduction_pct']:.1f}%")
            try:
                n_comp_ica3 = determine_n_components(
                    residual_c, label="ICA3/template_residual")
                ICs3_ref, _ = run_ica(residual_c, n_components=n_comp_ica3)
                mat_residual_idx_c = _find_maternal_residual_idx(
                    ICs3_ref, maternal_ic, cfg)
                c_sig, c_idx, c_peaks, c_hr, c_stability, c_score = \
                    _best_ic_ensemble_with_retry(
                        residual_c, mat_residual_idx_c, maternal_hr, fs, cfg,
                        label="Path C", expected_hr=expected_fhr,
                        min_peaks=min_peaks, maternal_ic=maternal_ic,
                        maternal_peaks=maternal_peaks, path_b=False,
                        n_components=n_comp_ica3)
                c_n = len(c_peaks)
                c_valid = _is_fetal_hr(c_hr, maternal_hr, cfg)
                self._log(f"  Path C: IC{c_idx+1}, {c_n} peaks, "
                          f"HR={c_hr:.1f} BPM, valid={'YES' if c_valid else 'NO'}, "
                          f"stability={c_stability:.3f}, score={c_score:.4f}")
            except ValueError as e:
                self._log(f"  Path C: failed ({e}) -- excluded from fusion")
                c_sig = c_idx = c_peaks = c_hr = c_stability = c_score = None
                c_valid = False
        else:
            residual_c = None

        # Steps 5-7 (Gaussian weights / AW-WSVD maternal reconstruction /
        # subtraction) previously fed Path B, which has been removed from
        # candidate generation entirely. They are no longer needed for the
        # chosen output -- only _save_figures' diagnostic plot still uses
        # maternal_recon/residual, so only compute them when a figure will
        # actually be drawn, instead of paying for the SVD on every run.
        if save_figures:
            self._log("Step 5: Gaussian weight matrix (for diagnostic figure only)...")
            weights = gaussian_weight_matrix(abd_proc.shape[1], maternal_peaks, fs)

            self._log("Step 6: AW-WSVD maternal reconstruction (for diagnostic figure only)...")
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

            self._log("Step 7: Maternal cancellation (for diagnostic figure only)...")
            residual = subtract_maternal(abd_proc, maternal_recon)
        else:
            weights = maternal_recon = residual = None

        # Step 9: Set Path A as the incumbent (Path B has been removed)
        self._log("Step 9: Setting Path A as incumbent...")
        if a_valid:
            chosen_sig, chosen_peaks = a_sig, a_peaks
            chosen_path = f"A_ICA1_direct_IC{a_idx+1}_{a_hr:.0f}bpm"
            self._log(f"  Path A valid -> selected as incumbent")
        else:
            chosen_sig, chosen_peaks = a_sig, a_peaks
            chosen_path = f"A_fallback_IC{a_idx+1}_{a_hr:.0f}bpm"
            self._log(f"  Path A selected as fallback")

        chosen_score = a_score
        chosen_hr    = a_hr

        # Step 9b: Path C fusion [NEW - Rank 1 + Rank 2]
        # Path C can override the incumbent Path A -- it must win on (optionally
        # SQI-weighted) score. This keeps Path A as the base and adds Path C
        # strictly additively.
        chosen_sqi   = None
        incumbent_sqi = None
        if getattr(cfg, "PATH_C_ENABLED", False) and c_valid:
            if getattr(cfg, "SQI_FUSION_ENABLED", False):
                fused = sqi_weighted_fusion([
                    {"label": "incumbent", "signal": chosen_sig,
                     "peaks": chosen_peaks, "fs": fs, "score": chosen_score},
                    {"label": "C", "signal": c_sig, "peaks": c_peaks,
                     "fs": fs, "score": c_score},
                ], cfg=cfg)
                fused_map = {f["label"]: f for f in fused}
                incumbent_fused = fused_map["incumbent"]["fused_score"]
                incumbent_sqi   = fused_map["incumbent"]["sqi"]
                c_fused         = fused_map["C"]["fused_score"]
                chosen_sqi_c    = fused_map["C"]["sqi"]
                self._log(f"  SQI fusion: incumbent({chosen_path}) fused="
                          f"{incumbent_fused:.4f} (sqi={incumbent_sqi:.3f}) vs "
                          f"Path C fused={c_fused:.4f} (sqi={chosen_sqi_c:.3f})")
                if c_fused > incumbent_fused:
                    chosen_sig, chosen_peaks = c_sig, c_peaks
                    chosen_path  = f"C_TemplateSub_ICA3_IC{c_idx+1}_{c_hr:.0f}bpm"
                    chosen_score = c_score
                    chosen_hr    = c_hr
                    chosen_sqi   = chosen_sqi_c
                    self._log(f"  Path C selected via SQI-weighted fusion")
                else:
                    chosen_sqi = incumbent_sqi
            else:
                if c_score > chosen_score:
                    chosen_sig, chosen_peaks = c_sig, c_peaks
                    chosen_path  = f"C_TemplateSub_ICA3_IC{c_idx+1}_{c_hr:.0f}bpm"
                    chosen_score = c_score
                    chosen_hr    = c_hr
                    self._log(f"  Path C selected (higher raw score, "
                              f"SQI fusion disabled)")
        elif getattr(cfg, "PATH_C_ENABLED", False) and not c_valid and c_score is not None:
            self._log(f"  Path C not valid (score={c_score:.4f}) -- not considered")

        confidence_gate = getattr(cfg, "CONFIDENCE_GATE_THRESHOLD",
                                  _DEFAULT_CONFIDENCE_GATE)
        low_confidence = chosen_score < confidence_gate
        if low_confidence:
            self._log(f"  *** LOW CONFIDENCE *** score={chosen_score:.4f} "
                      f"< gate={confidence_gate} -- flagged (retry already attempted in ensemble)")

        
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
            ]
            if getattr(cfg, "PATH_C_ENABLED", False) and c_sig is not None:
                candidates_spectral.append((c_sig, c_peaks, c_hr, c_score, "C"))
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
                    if cand_label == "A":
                        chosen_path = f"A_spectral_IC{a_idx+1}_{a_hr:.0f}bpm"
                    else:
                        chosen_path = f"C_spectral_IC{c_idx+1}_{c_hr:.0f}bpm"
            chosen_sig, chosen_peaks = best_spec_sig, best_spec_peaks
            self._log(f"  Spectral fallback selected: {chosen_path} "
                      f"(band_ratio={best_spec_ratio:.4f})")
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

        # Step 11: Final QRS detection +  under-detection retry
        self._log("Step 11: Final fetal QRS detection...")
        if ann_path and ann_is_fetal:
            ref_peaks = load_wfdb_annotation(ann_path, ann_ext)
            self._log(f"  Reference: .{ann_ext} annotation -- {len(ref_peaks)} peaks")
        elif dir_proc is not None:
            ref_peaks = detect_reference_fetal_qrs(dir_proc, fs)
            self._log(f"  Reference: Direct_1 detector -- {len(ref_peaks)} peaks")
        else:
            ref_peaks = np.array([])
            self._log("  Reference: none available")
        fetal_peaks = detect_fetal_qrs(fetal_ecg, fs, cfg=cfg)

       
        _n_ref = len(ref_peaks) if len(ref_peaks) > 10 else int(
            duration * cfg.FETAL_HR_CENTRE / 60.0)
        _det_ratio  = len(fetal_peaks) / (_n_ref + 1e-10)
        _fp_est     = max(0.0, len(fetal_peaks) - _n_ref) / (len(fetal_peaks) + 1e-10)
        _retry_trigger = (_det_ratio < 0.90) or (_fp_est > 0.25)

        if _retry_trigger:
            self._log(f"  Retry trigger: det/ref={_det_ratio:.3f}, "
                      f"est_fp_rate={_fp_est:.2f} "
                      f"({'under-detection' if _det_ratio < 0.90 else 'over-detection'}) "
                      f"-- retrying with narrow-band force_low_threshold...")
            fetal_peaks_retry = detect_fetal_qrs(
                fetal_ecg, fs, cfg=cfg, force_low_threshold=True)
            _retry_ratio = len(fetal_peaks_retry) / (_n_ref + 1e-10)
            # Only accept retry if it moved the det/ref ratio closer to 1.0
            if abs(_retry_ratio - 1.0) < abs(_det_ratio - 1.0):
                self._log(f"  Retry accepted: {len(fetal_peaks_retry)} peaks "
                          f"(was {len(fetal_peaks)}), "
                          f"new det/ref={_retry_ratio:.3f}")
                fetal_peaks = fetal_peaks_retry
            else:
                self._log(f"  Retry rejected: {len(fetal_peaks_retry)} peaks "
                          f"would worsen det/ref ({_retry_ratio:.3f} vs {_det_ratio:.3f})")

        fet_hr = compute_hr_stats(fetal_peaks, fs)
        self._log(f"  {len(fetal_peaks)} peaks, HR = {fet_hr['mean_hr']:.1f} BPM")
        n_before_merge = len(fetal_peaks)
        if len(fetal_peaks) > 2:
            merge_radius = int(0.080 * fs)
            merged = [int(fetal_peaks[0])]
            for p in fetal_peaks[1:]:
                if int(p) - merged[-1] < merge_radius:
                    if abs(float(chosen_sig[int(p)])) > abs(float(chosen_sig[merged[-1]])):
                        merged[-1] = int(p)
                else:
                    merged.append(int(p))
            fetal_peaks = np.array(merged, dtype=int)
        if len(fetal_peaks) != n_before_merge:
            self._log(f"  Peak merging: {n_before_merge} -> {len(fetal_peaks)} peaks "
                      f"({n_before_merge - len(fetal_peaks)} doublets collapsed)")
            

        # Step 12: Evaluation
        self._log("Step 12: Evaluation...")   
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

        if self.dump_peaks and len(ref_peaks) > 0:
            dump_peak_positions(
                rec_id, fetal_peaks, ref_peaks, fs,
                out_dir=self.peak_dump_dir,
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
        chosen_stability = (a_stability if "A_" in chosen_path
                            else c_stability)

        metadata = {
            "ica1_pca_n_components"                  : int(n_comp_ica1),
            "ica1_maternal_ic_index"                 : int(maternal_ic_idx),
            "path_a_selected_ic_index"               : int(a_idx),
            "path_a_selected_ic_peak_count"          : int(a_n),
            "path_a_selected_ic_hr_bpm"              : float(a_hr),
            "path_a_selected_ic_is_valid"            : bool(a_valid),
            "path_a_selected_ic_stability"           : float(a_stability),
            "path_a_selected_ic_score"               : float(a_score),
            "path_c_enabled"                         : bool(getattr(cfg, "PATH_C_ENABLED", False)),
            "path_c_selected_ic_index"               : (int(c_idx) if c_idx is not None else None),
            "path_c_selected_ic_hr_bpm"              : (float(c_hr) if c_hr is not None else None),
            "path_c_selected_ic_is_valid"            : bool(c_valid),
            "path_c_selected_ic_stability"           : (float(c_stability) if c_stability is not None else None),
            "path_c_selected_ic_score"               : (float(c_score) if c_score is not None else None),
            "path_c_template_energy_reduction_pct"   : (float(ts_metrics["energy_reduction_pct"])
                                                        if ts_metrics is not None else None),
            "path_c_template_n_beats"                : (int(ts_metrics["n_beats_checked"])
                                                        if ts_metrics is not None else None),
            "sqi_fusion_enabled"                     : bool(getattr(cfg, "SQI_FUSION_ENABLED", False)),
            "chosen_sqi"                             : (float(chosen_sqi) if chosen_sqi is not None else None),
            "periodicity_score_enabled"              : bool(getattr(cfg, "PERIODICITY_SCORE_ENABLED", False)),
            "chosen_path_description"                : chosen_path,
            "chosen_ic_index"                        : int(a_idx if "A_" in chosen_path else c_idx),
            "chosen_ic_hr_bpm"                       : float(a_hr if "A_" in chosen_path else c_hr),
            "chosen_ic_peak_count"                   : int(len(chosen_peaks)),
            "chosen_ic_selection_score"              : float(chosen_score),
            "chosen_ic_stability"                    : float(chosen_stability),
            "low_confidence"                         : bool(low_confidence),
            "confidence_gate_threshold"              : float(confidence_gate),
            "harmonic_confusion"                     : bool(harmonic_confusion),
            "sparse_annotation"                      : bool(sparse_annotation),
            "ekf_used"                               : bool(ekf_used),
            "ekf_use_rts"                            : bool(self.use_rts),
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
            "residual_c"         : residual_c,
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

        # A local, mutable config copy is used for the incremental
        # [Rank 1-4] ablation legs below so that toggling feature flags for
        # a single ablation step (e.g. enabling periodicity scoring only
        # for step 9) never mutates self.cfg / affects self.run().
        cfg_ablation = copy.copy(cfg)
        cfg_ablation.PERIODICITY_SCORE_ENABLED = False

        ICs1, _          = run_ica(abd_proc)
        mat_idx_blind, _ = select_maternal_ic(ICs1, fs, cfg=cfg_ablation)
        mat_ic_blind     = get_ic_as_signal(ICs1, mat_idx_blind)
        mat_peaks_blind  = detect_maternal_qrs(mat_ic_blind, fs, cfg=cfg_ablation)
        mat_hr_blind     = compute_hr_stats(mat_peaks_blind, fs)["mean_hr"]
        weights_gauss    = gaussian_weight_matrix(abd_proc.shape[1], mat_peaks_blind, fs)

        def _eval(sig, peaks, label):
            return evaluate(sig, dir_proc, peaks, ref_peaks, fs, label=label)

        def _select(ICs, excl, mat_hr, mat_ic=None, mat_peaks=None, p_b=False,
                    use_cfg=None):
            sig, idx, peaks, hr, _ = _best_ic(
                ICs, excl, mat_hr, fs, use_cfg or cfg_ablation, min_peaks=min_peaks,
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
        excl_2      = _find_maternal_residual_idx(ICs2_2, mat_ic_blind, cfg_ablation)
        sig_2, pks_2 = _select(ICs2_2, excl_2, mat_hr_blind, mat_ic_blind, mat_peaks_blind)
        results["2_Blind_IC_Selection"] = _eval(sig_2, pks_2, "+Blind IC Selection")

        # Config 3: + Gaussian weights
        self._log("  Config 3: + Gaussian weights...")
        mat_recon_3 = _global_wsvd(abd_proc, weights_gauss)
        residual_3  = subtract_maternal(abd_proc, mat_recon_3)
        ICs2_3, _   = run_ica(residual_3)
        excl_3      = _find_maternal_residual_idx(ICs2_3, mat_ic_blind, cfg_ablation)
        sig_3, pks_3 = _select(ICs2_3, excl_3, mat_hr_blind, mat_ic_blind, mat_peaks_blind)
        results["3_Gaussian_Weights"] = _eval(sig_3, pks_3, "+Gaussian Weights")

        # Config 4: + Adaptive windowed WSVD
        self._log("  Config 4: + Adaptive Windowed WSVD...")
        channel_r2  = np.array([float(np.corrcoef(abd_proc[ch], mat_ic_blind)[0, 1] ** 2)
                                 for ch in range(abd_proc.shape[0])])
        mat_recon_4 = adaptive_windowed_wsvd(abd_proc, weights_gauss, fs,
                                              mat_ic=mat_ic_blind, channel_r2=channel_r2,
                                              duration_sec=duration,
                                              cfg=cfg_ablation)
        residual_4  = subtract_maternal(abd_proc, mat_recon_4)
        ICs2_4, _   = run_ica(residual_4)
        excl_4      = _find_maternal_residual_idx(ICs2_4, mat_ic_blind, cfg_ablation)
        sig_4, pks_4 = _select(ICs2_4, excl_4, mat_hr_blind, mat_ic_blind, mat_peaks_blind, p_b=True)
        results["4_Adaptive_WSVD"] = _eval(sig_4, pks_4, "+Adaptive WSVD")

        # Config 5: + EKF-RTS
        self._log("  Config 5: Full PHASE (+ EKF-RTS)...")
        fetal_ecg_5, ekf_ok = _apply_ekf(sig_4, pks_4, fs, use_rts=True, cfg=self.cfg)
        if not ekf_ok:
            fetal_ecg_5 = sig_4
        pks_5 = detect_fetal_qrs(fetal_ecg_5, fs, cfg=self.cfg)
        results["5_PHASE_Full"] = _eval(fetal_ecg_5, pks_5, "PHASE Full")

        # ── [Rank 1-4, NEW] Incremental enhancement ablation legs ──────────
        # Each leg below starts from Config 5's state and adds exactly one
        # roadmap enhancement, matching the dissertation results-chapter
        # ablation order: (6) +Path C, (7) +SQI-weighted fusion,
        # (8) +periodicity-constrained IC scoring, (9) +RLS/NLMS cleanup.
        # (Rank 5, EKF-RTS smoothing, is already included in Config 5;
        # Rank 6, per-dataset threshold tuning, is a run_experiment_new.py
        # / offline concern and does not change separation-stage code.)

        # Config 6: + Path C (AW-WSVD, epoch domain / template subtraction)
        self._log("  Config 6: + Path C (AW-WSVD, epoch domain)...")
        try:
            residual_c = adaptive_template_subtraction(
                abd_proc, mat_peaks_blind, fs,
                half_window_sec=cfg.TEMPLATE_HALF_WINDOW_SEC,
                update_every=cfg.TEMPLATE_UPDATE_EVERY_BEATS,
                context_beats=cfg.TEMPLATE_CONTEXT_BEATS,
                min_beats_for_template=cfg.TEMPLATE_MIN_BEATS,
                estimator=getattr(cfg, "TEMPLATE_ESTIMATOR", "median"),
                svd_n_components=getattr(cfg, "TEMPLATE_SVD_N_COMPONENTS", 1))
            ICs2_c, _ = run_ica(residual_c)
            excl_c    = _find_maternal_residual_idx(ICs2_c, mat_ic_blind, cfg_ablation)
            sig_c, pks_c = _select(ICs2_c, excl_c, mat_hr_blind, mat_ic_blind, mat_peaks_blind)
            score_5 = _score_ic_unified(sig_4, pks_4, mat_peaks_blind, mat_ic_blind, mat_hr_blind, fs, path_b=True)
            score_c = _score_ic_unified(sig_c, pks_c, mat_peaks_blind, mat_ic_blind, mat_hr_blind, fs, path_b=False)
            if score_c > score_5:
                sig_6, pks_6 = sig_c, pks_c
            else:
                sig_6, pks_6 = sig_4, pks_4
            fetal_ecg_6, ekf_ok = _apply_ekf(sig_6, pks_6, fs, use_rts=True, cfg=self.cfg)
            if not ekf_ok:
                fetal_ecg_6 = sig_6
            pks_6_final = detect_fetal_qrs(fetal_ecg_6, fs, cfg=self.cfg)
            results["6_PathC_TemplateSub"] = _eval(fetal_ecg_6, pks_6_final, "+Path C")
        except Exception as e:
            self._log(f"  Config 6 failed ({e}) -- skipping")
            sig_6, pks_6 = sig_4, pks_4

        # Config 7: + SQI-weighted fusion (Config 5 vs Config 6 candidates,
        # fused by trust-weighted score rather than raw score)
        self._log("  Config 7: + SQI-weighted fusion...")
        try:
            fused = sqi_weighted_fusion([
                {"label": "5", "signal": sig_4, "peaks": pks_4, "fs": fs,
                 "score": _score_ic_unified(sig_4, pks_4, mat_peaks_blind, mat_ic_blind, mat_hr_blind, fs, path_b=True)},
                {"label": "6", "signal": sig_6, "peaks": pks_6, "fs": fs,
                 "score": _score_ic_unified(sig_6, pks_6, mat_peaks_blind, mat_ic_blind, mat_hr_blind, fs, path_b=False)},
            ], cfg=cfg)
            best = max(fused, key=lambda c: c["fused_score"])
            sig_7, pks_7 = best["signal"], best["peaks"]
            fetal_ecg_7, ekf_ok = _apply_ekf(sig_7, pks_7, fs, use_rts=True, cfg=self.cfg)
            if not ekf_ok:
                fetal_ecg_7 = sig_7
            pks_7_final = detect_fetal_qrs(fetal_ecg_7, fs, cfg=self.cfg)
            results["7_SQI_Weighted_Fusion"] = _eval(fetal_ecg_7, pks_7_final, "+SQI Fusion")
        except Exception as e:
            self._log(f"  Config 7 failed ({e}) -- skipping")
            sig_7, pks_7 = sig_6, pks_6

        # Config 8: + Periodicity-constrained IC scoring (re-select Path B's
        # winning IC with PERIODICITY_SCORE_ENABLED=True)
        self._log("  Config 8: + Periodicity-constrained IC scoring...")
        try:
            cfg_periodicity = copy.copy(cfg)
            cfg_periodicity.PERIODICITY_SCORE_ENABLED = True
            sig_8, pks_8 = _select(ICs2_4, excl_4, mat_hr_blind, mat_ic_blind,
                                    mat_peaks_blind, p_b=True, use_cfg=cfg_periodicity)
            fetal_ecg_8, ekf_ok = _apply_ekf(sig_8, pks_8, fs, use_rts=True, cfg=self.cfg)
            if not ekf_ok:
                fetal_ecg_8 = sig_8
            pks_8_final = detect_fetal_qrs(fetal_ecg_8, fs, cfg=self.cfg)
            results["8_Periodicity_Scoring"] = _eval(fetal_ecg_8, pks_8_final, "+Periodicity Scoring")
        except Exception as e:
            self._log(f"  Config 8 failed ({e}) -- skipping")

        # Config 9: + RLS/NLMS adaptive filter residual cleanup on Path B
        self._log("  Config 9: + Adaptive filter (RLS/NLMS) cleanup...")
        try:
            residual_4_clean = adaptive_residual_cleanup(
                residual_4, mat_recon_4, fs,
                method=cfg.ADAPTIVE_FILTER_METHOD,
                n_taps=cfg.ADAPTIVE_FILTER_N_TAPS,
                forgetting_factor=cfg.ADAPTIVE_FILTER_FORGETTING,
                delta=cfg.ADAPTIVE_FILTER_DELTA,
                step_size=cfg.ADAPTIVE_FILTER_STEP_SIZE,
                eps=cfg.ADAPTIVE_FILTER_EPS)
            ICs2_9, _ = run_ica(residual_4_clean)
            excl_9    = _find_maternal_residual_idx(ICs2_9, mat_ic_blind, cfg_ablation)
            sig_9, pks_9 = _select(ICs2_9, excl_9, mat_hr_blind, mat_ic_blind, mat_peaks_blind, p_b=True)
            fetal_ecg_9, ekf_ok = _apply_ekf(sig_9, pks_9, fs, use_rts=True, cfg=self.cfg)
            if not ekf_ok:
                fetal_ecg_9 = sig_9
            pks_9_final = detect_fetal_qrs(fetal_ecg_9, fs, cfg=self.cfg)
            results["9_Adaptive_Filter_Cleanup"] = _eval(fetal_ecg_9, pks_9_final, "+Adaptive Filter")
        except Exception as e:
            self._log(f"  Config 9 failed ({e}) -- skipping")

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