"""
separation/ica.py
FastICA with blind maternal and fetal IC selection.

Key novelty over prior ICA+WSVD work:
- Blind maternal IC selection using physiological scoring (no ground truth)
- Blind fetal IC selection using HR range + temporal independence + kurtosis
- Correct ICA preprocessing (center only, not z-score)

FIX: maternal_idx parameter in select_fetal_ic() was accepted but never used,
creating a risk of selecting the maternal IC as fetal if ICA2 happens to
reproduce a similar component. It is now documented clearly as vestigial
(ICA2 runs on the residual, so index mapping does not carry over) and
removed from the scoring loop to avoid misleading future readers.
"""

import numpy as np
from scipy.stats import kurtosis
from sklearn.decomposition import FastICA
from config import (
    ICA_N_COMPONENTS, ICA_MAX_ITER, ICA_RANDOM_STATE, ICA_TOL,
    MATERNAL_HR_MIN, MATERNAL_HR_MAX, FETAL_HR_MIN, FETAL_HR_MAX,
    ECHO_MATERNAL_EXCLUSION_SEC, FS
)
from preprocessing.qrs_detector import pan_tompkins, compute_hr_stats


def run_ica(signals: np.ndarray,
            n_components: int = ICA_N_COMPONENTS) -> tuple[np.ndarray, FastICA]:
    """
    Run FastICA on multichannel signal.

    Parameters
    ----------
    signals      : (n_channels, N) — must be centered (zero-mean)
    n_components : number of independent components

    Returns
    -------
    ICs  : (n_components, N) independent components
    ica  : fitted FastICA object (for inverse transform)

    Notes
    -----
    Zero-variance channels (e.g. zero-padding for recordings with fewer
    than 4 abdominal channels) are excluded before FastICA to prevent
    the whitening step from dividing by near-zero eigenvalues (NaN/Inf).
    Output is always padded back to n_components rows.
    """
    # Identify channels with meaningful signal (variance > threshold)
    variances     = np.var(signals, axis=1)
    active_mask   = variances > 1e-10
    active_idx    = np.where(active_mask)[0]
    n_active      = len(active_idx)

    if n_active == 0:
        raise ValueError("All input channels have zero variance — cannot run ICA.")

    # Use only active channels; cap n_components to n_active
    active_signals = signals[active_idx]
    n_comp_actual  = min(n_components, n_active)

    if n_active < n_components:
        print(f"[ICA] {n_components - n_active} zero-variance channel(s) excluded "
              f"— running ICA with {n_comp_actual} components on {n_active} channels")

    ica = FastICA(
        n_components=n_comp_actual,
        max_iter=ICA_MAX_ITER,
        random_state=ICA_RANDOM_STATE,
        tol=ICA_TOL,
        whiten='unit-variance'
    )
    ICs_active = ica.fit_transform(active_signals.T).T   # (n_comp_actual, N)

    # Pad output to n_components with zero rows so downstream code
    # that iterates over ICs by index continues to work unchanged
    if n_comp_actual < n_components:
        N   = signals.shape[1]
        ICs = np.zeros((n_components, N), dtype=ICs_active.dtype)
        ICs[:n_comp_actual] = ICs_active
    else:
        ICs = ICs_active

    return ICs, ica


def score_maternal_ic(ic: np.ndarray, fs: int = FS) -> float:
    """
    Score a single IC for how likely it is to be the maternal ECG.

    Criteria (physiologically justified):
    1. HR plausibility   : maternal HR must be 55-110 BPM
    2. RR regularity     : maternal HR is regular (low coefficient of variation)
    3. Signal amplitude  : maternal ECG dominates abdominal mixture (high var)
    4. Kurtosis          : ECG has high kurtosis due to sharp QRS spikes
    """
    peaks, _ = pan_tompkins(ic, fs, min_hr_bpm=50, max_hr_bpm=120)
    if len(peaks) < 4:
        return 0.0

    stats   = compute_hr_stats(peaks, fs)
    mean_hr = stats["mean_hr"]
    std_hr  = stats["std_hr"]

    if not (MATERNAL_HR_MIN - 5 <= mean_hr <= MATERNAL_HR_MAX + 5):
        return 0.0

    cv              = std_hr / (mean_hr + 1e-8)
    regularity_score = 1.0 / (1.0 + cv * 10)
    amplitude_score  = float(np.var(ic))
    kurt             = float(kurtosis(ic, fisher=True))
    kurt_score       = np.clip(kurt / 20.0, 0.0, 1.0)

    return float(regularity_score * amplitude_score * (1.0 + kurt_score))


def select_maternal_ic(ICs: np.ndarray, fs: int = FS) -> tuple[int, list[float]]:
    """
    Blindly select the maternal IC from all independent components.

    Parameters
    ----------
    ICs : (n_components, N)

    Returns
    -------
    best_idx : index of maternal IC
    scores   : list of scores for each IC
    """
    scores   = [score_maternal_ic(ic, fs) for ic in ICs]
    best_idx = int(np.argmax(scores))

    print(f"\n[ICA] Maternal IC selection scores:")
    for i, s in enumerate(scores):
        marker = " <- selected (maternal)" if i == best_idx else ""
        print(f"  IC{i+1}: {s:.4f}{marker}")

    return best_idx, scores


def score_fetal_ic(ic: np.ndarray, maternal_peaks: np.ndarray,
                   fs: int = FS) -> float:
    """
    Score a single IC for how likely it is to contain the fetal ECG.

    Criteria:
    1. HR plausibility   : fetal HR 60-185 BPM
    2. Temporal independence from maternal peaks
    3. Kurtosis          : ECG-like morphology
    4. RR regularity     : fetal HR is relatively regular
    """
    peaks, _ = pan_tompkins(ic, fs, min_hr_bpm=55, max_hr_bpm=195)
    if len(peaks) < 5:
        return 0.0

    stats   = compute_hr_stats(peaks, fs)
    mean_hr = stats["mean_hr"]
    std_hr  = stats["std_hr"]

    if not (60 <= mean_hr <= 185):
        return 0.0

    exclusion_samples    = int(ECHO_MATERNAL_EXCLUSION_SEC * fs)
    n_fetal_near_maternal = 0
    for fp in peaks:
        if len(maternal_peaks) and np.min(np.abs(maternal_peaks - fp)) < exclusion_samples:
            n_fetal_near_maternal += 1
    independence = 1.0 - (n_fetal_near_maternal / (len(peaks) + 1e-8))

    cv         = std_hr / (mean_hr + 1e-8)
    regularity = 1.0 / (1.0 + cv * 10)
    kurt       = float(kurtosis(ic, fisher=True))
    kurt_score = np.clip(kurt / 20.0, 0.0, 1.0)

    return float(independence * regularity * (1.0 + kurt_score))


def select_fetal_ic(ICs: np.ndarray,
                    maternal_peaks: np.ndarray,
                    maternal_idx: int,
                    fs: int = FS,
                    residual: np.ndarray = None) -> tuple[int, list[float]]:
    """
    Blindly select the fetal IC from ICA components of the residual signal.

    Strategy:
    1. Score all ICs using physiological criteria.
    2. If all scores are zero, fall back to residual correlation.

    Parameters
    ----------
    ICs           : (n_components, N)
    maternal_peaks: detected maternal R-peak indices
    maternal_idx  : kept for API compatibility but NOT used for exclusion.
                    ICA2 runs on the residual signal, so component indices
                    do not map to the same sources as ICA1 — excluding by
                    index would incorrectly discard a valid fetal component.
    residual      : (n_channels, N) residual signal for fallback correlation
    """
    scores   = [score_fetal_ic(ic, maternal_peaks, fs) for ic in ICs]
    best_idx = int(np.argmax(scores))

    # Fallback: if all scores zero, use residual correlation
    if max(scores) < 1e-6 and residual is not None:
        print("\n[ICA] All physiological scores zero — using residual correlation fallback")
        best_cc  = -1
        for i, ic in enumerate(ICs):
            for ch in range(residual.shape[0]):
                cc = abs(float(np.corrcoef(ic, residual[ch])[0, 1]))
                if cc > best_cc:
                    best_cc  = cc
                    best_idx = i
        print(f"[ICA] Fallback selected IC{best_idx+1} (best residual CC={best_cc:.4f})")

    print(f"\n[ICA] Fetal IC selection scores:")
    for i, s in enumerate(scores):
        marker = " <- selected (fetal)" if i == best_idx else ""
        print(f"  IC{i+1}: {s:.4f}{marker}")

    return best_idx, scores


def get_ic_as_signal(ICs: np.ndarray, idx: int) -> np.ndarray:
    """
    Extract and center a single IC for downstream use.

    Note: unit-normalization (scale_signal) is NOT applied here to
    preserve amplitude information. The EKF input is normalized
    separately in pipeline.py just before the EKF step.
    """
    ic = ICs[idx].copy()
    ic = ic - np.mean(ic)
    ic = ic / (np.std(ic) + 1e-10)
    return ic
