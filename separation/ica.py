"""
separation/ica.py
FastICA with blind maternal and fetal IC selection.

CHANGES FROM ORIGINAL:
  [FIX-1] FastICA whitening: 'unit-variance' -> 'arbitrary-variance'.
  [FIX-2] score_maternal_ic(): removed np.var amplitude criterion (collapses to
          same value for all ICs with arbitrary-variance whitening). Now uses
          regularity + kurtosis + peak count.
  [FIX-3] score_maternal_ic(): added progressive threshold search (same pattern
          as detect_fetal_qrs). When PT_THRESHOLD_FACTOR=1.0 finds no peaks on
          noisy NIFECGDB ICA components, progressively lowers the threshold
          WITHIN the maternal HR window until peaks are found. This is the fix
          for all-zero maternal scores on ecgca102.
  [FIX-4] select_maternal_ic(): fallback to highest-variance IC only when all
          physiological scores are zero, with clear WARNING print.
  [FIX-5] score_fetal_ic() HR lower bound: 60 -> FETAL_HR_MIN (100 BPM).
  [FIX-6] All pan_tompkins() calls use bp_low/bp_high parameter names.
"""

import numpy as np
from scipy.stats import kurtosis
from sklearn.decomposition import FastICA
from configs import BaseConfig

# Use BaseConfig defaults (shared across all datasets)
_cfg = BaseConfig()
ICA_N_COMPONENTS = _cfg.ICA_N_COMPONENTS
ICA_MAX_ITER = _cfg.ICA_MAX_ITER
ICA_RANDOM_STATE = _cfg.ICA_RANDOM_STATE
ICA_TOL = _cfg.ICA_TOL
MATERNAL_HR_MIN = _cfg.MATERNAL_HR_MIN
MATERNAL_HR_MAX = _cfg.MATERNAL_HR_MAX
FETAL_HR_MIN = _cfg.FETAL_HR_MIN
FETAL_HR_MAX = _cfg.FETAL_HR_MAX
ECHO_MATERNAL_EXCLUSION_SEC = _cfg.ECHO_MATERNAL_EXCLUSION_SEC
FS = _cfg.FS
PT_MATERNAL_BANDPASS_LOW = _cfg.PT_MATERNAL_BANDPASS_LOW
PT_MATERNAL_BANDPASS_HIGH = _cfg.PT_MATERNAL_BANDPASS_HIGH
PT_FETAL_BANDPASS_LOW = _cfg.PT_FETAL_BANDPASS_LOW
PT_FETAL_BANDPASS_HIGH = _cfg.PT_FETAL_BANDPASS_HIGH
PT_INTEGRATION_WINDOW_SEC = _cfg.PT_INTEGRATION_WINDOW_SEC
PT_THRESHOLD_FACTOR = _cfg.PT_THRESHOLD_FACTOR

from scipy.signal import butter, filtfilt, find_peaks
from preprocessing.qrs_detector import pan_tompkins, compute_hr_stats


def run_ica(signals: np.ndarray,
            n_components: int = ICA_N_COMPONENTS) -> tuple[np.ndarray, FastICA]:
    """
    Run FastICA on multichannel signal.

    [FIX-1] Uses whiten='arbitrary-variance' instead of 'unit-variance'.
    Preserves inter-channel amplitude ratios that encode mixing coefficients.
    """
    variances   = np.var(signals, axis=1)
    active_mask = variances > 1e-10
    active_idx  = np.where(active_mask)[0]
    n_active    = len(active_idx)

    if n_active == 0:
        raise ValueError("All input channels have zero variance -- cannot run ICA.")

    active_signals = signals[active_idx]
    n_comp_actual  = min(n_components, n_active)

    if n_active < n_components:
        print(f"[ICA] {n_components - n_active} zero-variance channel(s) excluded "
              f"-- running ICA with {n_comp_actual} components on {n_active} channels")

    ica = FastICA(
        n_components=n_comp_actual,
        max_iter=ICA_MAX_ITER,
        random_state=ICA_RANDOM_STATE,
        tol=ICA_TOL,
        whiten='arbitrary-variance',   # [FIX-1]
    )
    ICs_active = ica.fit_transform(active_signals.T).T

    if n_comp_actual < n_components:
        N   = signals.shape[1]
        ICs = np.zeros((n_components, N), dtype=ICs_active.dtype)
        ICs[:n_comp_actual] = ICs_active
    else:
        ICs = ICs_active

    return ICs, ica


def _detect_peaks_adaptive(ic: np.ndarray, fs: int,
                            bp_low: float, bp_high: float,
                            min_hr_bpm: float, max_hr_bpm: float,
                            hr_lo_gate: float, hr_hi_gate: float
                            ) -> tuple[np.ndarray, float]:
    """
    [FIX-3] Adaptive threshold Pan-Tompkins peak detector.

    Tries progressively lower thresholds (like detect_fetal_qrs) but ONLY
    accepts a threshold if the resulting peaks have a mean HR inside
    [hr_lo_gate, hr_hi_gate]. This prevents noise/T-wave pickup from passing
    the HR gate by accident.

    Returns (best_peaks, mean_hr).  best_peaks may be empty if nothing found.
    """
    nyq      = 0.5 * fs
    b, a     = butter(2, [bp_low / nyq, bp_high / nyq], btype='band')
    filtered = filtfilt(b, a, ic)
    diff     = np.gradient(filtered)
    squared  = diff ** 2
    win      = max(1, int(PT_INTEGRATION_WINDOW_SEC * fs))
    intg     = np.convolve(squared, np.ones(win) / win, mode='same')

    sig_mean = np.mean(intg)
    sig_std  = np.std(intg)
    min_dist = int((60.0 / max_hr_bpm) * fs)

    best_peaks  = np.array([])
    best_mean_hr = np.nan

    # Start at configured threshold and progressively relax
    for factor in [PT_THRESHOLD_FACTOR, 0.5, 0.2, 0.08, 0.03, 0.01]:
        thr    = sig_mean + factor * sig_std
        pks, _ = find_peaks(intg, height=thr, distance=min_dist)
        if len(pks) < 4:
            continue
        rr = np.diff(pks) / fs
        if len(rr) == 0:
            continue
        mean_hr = float(60.0 / np.mean(rr))
        if hr_lo_gate <= mean_hr <= hr_hi_gate:
            # First in-range candidate wins (threshold is highest possible = cleanest)
            best_peaks   = pks
            best_mean_hr = mean_hr
            break

    return best_peaks, best_mean_hr


def score_maternal_ic(ic: np.ndarray, fs: int = FS) -> float:
    """
    Score an IC for how likely it is to be the maternal ECG.

    [FIX-2] np.var amplitude criterion removed (all ICs have similar variance
            under arbitrary-variance whitening).
    [FIX-3] Uses _detect_peaks_adaptive() with progressive threshold relaxation
            so that noisy NIFECGDB components (where PT_THRESHOLD_FACTOR=1.0
            finds no peaks at all) still get a meaningful score.

    Score = regularity * peak_completeness * (1 + kurtosis_bonus)
    """
    peaks, mean_hr = _detect_peaks_adaptive(
        ic, fs,
        bp_low=PT_MATERNAL_BANDPASS_LOW,
        bp_high=PT_MATERNAL_BANDPASS_HIGH,
        min_hr_bpm=50, max_hr_bpm=120,
        hr_lo_gate=MATERNAL_HR_MIN - 5,
        hr_hi_gate=MATERNAL_HR_MAX + 5,
    )

    if len(peaks) < 4 or np.isnan(mean_hr):
        return 0.0

    stats  = compute_hr_stats(peaks, fs)
    std_hr = stats["std_hr"]

    # Regularity: maternal HR is very regular (low CV)
    cv               = std_hr / (mean_hr + 1e-8)
    regularity_score = 1.0 / (1.0 + cv * 10)

    # Peak completeness: how many peaks did we find vs expected?
    n_samples      = len(ic)
    expected_peaks = (mean_hr / 60.0) * (n_samples / fs)
    peak_ratio     = min(1.0, len(peaks) / (expected_peaks + 1e-6))

    # Kurtosis bonus: ECG QRS spikes give super-Gaussian distribution
    kurt       = float(kurtosis(ic, fisher=True))
    kurt_score = np.clip(kurt / 20.0, 0.0, 1.0)

    return float(regularity_score * peak_ratio * (1.0 + kurt_score))


def select_maternal_ic(ICs: np.ndarray, fs: int = FS) -> tuple[int, list[float]]:
    """
    Select the maternal IC blindly from ICA1 components.

    [FIX-4] If all physiological scores are zero (very noisy recording),
    falls back to highest-variance IC and prints a WARNING so the user knows.
    """
    scores   = [score_maternal_ic(ic, fs) for ic in ICs]
    best_idx = int(np.argmax(scores))

    # [FIX-4] Variance fallback only when all scores are zero
    if max(scores) < 1e-9:
        variances = [float(np.var(ic)) for ic in ICs]
        best_idx  = int(np.argmax(variances))
        print(f"[ICA] WARNING: all maternal IC scores zero -- "
              f"falling back to highest-variance IC (IC{best_idx+1}). "
              f"Maternal detection may be unreliable.")

    print(f"\n[ICA] Maternal IC selection scores:")
    for i, s in enumerate(scores):
        marker = " <- selected (maternal)" if i == best_idx else ""
        print(f"  IC{i+1}: {s:.4f}{marker}")

    return best_idx, scores


def score_fetal_ic(ic: np.ndarray, maternal_peaks: np.ndarray,
                   fs: int = FS) -> float:
    """
    Score an IC for how likely it is to be the fetal ECG.

    [FIX-3] Uses _detect_peaks_adaptive() for robust peak finding on noisy
            ICA2 residual components.
    [FIX-5] HR lower bound: 60 -> FETAL_HR_MIN (100 BPM).

    Score = independence * regularity * (1 + kurtosis_bonus)
    """
    peaks, mean_hr = _detect_peaks_adaptive(
        ic, fs,
        bp_low=PT_FETAL_BANDPASS_LOW,
        bp_high=PT_FETAL_BANDPASS_HIGH,
        min_hr_bpm=FETAL_HR_MIN - 10,
        max_hr_bpm=FETAL_HR_MAX + 10,
        hr_lo_gate=FETAL_HR_MIN,         # [FIX-5] was 60
        hr_hi_gate=FETAL_HR_MAX,
    )

    if len(peaks) < 5 or np.isnan(mean_hr):
        return 0.0

    stats  = compute_hr_stats(peaks, fs)
    std_hr = stats["std_hr"]

    # Temporal independence from maternal beats
    exclusion_samples     = int(ECHO_MATERNAL_EXCLUSION_SEC * fs)
    n_fetal_near_maternal = 0
    if len(maternal_peaks) > 0:
        for fp in peaks:
            if np.min(np.abs(maternal_peaks - fp)) < exclusion_samples:
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
    Select the fetal IC from ICA components.
    maternal_idx kept for API compatibility; exclusion is handled upstream.
    """
    scores   = [score_fetal_ic(ic, maternal_peaks, fs) for ic in ICs]
    best_idx = int(np.argmax(scores))

    if max(scores) < 1e-6 and residual is not None:
        print("\n[ICA] All fetal scores zero -- using residual correlation fallback")
        best_cc = -1
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
    """Extract, center and unit-normalise a single IC."""
    ic = ICs[idx].copy()
    ic = ic - np.mean(ic)
    ic = ic / (np.std(ic) + 1e-10)
    return ic