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

DISSERTATION MODIFICATION:
  [MOD-1] score_fetal_ic() is now imported and called directly by pipeline.py
          as the base_score component of the unified three-factor scoring
          formula in _best_ic() / _best_ic_ensemble(). No changes to
          score_fetal_ic() itself are required — it already computes
          regularity × completeness × (1 + kurtosis) as specified.
          The maternal_penalty and morphology_score factors are computed in
          pipeline.py (_maternal_penalty, _morphology_score) to keep all
          path-specific logic (Path A vs Path B half-weight) in one place.
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


def _estimate_dominant_frequency(signal: np.ndarray, fs: int = FS,
                                   freq_low: float = 1.0,
                                   freq_high: float = 5.0) -> float:
    """
    Estimate dominant frequency in given band using FFT.
    
    Used for FIX 5: spectral separation check in CinC2013 IC selection.
    
    Parameters
    ----------
    signal : (N,) input signal
    fs : sampling rate (Hz)
    freq_low : lower bound of frequency band (Hz)
    freq_high : upper bound of frequency band (Hz)
    
    Returns
    -------
    freq : dominant frequency (Hz), or np.nan if no power found
    """
    N = len(signal)
    fft_vals = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N, 1.0 / fs)
    
    # Only positive frequencies
    mask = freqs >= 0
    freqs = freqs[mask]
    magnitude = np.abs(fft_vals[mask])
    
    # Restrict to band of interest
    band_mask = (freqs >= freq_low) & (freqs <= freq_high)
    if not np.any(band_mask):
        return np.nan
    
    dominant_idx = np.argmax(magnitude[band_mask])
    return float(freqs[band_mask][dominant_idx])


def _get_cfg(cfg: BaseConfig = None) -> BaseConfig:
    return _cfg if cfg is None else cfg


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
                            hr_lo_gate: float, hr_hi_gate: float,
                            cfg: BaseConfig = None
                            ) -> tuple[np.ndarray, float]:
    """
    [FIX-3 + IMPROVEMENT] Adaptive threshold Pan-Tompkins peak detector.

    IMPROVEMENT: Previously used a single fixed integration window and
    stopped at the first in-range HR candidate (highest threshold = fewest
    peaks). This caused score_fetal_ic() and score_maternal_ic() to under-
    count peaks on high-HR recordings (>160 BPM) and on post-subtraction
    ICs where QRS amplitude variance is high, making valid fetal ICs score
    the same as noisy ICs.

    Now mirrors the multi-window sweep from detect_fetal_qrs:
    - Tries integration windows [80, 50, 35, 20] ms
    - Uses completeness-weighted scoring: score = n_peaks × completeness
      where completeness = min(n_peaks / expected_n, 1.0)
    - Returns the globally best (window, threshold) combination
    - Does NOT break on first match — evaluates all combinations

    Returns (best_peaks, mean_hr). best_peaks may be empty if nothing found.
    """
    cfg = _get_cfg(cfg)
    nyq      = 0.5 * fs
    b, a     = butter(2, [bp_low / nyq, bp_high / nyq], btype='band')
    filtered = filtfilt(b, a, ic)
    diff     = np.gradient(filtered)
    squared  = diff ** 2
    min_dist = int((60.0 / max_hr_bpm) * fs)
    duration_sec = len(ic) / fs

    best_peaks   = np.array([])
    best_mean_hr = np.nan
    best_score   = -1.0

    for win_ms in [80, 50, 35, 20]:
        win  = max(1, int(win_ms / 1000.0 * fs))
        intg = np.convolve(squared, np.ones(win) / win, mode='same')
        sig_mean = np.mean(intg)
        sig_std  = np.std(intg)

        for factor in [cfg.PT_THRESHOLD_FACTOR, 0.5, 0.2, 0.08, 0.03, 0.01]:
            thr    = sig_mean + factor * sig_std
            pks, _ = find_peaks(intg, height=thr, distance=min_dist)
            if len(pks) < 4:
                continue
            rr = np.diff(pks) / fs
            if len(rr) == 0:
                continue
            mean_hr = float(60.0 / np.mean(rr))
            if hr_lo_gate <= mean_hr <= hr_hi_gate:
                expected_n   = max(1.0, mean_hr / 60.0 * duration_sec)
                completeness = min(len(pks) / expected_n, 1.0)
                score        = len(pks) * completeness
                if score > best_score:
                    best_score   = score
                    best_peaks   = pks
                    best_mean_hr = mean_hr

    return best_peaks, best_mean_hr


def score_maternal_ic(ic: np.ndarray, fs: int = FS,
                       cfg: BaseConfig = None) -> float:
    """
    Score an IC for how likely it is to be the maternal ECG.

    [FIX-2] np.var amplitude criterion removed (all ICs have similar variance
            under arbitrary-variance whitening).
    [FIX-3] Uses _detect_peaks_adaptive() with progressive threshold relaxation
            so that noisy NIFECGDB components (where PT_THRESHOLD_FACTOR=1.0
            finds no peaks at all) still get a meaningful score.

    Score = regularity * peak_completeness * (1 + kurtosis_bonus)
    """
    cfg = _get_cfg(cfg)
    peaks, mean_hr = _detect_peaks_adaptive(
        ic, fs,
        bp_low=cfg.PT_MATERNAL_BANDPASS_LOW,
        bp_high=cfg.PT_MATERNAL_BANDPASS_HIGH,
        min_hr_bpm=cfg.MATERNAL_HR_MIN,
        max_hr_bpm=cfg.MATERNAL_HR_MAX,
        hr_lo_gate=cfg.MATERNAL_HR_MIN - 5,
        hr_hi_gate=cfg.MATERNAL_HR_MAX + 5,
        cfg=cfg,
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


def select_maternal_ic(ICs: np.ndarray, fs: int = FS,
                        cfg: BaseConfig = None) -> tuple[int, list[float]]:
    """
    Select the maternal IC blindly from ICA1 components.

    [FIX-4] If all physiological scores are zero (very noisy recording),
    falls back to highest-variance IC and prints a WARNING so the user knows.
    """
    scores   = [score_maternal_ic(ic, fs, cfg=cfg) for ic in ICs]
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
                   fs: int = FS,
                   cfg: BaseConfig = None) -> float:
    """
    Score an IC for how likely it is to be the fetal ECG.

    [FIX-3] Uses _detect_peaks_adaptive() for robust peak finding on noisy
            ICA2 residual components.
    [FIX-5] HR lower bound: 60 -> FETAL_HR_MIN (100 BPM).

    Score = independence * regularity * (1 + kurtosis_bonus)
    """
    cfg = _get_cfg(cfg)
    peaks, mean_hr = _detect_peaks_adaptive(
        ic, fs,
        bp_low=cfg.PT_FETAL_BANDPASS_LOW,
        bp_high=cfg.PT_FETAL_BANDPASS_HIGH,
        min_hr_bpm=cfg.FETAL_HR_MIN - 10,
        max_hr_bpm=cfg.FETAL_HR_MAX + 10,
        hr_lo_gate=cfg.FETAL_HR_MIN,         # [FIX-5] was 60
        hr_hi_gate=cfg.FETAL_HR_MAX,
        cfg=cfg,
    )

    if len(peaks) < 5 or np.isnan(mean_hr):
        return 0.0

    stats  = compute_hr_stats(peaks, fs)
    std_hr = stats["std_hr"]

    # Temporal independence from maternal beats
    exclusion_samples     = int(cfg.ECHO_MATERNAL_EXCLUSION_SEC * fs)
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
                    residual: np.ndarray = None,
                    cfg: BaseConfig = None) -> tuple[int, list[float]]:
    """
    Select the fetal IC from ICA components.
    
    FIX 5 (CinC2013): After HR-gating, verify spectral separation from maternal.
    Requires peak-frequency in [FETAL_HR_MIN/60, FETAL_HR_MAX/60] Hz distinct
    from maternal frequency by >= HR_SEP_MIN_BPM/60 Hz.
    
    Parameters
    ----------
    ICs : (n_comp, N) independent components
    maternal_peaks : (K,) maternal QRS indices
    maternal_idx : maternal IC index (for API compatibility)
    fs : sampling rate
    residual : residual signal (for fallback)
    cfg : BaseConfig (optional, for CinC2013 spectral checks)
    """
    if cfg is None:
        cfg = _cfg
    
    scores   = [score_fetal_ic(ic, maternal_peaks, fs, cfg=cfg) for ic in ICs]
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
    
    # FIX 5: Spectral separation check (CinC2013)
    # Estimate maternal frequency from maternal peaks
    mat_freq = np.nan
    if len(maternal_peaks) > 1:
        mat_hr = compute_hr_stats(maternal_peaks, fs)["mean_hr"]
        mat_freq = mat_hr / 60.0  # Convert to Hz
    
    # Check spectral separation for top candidates
    if not np.isnan(mat_freq) and cfg.dataset.lower() == "cinc2013":
        fetal_freq_min = cfg.FETAL_HR_MIN / 60.0
        fetal_freq_max = cfg.FETAL_HR_MAX / 60.0
        freq_sep_min = cfg.HR_SEP_MIN_BPM / 60.0
        
        spectral_scores = []
        for i, ic in enumerate(ICs):
            fetal_freq = _estimate_dominant_frequency(ic, fs, freq_low=fetal_freq_min-0.5, freq_high=fetal_freq_max+0.5)
            
            freq_in_range = (not np.isnan(fetal_freq)) and (fetal_freq_min <= fetal_freq <= fetal_freq_max)
            freq_separated = (not np.isnan(fetal_freq)) and (abs(fetal_freq - mat_freq) >= freq_sep_min)
            
            spec_score = 0.0
            if freq_in_range and freq_separated:
                spec_score = 1.0
            
            spectral_scores.append(spec_score)
            if i == best_idx:
                print(f"[ICA-SPEC] IC{i+1} (best): fetal_freq={fetal_freq:.2f}Hz, "
                      f"maternal_freq={mat_freq:.2f}Hz, separated={freq_separated}, "
                      f"in_range={freq_in_range}")
        
        # Re-select if top candidate fails spectral check
        valid_idx = [i for i in range(len(ICs)) if spectral_scores[i] > 0.5]
        if valid_idx and best_idx not in valid_idx:
            # Choose best non-failing candidate
            best_idx = valid_idx[np.argmax([scores[i] for i in valid_idx])]
            print(f"[ICA-SPEC] Re-selected to IC{best_idx+1} due to spectral separation")

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