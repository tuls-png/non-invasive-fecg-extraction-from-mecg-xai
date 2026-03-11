"""
preprocessing/qrs_detector.py
Pan-Tompkins QRS detector — maternal and fetal variants.
Also includes a .qrs annotation file loader for ADFECGDB/NIFECGDB ground truth.

DATASET-SPECIFIC CONFIGURATION
==============================
By default, uses ADFECGDB base config (FETAL_HR_MIN=100, FETAL_HR_MAX=185).
To use dataset-specific values from YAML:

  Option 1 — Module-level initialization (affects all subsequent calls):
    from preprocessing.qrs_detector import initialize_qrs_detector
    initialize_qrs_detector("cinc2013")  # Loads cinc2013.yaml overrides (e.g., FETAL_HR_MAX=200)

  Option 2 — Per-function config (recommended for pipeline clarity):
    from configs import get_config
    cfg = get_config("cinc2013")
    peaks = detect_fetal_qrs(signal, cfg=cfg)

If a dataset YAML does not define FETAL_HR_MIN/MAX, base.py defaults are used.

CHANGES (per codebase review)
=============================
1. Separated maternal and fetal Pan-Tompkins bandpass configurations.
   - Maternal: 5–15 Hz (standard adult QRS band, order-2 Butterworth).
   - Fetal:   10–40 Hz (fetal QRS is short-duration and broadband; using
     5–15 Hz preferentially detects maternal energy that dominates ICA
     components with residual maternal contamination).

2. FIX — detect_fetal_qrs: added HR gating inside the adaptive threshold
   loop. Previously the detector lowered its threshold until it found 200
   peaks, keeping whichever threshold gave the most peaks regardless of
   whether those peaks had a fetal-range heart rate. At very low thresholds
   (factor=0.005) this detects T-waves and P-waves from residual maternal
   signal, producing an apparent "HR" of 130–160 BPM that passes the fetal
   HR filter downstream — the primary cause of the maternal-ECG extraction
   failure. The loop now prefers solutions with HR inside FETAL_HR_MIN–MAX.

3. FIX — Butterworth filter order raised from 1 → 2.
   Order-1 has very poor roll-off (~20 dB/decade); order-2 gives much
   better stopband rejection (~40 dB/decade) with minimal phase distortion
   after the filtfilt zero-phase application.

4. Enhanced config system support:
   - Added initialize_qrs_detector(dataset) for global module configuration.
   - detect_fetal_qrs() and detect_reference_fetal_qrs() now accept optional
     config parameter for per-call dataset overrides.
"""

import struct
import wfdb
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt, find_peaks
from configs import BaseConfig

# Module-level config (default). Can be overridden via initialize_qrs_detector(dataset).
_cfg = BaseConfig()

# Extract default parameters from base config
FS = _cfg.FS
PT_MATERNAL_BANDPASS_LOW = _cfg.PT_MATERNAL_BANDPASS_LOW
PT_MATERNAL_BANDPASS_HIGH = _cfg.PT_MATERNAL_BANDPASS_HIGH
PT_MATERNAL_BANDPASS_ORDER = _cfg.PT_MATERNAL_BANDPASS_ORDER
PT_FETAL_BANDPASS_LOW = _cfg.PT_FETAL_BANDPASS_LOW
PT_FETAL_BANDPASS_HIGH = _cfg.PT_FETAL_BANDPASS_HIGH
PT_FETAL_BANDPASS_ORDER = _cfg.PT_FETAL_BANDPASS_ORDER
PT_INTEGRATION_WINDOW_SEC = _cfg.PT_INTEGRATION_WINDOW_SEC
PT_THRESHOLD_FACTOR = _cfg.PT_THRESHOLD_FACTOR
FETAL_HR_MIN = _cfg.FETAL_HR_MIN
FETAL_HR_MAX = _cfg.FETAL_HR_MAX


def initialize_qrs_detector(dataset: str = "adfecgdb"):
    """
    Initialize QRS detector with dataset-specific configuration.
    Call this before processing to load dataset-specific values from YAML.
    
    Parameters
    ----------
    dataset : str
        Dataset name ('adfecgdb', 'cinc2013', 'nifecgdb', etc.).
        Loads dataset.yaml if it exists and applies overrides to module globals.
    
    Examples
    --------
    >>> initialize_qrs_detector("cinc2013")  # Uses cinc2013.yaml values
    """
    global _cfg, FS
    global PT_MATERNAL_BANDPASS_LOW, PT_MATERNAL_BANDPASS_HIGH, PT_MATERNAL_BANDPASS_ORDER
    global PT_FETAL_BANDPASS_LOW, PT_FETAL_BANDPASS_HIGH, PT_FETAL_BANDPASS_ORDER
    global PT_INTEGRATION_WINDOW_SEC, PT_THRESHOLD_FACTOR
    global FETAL_HR_MIN, FETAL_HR_MAX
    
    _cfg = BaseConfig(dataset=dataset)
    
    # Update all module-level variables from the new config
    FS = _cfg.FS
    PT_MATERNAL_BANDPASS_LOW = _cfg.PT_MATERNAL_BANDPASS_LOW
    PT_MATERNAL_BANDPASS_HIGH = _cfg.PT_MATERNAL_BANDPASS_HIGH
    PT_MATERNAL_BANDPASS_ORDER = _cfg.PT_MATERNAL_BANDPASS_ORDER
    PT_FETAL_BANDPASS_LOW = _cfg.PT_FETAL_BANDPASS_LOW
    PT_FETAL_BANDPASS_HIGH = _cfg.PT_FETAL_BANDPASS_HIGH
    PT_FETAL_BANDPASS_ORDER = _cfg.PT_FETAL_BANDPASS_ORDER
    PT_INTEGRATION_WINDOW_SEC = _cfg.PT_INTEGRATION_WINDOW_SEC
    PT_THRESHOLD_FACTOR = _cfg.PT_THRESHOLD_FACTOR
    FETAL_HR_MIN = _cfg.FETAL_HR_MIN
    FETAL_HR_MAX = _cfg.FETAL_HR_MAX

# ── Internal helpers ──────────────────────────────────────────────────────────

def _pt_integrate(signal: np.ndarray, fs: int,
                  bp_low: float, bp_high: float, cfg: BaseConfig = None) -> np.ndarray:
    """
    Shared Pan-Tompkins integration step.

    bandpass → differentiate → square → moving-window integrate.
    Returns the integrated signal ready for peak detection.
    Uses order-2 Butterworth for adequate stopband rejection.
    
    Parameters
    ----------
    signal : input signal
    fs : sampling rate
    bp_low : bandpass low cutoff (Hz)
    bp_high : bandpass high cutoff (Hz)
    cfg : BaseConfig, optional — uses cfg.PT_INTEGRATION_WINDOW_SEC if provided
    """
    if cfg is None:
        integration_window = PT_INTEGRATION_WINDOW_SEC
    else:
        integration_window = cfg.PT_INTEGRATION_WINDOW_SEC
    
    nyq      = 0.5 * fs
    b, a     = butter(2, [bp_low / nyq, bp_high / nyq], btype='band')  # FIX: order 2
    filtered = filtfilt(b, a, signal)
    diff     = np.gradient(filtered)
    squared  = diff ** 2
    win      = int(integration_window * fs)
    return np.convolve(squared, np.ones(win) / win, mode='same')


# ── Pan-Tompkins core (generic) ───────────────────────────────────────────────

def pan_tompkins(signal: np.ndarray, fs: int = FS,
                 min_hr_bpm: float = 40,
                 max_hr_bpm: float = 200,
                 bp_low: float = None,
                 bp_high: float = None,
                 cfg: BaseConfig = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Pan-Tompkins QRS detector.

    Parameters
    ----------
    signal      : (N,) input signal
    fs          : sampling rate
    min_hr_bpm  : minimum expected HR (used for minimum peak distance)
    max_hr_bpm  : maximum expected HR (used for minimum peak distance)
    bp_low      : bandpass lower cutoff Hz (defaults to maternal band)
    bp_high     : bandpass upper cutoff Hz (defaults to maternal band)
    cfg         : BaseConfig, optional — uses config values if provided

    Returns (peaks, integrated_signal).
    """
    if bp_low is None:
        bp_low = cfg.PT_MATERNAL_BANDPASS_LOW if cfg is not None else PT_MATERNAL_BANDPASS_LOW
    if bp_high is None:
        bp_high = cfg.PT_MATERNAL_BANDPASS_HIGH if cfg is not None else PT_MATERNAL_BANDPASS_HIGH

    if cfg is None:
        threshold_factor = PT_THRESHOLD_FACTOR
    else:
        threshold_factor = cfg.PT_THRESHOLD_FACTOR

    integrated = _pt_integrate(signal, fs, bp_low, bp_high, cfg=cfg)
    threshold  = np.mean(integrated) + threshold_factor * np.std(integrated)
    min_dist   = int((60.0 / max_hr_bpm) * fs)
    peaks, _   = find_peaks(integrated, height=threshold, distance=min_dist)
    return peaks, integrated


# ── Maternal detection ────────────────────────────────────────────────────────

def detect_maternal_qrs(maternal_ic: np.ndarray, fs: int = FS, cfg: BaseConfig = None) -> np.ndarray:
    """
    Detect maternal QRS peaks.

    Uses the maternal bandpass (5–15 Hz, standard adult QRS band).
    Maternal HR range: 50–115 BPM (or dataset-specific if cfg provided).
    
    Parameters
    ----------
    maternal_ic : input signal
    fs : sampling rate
    cfg : BaseConfig, optional — uses config values if provided
    """
    if cfg is None:
        bp_low = PT_MATERNAL_BANDPASS_LOW
        bp_high = PT_MATERNAL_BANDPASS_HIGH
    else:
        bp_low = cfg.PT_MATERNAL_BANDPASS_LOW
        bp_high = cfg.PT_MATERNAL_BANDPASS_HIGH
    
    peaks, _ = pan_tompkins(
        maternal_ic, fs,
        min_hr_bpm=50, max_hr_bpm=115,
        bp_low=bp_low,
        bp_high=bp_high,
        cfg=cfg,
    )
    return peaks


# ── Fetal detection ───────────────────────────────────────────────────────────

def _pt_integrate_window(signal, fs, bp_low, bp_high, window_ms=80, cfg: BaseConfig = None):
    """
    Pan-Tompkins integration with custom window size.
    
    Parameters
    ----------
    signal : input signal
    fs : sampling rate
    bp_low : bandpass low cutoff (Hz)
    bp_high : bandpass high cutoff (Hz)
    window_ms : integration window in milliseconds
    cfg : BaseConfig, optional
    """
    nyq = 0.5 * fs
    b, a = butter(2, [bp_low / nyq, bp_high / nyq], btype='band')
    filtered = filtfilt(b, a, signal)
    diff = np.gradient(filtered)
    squared = diff ** 2
    win = max(1, int(window_ms / 1000.0 * fs))
    return np.convolve(squared, np.ones(win) / win, mode='same')

def detect_fetal_qrs(fetal_signal: np.ndarray, fs: int = FS, 
                      cfg: BaseConfig = None) -> np.ndarray:
    """
    Detect fetal QRS peaks using adaptive Pan-Tompkins with HR gating.
    
    Parameters
    ----------
    fetal_signal : (N,) input signal
    fs : sampling rate (default from config)
    cfg : BaseConfig, optional
        Dataset-specific configuration. If provided, uses all fetal-specific parameters
        from this config (PT_FETAL_BANDPASS_LOW/HIGH, FETAL_HR_MIN/MAX, etc.).
        If None, uses module-level values (set via initialize_qrs_detector).
    
    Returns
    -------
    np.ndarray — sample indices of detected QRS peaks
    """
    # Use provided config or fall back to module level
    if cfg is None:
        fetal_hr_min = FETAL_HR_MIN
        fetal_hr_max = FETAL_HR_MAX
        bandpass_low = PT_FETAL_BANDPASS_LOW
        bandpass_high = PT_FETAL_BANDPASS_HIGH
    else:
        fetal_hr_min = cfg.FETAL_HR_MIN
        fetal_hr_max = cfg.FETAL_HR_MAX
        bandpass_low = cfg.PT_FETAL_BANDPASS_LOW
        bandpass_high = cfg.PT_FETAL_BANDPASS_HIGH
    print('fetal_hr_min', fetal_hr_min)
    print('fetal_hr_max', fetal_hr_max)
    best_peaks = np.array([])
    best_score = -1
    fallback_peaks = np.array([])
    fallback_hr_dist = np.inf

    # Try multiple integration windows — shorter windows help high HR detection
    for window_ms in [80, 50, 35]:
        integrated = _pt_integrate_window(
            fetal_signal, fs,
            bandpass_low, bandpass_high,
            window_ms=window_ms, cfg=cfg
        )
        min_dist = int((60.0 / fetal_hr_max) * fs)

        for factor in [0.50, 0.30, 0.15, 0.08, 0.03, 0.01, 0.005]:
            threshold = np.mean(integrated) + factor * np.std(integrated)
            p, _ = find_peaks(integrated, height=threshold, distance=min_dist)
            if len(p) < 3:
                continue
            rr = np.diff(p) / fs
            hr_vals = 60.0 / (rr + 1e-8)
            mean_hr = float(np.mean(hr_vals))
            in_fetal_range = fetal_hr_min <= mean_hr <= fetal_hr_max

            if in_fetal_range:
                score = len(p)
                if score > best_score:
                    best_score = score
                    best_peaks = p
                if len(p) >= 200:
                    break
            else:
                hr_dist = min(
                    abs(mean_hr - fetal_hr_min),
                    abs(mean_hr - fetal_hr_max)
                )
                if hr_dist < fallback_hr_dist and len(p) > len(fallback_peaks):
                    fallback_hr_dist = hr_dist
                    fallback_peaks = p

    if len(best_peaks) == 0:
        best_peaks = fallback_peaks

    # Strip physically impossible peaks
    if len(best_peaks) > 1:
        min_ibi = int((60.0 / fetal_hr_max) * fs)
        keep = [best_peaks[0]]
        for p in best_peaks[1:]:
            if (p - keep[-1]) >= min_ibi:
                keep.append(p)
        best_peaks = np.array(keep)

    return best_peaks


# ── Reference fetal detection (Direct_1 only) ─────────────────────────────────

def detect_reference_fetal_qrs(direct_signal: np.ndarray,
                                fs: int = FS,
                                cfg: BaseConfig = None) -> np.ndarray:
    """
    Polarity-agnostic fetal QRS detector for the Direct_1 reference electrode.
    Used ONLY for evaluation — never inside the blind separation pipeline.

    Runs both the positive and negative signal through Pan-Tompkins, merges
    the results, and filters by HR plausibility.
    
    Parameters
    ----------
    direct_signal : (N,) reference electrode signal
    fs : sampling rate
    cfg : BaseConfig, optional
        Dataset-specific configuration. If None, uses module-level defaults.
    """
    def _pt_one_pass(sig, threshold_factor, min_dist_samples):
        nyq  = 0.5 * fs
        b, a = butter(2, [5 / nyq, 20 / nyq], btype='band')   # FIX: order 2
        filt = filtfilt(b, a, sig)
        diff = np.gradient(filt)
        sq   = diff ** 2
        win  = int(0.08 * fs)
        intg = np.convolve(sq, np.ones(win) / win, mode='same')
        thr  = np.mean(intg) + threshold_factor * np.std(intg)
        pks, _ = find_peaks(intg, height=thr, distance=min_dist_samples)
        return pks

    def _merge(peaks_a, peaks_b, min_sep):
        combined = np.sort(np.concatenate([peaks_a, peaks_b]))
        if len(combined) == 0:
            return combined
        keep = [combined[0]]
        for p in combined[1:]:
            if p - keep[-1] >= min_sep:
                keep.append(p)
        return np.array(keep)

    def _filter_hr(peaks, min_bpm=90, max_bpm=210):
        if len(peaks) < 2:
            return peaks
        valid = [peaks[0]]
        for p in peaks[1:]:
            hr = 60.0 / ((p - valid[-1]) / fs)
            if min_bpm <= hr <= max_bpm:
                valid.append(p)
        return np.array(valid)

    min_dist   = int(0.28 * fs)
    best_peaks = np.array([])

    for factor in [0.20, 0.10, 0.05, 0.02, 0.01]:
        pks_pos  = _pt_one_pass(direct_signal,  factor, min_dist)
        pks_neg  = _pt_one_pass(-direct_signal, factor, min_dist)
        merged   = _merge(pks_pos, pks_neg, min_dist // 2)
        filtered = _filter_hr(merged)
        if len(filtered) > len(best_peaks):
            best_peaks = filtered
        if len(filtered) >= 400:
            break

    return best_peaks


# ── ADFECGDB / NIFECGDB ground truth loader ──────────────────────────────────
def compute_hr_from_samples(peaks, fs):
    if len(peaks) < 2:
        return np.nan

    rr_samples = np.diff(peaks)
    rr_sec = rr_samples / fs

    hr = 60.0 / np.mean(rr_sec)
    return hr

def load_adfecgdb_annotation(ann_path: str):
    file_path = ann_path[:-4]
    ann = wfdb.rdann(file_path, 'qrs')
    peaks = ann.sample
    return peaks

def load_wfdb_annotation(record_stem: str, extension: str = 'qrs') -> np.ndarray:
    """
    General WFDB annotation loader for any extension.

    Parameters
    ----------
    record_stem : str
        Path WITHOUT extension (e.g. '/path/to/set-a/a01').
        For ADFECGDB: pass ann_path[:-4] (strips '.qrs' from full path).
        For CinC2013: pass annotation_path directly (already a stem).
    extension : str
        Annotation extension without dot.
        'qrs'  — ADFECGDB fetal ground truth
        'fqrs' — CinC2013 fetal ground truth

    Returns
    -------
    np.ndarray of int — sample indices of annotated beats
    """
    ann = wfdb.rdann(record_stem, extension)
    return ann.sample
# ── Utility ───────────────────────────────────────────────────────────────────

def compute_hr_stats(peaks: np.ndarray, fs: int = FS, cfg: BaseConfig = None) -> dict:
    """
    Compute heart rate statistics from detected peaks.
    
    Parameters
    ----------
    peaks : array of peak indices
    fs : sampling rate
    cfg : BaseConfig, optional — not currently needed but included for API consistency
    """
    if len(peaks) < 2:
        return {"mean_hr": np.nan, "std_hr": np.nan,
                "min_hr": np.nan, "max_hr": np.nan,
                "n_peaks": len(peaks), "hr_series": np.array([])}
    rr_intervals = np.diff(peaks) / fs
    rr_intervals = rr_intervals[rr_intervals > 0]
    hr_series    = 60.0 / rr_intervals
    return {
        "mean_hr"  : float(np.mean(hr_series)),
        "std_hr"   : float(np.std(hr_series)),
        "min_hr"   : float(np.min(hr_series)),
        "max_hr"   : float(np.max(hr_series)),
        "n_peaks"  : len(peaks),
        "hr_series": hr_series,
    }