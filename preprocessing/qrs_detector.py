"""
preprocessing/qrs_detector.py

"""

import struct
import csv
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
PT_THRESHOLD_FACTOR = _cfg.PT_THRESHOLD_FACTOR


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
        min_hr_bpm = 50
        max_hr_bpm = 115
    else:
        bp_low = cfg.PT_MATERNAL_BANDPASS_LOW
        bp_high = cfg.PT_MATERNAL_BANDPASS_HIGH
        min_hr_bpm = cfg.MATERNAL_HR_MIN
        max_hr_bpm = cfg.MATERNAL_HR_MAX
    
    peaks, _ = pan_tompkins(
        maternal_ic, fs,
        min_hr_bpm=min_hr_bpm, max_hr_bpm=max_hr_bpm,
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
                      cfg: BaseConfig = None,
                      force_low_threshold: bool = False) -> np.ndarray:
    """
    Detect fetal QRS peaks using adaptive Pan-Tompkins with HR gating.

    Parameters
    ----------
    fetal_signal        : (N,) input signal
    fs                  : sampling rate (default from config)
    cfg                 : BaseConfig, optional. If provided, uses dataset-
                          specific parameters (PT_FETAL_BANDPASS_LOW/HIGH,
                          FETAL_HR_MIN/MAX). If None, uses module-level values.
    force_low_threshold : [FIX-RETRY] When True, the threshold search skips
                          the high factors and starts directly from the
                          sensitive range [0.08 .. 0.002]. Called by
                          pipeline.py Step 11 when initial detection yields
                          < 80% of expected beats. Recovers weak fetal beats
                          in under-detected Type A failure recordings (a06,
                          a11, a62) without globally lowering sensitivity.

    Returns
    -------
    np.ndarray — sample indices of detected QRS peaks
    """
    if cfg is None:
        fetal_hr_min  = FETAL_HR_MIN
        fetal_hr_max  = FETAL_HR_MAX
        bandpass_low  = PT_FETAL_BANDPASS_LOW
        bandpass_high = PT_FETAL_BANDPASS_HIGH
    else:
        fetal_hr_min  = cfg.FETAL_HR_MIN
        fetal_hr_max  = cfg.FETAL_HR_MAX
        bandpass_low  = cfg.PT_FETAL_BANDPASS_LOW
        bandpass_high = cfg.PT_FETAL_BANDPASS_HIGH

    best_peaks       = np.array([])
    best_score       = -1
    fallback_peaks   = np.array([])
    fallback_hr_dist = np.inf

    # [FIX-RETRY] Skip high threshold factors when force_low_threshold=True
    if force_low_threshold:
        threshold_factors = [0.08, 0.03, 0.01, 0.005, 0.003, 0.002]
    else:
        threshold_factors = [0.50, 0.30, 0.15, 0.08, 0.03, 0.01, 0.005]

    for window_ms in [80, 50, 35]:
        integrated = _pt_integrate_window(
            fetal_signal, fs,
            bandpass_low, bandpass_high,
            window_ms=window_ms, cfg=cfg
        )
        min_dist = int((60.0 / fetal_hr_max) * fs)

        for factor in threshold_factors:
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

# ── Peak-position diagnostic dump ────────────────────────────────────────────

def dump_peak_positions(rec_id: str,
                         detected_peaks: np.ndarray,
                         reference_peaks: np.ndarray,
                         fs: int,
                         out_dir: str = "peak_dumps",
                         tolerance_ms: float = 50.0) -> str:
    """
    Write a per-peak diagnostic CSV comparing detected vs reference peaks.

    Uses the exact same matching logic as evaluation.metrics.match_peaks, so
    the TP/FP/FN counts here are the same ones the reported F1 is built from
    -- this is not a separate/approximate comparison.

    Output columns:
        peak_type          : 'TP', 'FP', or 'FN'
        detected_sample     : sample index of the detected peak (TP, FP)
        detected_time_sec   : detected_sample / fs
        reference_sample    : sample index of the matched/missed reference peak (TP, FN)
        offset_ms            : signed (detected - reference) timing error in ms (TP only)

    How to read it for the "HR looks right but F1 is low" cases:
        - Many TP rows with small, randomly-signed offset_ms (e.g. +/-10-30ms)
          -> genuine timing jitter/precision noise in the detector or EKF.
        - Many TP rows with one dominant, consistently-signed offset_ms
          -> systematic lag/lead (e.g. EKF phase burn-in or filter group delay).
        - Mostly FP+FN with very few TP despite detected count ~= reference count
          -> peaks are landing at the wrong phase entirely (e.g. every other
          beat, or locked onto a maternal harmonic) -- not a jitter problem.

    Returns the path to the written CSV.
    """
    from evaluation.metrics import match_peaks

    detected_peaks  = np.asarray(detected_peaks)
    reference_peaks = np.asarray(reference_peaks)

    match = match_peaks(detected_peaks, reference_peaks, fs=fs, tolerance_ms=tolerance_ms)
    matched_det = {int(dp) for dp, rp in match["tp_pairs"]}
    matched_ref = {int(rp) for dp, rp in match["tp_pairs"]}

    rows = []
    for dp, rp in match["tp_pairs"]:
        offset_ms = (int(dp) - int(rp)) / fs * 1000.0
        rows.append(("TP", int(dp), dp / fs, int(rp), round(offset_ms, 2)))
    for dp in np.sort(detected_peaks):
        if int(dp) not in matched_det:
            rows.append(("FP", int(dp), round(dp / fs, 4), "", ""))
    for rp in np.sort(reference_peaks):
        if int(rp) not in matched_ref:
            rows.append(("FN", "", "", int(rp), ""))

    rows.sort(key=lambda r: r[1] if r[1] != "" else r[3])

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_path / f"{rec_id}_peaks.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["peak_type", "detected_sample", "detected_time_sec",
                         "reference_sample", "offset_ms"])
        writer.writerows(rows)

    print(f"[PEAK-DUMP] {rec_id}: {match['TP']} TP, {match['FP']} FP, "
          f"{match['FN']} FN -> {csv_path}")
    return str(csv_path)


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