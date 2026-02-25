"""
preprocessing/qrs_detector.py
Pan-Tompkins QRS detector — maternal and fetal variants.
Also includes a .qrs annotation file loader for ADFECGDB ground truth.
"""

import struct
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt, find_peaks
from config import (
    FS, PT_BANDPASS_LOW, PT_BANDPASS_HIGH,
    PT_INTEGRATION_WINDOW_SEC, PT_MIN_PEAK_DISTANCE_SEC,
    PT_THRESHOLD_FACTOR
)


# ── Pan-Tompkins core ──────────────────────────────────────────────────────────

def pan_tompkins(signal: np.ndarray, fs: int = FS,
                 min_hr_bpm: float = 40,
                 max_hr_bpm: float = 200) -> tuple[np.ndarray, np.ndarray]:
    """
    Pan-Tompkins QRS detector.
    Returns (peaks, integrated_signal).
    """
    nyq = 0.5 * fs
    b, a = butter(1, [PT_BANDPASS_LOW / nyq, PT_BANDPASS_HIGH / nyq],
                  btype='band')
    filtered   = filtfilt(b, a, signal)
    diff       = np.gradient(filtered)
    squared    = diff ** 2
    win        = int(PT_INTEGRATION_WINDOW_SEC * fs)
    integrated = np.convolve(squared, np.ones(win) / win, mode='same')
    threshold  = np.mean(integrated) + PT_THRESHOLD_FACTOR * np.std(integrated)
    min_dist   = int((60.0 / max_hr_bpm) * fs)
    peaks, _   = find_peaks(integrated, height=threshold, distance=min_dist)
    return peaks, integrated


# ── Maternal detection ─────────────────────────────────────────────────────────

def detect_maternal_qrs(maternal_ic: np.ndarray, fs: int = FS) -> np.ndarray:
    """Detect maternal QRS. Maternal HR range: 50-115 BPM."""
    peaks, _ = pan_tompkins(maternal_ic, fs, min_hr_bpm=50, max_hr_bpm=115)
    return peaks


# ── Fetal detection ───────────────────────────────────────────────────────────

def detect_fetal_qrs(fetal_signal: np.ndarray, fs: int = FS) -> np.ndarray:
    """
    Detect fetal QRS with adaptive thresholding on extracted ICA/EKF signal.

    Uses progressively lower thresholds until enough peaks are found.
    Only removes peaks that are physically impossible (too close together),
    never removes peaks that are far apart (those just mean a beat was missed).
    """
    nyq = 0.5 * fs
    b, a = butter(1, [PT_BANDPASS_LOW / nyq, PT_BANDPASS_HIGH / nyq],
                  btype='band')
    filtered   = filtfilt(b, a, fetal_signal)
    diff       = np.gradient(filtered)
    squared    = diff ** 2
    win        = int(PT_INTEGRATION_WINDOW_SEC * fs)
    integrated = np.convolve(squared, np.ones(win) / win, mode='same')

    # Min distance = shortest plausible fetal beat (185 BPM max)
    min_dist = int((60.0 / 185) * fs)

    # Try progressively lower thresholds; keep best result
    peaks = np.array([])
    for factor in [0.50, 0.30, 0.15, 0.08, 0.03, 0.01, 0.005]:
        threshold = np.mean(integrated) + factor * np.std(integrated)
        p, _      = find_peaks(integrated, height=threshold, distance=min_dist)
        if len(p) > len(peaks):
            peaks = p
        if len(p) >= 200:
            break

    # Only strip peaks that are TOO CLOSE (false positives from noise spikes).
    # Never strip peaks that are far apart — distance just means a missed beat.
    if len(peaks) > 1:
        min_ibi = int((60.0 / 185) * fs)
        keep = [peaks[0]]
        for p in peaks[1:]:
            if (p - keep[-1]) >= min_ibi:
                keep.append(p)
        peaks = np.array(keep)

    return peaks


# ── Reference fetal detection (Direct_1 only) ─────────────────────────────────

def detect_reference_fetal_qrs(direct_signal: np.ndarray,
                                fs: int = FS) -> np.ndarray:
    """
    Polarity-agnostic fetal QRS detector for Direct_1 reference electrode.
    Used ONLY for evaluation — never inside the blind separation pipeline.
    """
    def _pt_one_pass(sig, threshold_factor, min_dist_samples):
        nyq = 0.5 * fs
        b, a = butter(2, [5 / nyq, 20 / nyq], btype='band')
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


# ── ADFECGDB ground truth loader ───────────────────────────────────────────────

def load_adfecgdb_annotation(qrs_path: str,
                              fs_annotation: int = None,   # kept for API compat, ignored
                              fs_signal: int = FS) -> np.ndarray:
    """
    Load fetal R-peak annotations from an ADFECGDB / NIFECGDB .edf.qrs file.

    Both databases use the standard WFDB binary annotation format:
      - ASCII text header ending with a null byte
        (contains "## time resolution: 1000" — deltas are in signal fs units)
      - Little-endian uint16 words, each encoding:
            bits 15-10 : annotation type (anntype)
            bits  9-0  : sample delta (advance from previous annotation)
        Special type 59 (SKIP): next 2 words = 32-bit signed long jump
        Type 0 / delta 0: end-of-file marker
      - anntype == 1 ('N') marks a normal fetal beat

    Sample deltas are already in the signal's native sampling rate (1000 Hz),
    so no scaling is applied. The fs_annotation parameter is retained only
    for backward API compatibility and is not used.

    Returns
    -------
    peaks : (K,) int array of fetal R-peak sample indices, or empty array.
    """
    raw = Path(qrs_path).read_bytes()

    # Locate end of ASCII text header (first null byte after byte 4)
    try:
        h_end = raw.index(b'\x00', 4)
    except ValueError:
        return np.array([])

    binary = raw[h_end + 1:]
    if len(binary) < 4:
        return np.array([])

    n     = len(binary) // 2
    words = struct.unpack(f'<{n}H', binary[:n * 2])

    sample = 0
    peaks  = []
    i      = 0

    while i < len(words):
        w       = words[i]
        anntype = (w >> 10) & 0x3F
        delta   = w & 0x3FF

        # SKIP annotation: next 2 words encode a 32-bit absolute time jump
        if anntype == 59:
            i += 1
            lo = words[i]     if i     < len(words) else 0
            i += 1
            hi = words[i]     if i     < len(words) else 0
            sample += struct.unpack('<i', struct.pack('<HH', lo, hi))[0]
            i += 1
            continue

        # End-of-file marker
        if anntype == 0 and delta == 0:
            break

        sample += delta

        # anntype 1 = 'N' (normal beat) — the fetal R-peak annotation
        if anntype == 1:
            peaks.append(sample)

        i += 1

    return np.array(peaks, dtype=int)


# ── Utility ───────────────────────────────────────────────────────────────────

def compute_hr_stats(peaks: np.ndarray, fs: int = FS) -> dict:
    """Compute heart rate statistics from detected peaks."""
    if len(peaks) < 2:
        return {"mean_hr": np.nan, "std_hr": np.nan,
                "min_hr": np.nan, "max_hr": np.nan,
                "n_peaks": len(peaks), "hr_series": np.array([])}
    rr_intervals = np.diff(peaks) / fs
    hr_series    = 60.0 / rr_intervals
    return {
        "mean_hr"  : float(np.mean(hr_series)),
        "std_hr"   : float(np.std(hr_series)),
        "min_hr"   : float(np.min(hr_series)),
        "max_hr"   : float(np.max(hr_series)),
        "n_peaks"  : len(peaks),
        "hr_series": hr_series,
    }
