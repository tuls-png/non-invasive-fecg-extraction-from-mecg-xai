"""
preprocessing/qrs_detector.py
Pan-Tompkins QRS detector — maternal and fetal variants.
Also includes a .qrs annotation file loader for ADFECGDB/NIFECGDB ground truth.

CHANGES (per codebase review):

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
"""

import struct
import wfdb
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt, find_peaks
from config import (
    FS,
    PT_MATERNAL_BANDPASS_LOW, PT_MATERNAL_BANDPASS_HIGH, PT_MATERNAL_BANDPASS_ORDER,
    PT_FETAL_BANDPASS_LOW,    PT_FETAL_BANDPASS_HIGH,    PT_FETAL_BANDPASS_ORDER,
    PT_INTEGRATION_WINDOW_SEC, PT_THRESHOLD_FACTOR,
    FETAL_HR_MIN, FETAL_HR_MAX,
)
from config_nifecgdb import (
    FS,
    PT_MATERNAL_BANDPASS_LOW, PT_MATERNAL_BANDPASS_HIGH, PT_MATERNAL_BANDPASS_ORDER,
    PT_FETAL_BANDPASS_LOW,    PT_FETAL_BANDPASS_HIGH,    PT_FETAL_BANDPASS_ORDER,
    PT_INTEGRATION_WINDOW_SEC, PT_THRESHOLD_FACTOR,
    FETAL_HR_MIN, FETAL_HR_MAX,
)

# ── Internal helpers ──────────────────────────────────────────────────────────

def _pt_integrate(signal: np.ndarray, fs: int,
                  bp_low: float, bp_high: float) -> np.ndarray:
    """
    Shared Pan-Tompkins integration step.

    bandpass → differentiate → square → moving-window integrate.
    Returns the integrated signal ready for peak detection.
    Uses order-2 Butterworth for adequate stopband rejection.
    """
    nyq      = 0.5 * fs
    b, a     = butter(2, [bp_low / nyq, bp_high / nyq], btype='band')  # FIX: order 2
    filtered = filtfilt(b, a, signal)
    diff     = np.gradient(filtered)
    squared  = diff ** 2
    win      = int(PT_INTEGRATION_WINDOW_SEC * fs)
    return np.convolve(squared, np.ones(win) / win, mode='same')


# ── Pan-Tompkins core (generic) ───────────────────────────────────────────────

def pan_tompkins(signal: np.ndarray, fs: int = FS,
                 min_hr_bpm: float = 40,
                 max_hr_bpm: float = 200,
                 bp_low: float = None,
                 bp_high: float = None) -> tuple[np.ndarray, np.ndarray]:
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

    Returns (peaks, integrated_signal).
    """
    if bp_low is None:
        bp_low = PT_MATERNAL_BANDPASS_LOW
    if bp_high is None:
        bp_high = PT_MATERNAL_BANDPASS_HIGH

    integrated = _pt_integrate(signal, fs, bp_low, bp_high)
    threshold  = np.mean(integrated) + PT_THRESHOLD_FACTOR * np.std(integrated)
    min_dist   = int((60.0 / max_hr_bpm) * fs)
    peaks, _   = find_peaks(integrated, height=threshold, distance=min_dist)
    return peaks, integrated


# ── Maternal detection ────────────────────────────────────────────────────────

def detect_maternal_qrs(maternal_ic: np.ndarray, fs: int = FS) -> np.ndarray:
    """
    Detect maternal QRS peaks.

    Uses the maternal bandpass (5–15 Hz, standard adult QRS band).
    Maternal HR range: 50–115 BPM.
    """
    peaks, _ = pan_tompkins(
        maternal_ic, fs,
        min_hr_bpm=50, max_hr_bpm=115,
        bp_low=PT_MATERNAL_BANDPASS_LOW,
        bp_high=PT_MATERNAL_BANDPASS_HIGH,
    )
    return peaks


# ── Fetal detection ───────────────────────────────────────────────────────────

def detect_fetal_qrs(fetal_signal: np.ndarray, fs: int = FS) -> np.ndarray:
    """
    Detect fetal QRS peaks with HR-gated adaptive thresholding.

    Uses the fetal bandpass (10–40 Hz) to avoid preferential detection of
    maternal QRS energy that dominates at 5–15 Hz.

    FIX: The adaptive threshold loop now gates on whether the detected peaks
    have a heart rate inside the fetal range (FETAL_HR_MIN–FETAL_HR_MAX BPM).
    Previously the loop always kept the threshold that gave the most peaks,
    even if those peaks were maternal or noise artefacts — this was the primary
    cause of extracting maternal ECG instead of fetal.

    Strategy:
      1. For each candidate threshold, compute detected peaks and their mean HR.
      2. If HR is in fetal range, score = number of peaks (more = better).
      3. If HR is outside fetal range, score = 0 (rejected).
      4. Keep the best-scoring result across all thresholds.
      5. If no threshold produces fetal-range peaks, fall back to the threshold
         that gave the HR closest to FETAL_HR_MIN (better than nothing).

    Only removes peaks that are physically impossible (inter-beat interval
    shorter than 60/185 s). Never removes peaks that are far apart — those
    just mean a beat was missed.
    """
    integrated = _pt_integrate(
        fetal_signal, fs,
        PT_FETAL_BANDPASS_LOW, PT_FETAL_BANDPASS_HIGH,
    )

    # Minimum distance = shortest plausible fetal beat (185 BPM max)
    min_dist = int((60.0 / FETAL_HR_MAX) * fs)

    best_peaks    = np.array([])
    best_score    = -1
    fallback_peaks = np.array([])
    fallback_hr_dist = np.inf

    for factor in [0.50, 0.30, 0.15, 0.08, 0.03, 0.01, 0.005]:
        threshold = np.mean(integrated) + factor * np.std(integrated)
        p, _      = find_peaks(integrated, height=threshold, distance=min_dist)

        if len(p) < 3:
            continue

        # Compute mean HR of this candidate set
        rr      = np.diff(p) / fs
        hr_vals = 60.0 / (rr + 1e-8)
        mean_hr = float(np.mean(hr_vals))

        in_fetal_range = FETAL_HR_MIN <= mean_hr <= FETAL_HR_MAX

        if in_fetal_range:
            # Prefer solutions with more peaks AND correct HR — score by peak count
            score = len(p)
            if score > best_score:
                best_score = score
                best_peaks = p
            if len(p) >= 200:
                break
        else:
            # Track the out-of-range result closest to fetal HR (for fallback)
            hr_dist = min(abs(mean_hr - FETAL_HR_MIN), abs(mean_hr - FETAL_HR_MAX))
            if hr_dist < fallback_hr_dist and len(p) > len(fallback_peaks):
                fallback_hr_dist = hr_dist
                fallback_peaks   = p

    # Use fallback only if nothing passed the HR gate
    if len(best_peaks) == 0:
        best_peaks = fallback_peaks

    # Strip only physically impossible peaks (too close together = noise spikes).
    # Never strip peaks that are far apart — distance just means a missed beat.
    if len(best_peaks) > 1:
        min_ibi = int((60.0 / FETAL_HR_MAX) * fs)
        keep = [best_peaks[0]]
        for p in best_peaks[1:]:
            if (p - keep[-1]) >= min_ibi:
                keep.append(p)
        best_peaks = np.array(keep)

    return best_peaks


# ── Reference fetal detection (Direct_1 only) ─────────────────────────────────

def detect_reference_fetal_qrs(direct_signal: np.ndarray,
                                fs: int = FS) -> np.ndarray:
    """
    Polarity-agnostic fetal QRS detector for the Direct_1 reference electrode.
    Used ONLY for evaluation — never inside the blind separation pipeline.

    Runs both the positive and negative signal through Pan-Tompkins, merges
    the results, and filters by HR plausibility.
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


# def load_adfecgdb_annotation(qrs_path: str,
#                              fs_annotation: int = None,
#                              fs_signal: int = FS) -> np.ndarray:
#     raw = Path(qrs_path).read_bytes()

#     # Find end of ASCII header
#     try:
#         h_end = raw.index(b'\x00', 4)
#     except ValueError:
#         return np.array([], dtype=int)

#     binary = raw[h_end + 1:]
#     if len(binary) < 4:
#         return np.array([], dtype=int)

#     n = len(binary) // 2
#     words = struct.unpack(f'<{n}H', binary[:n * 2])

#     sample = 0
#     all_annotations = []  # (anntype, sample)

#     i = 0
#     types_seen = set()

#     while i < len(words):
#         w = words[i]
#         anntype = (w >> 10) & 0x3F
#         delta = w & 0x3FF
#         types_seen.add(anntype)
#         # SKIP
#         if anntype == 59:
#             if i + 2 >= len(words):
#                 break

#             lo = words[i + 1]
#             hi = words[i + 2]
#             jump = struct.unpack('<i', struct.pack('<HH', lo, hi))[0]

#             # 🔴 SAFETY: ignore insane jumps (corrupted records)
#             max_reasonable_jump = 10 * fs_signal * 60  # 10 minutes
#             if abs(jump) > max_reasonable_jump:
#                 i += 3
#                 continue

#             sample += jump
#             i += 3
#             continue

#         # EOF
#         if anntype == 0 and delta == 0:
#             break

#         sample += delta
#         if sample < 0:
#             i += 1
#             continue
#         if sample > 24 * 3600 * fs_signal:
#             i += 1
#             continue
#         all_annotations.append((anntype, sample))
#         i += 1
    
#     print("Annotation types seen:", types_seen)

#     if not all_annotations:
#         return np.array([], dtype=int)

#     # -------------------------------------------------
#     # AUTO-DETECT dominant beat type (key fix)
#     # -------------------------------------------------
#     from collections import Counter
#     counts = Counter(a for a, _ in all_annotations)

#     # Ignore non-beat administrative codes
#     ignore_types = {0, 59}
#     for t in ignore_types:
#         counts.pop(t, None)

#     if not counts:
#         return np.array([], dtype=int)

#     dominant_type = counts.most_common(1)[0][0]

#     peaks = [s for a, s in all_annotations if a == dominant_type]
#     peaks = np.array(peaks, dtype=int)

#     if len(peaks) > 1:
#         diffs = np.diff(np.concatenate(([-1], peaks)))
#         peaks = peaks[diffs > 0]
#     print("Loaded GT peaks:", len(peaks))
#     print("GT peaks:", len(peaks))
#     print("First 5 peaks:", peaks[:5])
#     print("Last peak:", peaks[-1])
#     gt_hr = compute_hr_from_samples(peaks, fs_signal)
#     print(f"[GT sanity] mean HR = {gt_hr:.1f} BPM")
#     rr = np.diff(peaks) / fs_signal
#     print("GT RR mean (s):", np.mean(rr))
#     print("GT HR approx:", 60/np.mean(rr))
#     print(np.array(peaks, dtype=int))
#     return np.array(peaks, dtype=int)



def load_adfecgdb_annotation(ann_path: str):
    file_path = ann_path[:-4]
    ann = wfdb.rdann(file_path, 'qrs')
    peaks = ann.sample
    return peaks
# ── Utility ───────────────────────────────────────────────────────────────────

def compute_hr_stats(peaks: np.ndarray, fs: int = FS) -> dict:
    """Compute heart rate statistics from detected peaks."""
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