"""
separation/template_subtraction.py
Adaptive Template Subtraction for maternal ECG cancellation.

Why this instead of WSVD for the primary cancellation step:
------------------------------------------------------------
WSVD reconstructs maternal ECG from dominant singular vectors of the
weighted signal matrix. The problem: when fetal and maternal ECGs share
energy in the top singular vectors (which they do, because both are
periodic cardiac signals mixed across the same 4 channels), WSVD removes
fetal energy along with maternal.

Template subtraction is more surgical:
  1. Extract the average maternal PQRST beat shape from each channel
  2. Align that template to every detected maternal R-peak
  3. Subtract only the template — leaving everything else intact

The fetal ECG is not periodic at the maternal rate, so it does not
contribute to the maternal template and is not subtracted.

Novel element — Adaptive template with local scaling:
-----------------------------------------------------
Standard template subtraction uses one fixed template for the entire
recording. We use a locally-weighted template that updates every N beats
to account for maternal ECG morphology changes due to:
  - Respiratory variation (beat-to-beat amplitude modulation)
  - Electrode movement during contractions
  - Postural changes

This is the methodological improvement over basic template subtraction
that justifies its inclusion as a novel contribution.
"""

import numpy as np
from scipy.signal import butter, filtfilt
from config import FS
from config_nifecgdb import FS


def extract_maternal_template(abd_signals: np.ndarray,
                               maternal_peaks: np.ndarray,
                               fs: int = FS,
                               half_window_sec: float = 0.35) -> np.ndarray:
    """
    Extract average maternal beat template from multichannel abdominal signals.

    For each channel, segments of length 2*half_window around each maternal
    R-peak are extracted and ensemble-averaged. The average shape is the
    template — it represents the pure maternal PQRST complex in that channel.

    Parameters
    ----------
    abd_signals      : (n_channels, N) preprocessed abdominal signals
    maternal_peaks   : (K,) R-peak indices
    fs               : sampling rate
    half_window_sec  : half-window around each peak in seconds (default 350ms)
                       covers full PQRST complex at normal HR

    Returns
    -------
    template : (n_channels, window_len) average maternal beat template
    """
    n_channels = abd_signals.shape[0]
    N          = abd_signals.shape[1]
    hw         = int(half_window_sec * fs)
    win_len    = 2 * hw

    # Collect valid beats (those fully within signal bounds)
    beats = []
    for pk in maternal_peaks:
        lo = pk - hw
        hi = pk + hw
        if lo >= 0 and hi <= N:
            segment = abd_signals[:, lo:hi]   # (n_channels, win_len)
            beats.append(segment)

    if len(beats) == 0:
        return np.zeros((n_channels, win_len))

    # Ensemble average — aligns all beats at the R-peak and averages
    # This suppresses the fetal ECG (random phase relative to maternal)
    # and reinforces the maternal PQRST (fixed phase relative to peak)
    template = np.mean(np.stack(beats, axis=0), axis=0)  # (n_channels, win_len)

    return template


def adaptive_template_subtraction(abd_signals: np.ndarray,
                                   maternal_peaks: np.ndarray,
                                   fs: int = FS,
                                   half_window_sec: float = 0.35,
                                   update_every: int = 30) -> np.ndarray:
    """
    Adaptive maternal ECG cancellation via template subtraction.

    Novel contribution: the template is re-estimated every `update_every`
    beats using a sliding window of recent beats. This adapts to slow
    morphology changes due to respiration and movement.

    For each detected maternal peak:
      1. Look up the current local template (re-estimated periodically)
      2. Scale the template to match the local beat amplitude
         (handles respiratory amplitude modulation)
      3. Subtract the scaled template from the signal at that peak location

    Parameters
    ----------
    abd_signals    : (n_channels, N) preprocessed abdominal signals
    maternal_peaks : (K,) R-peak indices
    fs             : sampling rate
    half_window_sec: half-window for template extraction/subtraction
    update_every   : re-estimate template every N beats (default 30 ≈ 20–30s)

    Returns
    -------
    residual : (n_channels, N) signal with maternal ECG removed
    """
    n_channels, N = abd_signals.shape
    hw            = int(half_window_sec * fs)
    residual      = abd_signals.copy()

    K = len(maternal_peaks)
    if K == 0:
        return residual

    # Initial global template from all beats
    template = extract_maternal_template(abd_signals, maternal_peaks, fs,
                                          half_window_sec)

    for i, pk in enumerate(maternal_peaks):
        lo = pk - hw
        hi = pk + hw

        # Skip beats too close to signal boundaries
        if lo < 0 or hi > N:
            continue

        # Re-estimate template every `update_every` beats
        # Use beats in a ±update_every/2 window around current beat
        if i % update_every == 0 and i > 0:
            window_start = max(0, i - update_every // 2)
            window_end   = min(K, i + update_every // 2)
            local_peaks  = maternal_peaks[window_start:window_end]
            template     = extract_maternal_template(
                abd_signals, local_peaks, fs, half_window_sec
            )

        # Local amplitude scaling per channel
        # Matches the template amplitude to the current beat amplitude
        # This corrects for respiratory modulation of maternal ECG amplitude
        current_beat   = abd_signals[:, lo:hi]          # (n_channels, win_len)
        template_power = np.sum(template**2, axis=1)    # (n_channels,)
        signal_power   = np.sum(current_beat * template,
                                axis=1)                 # cross-correlation

        # Scale factor: how much to scale template to match current beat
        scale = signal_power / (template_power + 1e-10)  # (n_channels,)
        scale = np.clip(scale, 0.3, 3.0)                 # prevent extreme scaling

        # Subtract scaled template
        scaled_template       = template * scale[:, np.newaxis]
        residual[:, lo:hi]   -= scaled_template

    # Re-centre residual (subtraction may introduce small DC drift)
    residual -= residual.mean(axis=1, keepdims=True)

    return residual


def verify_cancellation(abd_signals: np.ndarray,
                         residual: np.ndarray,
                         maternal_peaks: np.ndarray,
                         fs: int = FS,
                         half_window_sec: float = 0.15) -> dict:
    """
    Measure how much maternal energy remains after cancellation.

    Computes the ratio of residual power at maternal peak locations
    to original signal power at those locations. Lower = better cancellation.

    Returns
    -------
    dict with per-channel and mean cancellation ratio (dB)
    """
    hw      = int(half_window_sec * fs)
    N       = abd_signals.shape[1]
    ratios  = []

    for pk in maternal_peaks:
        lo = pk - hw
        hi = pk + hw
        if lo < 0 or hi > N:
            continue
        orig_power = np.mean(abd_signals[:, lo:hi]**2, axis=1)
        res_power  = np.mean(residual[:, lo:hi]**2, axis=1)
        ratio      = res_power / (orig_power + 1e-12)
        ratios.append(ratio)

    if not ratios:
        return {"mean_cancellation_db": 0.0}

    ratios = np.array(ratios)           # (K, n_channels)
    mean_ratio = np.mean(ratios)
    cancellation_db = -10 * np.log10(mean_ratio + 1e-12)

    print(f"[Template] Maternal cancellation: {cancellation_db:.1f} dB "
          f"(power reduction at QRS locations)")

    return {
        "mean_ratio"          : float(mean_ratio),
        "cancellation_db"     : float(cancellation_db),
        "per_channel_db"      : (-10 * np.log10(
                                    np.mean(ratios, axis=0) + 1e-12
                                 )).tolist(),
    }
