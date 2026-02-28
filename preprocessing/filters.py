"""
preprocessing/filters.py
Signal preprocessing: bandpass, notch, median filter, centering.

Design decisions:
- Butterworth bandpass (1-45 Hz): removes baseline drift and HF noise
  while preserving QRS (5-40 Hz), P and T waves.
- Notch at 50 Hz: suppresses power line interference (European standard).
- Median filter (kernel=3): removes impulsive spike artifacts without
  distorting QRS morphology.
- Centering only (zero-mean, preserve variance): ICA requires centered input.
  Dividing by std (z-score) discards amplitude ratios between channels that
  encode mixing coefficients — a common implementation error.

FIX: Removed scale_signal() call from preprocess_channel(). The docstring
rationale (preserve amplitude ratios for ICA) is now correctly implemented.
scale_signal() is retained as a utility for use AFTER ICA, e.g. before EKF.
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, medfilt
from config import (
    FS, BANDPASS_LOW, BANDPASS_HIGH, BANDPASS_ORDER,
    NOTCH_FREQ, NOTCH_QUALITY, MEDFILT_KERNEL
)
from config_nifecgdb import (
    FS, BANDPASS_LOW, BANDPASS_HIGH, BANDPASS_ORDER,
    NOTCH_FREQ, NOTCH_QUALITY, MEDFILT_KERNEL
)


def bandpass_filter(signal: np.ndarray, fs: int = FS,
                    lowcut: float = BANDPASS_LOW,
                    highcut: float = BANDPASS_HIGH,
                    order: int = BANDPASS_ORDER) -> np.ndarray:
    """
    Zero-phase Butterworth bandpass filter.
    Uses filtfilt (forward-backward) to eliminate phase distortion,
    which is critical for preserving QRS morphology.
    """
    nyq  = 0.5 * fs
    low  = lowcut  / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


def notch_filter(signal: np.ndarray, fs: int = FS,
                 freq: float = NOTCH_FREQ,
                 quality: float = NOTCH_QUALITY) -> np.ndarray:
    """
    IIR notch filter at power line frequency (50 Hz).
    Quality factor Q=30 gives a narrow notch (~1.7 Hz bandwidth).
    """
    b, a = iirnotch(w0=freq / (0.5 * fs), Q=quality)
    return filtfilt(b, a, signal)


def median_filter(signal: np.ndarray,
                  kernel_size: int = MEDFILT_KERNEL) -> np.ndarray:
    """
    Median filter for impulsive artifact removal.
    Kernel size 3 removes single-sample spikes without distorting QRS.
    """
    return medfilt(signal, kernel_size=kernel_size)


def center_signal(signal: np.ndarray) -> np.ndarray:
    """
    Remove DC offset (zero-mean). DO NOT scale variance.

    Rationale: FastICA requires zero-mean inputs. Dividing by std
    (z-score normalization) discards the amplitude ratios between
    channels that encode mixing coefficients — this is a subtle but
    important implementation error found in many published codebases.
    """
    return signal - np.mean(signal)


def scale_signal(signal: np.ndarray) -> np.ndarray:
    """
    Scale to unit variance.

    NOTE: Do NOT use this inside preprocess_channel() before ICA.
    Use only after ICA (e.g. before EKF) where amplitude ratios no
    longer matter and numerical stability is the priority.
    """
    std = np.std(signal)
    if std < 1e-10:
        return signal
    return signal / std


def preprocess_channel(signal: np.ndarray, fs: int = FS) -> np.ndarray:
    """
    Apply full preprocessing chain to a single channel.

    Pipeline:
        raw -> bandpass -> notch -> median -> center

    Amplitude variance is deliberately preserved (not z-scored) so that
    ICA can exploit inter-channel amplitude ratios when estimating the
    mixing matrix. See center_signal() docstring for full rationale.

    Parameters
    ----------
    signal : (N,) raw ECG channel
    fs     : sampling rate in Hz

    Returns
    -------
    (N,) preprocessed signal, zero-mean, original variance preserved
    """
    s = bandpass_filter(signal, fs)
    s = notch_filter(s, fs)
    s = median_filter(s)
    s = center_signal(s)
    # NOTE: scale_signal() intentionally NOT called here. See module docstring.
    return s


def preprocess_multichannel(signals: np.ndarray, fs: int = FS) -> np.ndarray:
    """
    Preprocess all channels independently.

    Parameters
    ----------
    signals : (n_channels, N)

    Returns
    -------
    (n_channels, N) preprocessed
    """
    return np.array([preprocess_channel(ch, fs) for ch in signals])


def normalize_for_display(signal: np.ndarray) -> np.ndarray:
    """
    Min-max normalize to [0, 1] — for visualization only.
    Never use this before ICA or metric computation.
    """
    rng = np.max(signal) - np.min(signal)
    if rng < 1e-10:
        return signal
    return (signal - np.min(signal)) / rng


def compute_snr_improvement(raw: np.ndarray, processed: np.ndarray,
                             reference: np.ndarray) -> dict:
    """
    Compute SNR before and after preprocessing against a reference signal.
    Used to validate that preprocessing improves signal quality.
    """
    def snr(est, ref):
        noise_power = np.var(est - ref)
        return 10 * np.log10(np.var(ref) / (noise_power + 1e-12))

    n = min(len(raw), len(processed), len(reference))
    raw_norm  = center_signal(raw[:n])
    proc_norm = processed[:n]
    ref_norm  = center_signal(reference[:n])

    return {
        "snr_raw_db"      : snr(raw_norm,  ref_norm),
        "snr_processed_db": snr(proc_norm, ref_norm),
    }
