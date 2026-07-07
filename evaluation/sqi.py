"""
evaluation/sqi.py
Signal Quality Index (SQI) for multi-channel ECG assessment.

Computes per-channel quality metrics:
  - Kurtosis (morphology sharpness, detects artifacts)
  - Variance (signal strength)
  - Band power 10-40 Hz (fetal QRS energy)
  - Maternal QRS consistency (if maternal peaks available)
  - Clipping ratio (amplitude saturation)
  - Flatline % (zero-variance sections)

Uses these to rank channels and gate bad channels before ICA.

DISSERTATION MODIFICATION [Enhancement Roadmap, Rank 2]:
  [NEW] compute_spectral_flatness / compute_rr_regularity / compute_candidate_sqi
        / sqi_weighted_fusion: promotes this module from a diagnostic /
        channel-ranking role to a first-class *trust-weighting* input for
        Path A / Path B / Path C fusion in pipeline.py, and for per-candidate
        weighting inside separation-stage IC ensembles. Rather than a single
        hard "chosen_path" selection, candidate scores are multiplied by a
        composite Signal Quality Index (kurtosis + spectral flatness +
        RR-regularity, following Mollakazemi et al. 2021) so that low-SNR
        candidates are down-weighted rather than winning ties purely on a
        noisy raw score. This targets exactly the CinC2013-specific
        heterogeneous-SNR gap identified in the literature review, while
        leaving well-behaved (typically ADFECGDB) recordings with near-
        uniform SQI weights and therefore near-unchanged behaviour.
"""

import numpy as np
from scipy.stats import kurtosis

_DEFAULT_FS = 1000


# ── Per-channel diagnostic sub-metrics (existing role) ─────────────────────

def compute_flatline_ratio(signal: np.ndarray, fs: int = _DEFAULT_FS,
                            window_sec: float = 1.0,
                            threshold: float = 1e-6) -> float:
    """
    Fraction of non-overlapping windows in which the signal is effectively
    flat (std below `threshold`). High values indicate electrode dropout
    or lead-off sections.
    """
    win = max(1, int(window_sec * fs))
    n_windows = len(signal) // win
    if n_windows == 0:
        return 0.0
    flat_count = 0
    for i in range(n_windows):
        seg = signal[i * win:(i + 1) * win]
        if np.std(seg) < threshold:
            flat_count += 1
    return float(flat_count / n_windows)


def compute_clipping_ratio(signal: np.ndarray,
                            threshold_pct: float = 0.98) -> float:
    """
    Fraction of samples whose absolute amplitude exceeds `threshold_pct`
    of the signal's peak absolute amplitude. High values indicate ADC
    saturation / amplitude clipping.
    """
    max_abs = float(np.max(np.abs(signal))) + 1e-12
    clip_thresh = threshold_pct * max_abs
    return float(np.mean(np.abs(signal) >= clip_thresh))


def compute_band_power(signal: np.ndarray, fs: int = _DEFAULT_FS,
                        freq_low: float = 10.0,
                        freq_high: float = 40.0) -> float:
    """
    Fraction of total spectral power falling within [freq_low, freq_high] Hz
    (default: fetal QRS energy band). Computed via a single-sided FFT PSD.
    """
    N = len(signal)
    if N < 2:
        return 0.0
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    psd = np.abs(np.fft.rfft(signal - np.mean(signal))) ** 2
    band_mask = (freqs >= freq_low) & (freqs <= freq_high)
    total = float(np.sum(psd)) + 1e-12
    return float(np.sum(psd[band_mask]) / total)


def compute_maternal_qrs_consistency(signal: np.ndarray,
                                      maternal_peaks: np.ndarray,
                                      fs: int = _DEFAULT_FS,
                                      window_ms: float = 100.0) -> float:
    """
    Beat-to-beat morphology consistency at the given maternal QRS locations:
    mean |correlation| of each beat window against the across-beat mean
    template. High values indicate a clean, physiologically consistent
    channel; low values indicate noise or motion artifact dominating the
    maternal QRS region.
    """
    if maternal_peaks is None or len(maternal_peaks) < 2:
        return 0.0
    hw = max(1, int(window_ms / 1000.0 * fs / 2))
    N = len(signal)
    windows = []
    for p in maternal_peaks:
        lo, hi = int(p) - hw, int(p) + hw
        if lo >= 0 and hi <= N:
            w = signal[lo:hi]
            w = w / (np.std(w) + 1e-10)
            windows.append(w)
    if len(windows) < 2:
        return 0.0
    min_len = min(len(w) for w in windows)
    arr = np.array([w[:min_len] for w in windows])
    template = np.mean(arr, axis=0)
    corrs = []
    for w in arr:
        try:
            c = float(np.corrcoef(w, template)[0, 1])
            if np.isfinite(c):
                corrs.append(abs(c))
        except Exception:
            pass
    return float(np.mean(corrs)) if corrs else 0.0


def compute_channel_sqi(signal: np.ndarray, fs: int = _DEFAULT_FS,
                         maternal_peaks: np.ndarray = None) -> float:
    """
    Composite per-channel Signal Quality Index in [0, 1], combining
    kurtosis (QRS sharpness), fetal-band power, maternal QRS consistency
    (if maternal_peaks provided) and a penalty for flatline/clipping.
    """
    kurt = float(kurtosis(signal, fisher=True))
    kurt_score = float(np.clip(kurt / 20.0, 0.0, 1.0))
    flat = compute_flatline_ratio(signal, fs)
    clip = compute_clipping_ratio(signal)
    band = compute_band_power(signal, fs, 10.0, 40.0)
    if maternal_peaks is not None and len(maternal_peaks) > 0:
        consistency = compute_maternal_qrs_consistency(signal, maternal_peaks, fs)
    else:
        consistency = 0.5  # neutral prior when no maternal reference is available

    quality_penalty = float(np.clip(flat + clip, 0.0, 1.0))
    raw = (0.30 * kurt_score + 0.25 * band + 0.25 * consistency
           + 0.20 * (1.0 - quality_penalty))
    return float(np.clip(raw, 0.0, 1.0))


def rank_channels_by_sqi(signals: np.ndarray, fs: int = _DEFAULT_FS,
                          maternal_peaks: np.ndarray = None) -> list:
    """
    Rank all channels of a (n_ch, N) multichannel signal by SQI, descending.
    Returns a list of (channel_index, sqi_score) tuples.
    """
    scores = [compute_channel_sqi(signals[ch], fs, maternal_peaks)
              for ch in range(signals.shape[0])]
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [(i, scores[i]) for i in order]


def select_best_channels(signals: np.ndarray, n_select: int,
                          fs: int = _DEFAULT_FS,
                          maternal_peaks: np.ndarray = None,
                          min_quality_thresh: float = None) -> list:
    """
    Select the indices of the `n_select` highest-SQI channels.
    If `min_quality_thresh` is given, channels below it are excluded first
    (falling back to the single best channel if that would leave none).
    Returns a sorted list of channel indices.
    """
    ranked = rank_channels_by_sqi(signals, fs, maternal_peaks)
    if min_quality_thresh is not None:
        filtered = [r for r in ranked if r[1] >= min_quality_thresh]
        ranked = filtered if filtered else ranked[:1]
    idxs = [i for i, _ in ranked[:n_select]]
    return sorted(idxs)


# ── [Rank 2, NEW] Candidate-level SQI and trust-weighted path fusion ───────

def compute_spectral_flatness(signal: np.ndarray, fs: int = _DEFAULT_FS,
                               freq_low: float = 0.5,
                               freq_high: float = 40.0) -> float:
    """
    Wiener entropy (spectral flatness) in [0, 1] over [freq_low, freq_high]:
    geometric_mean(PSD) / arithmetic_mean(PSD). Values near 1 indicate a
    noise-like (white) spectrum; values near 0 indicate a tonal/peaky
    spectrum, as expected from a periodic QRS-driven signal.
    """
    N = len(signal)
    if N < 2:
        return 1.0
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    psd = np.abs(np.fft.rfft(signal - np.mean(signal))) ** 2 + 1e-12
    band_mask = (freqs >= freq_low) & (freqs <= freq_high)
    p = psd[band_mask]
    if p.size == 0:
        return 1.0
    log_mean = float(np.mean(np.log(p)))
    gm = np.exp(log_mean)
    am = float(np.mean(p))
    return float(np.clip(gm / (am + 1e-12), 0.0, 1.0))


def compute_rr_regularity(peaks: np.ndarray, fs: int = _DEFAULT_FS) -> float:
    """
    RR-interval regularity score in [0, 1]: 1 / (1 + 10 * CV(RR)).
    Used as the "RR-regularity" term of the composite candidate SQI.
    """
    if peaks is None or len(peaks) < 3:
        return 0.0
    rr = np.diff(peaks) / fs
    mean_rr = float(np.mean(rr))
    if mean_rr <= 0:
        return 0.0
    cv = float(np.std(rr)) / mean_rr
    return float(1.0 / (1.0 + cv * 10.0))


def compute_candidate_sqi(signal: np.ndarray, peaks: np.ndarray,
                           fs: int = _DEFAULT_FS, cfg=None) -> float:
    """
    Composite per-candidate Signal Quality Index in [0, 1], following
    Mollakazemi et al. (2021): kurtosis + spectral flatness + RR-regularity.

    Used to *weight* (not just gate) Path A / Path B / Path C candidates
    in pipeline.py's SQI-weighted fusion step, and to weight candidates
    inside the ICA ensemble scoring.
    """
    kurt_norm = getattr(cfg, "SQI_KURTOSIS_NORM", 20.0) if cfg is not None else 20.0
    kurt = float(kurtosis(signal, fisher=True))
    kurt_score = float(np.clip(kurt / (kurt_norm + 1e-12), 0.0, 1.0))
    # Low spectral flatness (tonal / peaky spectrum) is high quality for a
    # QRS-driven signal, so invert.
    flatness_score = 1.0 - compute_spectral_flatness(signal, fs)
    regularity_score = compute_rr_regularity(peaks, fs)
    composite = (kurt_score + flatness_score + regularity_score) / 3.0
    return float(np.clip(composite, 0.0, 1.0))


def sqi_weighted_fusion(candidates: list, cfg=None) -> list:
    """
    Blend each candidate's existing selection score with its composite SQI
    to produce a trust-weighted fused score, for path/candidate fusion.

    Parameters
    ----------
    candidates : list of dict, each with keys:
        "label"  : candidate identifier (str)
        "signal" : (N,) candidate waveform
        "peaks"  : (K,) candidate QRS peak indices
        "fs"     : sampling rate for this candidate
        "score"  : existing (unified three-factor) selection score
    cfg : BaseConfig, optional — supplies SQI_FUSION_WEIGHT / SQI_KURTOSIS_NORM

    Returns
    -------
    list of dict — same candidates, each augmented with "sqi" and
    "fused_score" keys. fused_score = score * (1 - alpha + alpha * sqi),
    i.e. SQI can only down-weight a candidate relative to its raw score
    (never manufacture confidence a candidate's raw score didn't earn).
    """
    alpha = float(getattr(cfg, "SQI_FUSION_WEIGHT", 0.35)) if cfg is not None else 0.35
    alpha = float(np.clip(alpha, 0.0, 1.0))

    enriched = []
    for c in candidates:
        sqi = compute_candidate_sqi(c["signal"], c["peaks"], c.get("fs", _DEFAULT_FS), cfg=cfg)
        trust_factor = (1.0 - alpha) + alpha * sqi
        fused_score = float(c["score"] * trust_factor)
        enriched.append({**c, "sqi": sqi, "fused_score": fused_score})
    return enriched