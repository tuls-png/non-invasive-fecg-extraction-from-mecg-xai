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

NEW FEATURES:
  [NEW-1] pca_n_components(): PCA pre-screening to determine how many ICA
          components to request per recording. Uses variance explained to retain
          only components explaining >PCA_VARIANCE_THRESHOLD of total variance,
          hard-capped at PCA_MAX_COMPONENTS. Addresses noise-IC contamination
          that dilutes the candidate pool.
  [NEW-2] run_ica_ensemble(): ICA ensemble with consensus scoring. Runs FastICA
          N=ICA_ENSEMBLE_RUNS times with different random seeds, applies
          three-factor scoring to all N x n_components candidates, picks the
          global winner. Addresses initialisation instability causing complete
          failures on difficult recordings (a60, a47, a34 patterns).
  [NEW-3] Three-factor IC scoring (base x penalty x morphology_boost):
          - base: independence * regularity * (1 + kurtosis_bonus)
          - penalty: suppresses ICs whose dominant frequency is at 2x maternal HR
            (second harmonic of maternal ECG residual)
          - morphology_boost: rewards ICs with a clean QRS morphology (sharp
            kurtosis peak), gated on minimum peak count to avoid noise bias.
"""

import numpy as np
from scipy.stats import kurtosis
from sklearn.decomposition import FastICA, PCA
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

# [NEW-1] PCA pre-screening parameters
PCA_VARIANCE_THRESHOLD = 0.05   # retain components explaining >5% variance
# No arbitrary component cap. The hard upper bound is n_active_channels,
# enforced inside pca_n_components() from the data itself. The variance
# threshold does the real filtering work.

# [NEW-2] ICA ensemble parameters
ICA_ENSEMBLE_RUNS = 7           # number of ICA runs per path (5-10 range)
ICA_ENSEMBLE_SEEDS = [0, 1, 2, 3, 42, 99, 137]  # distinct seeds, len == RUNS

# [NEW-3] Three-factor scoring parameters
MORPH_BOOST_MIN_PEAKS = 8       # gate morphology boost on minimum peak count
HARMONIC_PENALTY_BANDWIDTH = 5  # BPM bandwidth around 2x maternal HR for penalty

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


# ---------------------------------------------------------------------------
# [NEW-1] PCA pre-screening
# ---------------------------------------------------------------------------

def pca_n_components(signals: np.ndarray,
                     variance_threshold: float = PCA_VARIANCE_THRESHOLD) -> int:
    """
    Use PCA variance explained to determine how many ICA components to request.

    Runs PCA on the active (non-zero-variance) channels and counts how many
    principal components explain more than `variance_threshold` of total
    variance. The only hard upper bound is n_active_channels — the
    mathematical limit of ICA — which is derived from the data itself.
    No dataset-specific caps are applied; the variance threshold does all
    the filtering work.

    The minimum returned value is 2 (ICA needs at least 2 components to be
    meaningful).

    Parameters
    ----------
    signals            : (n_ch, N) preprocessed multichannel signal
    variance_threshold : minimum explained variance fraction to retain a PC

    Returns
    -------
    n_components : int >= 2
    """
    # Identify active (non-zero-variance) channels
    variances      = np.var(signals, axis=1)
    active_mask    = variances > 1e-10
    active_signals = signals[active_mask]
    n_active       = active_signals.shape[0]

    if n_active < 2:
        return 2

    # Fit PCA up to the maximum mathematically possible components
    n_pca = min(n_active, signals.shape[1] - 1)
    if n_pca < 2:
        return 2

    try:
        pca = PCA(n_components=n_pca)
        pca.fit(active_signals.T)
        var_ratios = pca.explained_variance_ratio_
    except Exception as e:
        print(f"[PCA] PCA fitting failed ({e}) -- defaulting to n_active={n_active}")
        return max(2, n_active)

    # The variance threshold does the filtering; n_active is the only cap
    n_above      = int(np.sum(var_ratios > variance_threshold))
    n_components = max(2, min(n_above, n_active))

    print(f"[PCA] Variance ratios: "
          f"{', '.join(f'PC{i+1}:{r*100:.1f}%' for i, r in enumerate(var_ratios))}")
    print(f"[PCA] Components explaining >{variance_threshold*100:.0f}% variance: "
          f"{n_above} → using {n_components} (n_active_channels={n_active})")

    return n_components


# ---------------------------------------------------------------------------
# Core ICA runner (single run)
# ---------------------------------------------------------------------------

def run_ica(signals: np.ndarray,
            n_components: int = ICA_N_COMPONENTS,
            random_state: int = ICA_RANDOM_STATE) -> tuple[np.ndarray, FastICA]:
    """
    Run FastICA on multichannel signal.

    [FIX-1] Uses whiten='arbitrary-variance' instead of 'unit-variance'.
    Preserves inter-channel amplitude ratios that encode mixing coefficients.

    Parameters
    ----------
    signals      : (n_ch, N) preprocessed multichannel signal
    n_components : number of ICA components to extract
    random_state : random seed for reproducibility

    Returns
    -------
    ICs : (n_components, N) independent components
    ica : fitted FastICA object
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
        random_state=random_state,
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


# ---------------------------------------------------------------------------
# [NEW-2] ICA ensemble runner
# ---------------------------------------------------------------------------

def run_ica_ensemble(signals: np.ndarray,
                     n_components: int,
                     n_runs: int = ICA_ENSEMBLE_RUNS,
                     seeds: list = None) -> list[np.ndarray]:
    """
    Run FastICA N times with different random seeds and return all IC sets.

    This addresses initialisation instability: on difficult recordings a single
    ICA run may converge to a poor local solution with no physiological ICs.
    By running N times we sample the solution space and the downstream
    three-factor scorer picks the global best candidate across all N x k ICs.

    Parameters
    ----------
    signals      : (n_ch, N) preprocessed multichannel signal
    n_components : number of components per run (PCA-determined)
    n_runs       : number of ICA runs
    seeds        : list of random seeds (defaults to ICA_ENSEMBLE_SEEDS[:n_runs])

    Returns
    -------
    ic_sets : list of n_runs arrays, each (n_components, N)
              Failed runs (convergence errors) are omitted silently.
    """
    if seeds is None:
        seeds = ICA_ENSEMBLE_SEEDS[:n_runs]
        # If n_runs > len(ICA_ENSEMBLE_SEEDS), extend with sequential seeds
        if len(seeds) < n_runs:
            seeds = seeds + list(range(200, 200 + n_runs - len(seeds)))

    ic_sets = []
    for i, seed in enumerate(seeds[:n_runs]):
        try:
            ICs, _ = run_ica(signals, n_components=n_components, random_state=seed)
            ic_sets.append(ICs)
        except Exception as e:
            print(f"[ICA-ENS] Run {i+1} (seed={seed}) failed: {e} -- skipped")

    if not ic_sets:
        raise ValueError("[ICA-ENS] All ensemble runs failed -- cannot continue.")

    n_success = len(ic_sets)
    if n_success < n_runs:
        print(f"[ICA-ENS] {n_success}/{n_runs} runs succeeded")
    else:
        print(f"[ICA-ENS] All {n_runs} runs succeeded  "
              f"(n_components={n_components}, pool={n_runs * n_components} candidates)")

    return ic_sets


# ---------------------------------------------------------------------------
# Peak detection helper (unchanged from original)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# [NEW-3] Three-factor IC scoring
# ---------------------------------------------------------------------------

def _harmonic_penalty(ic: np.ndarray, fs: int, maternal_hr_bpm: float,
                       bandwidth_bpm: float = HARMONIC_PENALTY_BANDWIDTH) -> float:
    """
    Compute a suppression penalty for ICs whose dominant frequency sits at
    2 x maternal_HR (the second harmonic of residual maternal ECG).

    The penalty is 1.0 (no suppression) when the IC's dominant frequency is
    well clear of the harmonic, and approaches 0.0 when it falls exactly on
    the harmonic. This is smoother than a hard reject and allows partial credit
    for ICs that happen to have some energy near the harmonic.

    Parameters
    ----------
    ic            : (N,) independent component signal
    fs            : sampling rate
    maternal_hr_bpm : maternal HR in BPM
    bandwidth_bpm : BPM half-bandwidth around 2x HR to apply penalty

    Returns
    -------
    penalty : float in (0, 1]  (1.0 = no penalty, ~0 = strong suppression)
    """
    if np.isnan(maternal_hr_bpm) or maternal_hr_bpm <= 0:
        return 1.0

    # 2x maternal HR in Hz — the explicit harmonic [NEW-3 patch (b)]
    harmonic_hz = (2.0 * maternal_hr_bpm) / 60.0
    bw_hz       = bandwidth_bpm / 60.0

    dom_freq = _estimate_dominant_frequency(
        ic, fs,
        freq_low=max(0.5, harmonic_hz - bw_hz * 3),
        freq_high=harmonic_hz + bw_hz * 3,
    )

    if np.isnan(dom_freq):
        return 1.0

    dist_hz = abs(dom_freq - harmonic_hz)
    # Smooth suppression: full penalty within bw_hz, falls off outside
    if dist_hz <= bw_hz:
        # Linear taper: 0.15 at centre, 1.0 at bw_hz edge
        penalty = 0.15 + 0.85 * (dist_hz / bw_hz)
    else:
        penalty = 1.0

    return float(np.clip(penalty, 0.15, 1.0))


def _morphology_boost(ic: np.ndarray, fs: int, peaks: np.ndarray,
                       min_peaks: int = MORPH_BOOST_MIN_PEAKS) -> float:
    """
    Compute a morphology quality boost for ICs with clean QRS-like waveform.

    Uses the kurtosis of local peak windows: high kurtosis means concentrated,
    spike-like peaks consistent with QRS morphology. Low kurtosis means broad,
    noisy, or multi-modal waveforms.

    [NEW-3 patch (a)] Gated on minimum peak count: if fewer than min_peaks are
    found, the boost is 0.0 (no boost). This prevents noise-ICs with a small
    number of large artifacts from receiving an undeserved morphology reward.

    Parameters
    ----------
    ic        : (N,) independent component signal
    fs        : sampling rate
    peaks     : (K,) detected peak indices
    min_peaks : minimum peaks required to compute boost

    Returns
    -------
    boost : float >= 1.0 (1.0 = no boost, up to ~2.0 for excellent morphology)
    """
    # [NEW-3 patch (a)]: gate on minimum peak count
    if len(peaks) < min_peaks:
        return 1.0

    hw = int(0.25 * fs)   # 250 ms half-window around each peak
    window_kurtoses = []
    for p in peaks:
        lo = max(0, p - hw)
        hi = min(len(ic), p + hw)
        seg = ic[lo:hi]
        if len(seg) < 10:
            continue
        try:
            k = float(kurtosis(seg, fisher=True))
            window_kurtoses.append(k)
        except Exception:
            continue

    if not window_kurtoses:
        return 1.0

    median_kurt = float(np.median(window_kurtoses))
    # Soft boost: kurtosis of 3+ is typical QRS; cap boost at 2.0
    boost = 1.0 + float(np.clip(median_kurt / 10.0, 0.0, 1.0))
    return boost


def score_fetal_ic_three_factor(ic: np.ndarray, maternal_peaks: np.ndarray,
                                  fs: int, maternal_hr_bpm: float = np.nan) -> float:
    """
    [NEW-3] Three-factor IC scoring: base x penalty x morphology_boost.

    Factor 1 (base): independence * regularity * (1 + kurtosis_bonus)
      - independence: fraction of fetal peaks NOT temporally overlapping
        maternal beats
      - regularity: inverse coefficient of variation of RR intervals
      - kurtosis_bonus: super-Gaussian distribution typical of ECG QRS spikes

    Factor 2 (penalty): suppresses ICs at 2x maternal HR (harmonic artifact)
      - explicitly uses 2 * maternal_HR_BPM as the harmonic frequency

    Factor 3 (morphology_boost): rewards ICs with clean QRS morphology
      - gated: requires at least MORPH_BOOST_MIN_PEAKS detected peaks

    Parameters
    ----------
    ic              : (N,) independent component (unit-normalised)
    maternal_peaks  : (K,) maternal QRS indices
    fs              : sampling rate
    maternal_hr_bpm : maternal HR in BPM (needed for harmonic penalty)

    Returns
    -------
    score : float >= 0
    """
    peaks, mean_hr = _detect_peaks_adaptive(
        ic, fs,
        bp_low=PT_FETAL_BANDPASS_LOW,
        bp_high=PT_FETAL_BANDPASS_HIGH,
        min_hr_bpm=FETAL_HR_MIN - 10,
        max_hr_bpm=FETAL_HR_MAX + 10,
        hr_lo_gate=FETAL_HR_MIN,
        hr_hi_gate=FETAL_HR_MAX,
    )

    if len(peaks) < 5 or np.isnan(mean_hr):
        return 0.0

    # --- Factor 1: base score (independence * regularity * kurtosis_bonus) ---
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

    base = float(independence * regularity * (1.0 + kurt_score))

    # --- Factor 2: harmonic penalty (2 x maternal HR) ---
    penalty = _harmonic_penalty(ic, fs, maternal_hr_bpm)

    # --- Factor 3: morphology boost (gated on min peak count) ---
    boost = _morphology_boost(ic, fs, peaks)

    score = base * penalty * boost
    return float(score)


# ---------------------------------------------------------------------------
# Maternal IC scoring (unchanged in substance)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Fetal IC scoring (legacy single-score, kept for backward compat)
# ---------------------------------------------------------------------------

def score_fetal_ic(ic: np.ndarray, maternal_peaks: np.ndarray,
                   fs: int = FS) -> float:
    """
    Score an IC for how likely it is to be the fetal ECG.

    [FIX-3] Uses _detect_peaks_adaptive() for robust peak finding on noisy
            ICA2 residual components.
    [FIX-5] HR lower bound: 60 -> FETAL_HR_MIN (100 BPM).

    Score = independence * regularity * (1 + kurtosis_bonus)

    NOTE: For the main pipeline, prefer score_fetal_ic_three_factor() which
    adds harmonic penalty and morphology boost [NEW-3].
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


# ---------------------------------------------------------------------------
# [NEW-2] Ensemble-aware fetal IC selection
# ---------------------------------------------------------------------------

def select_best_ic_from_ensemble(ic_sets: list[np.ndarray],
                                  maternal_peaks: np.ndarray,
                                  maternal_hr_bpm: float,
                                  fs: int,
                                  exclude_fn=None,
                                  label: str = "") -> tuple[np.ndarray, int, int, float]:
    """
    Apply three-factor scoring across all ICs from all ensemble runs and
    return the global best candidate.

    Parameters
    ----------
    ic_sets         : list of (n_components, N) arrays from run_ica_ensemble()
    maternal_peaks  : (K,) maternal QRS indices
    maternal_hr_bpm : maternal HR in BPM (for harmonic penalty)
    fs              : sampling rate
    exclude_fn      : optional callable(ic) -> bool; if True, skip this IC
                      (used to exclude maternal residual ICs in Path B)
    label           : label string for logging

    Returns
    -------
    best_ic   : (N,) best independent component (unit-normalised)
    run_idx   : index of the winning ensemble run (0-based)
    comp_idx  : component index within that run (0-based)
    best_score: the three-factor score of the winner
    """
    best_score = -np.inf
    best_ic    = None
    best_run   = -1
    best_comp  = -1

    n_scored = 0
    n_skipped = 0

    for run_i, ICs in enumerate(ic_sets):
        for comp_j, ic in enumerate(ICs):
            # Skip zero-variance padding ICs
            if np.var(ic) < 1e-10:
                n_skipped += 1
                continue

            # Apply exclusion criterion (e.g. maternal residual check)
            if exclude_fn is not None and exclude_fn(ic):
                n_skipped += 1
                continue

            ic_norm = ic - np.mean(ic)
            std = np.std(ic_norm)
            if std > 1e-10:
                ic_norm = ic_norm / std

            sc = score_fetal_ic_three_factor(
                ic_norm, maternal_peaks, fs, maternal_hr_bpm)
            n_scored += 1

            if sc > best_score:"""
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
  [C3]    score_fetal_ic(): maternal residual suppression penalty.
          Penalises ICs correlated with maternal IC at zero lag or one
          maternal beat period lag (catches harmonics).
  [C4]    score_fetal_ic(): waveform morphology consistency score.
          Rewards ICs whose beat templates are mutually similar (real
          fetal ECG has consistent QRS morphology; noise does not).
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



def _maternal_residual_penalty(ic: np.ndarray,
                               maternal_ic: np.ndarray,
                               maternal_peaks: np.ndarray) -> float:
    """
    Criterion 3 — Maternal Residual Suppression Penalty.

    A genuine fetal IC should be statistically independent from the maternal
    IC. Components that are maternal harmonics or carry residual maternal
    structure will correlate with the maternal IC either at zero lag (direct
    residual) or at a lag of one maternal beat period (harmonic check).

    WHY THIS HELPS:
      Root-cause analysis showed that on high-SNR CinC2013 recordings, ICA
      produces ghost components at ~2x maternal HR that sit inside the fetal
      HR range. These score well on regularity and HR but are mathematically
      derived from the maternal signal. Zero-lag correlation catches direct
      residual; one-beat-lag correlation catches the harmonic structure.

    Returns
    -------
    penalty : float in [0, 1]
        1.0 = completely independent from maternal (no penalty)
        0.0 = completely correlated with maternal (maximum penalty)
    """
    if len(maternal_ic) != len(ic):
        return 1.0  # cannot compute — do not penalise

    # Zero-lag normalised correlation
    ic_z  = (ic - np.mean(ic))  / (np.std(ic)  + 1e-10)
    mat_z = (maternal_ic - np.mean(maternal_ic)) / (np.std(maternal_ic) + 1e-10)
    corr_zero = float(np.abs(np.dot(ic_z, mat_z) / len(ic_z)))

    # One-maternal-beat-lag correlation (harmonic detection)
    corr_harmonic = 0.0
    if len(maternal_peaks) > 1:
        mean_mat_period = int(np.mean(np.diff(maternal_peaks)))
        if 0 < mean_mat_period < len(ic) // 2:
            mat_shifted = maternal_ic[mean_mat_period:]
            ic_trimmed  = ic[:len(mat_shifted)]
            if len(ic_trimmed) > 10:
                ic_s  = (ic_trimmed  - np.mean(ic_trimmed))  / (np.std(ic_trimmed)  + 1e-10)
                mat_s = (mat_shifted - np.mean(mat_shifted)) / (np.std(mat_shifted) + 1e-10)
                corr_harmonic = float(np.abs(np.dot(ic_s, mat_s) / len(ic_s)))

    penalty = 1.0 - float(np.clip(max(corr_zero, corr_harmonic), 0.0, 1.0))
    return penalty


def _morphology_consistency_score(ic: np.ndarray,
                                   peaks: np.ndarray,
                                   fs: int,
                                   window_ms: float = 120.0) -> float:
    """
    Criterion 4 — Waveform Morphology Consistency Score.

    Real fetal ECG beats all look similar to each other — the QRS complex
    repeats with consistent shape at every beat. Noise components and maternal
    harmonic artefacts produce peaks with inconsistent beat shapes.

    WHY THIS HELPS:
      Current scoring uses HR regularity (are beats evenly spaced?) but not
      morphology regularity (do beats look the same?). A noise component can
      produce evenly-spaced local maxima with random shapes — it scores well
      on regularity but poorly on morphology. This directly selects for ICs
      that will work best with EKF-RTS, which assumes consistent morphology.

    Parameters
    ----------
    window_ms : half-window in milliseconds around each detected peak

    Returns
    -------
    score : float in [0, 1]
        1.0 = all beats look identical
        0.0 = beats are inconsistent (noise)
    """
    if len(peaks) < 4:
        return 0.0

    half_w = int(window_ms * fs / 1000)
    N      = len(ic)

    templates = []
    for p in peaks:
        lo, hi = p - half_w, p + half_w
        if lo >= 0 and hi <= N:
            window = ic[lo:hi].copy()
            std    = np.std(window)
            if std < 1e-10:
                continue
            window = (window - np.mean(window)) / std
            templates.append(window)

    if len(templates) < 3:
        return 0.0

    # Vectorised pairwise correlation via matrix multiply
    T  = np.stack(templates)                                     # (n, w)
    T  = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-10) # unit rows
    CC = T @ T.T                                                  # (n, n)
    n  = len(templates)
    off_diag_mean = (CC.sum() - np.trace(CC)) / (n * (n - 1))

    return float(np.clip(off_diag_mean, 0.0, 1.0))


def score_fetal_ic(ic: np.ndarray, maternal_peaks: np.ndarray,
                   fs: int = FS,
                   maternal_ic: np.ndarray = None) -> float:
    """
    Score an IC for how likely it is to be the fetal ECG.

    [FIX-3] Uses _detect_peaks_adaptive() for robust peak finding on noisy
            ICA2 residual components.
    [FIX-5] HR lower bound: 60 -> FETAL_HR_MIN (100 BPM).
    [C3]    Maternal residual suppression penalty — penalises ICs that
            correlate with maternal IC at zero or one-beat lag.
    [C4]    Morphology consistency score — rewards ICs with consistent
            QRS beat shapes across all detected beats.

    Final score = base_score
                  * maternal_penalty       [C3]
                  * (1 + morphology_score) [C4]
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

    base_score = float(independence * regularity * (1.0 + kurt_score))

    # [C3] Maternal residual suppression penalty
    if maternal_ic is not None and len(maternal_peaks) > 0:
        mat_penalty = _maternal_residual_penalty(ic, maternal_ic, maternal_peaks)
    else:
        mat_penalty = 1.0  # no penalty if maternal_ic not provided

    # [C4] Morphology consistency score
    morph_score = _morphology_consistency_score(ic, peaks, fs)

    return float(base_score * mat_penalty * (1.0 + morph_score))


def select_fetal_ic(ICs: np.ndarray,
                    maternal_peaks: np.ndarray,
                    maternal_idx: int,
                    fs: int = FS,
                    residual: np.ndarray = None,
                    cfg: BaseConfig = None,
                    maternal_ic: np.ndarray = None) -> tuple[int, list[float]]:
    """
    Select the fetal IC from ICA components.

    FIX 5 (CinC2013): After HR-gating, verify spectral separation from maternal.
    Requires peak-frequency in [FETAL_HR_MIN/60, FETAL_HR_MAX/60] Hz distinct
    from maternal frequency by >= HR_SEP_MIN_BPM/60 Hz.

    [C3][C4] maternal_ic passed into score_fetal_ic for enhanced scoring.

    Parameters
    ----------
    ICs          : (n_comp, N) independent components
    maternal_peaks : (K,) maternal QRS indices
    maternal_idx : maternal IC index (for API compatibility)
    fs           : sampling rate
    residual     : residual signal (for fallback)
    cfg          : BaseConfig (optional, for CinC2013 spectral checks)
    maternal_ic  : (N,) maternal IC waveform — enables C3 and C4 scoring
    """
    if cfg is None:
        cfg = _cfg

    scores   = [score_fetal_ic(ic, maternal_peaks, fs, maternal_ic=maternal_ic)
                for ic in ICs]
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
                best_score = sc
                best_ic    = ic_norm
                best_run   = run_i
                best_comp  = comp_j

    if best_ic is None:
        raise ValueError(
            f"[ICA-ENS] {label}: no scorable IC candidates found across "
            f"{len(ic_sets)} runs.")

    print(f"[ICA-ENS] {label}: scored {n_scored} ICs "
          f"(skipped {n_skipped}) across {len(ic_sets)} runs")
    print(f"[ICA-ENS] {label}: winner → run={best_run+1}, "
          f"comp={best_comp+1}, three-factor score={best_score:.4f}")

    return best_ic, best_run, best_comp, best_score


def get_ic_as_signal(ICs: np.ndarray, idx: int) -> np.ndarray:
    """Extract, center and unit-normalise a single IC."""
    ic = ICs[idx].copy()
    ic = ic - np.mean(ic)
    ic = ic / (np.std(ic) + 1e-10)
    return ic