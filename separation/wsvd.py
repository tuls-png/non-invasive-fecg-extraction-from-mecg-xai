"""
separation/wsvd.py
Novel Adaptive Windowed Weighted SVD (AW-WSVD) for maternal ECG cancellation.

FIX: Removed two duplicate function bodies that appeared after the first
return statement in adaptive_windowed_wsvd. Only the most complete version
(with per-channel subtraction gating and energy protection) is kept.
"""

import numpy as np
from configs import BaseConfig

# Use BaseConfig defaults (shared across all datasets)
_cfg = BaseConfig()
FS = _cfg.FS
QRS_SIGMA_SEC = _cfg.QRS_SIGMA_SEC
QRS_BASELINE_WEIGHT = _cfg.QRS_BASELINE_WEIGHT
WSVD_WINDOW_SEC = _cfg.WSVD_WINDOW_SEC
WSVD_OVERLAP = _cfg.WSVD_OVERLAP
WSVD_N_COMPONENTS = _cfg.WSVD_N_COMPONENTS
WSVD_COMPONENT_CORR_THRESH = _cfg.WSVD_COMPONENT_CORR_THRESH
WSVD_MAX_ENERGY_REMOVAL = _cfg.WSVD_MAX_ENERGY_REMOVAL
WSVD_CHANNEL_R2_MIN = _cfg.WSVD_CHANNEL_R2_MIN

def gaussian_weight_matrix(n_samples: int, qrs_peaks: np.ndarray,
                            fs: int = FS,
                            sigma_sec: float = QRS_SIGMA_SEC,
                            baseline: float = QRS_BASELINE_WEIGHT) -> np.ndarray:
    """
    Construct a Gaussian physiological weight vector.

    Each QRS peak contributes a Gaussian bump of width sigma.
    The baseline weight ensures non-QRS regions are not completely ignored.

    Parameters
    ----------
    n_samples  : length of signal
    qrs_peaks  : (K,) sample indices of maternal QRS peaks
    fs         : sampling rate
    sigma_sec  : Gaussian sigma in seconds (default 40 ms = QRS duration)
    baseline   : minimum weight for non-QRS regions

    Returns
    -------
    weights : (n_samples,) weight vector in [baseline, 1.0]
    """
    weights = np.full(n_samples, baseline, dtype=np.float64)
    sigma_s = sigma_sec * fs
    radius  = int(4 * sigma_s)

    for peak in qrs_peaks:
        lo = max(0, peak - radius)
        hi = min(n_samples, peak + radius + 1)
        t_local  = np.arange(lo, hi) - peak
        gaussian = np.exp(-t_local**2 / (2 * sigma_s**2))
        weights[lo:hi] = np.maximum(weights[lo:hi], gaussian)

    return weights


def adaptive_windowed_wsvd(abd: np.ndarray,
                            weights: np.ndarray,
                            fs: int = FS,
                            mat_ic: np.ndarray = None,
                            n_components: int = WSVD_N_COMPONENTS,
                            corr_thresh: float = WSVD_COMPONENT_CORR_THRESH,
                            channel_r2: np.ndarray = None) -> np.ndarray:
    """
    Adaptive Windowed WSVD with per-window maternal correlation validation
    and per-channel subtraction gating.

    Per-channel gating: only subtract from channels where maternal IC R^2
    is above WSVD_CHANNEL_R2_MIN. Channels with low maternal R^2 are
    fetal-dominant and are left untouched.

    Per-window validation: only SVD components whose reconstructed waveform
    correlates with the maternal IC above corr_thresh are subtracted.
    Windows where nothing passes are left unchanged — safer than
    over-subtraction.

    Parameters
    ----------
    abd         : (n_ch, N) preprocessed abdominal signal
    weights     : (N,) Gaussian weight matrix
    fs          : sampling rate
    mat_ic      : (N,) maternal IC for per-window correlation validation
    n_components: max SVD components to consider per window
    corr_thresh : minimum |correlation| to accept component as maternal
    channel_r2  : (n_ch,) maternal IC R^2 per channel

    Returns
    -------
    recon : (n_ch, N) reconstructed maternal signal
    """
    n_ch, N = abd.shape
    win_len = int(WSVD_WINDOW_SEC * fs)
    hop     = int(win_len * (1.0 - WSVD_OVERLAP))
    H       = np.hanning(win_len)

    recon      = np.zeros_like(abd)
    weight_pad = np.zeros(N)

    win_count     = 0
    skipped_count = 0

    # Determine which channels to subtract from
    if channel_r2 is not None:
        subtract_mask = np.array(channel_r2) >= WSVD_CHANNEL_R2_MIN
        if not subtract_mask.any():
            subtract_mask[:] = True   # fallback: use all channels
        print(f"[AW-WSVD] Channel subtraction mask (R^2>={WSVD_CHANNEL_R2_MIN}): "
              f"{[f'ch{i+1}:{subtract_mask[i]}(R^2={channel_r2[i]:.3f})' for i in range(n_ch)]}")
    else:
        subtract_mask = np.ones(n_ch, dtype=bool)

    for start in range(0, N - win_len + 1, hop):
        stop = start + win_len
        w    = H * weights[start:stop]
        Xw   = abd[:, start:stop] * w[None, :]

        try:
            U, S, Vt = np.linalg.svd(Xw, full_matrices=False)
        except np.linalg.LinAlgError:
            weight_pad[start:stop] += H ** 2
            win_count += 1
            continue

        r = min(n_components, S.size)

        # Per-component correlation check against maternal IC
        keep_mask = np.zeros(r, dtype=bool)
        if mat_ic is not None:
            mat_seg = mat_ic[start:stop]
            for k in range(r):
                comp   = (U[:, k:k+1] * S[k]) @ Vt[k:k+1, :]
                scalar = comp.mean(axis=0)
                if len(mat_seg) == len(scalar):
                    try:
                        c = np.corrcoef(scalar, mat_seg)[0, 1]
                        if np.isfinite(c) and abs(c) >= corr_thresh:
                            keep_mask[k] = True
                    except Exception:
                        pass
        else:
            keep_mask[:] = True

        if keep_mask.any():
            Xrec = np.zeros((n_ch, win_len))
            for k in range(r):
                if keep_mask[k]:
                    Xrec += (U[:, k:k+1] * S[k]) @ Vt[k:k+1, :]

            # Energy protection: skip if reconstruction removes too much energy
            orig_energy  = np.sum(abd[:, start:stop] ** 2) + 1e-12
            recon_energy = np.sum(Xrec ** 2)
            if recon_energy / orig_energy > WSVD_MAX_ENERGY_REMOVAL:
                Xrec = np.zeros((n_ch, win_len))
                skipped_count += 1
            else:
                # Per-channel gating: zero out protected (fetal-dominant) channels
                Xrec[~subtract_mask, :] = 0.0
        else:
            Xrec = np.zeros((n_ch, win_len))
            skipped_count += 1

        recon[:, start:stop]  += Xrec * H[None, :]
        weight_pad[start:stop] += H ** 2
        win_count += 1

    nonzero = weight_pad > 1e-8
    recon[:, nonzero] /= weight_pad[None, nonzero]

    print(f"[AW-WSVD] Processed {win_count} windows "
          f"(window={WSVD_WINDOW_SEC}s, overlap={WSVD_OVERLAP*100:.0f}%, "
          f"components={n_components}, corr_thresh={corr_thresh})")
    if skipped_count > 0:
        pct = 100 * skipped_count / (win_count + 1e-8)
        print(f"[AW-WSVD] {skipped_count} windows ({pct:.1f}%) had no "
              f"maternal component above threshold — left unchanged")

    return recon


def subtract_maternal(abd_signals: np.ndarray,
                       maternal_recon: np.ndarray) -> np.ndarray:
    """
    Subtract maternal ECG reconstruction from abdominal signals.

    fetal_estimate(t) = abdomen(t) - maternal_recon(t)

    The residual contains fetal ECG + noise + residual maternal artifact.
    The subsequent ICA step further separates the fetal component.
    """
    residual = abd_signals - maternal_recon
    # Re-centre residual channels (subtraction may introduce small DC offset)
    residual = residual - residual.mean(axis=1, keepdims=True)
    return residual


def svd_explained_variance(signals: np.ndarray, n_top: int = 4) -> np.ndarray:
    """
    Compute explained variance ratio of SVD components.
    Used to justify choice of n_components in the dissertation.
    """
    _, S, _ = np.linalg.svd(signals, full_matrices=False)
    var_ratio = (S**2) / (np.sum(S**2) + 1e-12)
    print(f"[SVD] Explained variance by top {n_top} components:")
    for i in range(min(n_top, len(var_ratio))):
        print(f"  SV{i+1}: {var_ratio[i]*100:.2f}%  (cumulative: "
              f"{np.sum(var_ratio[:i+1])*100:.2f}%)")
    return var_ratio
