"""
separation/adaptive_filter.py
Adaptive noise cancellation (RLS / NLMS) for residual maternal leakage
cleanup.

DISSERTATION MODIFICATION [Enhancement Roadmap, Rank 4]:
  Runs a short adaptive filter, using the AW-WSVD maternal reconstruction
  as the reference signal, on the WSVD residual before the Path B ICA2
  stage. This mops up residual periodic maternal leakage that survives
  the correlation-gated AW-WSVD subtraction (separation/wsvd.py), which
  only accepts or rejects whole SVD components per window rather than
  adaptively tracking a continuously-varying leakage waveform.

  This mirrors the RLS + ICA cascade in Barnova et al. (2021), who report
  ~11 F1 points of improvement over EMD alone specifically attributable
  to an adaptive filter recovering residual periodic leakage that a
  single decomposition pass leaves behind. Because the AW-WSVD maternal
  reconstruction (`maternal_recon`) is already computed and already
  correlation-validated against the maternal IC, it can be reused
  directly as the adaptive-filter reference signal at negligible extra
  cost — no new signal needs to be estimated.

Two classical adaptive filter algorithms are provided:
  - RLS  (Recursive Least Squares): faster convergence, more expensive
    per-sample update (O(n_taps^2)). Default.
  - NLMS (Normalized Least Mean Squares): cheaper per-sample update
    (O(n_taps)), slower convergence. Selectable via
    cfg.ADAPTIVE_FILTER_METHOD = "nlms".
"""

import numpy as np


def rls_filter(desired: np.ndarray, reference: np.ndarray,
                n_taps: int = 5, forgetting_factor: float = 0.995,
                delta: float = 1.0) -> tuple:
    """
    Recursive Least Squares adaptive filter.

    Adapts an n_taps-length FIR filter w such that (w . reference_window)
    approximates the leakage component present in `desired`. Returns the
    filter's running estimate of that leakage and the resulting error
    (cleaned) signal.

    Parameters
    ----------
    desired            : (N,) primary signal (residual containing leakage
                         + fetal ECG + noise)
    reference          : (N,) reference signal correlated with the
                         leakage (e.g. the AW-WSVD maternal reconstruction
                         for the same channel)
    n_taps             : FIR filter length
    forgetting_factor  : RLS lambda in (0, 1]; close to 1.0 for slowly
                         time-varying leakage statistics
    delta              : inverse-correlation-matrix initialisation constant

    Returns
    -------
    estimate : (N,) filter's estimate of the leakage component
    error    : (N,) desired - estimate (the cleaned signal)
    """
    N = len(desired)
    w = np.zeros(n_taps)
    P = np.eye(n_taps) / delta
    estimate = np.zeros(N)
    error = np.zeros(N)
    ref_padded = np.concatenate([np.zeros(n_taps - 1), reference])

    for n in range(N):
        x = ref_padded[n:n + n_taps][::-1]
        y = float(np.dot(w, x))
        e = desired[n] - y
        Px = P @ x
        denom = forgetting_factor + float(x @ Px)
        k = Px / (denom + 1e-12)
        w = w + k * e
        P = (P - np.outer(k, Px)) / forgetting_factor
        estimate[n] = y
        error[n] = e

    return estimate, error


def nlms_filter(desired: np.ndarray, reference: np.ndarray,
                 n_taps: int = 5, step_size: float = 0.02,
                 eps: float = 1e-6) -> tuple:
    """
    Normalized Least Mean Squares adaptive filter.

    Lower per-sample computational cost than RLS; used as a lighter-weight
    alternative when cfg.ADAPTIVE_FILTER_METHOD == "nlms".

    Parameters
    ----------
    desired   : (N,) primary signal
    reference : (N,) reference signal
    n_taps    : FIR filter length
    step_size : NLMS step size (mu); larger values converge faster but are
                less stable
    eps       : regularisation constant guarding against division by zero
                when the reference window has near-zero energy

    Returns
    -------
    estimate, error : as in rls_filter()
    """
    N = len(desired)
    w = np.zeros(n_taps)
    estimate = np.zeros(N)
    error = np.zeros(N)
    ref_padded = np.concatenate([np.zeros(n_taps - 1), reference])

    for n in range(N):
        x = ref_padded[n:n + n_taps][::-1]
        y = float(np.dot(w, x))
        e = desired[n] - y
        norm = float(np.dot(x, x)) + eps
        w = w + (step_size / norm) * e * x
        estimate[n] = y
        error[n] = e

    return estimate, error


def adaptive_residual_cleanup(residual: np.ndarray,
                               maternal_recon: np.ndarray,
                               fs: int,
                               method: str = "rls",
                               n_taps: int = 5,
                               forgetting_factor: float = 0.995,
                               delta: float = 1.0,
                               step_size: float = 0.02,
                               eps: float = 1e-6) -> np.ndarray:
    """
    Apply per-channel adaptive filtering to a multichannel WSVD residual,
    using the AW-WSVD maternal reconstruction (same shape) as the
    reference signal for each channel.

    Only channels where AW-WSVD actually estimated a non-trivial maternal
    contribution are filtered; channels with a near-silent reference are
    passed through unchanged, since an adaptive filter given no
    informative reference cannot safely remove anything and risks
    distorting the fetal signal (the same over-subtraction guard
    philosophy already used by separation/wsvd.py's per-window/per-channel
    gating).

    Parameters
    ----------
    residual        : (n_ch, N) AW-WSVD residual (fetal + noise + leakage)
    maternal_recon  : (n_ch, N) AW-WSVD maternal reconstruction, same shape
    fs              : sampling rate (kept for API symmetry / future
                      frequency-dependent tap sizing; not currently used
                      directly by the filters themselves)
    method          : "rls" or "nlms"
    n_taps, forgetting_factor, delta, step_size, eps :
                      filter hyperparameters, see rls_filter()/nlms_filter()

    Returns
    -------
    cleaned : (n_ch, N) residual after adaptive leakage cleanup
    """
    n_ch, N = residual.shape
    cleaned = residual.copy()
    ref_energy = np.sum(maternal_recon ** 2, axis=1)
    total_energy = np.sum(residual ** 2, axis=1) + 1e-12
    n_filtered = 0

    for ch in range(n_ch):
        if ref_energy[ch] < 1e-8 * (total_energy[ch] + 1e-8):
            continue
        ref = maternal_recon[ch]
        des = residual[ch]
        if method == "nlms":
            _, err = nlms_filter(des, ref, n_taps=n_taps,
                                  step_size=step_size, eps=eps)
        else:
            _, err = rls_filter(des, ref, n_taps=n_taps,
                                 forgetting_factor=forgetting_factor,
                                 delta=delta)
        cleaned[ch] = err
        n_filtered += 1

    cleaned = cleaned - cleaned.mean(axis=1, keepdims=True)
    print(f"[ADAPTIVE-FILTER] {method.upper()} cleanup applied to "
          f"{n_filtered}/{n_ch} channels (n_taps={n_taps})")
    return cleaned