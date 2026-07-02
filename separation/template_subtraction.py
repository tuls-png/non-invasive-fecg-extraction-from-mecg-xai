"""
separation/template_subtraction.py
Adaptive Windowed Weighted SVD — epoch domain  ("Path C")

RELATIONSHIP TO separation/wsvd.py (the spatial-temporal AW-WSVD, "Path B"):
-----------------------------------------------------------------------------
Both modules perform the same underlying operation -- an adaptive, windowed,
weighted low-rank decomposition used to estimate the maternal ECG component
so it can be subtracted -- but along two different orientations of the same
abdominal recording:

  Path B (separation/wsvd.py):
      Matrix orientation : channels x time-window
      Decomposition       : weighted SVD (Gaussian QRS/PQRST weighting)
      Adaptivity           : window LENGTH adapts to recording duration;
                              which singular components are kept adapts via
                              a correlation gate against the maternal IC
      Output               : a maternal reconstruction, subtracted from the
                              full-length signal, then residual -> ICA2

  Path C (this module):
      Matrix orientation : aligned-beat-epoch x within-beat-sample
                            (each row is one maternal PQRST window, rows are
                            beats drawn from a local temporal context)
      Decomposition       : rank-1 low-rank estimate of the recurring beat
                            shape across that beat-epoch matrix, via either
                            a robust median estimator (default; see
                            `extract_maternal_template`) or an explicit
                            weighted SVD of the beat-epoch matrix (see
                            `extract_maternal_template_svd`) -- selected by
                            cfg.TEMPLATE_ESTIMATOR
      Adaptivity           : which BEATS are pooled into the estimate
                            adapts (local re-estimation window, every
                            TEMPLATE_UPDATE_EVERY_BEATS beats, using
                            TEMPLATE_CONTEXT_BEATS beats of context); the
                            per-beat AMPLITUDE fit adapts via a least-
                            squares scalar `alpha`
      Output               : a per-beat, amplitude-scaled template
                            subtraction directly on the full-length signal,
                            then residual -> ICA3

Framed this way, Path C is not an unrelated bolt-on: it is the epoch-domain
counterpart of the same adaptive-windowed-weighted-SVD principle stated in
the dissertation title, applied to a beat-epoch matrix instead of a
channel-time matrix. The classical precedent for SVD-based template
construction from a beat-epoch matrix is Kanjilal, Palit & Saha (1997),
"Fetal ECG extraction from single-channel maternal ECG using singular value
decomposition," IEEE TBME 44(1):51-59 -- there, SVD is used to estimate the
template itself from a matrix of aligned QRS complexes, which is exactly
the role `extract_maternal_template_svd` below plays when
TEMPLATE_ESTIMATOR="svd". The default estimator remains the median (more
outlier-robust to occasional ectopic/motion-artifact beats than an SVD
top-singular-vector, which is optimal in an L2 sense but not necessarily
in a robustness sense); TEMPLATE_ESTIMATOR="svd" is provided so the two
estimators can be compared directly (see scripts/compare_template_estimators.py)
rather than the choice being asserted without evidence.

Why this axis is needed in addition to Path B:
------------------------------------------------------------
WSVD (Path B) reconstructs maternal ECG from dominant singular vectors of
the weighted channel-time signal matrix. The problem: when fetal and
maternal ECGs share energy in the top singular vectors (which they do,
because both are periodic cardiac signals mixed across the same 4
channels), WSVD removes fetal energy along with maternal, and a
fixed-rank channel-time decomposition structurally cannot represent
beat-to-beat morphology drift (respiration, electrode movement) within a
single window.

Template subtraction (this axis) is more surgical along a different
dimension:
  1. Estimate the recurring maternal PQRST beat shape from a LOCAL set of
     aligned beat epochs (median, or top-singular-vector of the beat-epoch
     matrix)
  2. Align that estimate to every detected maternal R-peak
  3. Subtract only that estimate, per beat, at a least-squares-optimal
     amplitude scale -- leaving everything else (including any fetal
     energy coincidentally present in the maternal-beat window) intact

The fetal ECG is not periodic at the maternal rate, so it does not
contribute to the maternal beat-epoch estimate and is not subtracted.

Adaptive template with local re-estimation and per-beat scaling:
-----------------------------------------------------------------
Standard (non-adaptive) template subtraction uses one fixed template,
estimated once, for the entire recording. This module instead uses a
locally-re-estimated template (updated every N beats from local context)
with a least-squares per-beat amplitude scale `alpha`, so it tracks
maternal ECG morphology changes due to respiratory modulation, electrode
movement during contractions, and postural changes without needing a full
re-estimation at every single beat.

DISSERTATION MODIFICATION [Enhancement Roadmap, Rank 1]:
  This module implements "Path C": a third, independent maternal-
  cancellation + IC-extraction path fed into pipeline.py alongside the
  existing Path A (ICA1-direct) and Path B (AW-WSVD, channel-time axis).
  Path C is amplitude/morphology-based rather than kurtosis- or subspace-
  based, and is therefore expected to resolve exactly the "symmetric IC
  selection failure" cases where ICA's maternal component overlaps in
  frequency/kurtosis with the fetal component (see pipeline.py
  `_find_maternal_residual_idx` / `_score_ic_unified` docstrings for the
  existing kurtosis/subspace-based approach this complements).

  Path A/B/C run in PARALLEL on independently-derived residuals/candidates
  (Path C does NOT consume Path B's residual) -- this is a deliberate
  architectural choice: parallel best-of-N candidate generation followed
  by unified scoring is more robust than a fixed serial pipeline, because
  no single path's failure on a given recording can propagate into and
  contaminate the others. See pipeline.py Step 9 / Step 9b for the
  selection/fusion logic across all three.
"""

import numpy as np


def extract_maternal_template(abd_signals: np.ndarray,
                               maternal_peaks: np.ndarray,
                               fs: int,
                               half_window_sec: float = 0.15) -> tuple:
    """
    Extract a robust (median-synchronised) maternal PQRST template per
    channel from the given set of maternal R-peak locations.

    Parameters
    ----------
    abd_signals     : (n_ch, N) preprocessed abdominal signal
    maternal_peaks  : (K,) sample indices of maternal QRS peaks used to
                      build the template (a subset of all detected beats
                      when called from adaptive_template_subtraction's
                      local-context windows)
    fs              : sampling rate
    half_window_sec : half-window length in seconds around each R-peak

    Returns
    -------
    template : (n_ch, 2*hw+1) robust per-channel maternal beat template.
               All-zero if fewer than 3 valid (in-bounds) peaks are given.
    hw       : half-window length in samples (so callers can reuse it
               without recomputing int(half_window_sec * fs))
    """
    n_ch, N = abd_signals.shape
    hw = max(1, int(half_window_sec * fs))
    win_len = 2 * hw + 1

    valid_peaks = [int(p) for p in maternal_peaks
                   if int(p) - hw >= 0 and int(p) + hw < N]

    if len(valid_peaks) < 3:
        return np.zeros((n_ch, win_len)), hw

    windows = np.zeros((n_ch, len(valid_peaks), win_len))
    for ch in range(n_ch):
        for i, p in enumerate(valid_peaks):
            windows[ch, i] = abd_signals[ch, p - hw:p + hw + 1]

    # Median across beats: robust to occasional outlier beats (ectopic
    # beats, motion artifact, contraction-related electrode shift) that
    # would otherwise distort a simple mean template.
    template = np.median(windows, axis=1)
    return template, hw


def extract_maternal_template_svd(abd_signals: np.ndarray,
                                   maternal_peaks: np.ndarray,
                                   fs: int,
                                   half_window_sec: float = 0.15,
                                   n_components: int = 1,
                                   recency_weighted: bool = True) -> tuple:
    """
    Extract the maternal PQRST template per channel via weighted SVD of the
    beat-epoch matrix (Kanjilal, Palit & Saha, 1997, IEEE TBME 44(1):51-59)
    -- the epoch-domain counterpart of separation/wsvd.py's channel-time
    weighted SVD.

    For each channel, build a (K beats x win_len samples) matrix of aligned
    maternal-beat windows, optionally row-weighted by recency (more recent
    beats within the local context weighted higher, matching the
    "adaptive" / locally-tracking design used elsewhere in this module),
    take its SVD, and reconstruct the template from the top
    `n_components` left-singular-vector-weighted mode(s). With
    n_components=1 this is the rank-1 (dominant recurring beat shape)
    estimate -- the direct analogue of Path B's dominant-maternal-subspace
    assumption, applied along the beat-epoch axis instead of the
    channel-time axis.

    Parameters
    ----------
    abd_signals      : (n_ch, N) preprocessed abdominal signal
    maternal_peaks   : (K,) sample indices of maternal QRS peaks used to
                       build the template
    fs               : sampling rate
    half_window_sec  : half-window length in seconds around each R-peak
    n_components     : number of leading SVD modes summed into the
                       template (1 = pure rank-1 estimate)
    recency_weighted : if True, weight beat-epoch rows by an exponential
                       recency taper (more recent beats weighted higher)
                       before the SVD, matching the "local context"
                       philosophy of `adaptive_template_subtraction`;
                       if False, all beats in the window are weighted
                       equally (closest to the original Kanjilal
                       formulation)

    Returns
    -------
    template : (n_ch, 2*hw+1) maternal beat template reconstructed from
               the top SVD mode(s) of the beat-epoch matrix. All-zero if
               fewer than 3 valid (in-bounds) peaks are given.
    hw       : half-window length in samples
    """
    n_ch, N = abd_signals.shape
    hw = max(1, int(half_window_sec * fs))
    win_len = 2 * hw + 1

    valid_peaks = [int(p) for p in maternal_peaks
                   if int(p) - hw >= 0 and int(p) + hw < N]

    if len(valid_peaks) < 3:
        return np.zeros((n_ch, win_len)), hw

    n_beats = len(valid_peaks)
    n_comp = max(1, min(n_components, n_beats, win_len))

    if recency_weighted and n_beats > 1:
        # Exponential recency taper: last beat in the context window has
        # weight 1.0, earliest has weight ~0.5, matching the same
        # "adapt to recent morphology" spirit as the median estimator's
        # local re-estimation cadence.
        row_weights = np.exp(-0.5 * (n_beats - 1 - np.arange(n_beats)) / max(n_beats, 1))
    else:
        row_weights = np.ones(n_beats)

    template = np.zeros((n_ch, win_len))
    for ch in range(n_ch):
        M = np.zeros((n_beats, win_len))
        for i, p in enumerate(valid_peaks):
            M[i] = abd_signals[ch, p - hw:p + hw + 1]

        # Remove the per-row (per-beat) mean before SVD so the leading
        # singular vector captures beat SHAPE rather than being dominated
        # by any residual DC offset across beats.
        row_means = M.mean(axis=1, keepdims=True)
        Mc = (M - row_means) * row_weights[:, None]

        try:
            U, S, Vt = np.linalg.svd(Mc, full_matrices=False)
        except np.linalg.LinAlgError:
            # Fall back to the robust median estimator for this channel if
            # SVD fails to converge (e.g. degenerate/near-constant rows).
            template[ch] = np.median(M, axis=0)
            continue

        # Reconstruct the template as the energy-weighted sum of the top
        # n_comp right-singular vectors (each Vt[k] is a within-beat
        # waveform shape common across beats), scaled back by the mean
        # per-beat singular value contribution so the template is on the
        # same amplitude scale as a single representative beat.
        weights = S[:n_comp] / (np.sum(S[:n_comp]) + 1e-12)
        shape = np.sum(weights[:, None] * Vt[:n_comp], axis=0)
        # Restore an amplitude/offset consistent with the (weighted) mean
        # beat, since the SVD above was computed on mean-removed rows.
        mean_beat = np.average(M, axis=0, weights=row_weights)
        # Scale `shape` (unit-norm-ish from SVD) to best match mean_beat's
        # energy via a least-squares scalar, then add back the DC offset.
        denom = float(np.dot(shape, shape)) + 1e-12
        scale = float(np.dot(mean_beat - mean_beat.mean(), shape)) / denom
        template[ch] = scale * shape + mean_beat.mean()

    return template, hw


def adaptive_template_subtraction(abd_signals: np.ndarray,
                                   maternal_peaks: np.ndarray,
                                   fs: int,
                                   half_window_sec: float = 0.15,
                                   update_every: int = 20,
                                   context_beats: int = 15,
                                   min_beats_for_template: int = 5,
                                   estimator: str = "median",
                                   svd_n_components: int = 1) -> np.ndarray:
    """
    Subtract a locally-adaptive maternal PQRST template at every detected
    maternal beat location.

    For every block of `update_every` consecutive beats, a fresh template
    is estimated from a wider local context (`context_beats` beats on
    each side of the block, so the template is stable but still tracks
    slow morphology drift). Within the block, each individual beat window
    is scaled by a least-squares-optimal scalar `alpha` before
    subtraction, so that beat-to-beat amplitude modulation (breathing,
    electrode movement) does not cause over- or under-subtraction.

    Parameters
    ----------
    abd_signals             : (n_ch, N) preprocessed abdominal signal
    maternal_peaks          : (K,) sample indices of ALL detected maternal
                               QRS peaks for this recording
    fs                      : sampling rate
    half_window_sec         : +/- window (seconds) around each R-peak
                               subtracted at every beat
    update_every            : number of beats between template re-estimates
    context_beats           : beats on each side of the current block used
                               to build that block's local template
    min_beats_for_template  : minimum number of usable maternal beats
                               required to attempt subtraction at all
    estimator               : "median" (default; robust to outlier beats)
                               or "svd" (weighted SVD of the beat-epoch
                               matrix -- see extract_maternal_template_svd
                               and the module docstring's two-axis
                               AW-WSVD framing). Controlled by
                               cfg.TEMPLATE_ESTIMATOR at the call site.
    svd_n_components        : number of SVD modes used when
                               estimator="svd" (ignored otherwise)

    Returns
    -------
    residual : (n_ch, N) abd_signals with the locally-adaptive maternal
               template subtracted at every beat location. If there are
               too few usable beats, returns abd_signals unchanged
               (re-centred), matching the "safer to leave unchanged than
               over-subtract" convention used elsewhere in this codebase
               (see separation/wsvd.py's per-window correlation gate).
    """
    n_ch, N = abd_signals.shape
    residual = abd_signals.copy()

    hw = max(1, int(half_window_sec * fs))
    peaks = np.asarray(sorted(int(p) for p in maternal_peaks
                               if hw <= int(p) < N - hw))

    if len(peaks) < min_beats_for_template:
        print(f"[TEMPLATE-SUB] Only {len(peaks)} usable maternal beats "
              f"(< {min_beats_for_template}) -- skipping Path C subtraction")
        residual = residual - residual.mean(axis=1, keepdims=True)
        return residual

    if estimator not in ("median", "svd"):
        raise ValueError(f"Unknown TEMPLATE_ESTIMATOR '{estimator}' "
                          f"(expected 'median' or 'svd')")
    template_fn = (
        (lambda sig, ctx: extract_maternal_template(sig, ctx, fs, half_window_sec))
        if estimator == "median" else
        (lambda sig, ctx: extract_maternal_template_svd(
            sig, ctx, fs, half_window_sec, n_components=svd_n_components))
    )

    n_beats = len(peaks)
    win_len = 2 * hw + 1
    templates_used = 0
    alphas_all = []

    for start_i in range(0, n_beats, max(1, update_every)):
        end_i = min(start_i + update_every, n_beats)

        ctx_lo = max(0, start_i - context_beats)
        ctx_hi = min(n_beats, end_i + context_beats)
        ctx_peaks = peaks[ctx_lo:ctx_hi]

        template, _ = template_fn(abd_signals, ctx_peaks)
        templates_used += 1

        for i in range(start_i, end_i):
            p = peaks[i]
            for ch in range(n_ch):
                seg = abd_signals[ch, p - hw:p + hw + 1]
                t = template[ch]
                denom = float(np.dot(t, t)) + 1e-10
                if denom < 1e-9:
                    continue
                # Least-squares-optimal scalar amplitude fit of the
                # template to this individual beat window.
                alpha = float(np.dot(seg, t)) / denom
                # Sanity bound: never invert polarity or blow up on a
                # spurious/near-zero-energy template match.
                alpha = float(np.clip(alpha, 0.0, 2.0))
                residual[ch, p - hw:p + hw + 1] = seg - alpha * t
                alphas_all.append(alpha)

    residual = residual - residual.mean(axis=1, keepdims=True)

    mean_alpha = float(np.mean(alphas_all)) if alphas_all else 0.0
    print(f"[TEMPLATE-SUB] Adaptive template subtraction (estimator={estimator}): "
          f"{n_beats} beats, {templates_used} template updates "
          f"(every {update_every} beats, context={context_beats} beats), "
          f"mean beat-scale alpha={mean_alpha:.3f}")

    return residual


def verify_cancellation(abd_signals: np.ndarray, residual: np.ndarray,
                         maternal_peaks: np.ndarray, fs: int,
                         half_window_sec: float = 0.15) -> dict:
    """
    Diagnostic check: how much energy was removed at maternal beat
    locations by the template subtraction, comparing `abd_signals`
    (pre-subtraction) to `residual` (post-subtraction).

    Returns
    -------
    dict with:
        energy_reduction_pct : percent energy reduction at beat windows
        n_beats_checked      : number of in-bounds beats used
        pre_beat_energy      : summed squared amplitude pre-subtraction
        post_beat_energy     : summed squared amplitude post-subtraction
    """
    hw = max(1, int(half_window_sec * fs))
    n_ch, N = abd_signals.shape
    peaks = [int(p) for p in maternal_peaks if hw <= int(p) < N - hw]

    if len(peaks) == 0:
        return {
            "energy_reduction_pct": 0.0,
            "n_beats_checked": 0,
            "pre_beat_energy": 0.0,
            "post_beat_energy": 0.0,
        }

    pre_energy = 0.0
    post_energy = 0.0
    for p in peaks:
        pre_energy += float(np.sum(abd_signals[:, p - hw:p + hw + 1] ** 2))
        post_energy += float(np.sum(residual[:, p - hw:p + hw + 1] ** 2))

    reduction = 1.0 - post_energy / (pre_energy + 1e-12)
    return {
        "energy_reduction_pct": float(reduction * 100.0),
        "n_beats_checked": len(peaks),
        "pre_beat_energy": float(pre_energy),
        "post_beat_energy": float(post_energy),
    }