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

DISSERTATION MODIFICATION [Enhancement Roadmap, Rank 1]:
  This module implements "Path C": a third, independent maternal-
  cancellation + IC-extraction path fed into pipeline.py alongside the
  existing Path A (ICA1-direct) and Path B (AW-WSVD residual -> ICA2).
  Path C is amplitude/morphology-based rather than kurtosis- or subspace-
  based, and is therefore expected to resolve exactly the "symmetric IC
  selection failure" cases where ICA's maternal component overlaps in
  frequency/kurtosis with the fetal component (see pipeline.py
  `_find_maternal_residual_idx` / `_score_ic_unified` docstrings for the
  existing kurtosis/subspace-based approach this complements).

  Per-beat local amplitude scaling (`alpha` below, a least-squares scalar
  fit of the template to each individual beat window) is what makes the
  subtraction "adaptive" rather than a single fixed template: it absorbs
  beat-to-beat amplitude modulation (e.g. respiratory) without needing a
  full re-estimation of the template shape at every beat.
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


def adaptive_template_subtraction(abd_signals: np.ndarray,
                                   maternal_peaks: np.ndarray,
                                   fs: int,
                                   half_window_sec: float = 0.15,
                                   update_every: int = 20,
                                   context_beats: int = 15,
                                   min_beats_for_template: int = 5) -> np.ndarray:
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

    n_beats = len(peaks)
    win_len = 2 * hw + 1
    templates_used = 0
    alphas_all = []

    for start_i in range(0, n_beats, max(1, update_every)):
        end_i = min(start_i + update_every, n_beats)

        ctx_lo = max(0, start_i - context_beats)
        ctx_hi = min(n_beats, end_i + context_beats)
        ctx_peaks = peaks[ctx_lo:ctx_hi]

        template, _ = extract_maternal_template(
            abd_signals, ctx_peaks, fs, half_window_sec)
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
    print(f"[TEMPLATE-SUB] Adaptive template subtraction: {n_beats} beats, "
          f"{templates_used} template updates (every {update_every} beats, "
          f"context={context_beats} beats), mean beat-scale alpha={mean_alpha:.3f}")

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