"""
evaluation/metrics.py
Standard evaluation metrics for fetal ECG separation.

R-peak detection metrics (Se, PPV, F1):
  TP : detected peak within +/-tolerance ms of reference peak
  FN : reference peak with no detected match
  FP : detected peak with no reference match

  Sensitivity (Se)  = TP / (TP + FN)
  Precision   (PPV) = TP / (TP + FP)
  F1 score         = 2 * Se * PPV / (Se + PPV)

Signal quality metrics: SNR (dB), PRD (%), RMSE, CC (Pearson)

FHR metric: FHR_MAE — mean absolute error of fetal HR estimate (BPM)

FIX: compute_fhr_mae() previously returned a rough, unreliable estimate
when the overlap between detected and reference HR series was too small.
It now returns np.nan in that case so aggregate_results() can filter it
cleanly rather than silently polluting results tables.
"""

import numpy as np
from scipy.stats import pearsonr, wilcoxon
from config import FS, EVAL_TOLERANCE_MS


def match_peaks(detected: np.ndarray, reference: np.ndarray,
                fs: int = FS,
                tolerance_ms: float = EVAL_TOLERANCE_MS) -> dict:
    """
    Match detected peaks to reference peaks within a tolerance window.

    Parameters
    ----------
    detected     : (K,) detected peak indices
    reference    : (M,) reference (ground truth) peak indices
    tolerance_ms : matching tolerance in milliseconds

    Returns
    -------
    dict with TP, FP, FN, matched pairs
    """
    tol_samples = int((tolerance_ms / 1000.0) * fs)
    detected    = np.sort(detected)
    reference   = np.sort(reference)

    matched_ref = set()
    matched_det = set()
    tp_pairs    = []

    for i, dp in enumerate(detected):
        candidates = np.where(np.abs(reference - dp) <= tol_samples)[0]
        if len(candidates) == 0:
            continue
        candidates = [c for c in candidates if c not in matched_ref]
        if not candidates:
            continue
        closest = candidates[np.argmin(np.abs(reference[candidates] - dp))]
        matched_ref.add(closest)
        matched_det.add(i)
        tp_pairs.append((dp, reference[closest]))

    TP = len(tp_pairs)
    FP = len(detected)  - TP
    FN = len(reference) - TP

    return {
        "TP": TP, "FP": FP, "FN": FN,
        "tp_pairs"   : tp_pairs,
        "n_detected" : len(detected),
        "n_reference": len(reference),
    }


def compute_se_ppv_f1(match_result: dict) -> dict:
    """Compute Sensitivity, PPV, and F1 from matched peaks."""
    TP = match_result["TP"]
    FP = match_result["FP"]
    FN = match_result["FN"]

    Se  = TP / (TP + FN + 1e-10)
    PPV = TP / (TP + FP + 1e-10)
    F1  = 2 * Se * PPV / (Se + PPV + 1e-10)

    return {
        "Se" : float(Se),
        "PPV": float(PPV),
        "F1" : float(F1),
        "TP" : TP, "FP": FP, "FN": FN,
    }


def compute_snr(estimated: np.ndarray, reference: np.ndarray) -> float:
    """SNR = 10 * log10(var(reference) / var(reference - estimated)). Higher is better."""
    n   = min(len(estimated), len(reference))
    est = estimated[:n]
    ref = reference[:n]
    noise_power = np.var(ref - est)
    if noise_power < 1e-12:
        return np.inf
    return float(10 * np.log10(np.var(ref) / noise_power))


def compute_prd(estimated: np.ndarray, reference: np.ndarray) -> float:
    """PRD = 100 * sqrt(sum((ref-est)^2) / sum(ref^2)). Lower is better. Units: %."""
    n     = min(len(estimated), len(reference))
    est   = estimated[:n]
    ref   = reference[:n]
    denom = np.sum(ref**2)
    if denom < 1e-12:
        return np.inf
    return float(100.0 * np.sqrt(np.sum((ref - est)**2) / denom))


def compute_rmse(estimated: np.ndarray, reference: np.ndarray) -> float:
    """Root mean squared error between estimated and reference signals."""
    n = min(len(estimated), len(reference))
    return float(np.sqrt(np.mean((estimated[:n] - reference[:n])**2)))


def compute_cc(estimated: np.ndarray, reference: np.ndarray) -> float:
    """Pearson correlation coefficient."""
    n    = min(len(estimated), len(reference))
    cc, _ = pearsonr(estimated[:n], reference[:n])
    return float(cc)


def compute_fhr_mae(detected_peaks: np.ndarray,
                    reference_peaks: np.ndarray,
                    fs: int = FS) -> float:
    """
    Mean absolute error of fetal heart rate estimate (BPM).

    Computes instantaneous HR from both detected and reference peaks,
    interpolates to a common time axis and computes MAE.

    FIX: Previously returned a rough fallback estimate when the HR series
    overlap was too small, silently producing unreliable values in results
    tables. Now returns np.nan in that case, which aggregate_results()
    filters cleanly.

    Returns
    -------
    float : MAE in BPM, or np.nan if insufficient data
    """
    if len(detected_peaks) < 3 or len(reference_peaks) < 3:
        return np.nan

    def hr_series(peaks):
        rr = np.diff(peaks) / fs
        return 60.0 / rr, (peaks[:-1] + peaks[1:]) / 2.0

    hr_det, t_det = hr_series(detected_peaks)
    hr_ref, t_ref = hr_series(reference_peaks)

    t_min = max(t_det[0],  t_ref[0])
    t_max = min(t_det[-1], t_ref[-1])

    mask_ref = (t_ref >= t_min) & (t_ref <= t_max)
    mask_det = (t_det >= t_min) & (t_det <= t_max)

    # FIX: return nan instead of rough fallback when overlap is insufficient
    if np.sum(mask_ref) < 2 or np.sum(mask_det) < 2:
        return np.nan

    hr_det_interp = np.interp(t_ref[mask_ref], t_det[mask_det], hr_det[mask_det])
    mae = float(np.mean(np.abs(hr_det_interp - hr_ref[mask_ref])))
    return mae


def evaluate(estimated_signal: np.ndarray,
             reference_signal: np.ndarray | None,
             estimated_peaks: np.ndarray,
             reference_peaks: np.ndarray,
             fs: int = FS,
             label: str = "PHASE") -> dict:
    """
    Run full evaluation suite.

    Parameters
    ----------
    estimated_signal : (N,) extracted fetal ECG
    reference_signal : (N,) direct fetal ECG — OR None for NIFECGDB
                       (no direct electrode; waveform metrics will be NaN)
    estimated_peaks  : (K,) detected fetal R-peaks
    reference_peaks  : (M,) reference fetal R-peaks from .qrs annotation
    label            : name for this configuration (for logging)

    Returns
    -------
    dict of all metrics
    """
    # Waveform quality metrics — only when a reference waveform exists
    if reference_signal is not None:
        n   = min(len(estimated_signal), len(reference_signal))
        est = estimated_signal[:n]
        ref = reference_signal[:n]
        cc_pos = np.corrcoef(est, ref)[0, 1]
        cc_neg = np.corrcoef(-est, ref)[0, 1]
        if cc_neg > cc_pos:
            est = -est
        snr  = compute_snr(est, ref)
        prd  = compute_prd(est, ref)
        rmse = compute_rmse(est, ref)
        cc   = compute_cc(est, ref)
    else:
        snr = prd = rmse = cc = np.nan

    match   = match_peaks(estimated_peaks, reference_peaks, fs)
    clf     = compute_se_ppv_f1(match)
    fhr_mae = compute_fhr_mae(estimated_peaks, reference_peaks, fs)

    results = {
        "label"      : label,
        "SNR_dB"     : snr,
        "PRD_pct"    : prd,
        "RMSE"       : rmse,
        "CC"         : cc,
        "Se"         : clf["Se"]  * 100,
        "PPV"        : clf["PPV"] * 100,
        "F1"         : clf["F1"]  * 100,
        "TP"         : clf["TP"],
        "FP"         : clf["FP"],
        "FN"         : clf["FN"],
        "FHR_MAE_bpm": fhr_mae,
        "n_detected" : match["n_detected"],
        "n_reference": match["n_reference"],
    }

    _print_results(results)
    return results


def _print_results(r: dict) -> None:
    fhr_str = f"{r['FHR_MAE_bpm']:.2f} BPM" if not np.isnan(r['FHR_MAE_bpm']) else "N/A"
    print(f"\n{'─'*50}")
    print(f"  Results: {r['label']}")
    print(f"{'─'*50}")
    if not np.isnan(r['SNR_dB']):
        print(f"  Signal Quality:")
        print(f"    SNR    : {r['SNR_dB']:+.2f} dB")
        print(f"    PRD    : {r['PRD_pct']:.2f} %")
        print(f"    RMSE   : {r['RMSE']:.4f}")
        print(f"    CC     : {r['CC']:.4f}")
    else:
        print(f"  Signal Quality: N/A (no direct reference)")
    print(f"  R-peak Detection (tol=50ms):")
    print(f"    Se     : {r['Se']:.2f} %")
    print(f"    PPV    : {r['PPV']:.2f} %")
    print(f"    F1     : {r['F1']:.2f} %")
    print(f"    TP={r['TP']}, FP={r['FP']}, FN={r['FN']}")
    print(f"  FHR MAE  : {fhr_str}")
    print(f"{'─'*50}\n")


def wilcoxon_test(scores_a: list, scores_b: list,
                  metric_name: str = "F1") -> dict:
    """
    Wilcoxon signed-rank test between two methods across recordings.
    Non-parametric — appropriate for small sample sizes (ADFECGDB: n=5).
    """
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)

    if len(scores_a) < 2:
        print(f"[Stats] Too few samples for Wilcoxon test on {metric_name}")
        return {"statistic": np.nan, "p_value": np.nan, "significant": False}

    stat, p = wilcoxon(scores_a, scores_b, alternative='greater')
    sig     = p < 0.05

    print(f"\n[Stats] Wilcoxon test ({metric_name}): "
          f"statistic={stat:.3f}, p={p:.4f} "
          f"({'SIGNIFICANT' if sig else 'not significant'} at alpha=0.05)")

    return {
        "statistic"  : float(stat),
        "p_value"    : float(p),
        "significant": sig,
        "metric"     : metric_name,
    }


def aggregate_results(results_list: list[dict]) -> dict:
    """
    Aggregate per-recording metrics to mean +/- std for the paper's results table.

    NaN values (e.g. from compute_fhr_mae fallback) are excluded cleanly.
    """
    metrics = ["SNR_dB", "PRD_pct", "RMSE", "CC",
               "Se", "PPV", "F1", "FHR_MAE_bpm"]
    agg = {}
    for m in metrics:
        vals = [r[m] for r in results_list
                if not np.isnan(float(r.get(m, np.nan)))]
        if vals:
            agg[f"{m}_mean"] = float(np.mean(vals))
            agg[f"{m}_std"]  = float(np.std(vals))
            agg[f"{m}_ci95"] = 1.96 * float(np.std(vals)) / np.sqrt(len(vals))

    print(f"\n{'='*55}")
    print("  Aggregated Results (mean +/- std across recordings)")
    print(f"{'='*55}")
    for m in metrics:
        if f"{m}_mean" in agg:
            print(f"  {m:15s}: {agg[f'{m}_mean']:.3f} +/- {agg[f'{m}_std']:.3f}")
    print(f"{'='*55}\n")

    return agg
