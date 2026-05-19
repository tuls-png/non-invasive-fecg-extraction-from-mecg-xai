"""
evaluation/nifecgdb_evaluator.py
NIFECGDB-specific evaluation module.

Since NIFECGDB has no fetal ground truth (no direct electrode, no .fqrs
annotations), fetal F1/Se/PPV cannot be computed. Instead, the three
.edf.qrs maternal annotations are used to validate three distinct aspects
of the pipeline's performance:

CHECK 1 — Maternal detector accuracy
  Cross-validate the pipeline's maternal peak detector (from ICA1) against
  the ground-truth maternal peaks in the .qrs annotation.
  Metric: maternal Se, PPV, F1 @ ±50ms tolerance.
  What to look for: F1 >= 0.90 means the Gaussian WSVD weights are placed
  correctly and maternal cancellation is operating on the right windows.
  If maternal F1 is low, WSVD will miss the actual beats.

CHECK 2 — Maternal cancellation quality
  Measure residual maternal energy in the post-subtraction signal at annotated
  maternal beat positions vs. non-beat positions.
  Metric: maternal_residual_ratio = mean energy at beat positions /
                                    mean energy at non-beat positions.
  Ratio >> 1.0 → maternal structure still dominates residual → bad cancellation.
  Ratio ≈ 1.0 → beats are no longer the dominant structure → good cancellation.

CHECK 3 — Fetal HR plausibility
  Since there is no fetal ground truth, check that the detected fetal peaks
  have a plausible heart rate (100–200 BPM) and are sufficiently separated
  from the known maternal HR (>= 15 BPM separation).
  Metrics: detected fetal HR mean/std, HR separation from maternal reference,
           HR plausibility flag (True/False).

All three checks are logged to CSV alongside standard pipeline metrics (FHR
estimate, SNR=NaN, F1=NaN for NIFECGDB).

Usage
-----
    from evaluation.nifecgdb_evaluator import NIFECGDBEvaluator, NIFECGDBResultsLogger

    # After pipeline.run() returns result dict:
    evaluator = NIFECGDBEvaluator(fs=recording["fs"])
    checks = evaluator.run_all_checks(
        recording=recording,
        result=result,       # pipeline.run() output dict
        maternal_ref_peaks=mat_ref_peaks,
    )

    logger = NIFECGDBResultsLogger("results_nifecgdb")
    logger.log(recording["recording"], checks)
    logger.save()
"""

import csv
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from preprocessing.qrs_detector import load_wfdb_annotation, compute_hr_stats
from evaluation.metrics import match_peaks, compute_se_ppv_f1


class NIFECGDBEvaluator:
    """
    Evaluates NIFECGDB pipeline results using maternal .edf.qrs annotations.

    Parameters
    ----------
    fs : int
        Sampling frequency (Hz).
    tolerance_ms : float
        Peak matching tolerance (default 50 ms, standard in the field).
    fetal_hr_min : float
        Minimum plausible fetal HR (BPM). Default 100.
    fetal_hr_max : float
        Maximum plausible fetal HR (BPM). Default 200.
    hr_sep_min : float
        Minimum HR separation from maternal to be plausible fetal (BPM). Default 15.
    beat_window_ms : float
        Window around each maternal beat for energy measurement (ms). Default 80.
    """

    def __init__(self, fs: int = 1000,
                 tolerance_ms: float = 50.0,
                 fetal_hr_min: float = 100.0,
                 fetal_hr_max: float = 200.0,
                 hr_sep_min: float = 15.0,
                 beat_window_ms: float = 80.0):
        self.fs             = fs
        self.tolerance_ms   = tolerance_ms
        self.fetal_hr_min   = fetal_hr_min
        self.fetal_hr_max   = fetal_hr_max
        self.hr_sep_min     = hr_sep_min
        self.beat_window_ms = beat_window_ms

    # ------------------------------------------------------------------ #
    # CHECK 1: Maternal peak detector accuracy
    # ------------------------------------------------------------------ #

    def check_maternal_detector(self,
                                  detected_maternal_peaks: np.ndarray,
                                  ref_maternal_peaks: np.ndarray) -> Dict[str, Any]:
        """
        Cross-validate pipeline maternal peak detector vs .qrs annotation.

        Parameters
        ----------
        detected_maternal_peaks : peaks from pipeline ICA1 maternal IC
        ref_maternal_peaks      : peaks from .edf.qrs annotation

        Returns
        -------
        dict with keys: maternal_Se, maternal_PPV, maternal_F1, maternal_TP,
                        maternal_FP, maternal_FN, n_detected_maternal,
                        n_ref_maternal, check1_pass (F1 >= 0.90)
        """
        if len(ref_maternal_peaks) == 0:
            return {
                "maternal_Se": np.nan, "maternal_PPV": np.nan, "maternal_F1": np.nan,
                "maternal_TP": 0, "maternal_FP": 0, "maternal_FN": 0,
                "n_detected_maternal": len(detected_maternal_peaks),
                "n_ref_maternal": 0,
                "check1_pass": False,
                "check1_note": "No .qrs annotation available",
            }

        match = match_peaks(detected_maternal_peaks, ref_maternal_peaks,
                            fs=self.fs, tolerance_ms=self.tolerance_ms)
        clf   = compute_se_ppv_f1(match)
        f1    = clf["F1"]

        return {
            "maternal_Se"          : round(clf["Se"] * 100, 2),
            "maternal_PPV"         : round(clf["PPV"] * 100, 2),
            "maternal_F1"          : round(f1 * 100, 2),
            "maternal_TP"          : match["TP"],
            "maternal_FP"          : match["FP"],
            "maternal_FN"          : match["FN"],
            "n_detected_maternal"  : match["n_detected"],
            "n_ref_maternal"       : match["n_reference"],
            "check1_pass"          : f1 >= 0.90,
            "check1_note"          : ("PASS" if f1 >= 0.90
                                      else f"FAIL — maternal F1={f1*100:.1f}% < 90%"),
        }

    # ------------------------------------------------------------------ #
    # CHECK 2: Maternal cancellation quality
    # ------------------------------------------------------------------ #

    def check_cancellation_quality(self,
                                    residual: np.ndarray,
                                    ref_maternal_peaks: np.ndarray) -> Dict[str, Any]:
        """
        Measure residual maternal energy at annotated beat positions.

        Parameters
        ----------
        residual            : (n_ch, N) post-subtraction signal from pipeline
        ref_maternal_peaks  : maternal R-peak indices from .qrs annotation

        Returns
        -------
        dict with keys: maternal_residual_ratio, beat_energy_mean,
                        non_beat_energy_mean, check2_pass (ratio < 1.5),
                        check2_per_channel (list of per-channel ratios)
        """
        if len(ref_maternal_peaks) == 0:
            return {
                "maternal_residual_ratio"   : np.nan,
                "beat_energy_mean"          : np.nan,
                "non_beat_energy_mean"      : np.nan,
                "check2_pass"               : False,
                "check2_per_channel"        : [],
                "check2_note"               : "No .qrs annotation available",
            }

        n_ch, N   = residual.shape
        hw        = int(self.beat_window_ms / 1000.0 * self.fs / 2)

        # Build beat mask (True at beat positions)
        beat_mask = np.zeros(N, dtype=bool)
        for peak in ref_maternal_peaks:
            lo = max(0, peak - hw)
            hi = min(N, peak + hw)
            beat_mask[lo:hi] = True
        non_beat_mask = ~beat_mask

        per_channel_ratios = []
        beat_energies      = []
        non_beat_energies  = []

        for ch in range(n_ch):
            ch_sig = residual[ch]
            be  = float(np.mean(ch_sig[beat_mask] ** 2)) if beat_mask.any() else np.nan
            nbe = float(np.mean(ch_sig[non_beat_mask] ** 2)) if non_beat_mask.any() else np.nan
            ratio = be / (nbe + 1e-12) if (not np.isnan(be) and not np.isnan(nbe)) else np.nan
            per_channel_ratios.append(round(ratio, 3) if not np.isnan(ratio) else None)
            beat_energies.append(be)
            non_beat_energies.append(nbe)

        mean_ratio = float(np.nanmean(per_channel_ratios))
        pass_threshold = 1.5  # >1.5 → maternal structure still dominant
        passes = mean_ratio < pass_threshold

        return {
            "maternal_residual_ratio"   : round(mean_ratio, 3),
            "beat_energy_mean"          : round(float(np.nanmean(beat_energies)), 6),
            "non_beat_energy_mean"      : round(float(np.nanmean(non_beat_energies)), 6),
            "check2_pass"               : passes,
            "check2_per_channel"        : per_channel_ratios,
            "check2_note"               : (
                f"PASS — ratio={mean_ratio:.2f} < {pass_threshold}"
                if passes
                else f"FAIL — ratio={mean_ratio:.2f} >= {pass_threshold} (maternal beats still dominant)"
            ),
        }

    # ------------------------------------------------------------------ #
    # CHECK 3: Fetal HR plausibility
    # ------------------------------------------------------------------ #

    def check_fetal_hr_plausibility(self,
                                     fetal_peaks: np.ndarray,
                                     ref_maternal_peaks: np.ndarray) -> Dict[str, Any]:
        """
        Check detected fetal peaks for HR plausibility.

        Uses the .qrs maternal reference HR as the denominator for HR
        separation check (more reliable than the ICA-estimated maternal HR).

        Parameters
        ----------
        fetal_peaks         : detected fetal R-peak indices from pipeline
        ref_maternal_peaks  : maternal R-peak indices from .qrs annotation

        Returns
        -------
        dict with keys: fetal_hr_mean, fetal_hr_std, maternal_hr_ref,
                        hr_separation, n_fetal_peaks, check3_pass,
                        in_fetal_range, sufficient_hr_sep
        """
        fetal_stats = compute_hr_stats(fetal_peaks, self.fs)
        fetal_hr    = fetal_stats["mean_hr"]
        fetal_std   = fetal_stats["std_hr"]

        mat_stats   = compute_hr_stats(ref_maternal_peaks, self.fs)
        maternal_hr = mat_stats["mean_hr"]

        if np.isnan(fetal_hr):
            return {
                "fetal_hr_mean"   : np.nan, "fetal_hr_std": np.nan,
                "maternal_hr_ref" : round(maternal_hr, 1) if not np.isnan(maternal_hr) else np.nan,
                "hr_separation"   : np.nan, "n_fetal_peaks": len(fetal_peaks),
                "check3_pass"     : False, "in_fetal_range": False,
                "sufficient_hr_sep": False,
                "check3_note"     : "FAIL — could not compute fetal HR (< 2 peaks)",
            }

        in_range   = self.fetal_hr_min <= fetal_hr <= self.fetal_hr_max
        hr_sep     = abs(fetal_hr - maternal_hr) if not np.isnan(maternal_hr) else np.nan
        sep_ok     = (hr_sep >= self.hr_sep_min) if not np.isnan(hr_sep) else False
        passes     = in_range and sep_ok

        note_parts = []
        if not in_range:
            note_parts.append(
                f"HR={fetal_hr:.1f} outside [{self.fetal_hr_min},{self.fetal_hr_max}] BPM")
        if not sep_ok:
            note_parts.append(
                f"HR separation={hr_sep:.1f} < {self.hr_sep_min} BPM from maternal")

        return {
            "fetal_hr_mean"    : round(fetal_hr, 1),
            "fetal_hr_std"     : round(fetal_std, 1) if not np.isnan(fetal_std) else np.nan,
            "maternal_hr_ref"  : round(maternal_hr, 1) if not np.isnan(maternal_hr) else np.nan,
            "hr_separation"    : round(hr_sep, 1) if not np.isnan(hr_sep) else np.nan,
            "n_fetal_peaks"    : len(fetal_peaks),
            "check3_pass"      : passes,
            "in_fetal_range"   : in_range,
            "sufficient_hr_sep": sep_ok,
            "check3_note"      : ("PASS" if passes else "FAIL — " + "; ".join(note_parts)),
        }

    # ------------------------------------------------------------------ #
    # Run all checks
    # ------------------------------------------------------------------ #

    def run_all_checks(self,
                        recording: Dict[str, Any],
                        result: Dict[str, Any],
                        maternal_ref_peaks: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Run all three NIFECGDB checks and return a combined result dict.

        Parameters
        ----------
        recording           : recording dict from NIFECGDBHandler
        result              : output dict from PHASEPipeline.run()
        maternal_ref_peaks  : maternal peaks from .qrs annotation.
                              If None, tries to load from recording["annotation_path"].

        Returns
        -------
        dict combining check results with pipeline output metadata
        """
        rec_id  = recording["recording"]
        fs      = recording["fs"]

        # Load maternal reference peaks if not provided
        if maternal_ref_peaks is None:
            ann_path = recording.get("annotation_path")
            ann_ext  = recording.get("annotation_ext", "qrs")
            if ann_path:
                try:
                    maternal_ref_peaks = load_wfdb_annotation(ann_path, ann_ext)
                    print(f"[NIFECGDBEval] {rec_id}: loaded {len(maternal_ref_peaks)} "
                          f"maternal ref peaks from .{ann_ext}")
                except Exception as e:
                    print(f"[NIFECGDBEval] {rec_id}: WARNING annotation load failed: {e}")
                    maternal_ref_peaks = np.array([])
            else:
                maternal_ref_peaks = np.array([])

        fetal_peaks    = result["fetal_peaks"]
        maternal_peaks = result["maternal_peaks"]
        residual       = result["residual"]

        print(f"\n[NIFECGDBEval] {rec_id} — running 3 validation checks")
        print(f"  Maternal ref peaks from .qrs: {len(maternal_ref_peaks)}")
        print(f"  Pipeline maternal peaks: {len(maternal_peaks)}")
        print(f"  Pipeline fetal peaks:    {len(fetal_peaks)}")

        # Run checks
        check1 = self.check_maternal_detector(maternal_peaks, maternal_ref_peaks)
        check2 = self.check_cancellation_quality(residual, maternal_ref_peaks)
        check3 = self.check_fetal_hr_plausibility(fetal_peaks, maternal_ref_peaks)

        # Summary
        all_pass = check1["check1_pass"] and check2["check2_pass"] and check3["check3_pass"]
        print(f"\n  CHECK 1 (maternal detector):      {check1['check1_note']}")
        print(f"  CHECK 2 (cancellation quality):   {check2['check2_note']}")
        print(f"  CHECK 3 (fetal HR plausibility):  {check3['check3_note']}")
        print(f"  OVERALL: {'ALL PASS' if all_pass else 'ONE OR MORE CHECKS FAILED'}\n")

        combined = {
            "recording"              : rec_id,
            "chosen_path"            : result.get("chosen_path", ""),
            "duration_sec"           : recording.get("duration_sec", np.nan),
            "n_abd_channels"         : recording.get("n_abd_channels", np.nan),
            "overall_pass"           : all_pass,
            "maternal_ref_peaks"     : check1["n_ref_maternal"],
            "pipeline_maternal_peaks": check1["n_detected_maternal"],
            "pipeline_fetal_peaks"   : check3["n_fetal_peaks"],
            **check1,
            **check2,
            **check3,
        }
        return combined


# --------------------------------------------------------------------------- #
# NIFECGDB-specific CSV logger
# --------------------------------------------------------------------------- #

class NIFECGDBResultsLogger:
    """
    Logs NIFECGDB validation results to a timestamped summary CSV.

    Produces one row per recording with the three validation checks,
    pass/fail columns, peak counts, and summary check notes.

    Usage
    -----
        logger = NIFECGDBResultsLogger("results_nifecgdb")
        logger.log(checks_dict)          # repeated for each recording
        csv_path, json_path = logger.save()
    """

    # Columns in the exact order they appear in the CSV
    COLUMNS = [
        "timestamp", "recording",
        "C1:MatDet", "C2:Cancel", "C3:FHR", "Overall",
        "maternal_ref_peaks", "pipeline_maternal_peaks", "pipeline_fetal_peaks",
        "check1_note", "check2_note", "check3_note",
        "maternal_residual_ratio", "hr_separation",
    ]

    def __init__(self, results_dir: str = "results_nifecgdb"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.records   = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def log(self, checks: Dict[str, Any]) -> None:
        """
        Append one recording's check results.

        Parameters
        ----------
        checks : dict returned by NIFECGDBEvaluator.run_all_checks()
        """
        def _pass_fail(v):
            if v is True:
                return "PASS"
            if v is False:
                return "FAIL"
            return ""

        row = {
            "timestamp"               : self.timestamp,
            "recording"               : checks.get("recording", ""),
            "C1:MatDet"               : _pass_fail(checks.get("check1_pass")),
            "C2:Cancel"               : _pass_fail(checks.get("check2_pass")),
            "C3:FHR"                  : _pass_fail(checks.get("check3_pass")),
            "Overall"                 : _pass_fail(checks.get("overall_pass")),
            "maternal_ref_peaks"      : checks.get("n_ref_maternal", ""),
            "pipeline_maternal_peaks" : checks.get("n_detected_maternal", ""),
            "pipeline_fetal_peaks"    : checks.get("n_fetal_peaks", ""),
            "check1_note"             : checks.get("check1_note", ""),
            "check2_note"             : checks.get("check2_note", ""),
            "check3_note"             : checks.get("check3_note", ""),
            "maternal_residual_ratio"  : checks.get("maternal_residual_ratio", ""),
            "hr_separation"           : checks.get("hr_separation", ""),
        }

        self.records.append(row)
        print(f"[NIFECGDBLogger] Logged: {checks.get('recording', '?')}")

    def save(self):
        """
        Write the NIFECGDB validation summary to a CSV.

        Returns
        -------
        (csv_path, None) : str path to the saved file and None.
        """
        if not self.records:
            print("[NIFECGDBLogger] No records to save.")
            return None, None

        csv_path = self.results_dir / f"nifecgdb_validation_summary_{self.timestamp}.csv"

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
            writer.writeheader()
            writer.writerows(self.records)

        print(f"\n[NIFECGDBLogger] Saved NIFECGDB validation summary to:")
        print(f"  CSV  → {csv_path}")

        try:
            from utils.visualization import plot_nifecgdb_peak_count_summary
            plot_path = self.results_dir / f"nifecgdb_peak_counts_{self.timestamp}.png"
            plot_nifecgdb_peak_count_summary(self.records, save_path=str(plot_path))
            print(f"  Peak count plot → {plot_path}")
        except Exception as e:
            print(f"[NIFECGDBLogger] WARNING: could not save peak count plot: {e}")

        self._print_summary()

        return str(csv_path), None

    def _print_summary(self):
        """Print a compact pass/fail table to terminal."""
        if not self.records:
            return
        print(f"\n{'='*75}")
        print("  NIFECGDB VALIDATION SUMMARY")
        print(f"{'='*75}")
        hdr = f"{'Recording':<20} {'C1:MatDet':>10} {'C2:Cancel':>10} {'C3:FHR':>8} {'Overall':>8}"
        print(hdr)
        print(f"{'-'*75}")
        n_pass = 0
        for r in self.records:
            def pf(v):
                if v == "PASS":  return "PASS"
                if v == "FAIL":  return "FAIL"
                if v == "True":  return "PASS"
                if v == "False": return "FAIL"
                return str(v)
            row = (f"{str(r.get('recording','?')):<20} "
                   f"{pf(r.get('C1:MatDet','')):>10} "
                   f"{pf(r.get('C2:Cancel','')):>10} "
                   f"{pf(r.get('C3:FHR','')):>8} "
                   f"{pf(r.get('Overall','')):>8}")
            print(row)
            if pf(r.get("Overall", "")) == "PASS":
                n_pass += 1
        print(f"{'-'*75}")
        print(f"  Overall PASS: {n_pass}/{len(self.records)} recordings")
        print(f"{'='*75}\n")
