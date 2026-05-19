"""
utils/logger.py
Automatic results logging to CSV for reproducibility.

Every time the pipeline runs, results are appended to a CSV.
This builds your results table automatically — no manual copying of numbers.
"""

import csv
import json
import numpy as np
from pathlib import Path
from datetime import datetime


class ResultsLogger:
    """
    Logs per-recording and aggregated metrics to CSV and JSON.

    Usage:
        logger = ResultsLogger("results/")
        logger.log_recording("r01", "PHASE", metrics_dict)
        logger.save()
    """

    COLUMN_LABELS = {
        "timestamp": "Execution Timestamp",
        "recording": "Recording ID",
        "method": "Method/Phase",
        "SNR_dB": "Signal-to-Noise Ratio (dB)",
        "PRD_pct": "Percent Root Distortion (%)",
        "RMSE": "Root Mean Squared Error",
        "CC": "Correlation Coefficient",
        "Se": "Sensitivity (%)",
        "PPV": "Precision (%)",
        "F1": "F1 Score (%)",
        "TP": "True Positives (TP)",
        "FP": "False Positives (FP)",
        "FN": "False Negatives (FN)",
        "FHR_MAE_bpm": "Fetal HR Mean Absolute Error (BPM)",
        "n_detected": "Detected Fetal Peaks",
        "n_reference": "Reference Fetal Peaks",
        "chosen_path": "Selected Reconstruction Path",
        "duration_sec": "Processing Duration (seconds)",
        "n_abd_channels": "Number of Abdominal Channels",
        "overall_pass": "Overall Quality Pass",
    }

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.records = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def log_recording(self, recording_id: str, method: str,
                      metrics: dict) -> None:
        """
        Log metrics for one recording under one method configuration.

        Parameters
        ----------
        recording_id : e.g. "r01"
        method       : e.g. "PHASE_full", "Baseline_ICA_WSVD"
        metrics      : output of evaluation.metrics.evaluate()
        """
        def _safe_value(v):
            """Convert metric value to a JSON/CSV-safe scalar."""
            if v is None:
                return None
            try:
                f = float(v)
                if np.isnan(f) or np.isinf(f):
                    return None
                return round(f, 4)
            except (TypeError, ValueError):
                return str(v)

        row = {
            "timestamp" : self.timestamp,
            "recording" : recording_id,
            "method"    : method,
        }
        for k, v in metrics.items():
            if k in ("label", "tp_pairs"):
                continue
            row[k] = _safe_value(v)

        self.records.append(row)
        print(f"[Logger] Logged: {recording_id} / {method}")

    def _output_fieldnames(self) -> list[str]:
        return [self.COLUMN_LABELS.get(k, k) for k in self.records[0].keys()]

    def _output_records(self) -> list[dict]:
        return [
            {self.COLUMN_LABELS.get(k, k): v for k, v in row.items()}
            for row in self.records
        ]

    def save_summary(self, aggregate_metrics: dict) -> str:
        """Save a human-readable summary text file for the dataset run."""
        summary_lines = [
            f"Results summary generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Dataset output path: {self.results_dir}",
            f"Number of recordings: {len(self.records)}",
            "",
            "Aggregated metrics:",
        ]
        if not aggregate_metrics:
            summary_lines.append("  No aggregated metrics available.")
        else:
            for metric, value in aggregate_metrics.items():
                summary_lines.append(f"  {metric}: {value:.4f}")

        text_path = self.results_dir / f"results_summary_{self.timestamp}.txt"
        with open(text_path, "w") as f:
            f.write("\n".join(summary_lines) + "\n")

        print(f"[Logger] Saved summary text → {text_path}")
        return str(text_path)

    def save(self) -> tuple[str, str]:
        """
        Save all records to CSV and JSON.

        Returns paths to both files.
        """
        if not self.records:
            print("[Logger] No records to save.")
            return None, None

        csv_path  = self.results_dir / f"results_{self.timestamp}.csv"
        json_path = self.results_dir / f"results_{self.timestamp}.json"

        output_records = self._output_records()
        fieldnames = self._output_fieldnames()

        # CSV
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output_records)

        # JSON (for easy loading in analysis)
        with open(json_path, "w") as f:
            json.dump(output_records, f, indent=2, default=str)

        print(f"[Logger] Saved {len(self.records)} records to:")
        print(f"  CSV  → {csv_path}")
        print(f"  JSON → {json_path}")

        return str(csv_path), str(json_path)

    def print_ablation_table(self) -> None:
        """
        Print a formatted ablation table comparing configurations.
        Copy this directly into your paper.
        """
        methods = {}
        for r in self.records:
            m = r["method"]
            if m not in methods:
                methods[m] = []
            methods[m].append(r)

        metrics = ["Se", "PPV", "F1", "SNR_dB", "PRD_pct", "FHR_MAE_bpm"]

        header = f"{'Method':<35}" + "".join(f"{m:>12}" for m in metrics)
        print(f"\n{'='*len(header)}")
        print("ABLATION TABLE (mean ± std across recordings)")
        print(f"{'='*len(header)}")
        print(header)
        print(f"{'-'*len(header)}")

        for method_name, records in methods.items():
            row = f"{method_name:<35}"
            for m in metrics:
                vals = [r.get(m, np.nan) for r in records]
                vals = [v for v in vals if v is not None and not
                        (isinstance(v, float) and np.isnan(v))]
                if vals:
                    mean = np.mean(vals)
                    std  = np.std(vals)
                    row += f"{mean:>7.2f}±{std:.2f}"
                else:
                    row += f"{'N/A':>12}"
            print(row)

        print(f"{'='*len(header)}\n")


class ECHOResultsLogger:
    """
    DEPRECATED: Use xai.echo_master_table.ECHOMasterTableGenerator instead.

    This class was used to generate echo_summary_<timestamp>.csv files.
    It is retained for backward compatibility only and should not be used
    in new code. The Master Explainability Table provides a more structured
    and comprehensive approach to ECHO data export.
    """
    """Logs per-recording ECHO summaries to a dataset-level CSV."""

    COLUMNS = [
        "timestamp", "recording", "method",
        "n_beats", "mean_fetal_hr", "fetal_hr_std",
        "maternal_hr", "hr_separation",
        "normal_hr_beats", "normal_hr_pct",
        "bradycardia_beats", "bradycardia_pct",
        "tachycardia_beats", "tachycardia_pct",
        "mean_hr_contrast_pct", "mean_morphology_pct",
        "mean_temporal_independence_pct", "mean_confidence_pct",
        "has_morphology", "clinical_note", "clinical_explanation",
    ]

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.records = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _safe_value(self, v):
        if v is None:
            return ""
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return ""
        return v

    def log_recording(self, recording_id: str, method: str, summary: dict) -> None:
        row = {
            "timestamp" : self.timestamp,
            "recording" : recording_id,
            "method"    : method,
        }
        for col in self.COLUMNS:
            if col in row:
                continue
            row[col] = self._safe_value(summary.get(col, ""))

        self.records.append(row)
        print(f"[ECHOResultsLogger] Logged: {recording_id} / {method}")

    def save(self) -> str:
        if not self.records:
            print("[ECHOResultsLogger] No records to save.")
            return None

        csv_path = self.results_dir / f"echo_summary_{self.timestamp}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
            writer.writeheader()
            writer.writerows(self.records)

        print(f"[ECHOResultsLogger] Saved ECHO summary CSV → {csv_path}")
        return str(csv_path)
