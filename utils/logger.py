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

        # CSV
        fieldnames = list(self.records[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.records)

        # JSON (for easy loading in analysis)
        with open(json_path, "w") as f:
            json.dump(self.records, f, indent=2, default=str)

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
