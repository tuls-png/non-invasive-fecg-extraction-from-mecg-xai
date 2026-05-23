"""
evaluation/nifecgdb_evaluator.py

Indirect NIFECGDB validation using maternal annotation only.

When NIFECGDB does not provide a direct fetal reference, the evaluation
relies on the available maternal .qrs labels and the pipeline outputs.

Checks implemented:
- Maternal detector accuracy (annotation vs detected maternal peaks)
- Maternal cancellation quality (residual energy at maternal beats vs non-beat times)
- Fetal heart rate plausibility (100-200 BPM and at least 15 BPM separation from maternal)
"""

import csv
from datetime import datetime
from pathlib import Path

import numpy as np

from evaluation.metrics import match_peaks, compute_se_ppv_f1
from preprocessing.qrs_detector import compute_hr_stats, load_wfdb_annotation


class NIFECGDBEvaluator:
    def __init__(self, method: str | None = None,
                 output_dir: str = "results_nifecgdb",
                 tolerance_ms: float = 50.0,
                 beat_window_sec: float = 0.15):
        self.method = method
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.tolerance_ms = tolerance_ms
        self.beat_window_sec = beat_window_sec
        self.records = []

    def run_all_checks(self, recording_id: str, recording: dict,
                       maternal_peaks: np.ndarray,
                       residual: np.ndarray,
                       fetal_peaks: np.ndarray) -> dict:
        ann_path = recording.get("annotation_path")
        ann_ext  = recording.get("annotation_ext", "qrs")
        fs       = recording.get("fs")

        if ann_path is None:
            raise ValueError(f"No annotation available for {recording_id}")
        if fs is None:
            raise ValueError(f"Sampling rate missing for {recording_id}")

        annotated_peaks = load_wfdb_annotation(ann_path, ann_ext)

        # Maternal detection accuracy using annotated maternal peaks.
        maternal_match = match_peaks(
            detected=maternal_peaks,
            reference=annotated_peaks,
            fs=fs,
            tolerance_ms=self.tolerance_ms
        )
        maternal_scores = compute_se_ppv_f1(maternal_match)

        # Maternal cancellation quality using residual energy.
        cancellation_ratio = self._maternal_residual_energy_ratio(
            residual=residual,
            maternal_peaks=annotated_peaks,
            fs=fs,
            half_window_sec=self.beat_window_sec
        )

        # Fetal HR plausibility from detected fetal peaks.
        fetal_stats = compute_hr_stats(fetal_peaks, fs)
        maternal_stats = compute_hr_stats(maternal_peaks, fs)
        fetal_hr = fetal_stats["mean_hr"]
        maternal_hr = maternal_stats["mean_hr"]
        hr_separation = np.nan
        if not np.isnan(fetal_hr) and not np.isnan(maternal_hr):
            hr_separation = abs(fetal_hr - maternal_hr)

        fetal_hr_in_range = (100.0 <= fetal_hr <= 200.0) if not np.isnan(fetal_hr) else False
        hr_separation_ok = (hr_separation >= 15.0) if not np.isnan(hr_separation) else False
        fetal_hr_plausible = fetal_hr_in_range and hr_separation_ok

        result = {
            "timestamp"                    : self.timestamp,
            "method"                       : self.method or "N/A",
            "recording"                    : recording_id,
            "n_annotated_maternal_beats"   : int(len(annotated_peaks)),
            "n_detected_maternal_beats"    : int(len(maternal_peaks)),
            "maternal_Se"                  : float(maternal_scores["Se"] * 100.0),
            "maternal_PPV"                 : float(maternal_scores["PPV"] * 100.0),
            "maternal_F1"                  : float(maternal_scores["F1"] * 100.0),
            "maternal_residual_energy_ratio": float(cancellation_ratio),
            "fetal_hr_bpm"                 : float(fetal_hr) if not np.isnan(fetal_hr) else np.nan,
            "maternal_hr_bpm"              : float(maternal_hr) if not np.isnan(maternal_hr) else np.nan,
            "hr_separation_bpm"            : float(hr_separation) if not np.isnan(hr_separation) else np.nan,
            "fetal_hr_in_range"            : bool(fetal_hr_in_range),
            "maternal_fetal_hr_sep_ok"      : bool(hr_separation_ok),
            "fetal_hr_plausible"           : bool(fetal_hr_plausible),
        }

        self.records.append(result)
        self._print_record_summary(result)
        return result

    def save(self) -> str | None:
        if not self.records:
            print("[NIFECGDB] No validation records to save.")
            return None

        filename = f"nifecgdb_validation_summary"
        if self.method:
            filename += f"_{self.method}"
        filename += f"_{self.timestamp}.csv"

        summary_path = self.output_dir / filename
        fieldnames = list(self.records[0].keys())

        with open(summary_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.records)

        print(f"[NIFECGDB] Validation summary saved → {summary_path}")
        return str(summary_path)

    def _maternal_residual_energy_ratio(self, residual: np.ndarray,
                                        maternal_peaks: np.ndarray,
                                        fs: int,
                                        half_window_sec: float = 0.15) -> float:
        if residual.ndim == 1:
            residual = residual[np.newaxis, :]

        n_channels, N = residual.shape
        hw = int(half_window_sec * fs)
        mask = np.zeros(N, dtype=bool)

        for pk in maternal_peaks:
            lo = max(0, pk - hw)
            hi = min(N, pk + hw)
            mask[lo:hi] = True

        if not np.any(mask) or np.all(mask):
            return np.nan

        beat_power = np.mean(residual[:, mask]**2)
        nonbeat_power = np.mean(residual[:, ~mask]**2)
        return float(beat_power / (nonbeat_power + 1e-12))

    def _print_record_summary(self, row: dict) -> None:
        print(f"\n[NIFECGDB] Validation checks for {row['recording']}")
        print(f"  Maternal detection: Se={row['maternal_Se']:.2f}%, "
              f"PPV={row['maternal_PPV']:.2f}%, F1={row['maternal_F1']:.2f}%")
        print(f"  Residual energy ratio (beat/non-beat): {row['maternal_residual_energy_ratio']:.3f}")
        print(f"  Fetal HR={row['fetal_hr_bpm']:.1f} BPM, Maternal HR={row['maternal_hr_bpm']:.1f} BPM, "
              f"sep={row['hr_separation_bpm']:.1f} BPM")
        print(f"  Fetal HR plausible: {row['fetal_hr_plausible']} "
              f"(in_range={row['fetal_hr_in_range']}, sep_ok={row['maternal_fetal_hr_sep_ok']})\n")
