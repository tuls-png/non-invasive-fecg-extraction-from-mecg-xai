"""
run_experiment_new.py
Updated entry point using the new streamlined dataset/config architecture.

Usage:
    python run_experiment_new.py --dataset cinc2013 --mode full
    python run_experiment_new.py --dataset adfecgdb --mode full
    python run_experiment_new.py --dataset adfecgdb --mode ablation
    python run_experiment_new.py --dataset adfecgdb --mode single --recording r01.edf

METADATA / LOGGING NOTES:
  ResultsLogger signature is (self, results_dir: str = 'results') — no
  immediate_save, no csv_filename, no timestamp kwarg. This file works
  within that constraint:

  - Two separate ResultsLogger instances write to different subdirs:
      results/         <- main metrics CSV
      results_meta/    <- metadata CSV (path scores, flags, IC indices, etc.)

  - A shared timestamp string (generated once at run start) is prepended
    to each recording label so both CSVs are trivially joinable on that
    field even if ResultsLogger generates its own filenames internally.

  - PhaseLogger (phase_logger.py) writes structured debug logs to logs/:
      logs/phase_debug_<timestamp>.txt    human-readable per-recording
      logs/phase_debug_<timestamp>.jsonl  machine-readable JSON lines
    Called with the full result dict so all fields populate from
    result["metadata"] and result["metrics"] automatically.

  - all_metrics.append() is guarded inside the try block AFTER all logger
    calls, so a logger crash cannot skip the append and corrupt aggregates.
    The outer except catches pipeline errors only; logger errors are caught
    separately and do not abort the recording loop.
"""

import matplotlib
matplotlib.use('Agg')
import sys
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from configs import get_config
from dataset_handlers import get_dataset
from pipeline import PHASEPipeline
from evaluation.metrics import aggregate_results, wilcoxon_test
from utils.logger import ResultsLogger
from utils.visualization import plot_ablation_results, plot_sota_comparison
from phase_logger import PhaseLogger

log_path = f"stdout_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

def run_full_dataset(dataset_name: str, data_dir: str,
                     save_figures: bool = True,
                     max_recordings: int = None):
    """
    Run PHASE on all recordings in a dataset.

    Output files:
      results/<auto-named>.csv           -- per-recording metrics
      results_meta/<auto-named>.csv      -- per-recording metadata / diagnostics
      logs/phase_debug_<ts>.txt/.jsonl   -- structured debug logs
    """
    print("\n" + "=" * 70)
    print(f"  PHASE Pipeline — {dataset_name.upper()} Full Experiment")
    print("=" * 70 + "\n")

    config  = get_config(dataset_name)
    handler = get_dataset(dataset_name)

    recordings = handler.load_all_recordings(
        data_dir, max_recordings=max_recordings)
    if not recordings:
        print(f"[ERROR] No recordings found in {data_dir}")
        return []

    pipe = PHASEPipeline(verbose=True, dataset=dataset_name, stdout_log_path=log_path)

    # ResultsLogger only accepts results_dir — use separate dirs so the two
    # CSVs (metrics and metadata) land in different folders and don't clash.
    logger          = ResultsLogger(f"results")
    metadata_logger = ResultsLogger(f"results_meta")

    # PhaseLogger for structured debug output (txt + jsonl)
    phase_log = PhaseLogger(log_dir="logs")

    all_metrics = []

    for rec in recordings:
        handler.print_recording_summary(rec)

        try:
            result = pipe.run(
                rec,
                save_figures=save_figures,
                figures_dir=f"figures_{dataset_name}"
            )

            # --- Main metrics CSV ---
            try:
                logger.log_recording(
                    rec["recording"],
                    f"PHASE_{dataset_name}",
                    result["metrics"]
                )
            except Exception as log_err:
                print(f"[WARN] metrics logger failed for "
                      f"{rec['recording']}: {log_err}")

            # --- Metadata CSV ---
            # Pass result["metadata"] as the metrics dict — it is a flat dict
            # of diagnostic values (path scores, IC indices, HR estimates,
            # confidence flags, harmonic confusion flag, etc.) and writes
            # cleanly to CSV. The 4th kwarg `metadata=` is also passed for
            # compatibility with ResultsLogger implementations that support it.
            try:
                metadata_logger.log_recording(
                    rec["recording"],
                    f"PHASE_{dataset_name}_metadata",
                    result.get("metadata", {}),
                    metadata=result.get("metadata")
                )
            except TypeError:
                # ResultsLogger doesn't accept metadata kwarg — use 3-arg form
                try:
                    metadata_logger.log_recording(
                        rec["recording"],
                        f"PHASE_{dataset_name}_metadata",
                        result.get("metadata", {})
                    )
                except Exception as meta_err:
                    print(f"[WARN] metadata logger failed for "
                          f"{rec['recording']}: {meta_err}")
            except Exception as meta_err:
                print(f"[WARN] metadata logger failed for "
                      f"{rec['recording']}: {meta_err}")

            # --- PhaseLogger (debug txt + jsonl) ---
            try:
                phase_log.log_recording(rec["recording"], result)
            except Exception as plog_err:
                print(f"[WARN] phase_log failed for "
                      f"{rec['recording']}: {plog_err}")

            # Append AFTER all loggers — a logger error must not skip this
            all_metrics.append(result["metrics"])

        except Exception as e:
            print(f"[ERROR] {rec['recording']}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Aggregate summary
    print("\n" + "=" * 70)
    print(f"  {dataset_name.upper()} AGGREGATE RESULTS")
    print("=" * 70)
    if all_metrics:
        aggregate_results(all_metrics)
    else:
        print("  [WARNING] No results to aggregate.")

    try:
        logger.save()
    except Exception as e:
        print(f"[WARN] logger.save() failed: {e}")

    try:
        metadata_logger.save()
    except Exception as e:
        print(f"[WARN] metadata_logger.save() failed: {e}")

    phase_log.close()

    print(f"\n[OUTPUT] Metrics    -> results/")
    print(f"[OUTPUT] Metadata   -> results_meta/")
    print(f"[OUTPUT] Debug logs -> logs/")

    return all_metrics


def run_ablation_dataset(dataset_name: str, data_dir: str,
                          max_recordings: int = None):
    """Run ablation study on a dataset."""
    print("\n" + "=" * 70)
    print(f"  PHASE Pipeline — {dataset_name.upper()} Ablation Study")
    print("=" * 70 + "\n")

    config  = get_config(dataset_name)
    handler = get_dataset(dataset_name)

    recordings = handler.load_all_recordings(
        data_dir, max_recordings=max_recordings)

    pipe           = PHASEPipeline(verbose=True, dataset=dataset_name, stdout_log_path=log_path)
    logger         = ResultsLogger(f"results_ablation_{dataset_name}")
    config_metrics = {}

    for rec in recordings:
        handler.print_recording_summary(rec)

        try:
            ablation_results = pipe.run_with_ablation(rec)
            for config_name, metrics in ablation_results.items():
                logger.log_recording(rec["recording"], config_name, metrics)
                config_metrics.setdefault(config_name, []).append(metrics)
        except Exception as e:
            print(f"[ERROR] {rec['recording']}: {e}")
            continue

    logger.save()

    try:
        logger.print_ablation_table()
    except Exception as e:
        print(f"[WARN] print_ablation_table() failed: {e}")

    if ("1_Baseline_ICA_WSVD" in config_metrics and
            "5_PHASE_Full" in config_metrics):
        baseline_f1 = [r["F1"] for r in config_metrics["1_Baseline_ICA_WSVD"]]
        phase_f1    = [r["F1"] for r in config_metrics["5_PHASE_Full"]]
        wilcoxon_test(phase_f1, baseline_f1, metric_name="F1")

    ablation_mean, ablation_std = {}, {}
    for cfg_name, records in sorted(config_metrics.items()):
        f1_vals = [r["F1"] for r in records]
        short   = cfg_name.split("_", 1)[1].replace("_", " ")
        ablation_mean[short] = float(np.mean(f1_vals))
        ablation_std[short]  = float(np.std(f1_vals))

    Path("figures").mkdir(exist_ok=True)
    try:
        fig = plot_ablation_results(
            ablation_mean, metric="F1 (%)",
            std_data=ablation_std,
            save_path=f"figures/ablation_f1_{dataset_name}.png"
        )
        fig.show()
    except Exception as e:
        print(f"[WARN] plot_ablation_results() failed: {e}")

    return config_metrics


def run_single_recording(dataset_name: str, filepath: str):
    """Run PHASE on a single recording."""
    print("\n" + "=" * 70)
    print(f"  PHASE Pipeline — Single Recording ({dataset_name.upper()})")
    print("=" * 70 + "\n")

    config  = get_config(dataset_name)
    handler = get_dataset(dataset_name)

    rec    = handler.load_single_recording(filepath)
    handler.print_recording_summary(rec)

    pipe   = PHASEPipeline(verbose=True, dataset=dataset_name, stdout_log_path=log_path)
    result = pipe.run(rec, save_figures=True, figures_dir="figures")

    print("\nFinal Metrics:")
    for k, v in result["metrics"].items():
        if isinstance(v, float):
            print(f"  {k:<20}: {v:.4f}")

    print("\nMetadata:")
    for k, v in result.get("metadata", {}).items():
        print(f"  {k:<45}: {v}")

    return result


def main():
    default_adfecgdb = str(
        Path(__file__).parent / "dataset_handlers" /
        "abdominal-and-direct-fetal-ecg-database-1.0.0"
    )
    default_nifecgdb = str(
        Path(__file__).parent / "dataset_handlers" /
        "non-invasive-fetal-ecg-database-1.0.0"
    )
    default_cinc2013 = str(
        Path(__file__).parent / "dataset_handlers" / "set-a"
    )

    parser = argparse.ArgumentParser(
        description="PHASE Fetal ECG Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_experiment_new.py --dataset cinc2013 --mode full
    python run_experiment_new.py --dataset adfecgdb --mode full
    python run_experiment_new.py --dataset adfecgdb --mode ablation
    python run_experiment_new.py --dataset adfecgdb --mode single --recording r01.edf
        """
    )
    parser.add_argument(
        "--dataset", type=str, default="adfecgdb",
        choices=["adfecgdb", "nifecgdb", "cinc2013"],
        help="Dataset to use (default: adfecgdb)"
    )
    parser.add_argument(
        "--mode", type=str, default="full",
        choices=["full", "ablation", "single"],
        help="Experiment mode (default: full)"
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Dataset directory (auto-selected if not specified)"
    )
    parser.add_argument(
        "--recording", type=str, default=None,
        help="Recording file path (for mode=single)"
    )
    parser.add_argument(
        "--max_recordings", type=int, default=None,
        help="Limit number of recordings to process"
    )
    parser.add_argument(
        "--no_figures", action="store_true",
        help="Skip saving figures"
    )

    args = parser.parse_args()

    if args.data_dir:
        data_dir = args.data_dir
    elif args.dataset == "nifecgdb":
        data_dir = default_nifecgdb
    elif args.dataset == "cinc2013":
        data_dir = default_cinc2013
    else:
        data_dir = default_adfecgdb

    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"[ERROR] Dataset directory not found: {data_dir}")
        sys.exit(1)

    config = get_config(args.dataset)
    print(f"\n[CONFIG] Using {args.dataset.upper()}")
    print(f"[CONFIG] FETAL_HR_LOW            : {config.FETAL_HR_LOW}")
    print(f"[CONFIG] FETAL_HR_HIGH           : {config.FETAL_HR_HIGH}")
    print(f"[CONFIG] ICA_N_COMPONENTS        : {config.ICA_N_COMPONENTS}")
    print(f"[CONFIG] PATH_A_PREFERENCE       : {config.PATH_A_PREFERENCE}")
    print(f"[CONFIG] CONFIDENCE_GATE         : {config.CONFIDENCE_GATE_THRESHOLD}\n")

    if args.mode == "single":
        if args.recording:
            filepath = args.recording
        else:
            edfs = sorted(data_path.glob("*.edf"))
            if not edfs:
                print(f"[ERROR] No EDF files found in {data_dir}")
                sys.exit(1)
            filepath = str(edfs[0])
            print(f"[INFO] No --recording specified, using: {filepath}")
        run_single_recording(args.dataset, filepath)

    elif args.mode == "full":
        run_full_dataset(
            args.dataset, str(data_path),
            save_figures=not args.no_figures,
            max_recordings=args.max_recordings
        )

    elif args.mode == "ablation":
        run_ablation_dataset(
            args.dataset, str(data_path),
            max_recordings=args.max_recordings
        )


if __name__ == "__main__":
    main()