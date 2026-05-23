"""
run_experiment_new.py
Entry point for the PHASE fetal ECG pipeline.

Usage:
    # Single method
    python run_experiment_new.py --dataset cinc2013 --mode full --method original
    python run_experiment_new.py --dataset cinc2013 --mode full --method sequential
    python run_experiment_new.py --dataset cinc2013 --mode full --method ensemble_simple
    python run_experiment_new.py --dataset cinc2013 --mode full --method ensemble_weighted
    python run_experiment_new.py --dataset cinc2013 --mode full --method rescue
    python run_experiment_new.py --dataset cinc2013 --mode full --method peak_fusion

    # Run ALL methods in sequence + produce comparison summary CSV
    python run_experiment_new.py --dataset cinc2013 --mode full --method all

    # Single recording
    python run_experiment_new.py --dataset adfecgdb --mode single --recording r01.edf
"""
import matplotlib
matplotlib.use('Agg')
import sys
import csv
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from configs import get_config
from dataset_handlers import get_dataset
from pipeline import PHASEPipeline
from evaluation.metrics import aggregate_results, wilcoxon_test
from evaluation.nifecgdb_evaluator import NIFECGDBEvaluator
from utils.logger import ResultsLogger
from utils.visualization import plot_ablation_results, plot_sota_comparison


# ------------------------------------------------------------------ #
#  Method registry                                                     #
# ------------------------------------------------------------------ #
ALL_METHODS = [
    "original",
    "sequential",
    "ensemble_simple",
    "ensemble_weighted",
    "rescue",
    "peak_fusion",
]

METHOD_LABELS = {
    "original"          : "PHASE (original)",
    "sequential"        : "PHASE-SEQ",
    "ensemble_simple"   : "PHASE-ENS-SIMPLE",
    "ensemble_weighted" : "PHASE-ENS-WEIGHTED",
    "rescue"            : "PHASE-RESCUE",
    "peak_fusion"       : "PHASE-PEAK-FUSION",
}

def _get_runner(pipe, method):
    """Return the pipeline method corresponding to the method key."""
    return {
        "original"          : pipe.run,
        "sequential"        : pipe.run_sequential,
        "ensemble_simple"   : pipe.run_ensemble_simple,
        "ensemble_weighted" : pipe.run_ensemble_weighted,
        "rescue"            : pipe.run_rescue,
        "peak_fusion"       : pipe.run_peak_fusion,
    }[method]


# ------------------------------------------------------------------ #
#  Core runners                                                        #
# ------------------------------------------------------------------ #
def run_method_on_dataset(method, dataset_name, data_dir,
                          save_figures=True, max_recordings=None):
    """
    Run a single method on all recordings. Returns list of metric dicts
    and the path to the saved CSV.
    """
    label = METHOD_LABELS[method]
    print("\n" + "=" * 70)
    print(f"  {label}  —  {dataset_name.upper()}")
    print("=" * 70 + "\n")

    handler    = get_dataset(dataset_name)
    recordings = handler.load_all_recordings(data_dir,
                                             max_recordings=max_recordings)
    if not recordings:
        print(f"[ERROR] No recordings found in {data_dir}")
        return [], None

    pipe    = PHASEPipeline(verbose=True, dataset=dataset_name)
    runner  = _get_runner(pipe, method)
    logger  = ResultsLogger(f"results_{method}_{dataset_name}")
    nifecgdb_evaluator = (
        NIFECGDBEvaluator(method=method)
        if dataset_name == "nifecgdb" else None
    )
    all_metrics = []

    for rec in recordings:
        handler.print_recording_summary(rec)
        try:
            result = runner(
                rec,
                save_figures=save_figures,
                figures_dir=f"figures_{method}_{dataset_name}"
            )
            logger.log_recording(
                rec["recording"], label, result["metrics"])
            all_metrics.append(result["metrics"])

            if nifecgdb_evaluator is not None:
                try:
                    nifecgdb_evaluator.run_all_checks(
                        rec["recording"], rec,
                        result["maternal_peaks"],
                        result["residual"],
                        result["fetal_peaks"]
                    )
                except Exception as e:
                    print(f"[NIFECGDB] Validation failed for {rec['recording']}: {e}")
                    import traceback; traceback.print_exc()
        except Exception as e:
            print(f"[ERROR] {rec['recording']}: {e}")
            import traceback; traceback.print_exc()
            continue

    print("\n" + "=" * 70)
    print(f"  {dataset_name.upper()}  {label}  —  AGGREGATE")
    print("=" * 70)
    aggregate_results(all_metrics)
    csv_path, _ = logger.save()
    if nifecgdb_evaluator is not None:
        nifecgdb_evaluator.save()
    return all_metrics, csv_path


def run_all_methods(dataset_name, data_dir,
                    save_figures=True, max_recordings=None):
    """
    Run every method in ALL_METHODS sequentially, then write a
    summary CSV comparing aggregate metrics across methods.
    """
    summary_rows   = []
    per_method_csv = {}
    timestamp      = datetime.now().strftime("%Y%m%d_%H%M%S")

    for method in ALL_METHODS:
        metrics_list, csv_path = run_method_on_dataset(
            method, dataset_name, data_dir,
            save_figures=save_figures,
            max_recordings=max_recordings)

        per_method_csv[method] = csv_path

        if not metrics_list:
            continue

        # Aggregate
        def _mean(key):
            vals = [m[key] for m in metrics_list
                    if m.get(key) is not None and not
                    (isinstance(m[key], float) and np.isnan(m[key]))]
            return float(np.mean(vals)) if vals else float("nan")

        def _std(key):
            vals = [m[key] for m in metrics_list
                    if m.get(key) is not None and not
                    (isinstance(m[key], float) and np.isnan(m[key]))]
            return float(np.std(vals)) if vals else float("nan")

        summary_rows.append({
            "method"         : METHOD_LABELS[method],
            "n_recordings"   : len(metrics_list),
            "Se_mean"        : round(_mean("Se"),  3),
            "Se_std"         : round(_std("Se"),   3),
            "PPV_mean"       : round(_mean("PPV"), 3),
            "PPV_std"        : round(_std("PPV"),  3),
            "F1_mean"        : round(_mean("F1"),  3),
            "F1_std"         : round(_std("F1"),   3),
            "FHR_MAE_mean"   : round(_mean("FHR_MAE_bpm"), 3),
            "FHR_MAE_std"    : round(_std("FHR_MAE_bpm"),  3),
            "results_csv"    : csv_path or "",
        })

    # Write summary CSV
    summary_dir = Path(f"results_summary_{dataset_name}")
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"summary_{timestamp}.csv"

    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

        # Pretty-print the comparison table
        print("\n" + "=" * 80)
        print(f"  ALL-METHODS SUMMARY  —  {dataset_name.upper()}")
        print("=" * 80)
        print(f"  {'Method':<25} {'Se':>8} {'PPV':>8} {'F1':>8} {'FHR_MAE':>10}")
        print("  " + "-" * 62)
        for row in summary_rows:
            print(f"  {row['method']:<25} "
                  f"{row['Se_mean']:>7.2f}% "
                  f"{row['PPV_mean']:>7.2f}% "
                  f"{row['F1_mean']:>7.2f}% "
                  f"{row['FHR_MAE_mean']:>9.2f}")
        print("=" * 80)
        print(f"\n  Summary CSV saved → {summary_path}\n")

    return summary_rows


# ------------------------------------------------------------------ #
#  Existing modes (ablation, single) — unchanged                      #
# ------------------------------------------------------------------ #
def run_ablation_dataset(dataset_name, data_dir, max_recordings=None):
    print("\n" + "=" * 70)
    print(f"  PHASE Pipeline — {dataset_name.upper()} Ablation Study")
    print("=" * 70 + "\n")

    handler    = get_dataset(dataset_name)
    recordings = handler.load_all_recordings(data_dir,
                                             max_recordings=max_recordings)
    pipe           = PHASEPipeline(verbose=True, dataset=dataset_name)
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
    logger.print_ablation_table()

    if ("1_Baseline_ICA_WSVD" in config_metrics and
            "5_PHASE_Full" in config_metrics):
        baseline_f1 = [r["F1"] for r in config_metrics["1_Baseline_ICA_WSVD"]]
        phase_f1    = [r["F1"] for r in config_metrics["5_PHASE_Full"]]
        wilcoxon_test(phase_f1, baseline_f1, metric_name="F1")

    ablation_mean, ablation_std = {}, {}
    for config, records in sorted(config_metrics.items()):
        f1_vals = [r["F1"] for r in records]
        short   = config.split("_", 1)[1].replace("_", " ")
        ablation_mean[short] = float(np.mean(f1_vals))
        ablation_std[short]  = float(np.std(f1_vals))

    Path("figures").mkdir(exist_ok=True)
    fig = plot_ablation_results(
        ablation_mean, metric="F1 (%)",
        std_data=ablation_std,
        save_path=f"figures/ablation_f1_{dataset_name}.png")
    fig.show()
    return config_metrics


def run_single_recording(dataset_name, filepath, method="original"):
    print("\n" + "=" * 70)
    print(f"  PHASE Pipeline — Single Recording ({dataset_name.upper()}) "
          f"[{METHOD_LABELS.get(method, method)}]")
    print("=" * 70 + "\n")

    handler = get_dataset(dataset_name)
    rec     = handler.load_single_recording(filepath)
    handler.print_recording_summary(rec)

    pipe   = PHASEPipeline(verbose=True, dataset=dataset_name)
    runner = _get_runner(pipe, method)
    result = runner(rec, save_figures=True, figures_dir="figures")

    print("\nFinal Metrics:")
    for k, v in result["metrics"].items():
        if isinstance(v, float):
            print(f"  {k:<20}: {v:.4f}")
    return result


# ------------------------------------------------------------------ #
#  main()                                                              #
# ------------------------------------------------------------------ #
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
    python run_experiment_new.py --dataset cinc2013 --mode full --method original
    python run_experiment_new.py --dataset cinc2013 --mode full --method all
    python run_experiment_new.py --dataset adfecgdb --mode full --method peak_fusion
    python run_experiment_new.py --dataset adfecgdb --mode single --recording r01.edf --method rescue
    python run_experiment_new.py --dataset cinc2013 --mode full --method all --max_recordings 10
        """
    )

    parser.add_argument("--dataset", type=str, default="adfecgdb",
                        choices=["adfecgdb", "nifecgdb", "cinc2013"])
    parser.add_argument("--mode", type=str, default="full",
                        choices=["full", "ablation", "single"])
    parser.add_argument("--method", type=str, default="original",
                        choices=ALL_METHODS + ["all"],
                        help="Pipeline variant to run (default: original). "
                             "Use 'all' to run every method in sequence.")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--recording", type=str, default=None,
                        help="Recording file (for mode=single)")
    parser.add_argument("--max_recordings", type=int, default=None)
    parser.add_argument("--no_figures", action="store_true")

    args = parser.parse_args()

    # Resolve data directory
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
    print(f"\n[CONFIG] Dataset : {args.dataset.upper()}")
    print(f"[CONFIG] Method  : {args.method}")
    print(f"[CONFIG] Mode    : {args.mode}")
    print(f"[CONFIG] FETAL_HR_LOW  : {config.FETAL_HR_LOW}")
    print(f"[CONFIG] FETAL_HR_HIGH : {config.FETAL_HR_HIGH}")
    print(f"[CONFIG] ICA_N_COMPONENTS : {config.ICA_N_COMPONENTS}\n")

    if args.mode == "single":
        filepath = args.recording
        if not filepath:
            edfs = sorted(data_path.glob("*.edf"))
            if not edfs:
                print(f"[ERROR] No EDF files found in {data_dir}")
                sys.exit(1)
            filepath = str(edfs[0])
            print(f"[INFO] No --recording specified, using: {filepath}")
        run_single_recording(args.dataset, filepath,
                             method=args.method if args.method != "all" else "original")

    elif args.mode == "ablation":
        run_ablation_dataset(args.dataset, str(data_path),
                             max_recordings=args.max_recordings)

    elif args.mode == "full":
        if args.method == "all":
            run_all_methods(
                args.dataset, str(data_path),
                save_figures=not args.no_figures,
                max_recordings=args.max_recordings)
        else:
            run_method_on_dataset(
                args.method, args.dataset, str(data_path),
                save_figures=not args.no_figures,
                max_recordings=args.max_recordings)


if __name__ == "__main__":
    main()