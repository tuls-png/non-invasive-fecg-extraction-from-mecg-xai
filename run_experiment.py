"""
run_experiment.py
Main entry point for the PHASE fetal ECG separation experiment.

Usage:
    # ADFECGDB (with direct electrode + .qrs annotations)
    python run_experiment.py --mode full
    python run_experiment.py --mode ablation
    python run_experiment.py --mode single --recording r01.edf

    # NIFECGDB (no direct electrode, .qrs annotations only)
    python run_experiment.py --mode nifecgdb --nifecgdb_dir /path/to/nifecgdb
    python run_experiment.py --mode nifecgdb --nifecgdb_dir /path/to/nifecgdb --max_recordings 10

Modes:
    full       : ADFECGDB — run PHASE on all 5 recordings
    ablation   : ADFECGDB — run all 5 ablation configurations
    single     : ADFECGDB — run on one recording (for quick testing)
    nifecgdb   : NIFECGDB — run PHASE on all 55 recordings (or subset)
"""

import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data.loader import (
    load_edf, load_all_recordings, load_nifecgdb_edf,
    load_all_nifecgdb, print_recording_summary
)
from pipeline import PHASEPipeline
from evaluation.metrics import aggregate_results, wilcoxon_test
from utils.logger import ResultsLogger
from utils.visualization import plot_ablation_results, plot_sota_comparison


# ── ADFECGDB modes ─────────────────────────────────────────────────────────────

def run_full(data_dir: str, save_figures: bool = True):
    """Run PHASE on all ADFECGDB recordings and aggregate results."""
    print("\n" + "="*60)
    print("  PHASE Pipeline — ADFECGDB Full Experiment")
    print("="*60 + "\n")

    recordings  = load_all_recordings(data_dir)
    pipe = PHASEPipeline(verbose=True, dataset="ADFECGDB")
    logger      = ResultsLogger("results")
    all_metrics = []

    for rec in recordings:
        print_recording_summary(rec)
        result = pipe.run(rec, save_figures=save_figures, figures_dir="figures")
        logger.log_recording(rec["recording"], "PHASE_full", result["metrics"])
        all_metrics.append(result["metrics"])

    print("\n" + "="*60)
    print("  AGGREGATE RESULTS (for paper Table)")
    print("="*60)
    aggregate_results(all_metrics)
    logger.save()
    logger.print_ablation_table()
    return all_metrics


def run_ablation(data_dir: str):
    """Run all ablation configurations on ADFECGDB."""
    print("\n" + "="*60)
    print("  PHASE Pipeline — Ablation Study")
    print("="*60 + "\n")

    recordings     = load_all_recordings(data_dir)
    pipe           = PHASEPipeline(verbose=True)
    logger         = ResultsLogger("results")
    config_metrics = {}

    for rec in recordings:
        print_recording_summary(rec)
        ablation_results = pipe.run_with_ablation(rec)
        for config_name, metrics in ablation_results.items():
            logger.log_recording(rec["recording"], config_name, metrics)
            config_metrics.setdefault(config_name, []).append(metrics)

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
        std_data=ablation_std, save_path="figures/ablation_f1.png"
    )
    fig.show()
    return config_metrics


def run_single(filepath: str):
    """Run on a single ADFECGDB EDF file."""
    print("\n" + "="*60)
    print("  PHASE Pipeline — Single Recording")
    print("="*60 + "\n")

    rec = load_edf(filepath)
    print_recording_summary(rec)

    pipe   = PHASEPipeline(verbose=True)
    result = pipe.run(rec, save_figures=True, figures_dir="figures")

    print("\nFinal Metrics:")
    for k, v in result["metrics"].items():
        if isinstance(v, float):
            print(f"  {k:<20}: {v:.4f}")

    result["echo"].plot_attribution_heatmap(window_sec=10)
    import matplotlib.pyplot as plt
    plt.show()
    return result


# ── NIFECGDB mode ──────────────────────────────────────────────────────────────

def run_nifecgdb(nifecgdb_dir: str, save_figures: bool = False,
                 max_recordings: int = None):
    """
    Run PHASE on NIFECGDB recordings.

    Key differences vs ADFECGDB:
    - No direct electrode → SNR/PRD/CC are N/A
    - .edf.qrs annotations are the ONLY ground truth (used for evaluation)
    - .qrs was annotated from abdominal signals — not a privileged source
    - Variable duration recordings; variable number of abdominal channels

    Results reported: Se, PPV, F1, FHR_MAE (waveform metrics skipped)
    """
    print("\n" + "="*60)
    print("  PHASE Pipeline — NIFECGDB Experiment")
    print("="*60 + "\n")

    recordings  = load_all_nifecgdb(nifecgdb_dir, max_recordings=max_recordings)
    pipe = PHASEPipeline(verbose=True, dataset="NIFECGDB")
    logger      = ResultsLogger("results_nifecgdb")
    all_metrics = []
    skipped     = 0

    for rec in recordings:
        # Skip recordings with no annotation file — can't evaluate
        if not rec.get("annotation_path"):
            print(f"[skip] {rec['recording']} — no .qrs annotation found")
            skipped += 1
            continue

        print_recording_summary(rec)
        try:
            result = pipe.run(
                rec,
                save_figures=save_figures,
                figures_dir="figures_nifecgdb"
            )
            logger.log_recording(rec["recording"], "PHASE_nifecgdb",
                                result["metrics"])
            all_metrics.append(result["metrics"])
        except Exception as e:
            import traceback
            print(f"[ERROR] {rec['recording']}: {e}")
            traceback.print_exc()   # ← shows exact line
            skipped += 1
            continue

    print(f"\n[Summary] Processed {len(all_metrics)} recordings "
          f"({skipped} skipped)")
    print("\n" + "="*60)
    print("  NIFECGDB AGGREGATE RESULTS")
    print("="*60)
    if all_metrics:
        aggregate_results(all_metrics)
    else:
        print("[Summary] No successful recordings to aggregate.")
    logger.save()
    logger.print_ablation_table()
    return all_metrics


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    default_adfecgdb = str(
        Path(__file__).parent / "Datasets" /
        "abdominal-and-direct-fetal-ecg-database-1.0.0"
    )
    default_nifecgdb = str(
        Path(__file__).parent / "Datasets" / "non-invasive-fetal-ecg-database-1.0.0"
    )

    parser = argparse.ArgumentParser(description="PHASE Fetal ECG Pipeline")
    parser.add_argument("--data_dir", type=str, default=default_adfecgdb,
                        help="ADFECGDB directory (for full/ablation/single modes)")
    parser.add_argument("--nifecgdb_dir", type=str, default=default_nifecgdb,
                        help="NIFECGDB directory (for nifecgdb mode)")
    parser.add_argument("--mode", type=str, default="full",
                        choices=["full", "ablation", "single", "nifecgdb"],
                        help="Experiment mode")
    parser.add_argument("--recording", type=str, default=None,
                        help="EDF file path (for mode=single)")
    parser.add_argument("--max_recordings", type=int, default=None,
                        help="Limit number of recordings (for nifecgdb mode)")
    parser.add_argument("--no_figures", action="store_true",
                        help="Skip saving figures")
    args = parser.parse_args()

    if args.mode == "single":
        if args.recording:
            filepath = args.recording
        else:
            edfs = sorted(Path(args.data_dir).glob("*.edf"))
            if not edfs:
                raise FileNotFoundError(f"No EDF files in {args.data_dir}")
            filepath = str(edfs[0])
            print(f"No --recording specified, using: {filepath}")
        run_single(filepath)

    elif args.mode == "full":
        run_full(args.data_dir, save_figures=not args.no_figures)

    elif args.mode == "ablation":
        run_ablation(args.data_dir)

    elif args.mode == "nifecgdb":
        run_nifecgdb(
            args.nifecgdb_dir,
            save_figures=not args.no_figures,
            max_recordings=args.max_recordings
        )
