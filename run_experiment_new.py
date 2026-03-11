"""
run_experiment_new.py
Updated entry point using the new streamlined dataset/config architecture.

This is an example of the RECOMMENDED approach. The original run_experiment.py
still works (backward compatible), but this shows cleaner patterns.

Usage:
    python run_experiment_new.py --dataset adfecgdb --mode full
    python run_experiment_new.py --dataset nifecgdb --mode full --max_recordings 10
    python run_experiment_new.py --dataset adfecgdb --mode single --recording r01.edf
"""

import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from configs import get_config
from dataset_handlers import get_dataset
from pipeline import PHASEPipeline
from evaluation.metrics import aggregate_results, wilcoxon_test
from utils.logger import ResultsLogger
from utils.visualization import plot_ablation_results, plot_sota_comparison


def run_full_dataset(dataset_name: str, data_dir: str, save_figures: bool = True, max_recordings: int = None):
    """
    Run PHASE on all recordings in a dataset.
    
    Parameters
    ----------
    dataset_name : str
        Dataset identifier ('adfecgdb' or 'nifecgdb').
    data_dir : str
        Path to dataset directory.
    save_figures : bool
        Whether to save output figures.
    """
    print("\n" + "="*70)
    print(f"  PHASE Pipeline — {dataset_name.upper()} Full Experiment")
    print("="*70 + "\n")

    # Get dataset-specific config and handler
    config = get_config(dataset_name)
    handler = get_dataset(dataset_name)
    '''
    ablation = "recordings = handler.load_all_recordings(
    data_dir,
    max_recordings=getattr(args, 'max_recordings', None)
)"
    '''
    # Load all recordings from dataset
    recordings = handler.load_all_recordings(data_dir,max_recordings=max_recordings)
    if not recordings:
        print(f"[ERROR] No recordings found in {data_dir}")
        return []

    # Run pipeline on each recording
    pipe = PHASEPipeline(verbose=True, dataset=dataset_name)
    logger = ResultsLogger(f"results_{dataset_name}")
    all_metrics = []

    for rec in recordings:
        # Use handler's summary method
        handler.print_recording_summary(rec)
        
        try:
            result = pipe.run(
                rec,
                save_figures=save_figures,
                figures_dir=f"figures_{dataset_name}"
            )
            logger.log_recording(rec["recording"], f"PHASE_{dataset_name}", result["metrics"])
            all_metrics.append(result["metrics"])
        except Exception as e:
            print(f"[ERROR] {rec['recording']}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print summary
    print("\n" + "="*70)
    print(f"  {dataset_name.upper()} AGGREGATE RESULTS")
    print("="*70)
    aggregate_results(all_metrics)
    logger.save()
    
    return all_metrics


def run_ablation_dataset(dataset_name: str, data_dir: str, max_recordings: int = None):
    """
    Run ablation studies on a dataset.
    
    Parameters
    ----------
    dataset_name : str
        Dataset identifier.
    data_dir : str, identifier.
        Path to dataset directory.
    """
    print("\n" + "="*70)
    print(f"  PHASE Pipeline — {dataset_name.upper()} Ablation Study")
    print("="*70 + "\n")

    config = get_config(dataset_name)
    handler = get_dataset(dataset_name)
    
    recordings = handler.load_all_recordings(data_dir,max_recordings=max_recordings)
    pipe = PHASEPipeline(verbose=True, dataset=dataset_name)
    logger = ResultsLogger(f"results_ablation_{dataset_name}")
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

    # Statistical testing if baseline and full exist
    if ("1_Baseline_ICA_WSVD" in config_metrics and
            "5_PHASE_Full" in config_metrics):
        baseline_f1 = [r["F1"] for r in config_metrics["1_Baseline_ICA_WSVD"]]
        phase_f1 = [r["F1"] for r in config_metrics["5_PHASE_Full"]]
        wilcoxon_test(phase_f1, baseline_f1, metric_name="F1")

    # Generate ablation plot
    ablation_mean, ablation_std = {}, {}
    for config, records in sorted(config_metrics.items()):
        f1_vals = [r["F1"] for r in records]
        short = config.split("_", 1)[1].replace("_", " ")
        ablation_mean[short] = float(np.mean(f1_vals))
        ablation_std[short] = float(np.std(f1_vals))

    Path("figures").mkdir(exist_ok=True)
    fig = plot_ablation_results(
        ablation_mean, metric="F1 (%)",
        std_data=ablation_std, save_path=f"figures/ablation_f1_{dataset_name}.png"
    )
    fig.show()

    return config_metrics


def run_single_recording(dataset_name: str, filepath: str):
    """
    Run PHASE on a single recording.
    
    Parameters
    ----------
    dataset_name : str
        Dataset identifier.
    filepath : str
        Path to recording file.
    """
    print("\n" + "="*70)
    print(f"  PHASE Pipeline — Single Recording ({dataset_name.upper()})")
    print("="*70 + "\n")

    config = get_config(dataset_name)
    handler = get_dataset(dataset_name)
    
    rec = handler.load_single_recording(filepath)
    handler.print_recording_summary(rec)

    pipe = PHASEPipeline(verbose=True, dataset=dataset_name)
    result = pipe.run(rec, save_figures=True, figures_dir="figures")

    print("\nFinal Metrics:")
    for k, v in result["metrics"].items():
        if isinstance(v, float):
            print(f"  {k:<20}: {v:.4f}")

    # Show attribution heatmap if ECHO is available
    if hasattr(result.get("echo"), "plot_attribution_heatmap"):
        result["echo"].plot_attribution_heatmap(window_sec=10)
        import matplotlib.pyplot as plt
        plt.show()

    return result


def main():
    """Main entry point."""
    # Default paths
    default_adfecgdb = str(
        Path(__file__).parent / "dataset_handlers" /
        "abdominal-and-direct-fetal-ecg-database-1.0.0"
    )
    default_nifecgdb = str(
        Path(__file__).parent / "dataset_handlers" / "non-invasive-fetal-ecg-database-1.0.0"
    )
    default_cinc2013 = str(
        Path(__file__).parent / "dataset_handlers" / "set-a"
    )
    parser = argparse.ArgumentParser(
        description="PHASE Fetal ECG Pipeline — Streamlined Version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full experiment on ADFECGDB
    python run_experiment_new.py --dataset adfecgdb --mode full
    
    # Run ablation study on NIFECGDB (first 20 recordings)
    python run_experiment_new.py --dataset nifecgdb --mode ablation --max_recordings 20
    
    # Run single recording
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

    # Select data directory
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

    # Print configuration info
    config = get_config(args.dataset)
    print(f"\n[CONFIG] Using {args.dataset.upper()}")
    print(f"[CONFIG] FETAL_HR_LOW: {config.FETAL_HR_LOW}")
    print(f"[CONFIG] FETAL_HR_HIGH: {config.FETAL_HR_HIGH}")
    print(f"[CONFIG] ICA_N_COMPONENTS: {config.ICA_N_COMPONENTS}\n")

    # Execute selected mode
    if args.mode == "single":
        if args.recording:
            filepath = args.recording
        else:
            # Find first EDF
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
        run_ablation_dataset(args.dataset, str(data_path))


if __name__ == "__main__":
    main()
