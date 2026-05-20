"""
run_experiment_sequential.py
Experiment runner for the PHASE-SEQ pipeline.

PHASE-SEQ applies Path A and Path B *sequentially* rather than selecting
between them:
  1. ICA1 → identify maternal IC (Path A)
  2. Project and subtract maternal IC from all channels (Path A contribution)
  3. AW-WSVD on the Path-A-cleaned multichannel residual (Path B on better input)
  4. ICA2 on doubly-cleaned residual → final fetal IC
  5. EKF-RTS refinement

This mirrors run_experiment_new.py but calls pipe.run_sequential() instead
of pipe.run(), and saves results to separate results_seq_* directories for
easy side-by-side comparison.

Usage:
    python run_experiment_sequential.py --dataset adfecgdb --mode full
    python run_experiment_sequential.py --dataset cinc2013 --mode full
    python run_experiment_sequential.py --dataset nifecgdb --mode full
    python run_experiment_sequential.py --dataset adfecgdb --mode single --recording r01.edf
    python run_experiment_sequential.py --dataset cinc2013 --mode full --max_recordings 10
"""
import matplotlib
matplotlib.use('Agg')
import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from configs import get_config
from dataset_handlers import get_dataset
from pipeline import PHASEPipeline
from evaluation.metrics import aggregate_results
from utils.logger import ResultsLogger


def run_sequential_full(dataset_name: str, data_dir: str,
                        save_figures: bool = True,
                        max_recordings: int = None):
    """
    Run PHASE-SEQ on all recordings in a dataset and print aggregate metrics.

    Parameters
    ----------
    dataset_name   : 'adfecgdb', 'nifecgdb', or 'cinc2013'
    data_dir       : path to dataset directory
    save_figures   : whether to save output figures
    max_recordings : cap number of recordings (None = all)
    """
    print("\n" + "=" * 70)
    print(f"  PHASE-SEQ Pipeline — {dataset_name.upper()} Full Experiment")
    print("=" * 70 + "\n")

    handler    = get_dataset(dataset_name)
    recordings = handler.load_all_recordings(data_dir, max_recordings=max_recordings)
    if not recordings:
        print(f"[ERROR] No recordings found in {data_dir}")
        return []

    pipe       = PHASEPipeline(verbose=True, dataset=dataset_name)
    logger     = ResultsLogger(f"results_seq_{dataset_name}")
    all_metrics = []

    for rec in recordings:
        handler.print_recording_summary(rec)
        try:
            result = pipe.run_sequential(
                rec,
                save_figures=save_figures,
                figures_dir=f"figures_seq_{dataset_name}"
            )
            logger.log_recording(
                rec["recording"],
                f"PHASE_SEQ_{dataset_name}",
                result["metrics"]
            )
            all_metrics.append(result["metrics"])
        except Exception as e:
            print(f"[ERROR] {rec['recording']}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 70)
    print(f"  {dataset_name.upper()} PHASE-SEQ AGGREGATE RESULTS")
    print("=" * 70)
    aggregate_results(all_metrics)
    logger.save()

    return all_metrics


def run_sequential_single(dataset_name: str, filepath: str):
    """
    Run PHASE-SEQ on a single recording and print metrics.

    Parameters
    ----------
    dataset_name : 'adfecgdb', 'nifecgdb', or 'cinc2013'
    filepath     : path to recording file
    """
    print("\n" + "=" * 70)
    print(f"  PHASE-SEQ Pipeline — Single Recording ({dataset_name.upper()})")
    print("=" * 70 + "\n")

    handler = get_dataset(dataset_name)
    rec     = handler.load_single_recording(filepath)
    handler.print_recording_summary(rec)

    pipe   = PHASEPipeline(verbose=True, dataset=dataset_name)
    result = pipe.run_sequential(rec, save_figures=True,
                                 figures_dir="figures_seq")

    print("\nFinal Metrics (PHASE-SEQ):")
    for k, v in result["metrics"].items():
        if isinstance(v, float):
            print(f"  {k:<20}: {v:.4f}")
    print(f"\n  Chosen path : {result['chosen_path']}")
    return result


def compare_results(seq_metrics: list, orig_results_dir: str, dataset_name: str):
    """
    Load existing PHASE results and print a side-by-side F1 comparison.

    Parameters
    ----------
    seq_metrics      : list of metric dicts from PHASE-SEQ run
    orig_results_dir : directory containing original PHASE results JSON
    dataset_name     : used to label the comparison table
    """
    import json, glob

    orig_jsons = sorted(glob.glob(f"{orig_results_dir}/results_*.json"))
    if not orig_jsons:
        print(f"[COMPARE] No original results found in {orig_results_dir}")
        return

    with open(orig_jsons[-1]) as f:
        orig_records = json.load(f)

    orig_by_rec = {r["recording"]: r for r in orig_records}

    print("\n" + "=" * 65)
    print(f"  F1 Comparison: PHASE vs PHASE-SEQ  [{dataset_name.upper()}]")
    print("=" * 65)
    print(f"  {'Recording':<14} {'PHASE F1':>10} {'SEQ F1':>10} {'Δ F1':>10}")
    print("  " + "-" * 50)

    deltas = []
    for seq_m in seq_metrics:
        rec_id = seq_m.get("recording", seq_m.get("label", "?"))
        # strip dataset prefix if present in label
        rec_id_short = rec_id.split("(")[-1].rstrip(")") if "(" in rec_id else rec_id
        orig = orig_by_rec.get(rec_id_short)
        orig_f1 = orig["F1"] if orig else float("nan")
        seq_f1  = seq_m.get("F1", float("nan"))
        delta   = seq_f1 - orig_f1
        deltas.append(delta)
        marker = "▲" if delta > 0.5 else ("▼" if delta < -0.5 else " ")
        print(f"  {rec_id_short:<14} {orig_f1:>9.2f}% {seq_f1:>9.2f}% "
              f"{delta:>+9.2f}% {marker}")

    valid = [d for d in deltas if not np.isnan(d)]
    if valid:
        print("  " + "-" * 50)
        print(f"  {'Mean delta':<14} {' ':>10} {' ':>10} {np.mean(valid):>+9.2f}%")
        improved = sum(1 for d in valid if d > 0)
        print(f"\n  Improved: {improved}/{len(valid)} recordings "
              f"({100*improved/len(valid):.0f}%)")
    print("=" * 65 + "\n")


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
        description="PHASE-SEQ Fetal ECG Pipeline — Sequential A+B Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full experiment (both datasets)
    python run_experiment_sequential.py --dataset adfecgdb --mode full
    python run_experiment_sequential.py --dataset cinc2013 --mode full

    # Quick test on first N recordings
    python run_experiment_sequential.py --dataset cinc2013 --mode full --max_recordings 10

    # Single recording
    python run_experiment_sequential.py --dataset adfecgdb --mode single --recording r01.edf

    # Full run + automatic comparison against existing PHASE results
    python run_experiment_sequential.py --dataset adfecgdb --mode full --compare
        """
    )

    parser.add_argument("--dataset", type=str, default="adfecgdb",
                        choices=["adfecgdb", "nifecgdb", "cinc2013"],
                        help="Dataset to use (default: adfecgdb)")
    parser.add_argument("--mode", type=str, default="full",
                        choices=["full", "single"],
                        help="Experiment mode (default: full)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Dataset directory (auto-selected if not specified)")
    parser.add_argument("--recording", type=str, default=None,
                        help="Recording file path (for mode=single)")
    parser.add_argument("--max_recordings", type=int, default=None,
                        help="Limit number of recordings to process")
    parser.add_argument("--no_figures", action="store_true",
                        help="Skip saving figures")
    parser.add_argument("--compare", action="store_true",
                        help="After running, compare F1 against existing PHASE results")

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

    # Run
    if args.mode == "single":
        if not args.recording:
            print("[ERROR] --recording required for mode=single")
            sys.exit(1)
        run_sequential_single(args.dataset, args.recording)

    else:  # full
        seq_metrics = run_sequential_full(
            args.dataset, data_dir,
            save_figures=not args.no_figures,
            max_recordings=args.max_recordings
        )

        if args.compare and seq_metrics:
            orig_dir = f"results_{args.dataset}"
            compare_results(seq_metrics, orig_dir, args.dataset)


if __name__ == "__main__":
    main()
