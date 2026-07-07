"""
scripts/compare_template_estimators.py

Compares Path C's two beat-epoch template estimators -- "median" (default,
robust) vs "svd" (weighted SVD of the beat-epoch matrix, the epoch-domain
analogue of Path B's channel-time AW-WSVD; see
separation/template_subtraction.py module docstring) -- on a real dataset,
running the FULL parallel Path A/B/C pipeline unchanged for each estimator
and comparing final F1 / SNR / CC.

This exists so the dissertation's "Path C is literally an AW-WSVD applied
along a different axis" claim is backed by a measured result rather than
asserted. It does NOT change pipeline control flow: Path A/B/C remain
parallel and independent; only cfg.TEMPLATE_ESTIMATOR differs between runs.

Usage
-----
    python scripts/compare_template_estimators.py --dataset adfecgdb
    python scripts/compare_template_estimators.py --dataset cinc2013 --max_recordings 20

Output
------
Prints a per-recording table (F1/SNR/CC for median vs svd) plus aggregate
means/stds, and a paired Wilcoxon signed-rank test on F1 (reusing
evaluation/metrics.wilcoxon_test, already implemented in this codebase).
Writes results/template_estimator_comparison_<dataset>_<timestamp>.csv.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs import get_config
from dataset_handlers import get_dataset
from pipeline import PHASEPipeline
from evaluation.metrics import wilcoxon_test


DATASET_DIRS = {
    "adfecgdb": Path(__file__).parent.parent / "dataset_handlers" / "adfecgdb",
    "cinc2013": Path(__file__).parent.parent / "dataset_handlers" / "set-a",
    "nifecgdb": Path(__file__).parent.parent / "dataset_handlers" / "nifecgdb",
}


def _run_one(dataset_name: str, data_dir: Path, estimator: str,
             max_recordings=None, verbose=False):
    """Run the full pipeline over a dataset with a fixed TEMPLATE_ESTIMATOR."""
    handler = get_dataset(dataset_name)
    recordings = handler.load_all_recordings(data_dir, max_recordings=max_recordings)
    if not recordings:
        raise RuntimeError(f"No recordings found in {data_dir}")

    pipe = PHASEPipeline(verbose=verbose, dataset=dataset_name,
                          config_overrides={"TEMPLATE_ESTIMATOR": estimator})

    rows = []
    for rec in recordings:
        try:
            result = pipe.run(rec, save_figures=False)
            m = result["metrics"]
            rows.append({
                "recording": rec["recording"],
                "F1": m.get("F1"),
                "Se": m.get("Se"),
                "PPV": m.get("PPV"),
                "SNR_dB": m.get("SNR_dB"),
                "CC": m.get("CC"),
                "FHR_MAE_bpm": m.get("FHR_MAE_bpm"),
                "chosen_path": result["metadata"].get("chosen_path_description"),
            })
        except Exception as e:
            print(f"[WARN] {rec['recording']} failed under estimator="
                  f"{estimator}: {e}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASET_DIRS.keys()))
    ap.add_argument("--max_recordings", type=int, default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    data_dir = DATASET_DIRS[args.dataset]

    print(f"\n=== Running with TEMPLATE_ESTIMATOR='median' ===")
    rows_median = _run_one(args.dataset, data_dir, "median",
                            args.max_recordings, args.verbose)
    print(f"\n=== Running with TEMPLATE_ESTIMATOR='svd' ===")
    rows_svd = _run_one(args.dataset, data_dir, "svd",
                         args.max_recordings, args.verbose)

    by_rec_median = {r["recording"]: r for r in rows_median}
    by_rec_svd = {r["recording"]: r for r in rows_svd}
    common = sorted(set(by_rec_median) & set(by_rec_svd))

    print(f"\n{'Recording':<12}{'F1 (median)':<14}{'F1 (svd)':<12}"
          f"{'CC (median)':<14}{'CC (svd)':<10}")
    f1_med, f1_svd = [], []
    for rec in common:
        rm, rs = by_rec_median[rec], by_rec_svd[rec]
        print(f"{rec:<12}{rm['F1']:<14.2f}{rs['F1']:<12.2f}"
              f"{(rm['CC'] if rm['CC'] is not None else float('nan')):<14.3f}"
              f"{(rs['CC'] if rs['CC'] is not None else float('nan')):<10.3f}")
        f1_med.append(rm["F1"])
        f1_svd.append(rs["F1"])

    f1_med = np.array(f1_med, dtype=float)
    f1_svd = np.array(f1_svd, dtype=float)
    print(f"\n--- Aggregate (n={len(common)}) ---")
    print(f"F1 median-estimator : {f1_med.mean():.2f} +/- {f1_med.std():.2f}")
    print(f"F1 svd-estimator    : {f1_svd.mean():.2f} +/- {f1_svd.std():.2f}")

    if len(common) >= 2:
        try:
            # evaluation.metrics.wilcoxon_test is one-sided (alternative=
            # 'greater', testing scores_a > scores_b) and returns a dict,
            # not a tuple. A single one-sided call in an arbitrary
            # direction can't establish "these are equivalent" -- a
            # non-significant result there only fails to show median >
            # svd, it says nothing about svd > median. Run both
            # directions so the comparison is symmetric and interpretable
            # either way.
            res_med_gt_svd = wilcoxon_test(f1_med.tolist(), f1_svd.tolist(),
                                            metric_name="F1 (median > svd)")
            res_svd_gt_med = wilcoxon_test(f1_svd.tolist(), f1_med.tolist(),
                                            metric_name="F1 (svd > median)")
            print(f"Wilcoxon median>svd : statistic={res_med_gt_svd['statistic']:.4f}, "
                  f"p={res_med_gt_svd['p_value']:.4f}, "
                  f"significant={res_med_gt_svd['significant']}")
            print(f"Wilcoxon svd>median : statistic={res_svd_gt_med['statistic']:.4f}, "
                  f"p={res_svd_gt_med['p_value']:.4f}, "
                  f"significant={res_svd_gt_med['significant']}")
            if not res_med_gt_svd["significant"] and not res_svd_gt_med["significant"]:
                print("Neither direction significant at alpha=0.05 -- "
                      "consistent with the two estimators being "
                      "statistically indistinguishable on this cohort.")
        except Exception as e:
            print(f"[WARN] Wilcoxon test failed: {e}")

    out_dir = Path(__file__).parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"template_estimator_comparison_{args.dataset}_{ts}.csv"
    with open(out_path, "w") as f:
        f.write("recording,F1_median,F1_svd,CC_median,CC_svd,"
                "chosen_path_median,chosen_path_svd\n")
        for rec in common:
            rm, rs = by_rec_median[rec], by_rec_svd[rec]
            f.write(f"{rec},{rm['F1']},{rs['F1']},{rm['CC']},{rs['CC']},"
                    f"{rm['chosen_path']},{rs['chosen_path']}\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()