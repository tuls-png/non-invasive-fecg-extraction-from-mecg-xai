"""
batch_run_tuned.py
Batch run with EKF disabled for fast correlation analysis.

FIX: Previously used monkey-patching to replace FetalECGKalmanFilter
after import, which silently failed because pipeline.py binds the name
at import time. Now PHASEPipeline is instantiated with use_rts=False,
which skips the RTS smoother, and the pipeline's EKF forward-filter
still runs but is much faster than the full smoother.

If you want to skip EKF entirely for speed, pass ekf_bypass=True to
PHASEPipeline (see pipeline.py).
"""

import csv
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr

from data.loader import load_all_recordings
from pipeline import PHASEPipeline
from separation.ica import run_ica

# ── Configuration ─────────────────────────────────────────────────────────────
# Update DATA_DIR to point to your local ADFECGDB folder,
# or pass --data_dir on the command line via run_experiment.py.
DATA_DIR = Path(__file__).parent / "Datasets" / \
           "abdominal-and-direct-fetal-ecg-database-1.0.0"

OUT_DIR  = Path("results")
OUT_DIR.mkdir(exist_ok=True)
OUT_CSV  = OUT_DIR / "results_batch_tuned.csv"

print("Loading recordings...")
recs = load_all_recordings(str(DATA_DIR))
print(f"Found {len(recs)} recordings")

# FIX: use_rts=False skips the expensive RTS smoother but keeps the
# forward EKF pass. This is the correct way to speed up the pipeline —
# not monkey-patching the class after import.
pipeline = PHASEPipeline(use_rts=False, verbose=False)

rows = []
for rec in recs:
    rec_id = rec["recording"]
    print(f"Processing {rec_id}...")
    try:
        result = pipeline.run(rec, save_figures=False)
    except Exception as e:
        print(f"  ERROR running pipeline: {e}")
        continue

    # Compute residual vs direct correlation
    residual = result["residual"]
    dir_proc = result["dir_proc"]
    best_cc  = -1.0
    best_ch  = -1
    for ch in range(residual.shape[0]):
        cc = max(abs(pearsonr(residual[ch], dir_proc)[0]),
                 abs(pearsonr(-residual[ch], dir_proc)[0]))
        if cc > best_cc:
            best_cc = cc
            best_ch = ch + 1

    # ICA2 on residual — best IC correlation with direct
    ICs2, _      = run_ica(residual, n_components=4)
    best_ic_cc   = -1.0
    best_ic_idx  = -1
    for i, ic in enumerate(ICs2):
        cc = max(abs(pearsonr(ic, dir_proc)[0]),
                 abs(pearsonr(-ic, dir_proc)[0]))
        if cc > best_ic_cc:
            best_ic_cc  = cc
            best_ic_idx = i + 1

    metrics = result.get("metrics", {})
    rows.append({
        "recording"            : rec_id,
        "best_residual_channel": best_ch,
        "best_residual_cc"     : float(best_cc),
        "best_ica2_idx"        : best_ic_idx,
        "best_ica2_cc"         : float(best_ic_cc),
        "eval_metrics"         : str(metrics),
    })

# Write CSV
fieldnames = ["recording", "best_residual_channel", "best_residual_cc",
              "best_ica2_idx", "best_ica2_cc", "eval_metrics"]
with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Done. Results written to {OUT_CSV}")
