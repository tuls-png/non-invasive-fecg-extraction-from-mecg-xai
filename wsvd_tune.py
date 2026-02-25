"""
wsvd_tune.py
Grid search over AW-WSVD hyperparameters on a single recording.

FIX: Hardcoded Windows path replaced with relative path to bundled Datasets.
"""

import numpy as np
from pathlib import Path
from scipy.stats import pearsonr

from data.loader import load_edf
from preprocessing.filters import preprocess_multichannel, preprocess_channel
from separation.ica import run_ica, select_maternal_ic, get_ic_as_signal
from separation.wsvd import gaussian_weight_matrix, adaptive_windowed_wsvd, subtract_maternal
from preprocessing.qrs_detector import detect_maternal_qrs, detect_reference_fetal_qrs

DATASET_PATH = Path(__file__).parent / "Datasets" / \
               "abdominal-and-direct-fetal-ecg-database-1.0.0"

fp  = DATASET_PATH / "r01.edf"
rec = load_edf(str(fp))
fs  = rec["fs"]
abd = rec["abdomen"]
direct = rec["direct"]

abd_proc = preprocess_multichannel(abd, fs)
dir_proc = preprocess_channel(direct, fs)

ICs1, _ = run_ica(abd_proc, n_components=4)
mat_idx, scores = select_maternal_ic(ICs1, fs)
mat_ic   = get_ic_as_signal(ICs1, mat_idx)
mat_peaks = detect_maternal_qrs(mat_ic, fs)

print(f"Tuning AW-WSVD on {fp}")
print(f"Detected maternal peaks: {len(mat_peaks)}; maternal IC: IC{mat_idx+1}")

baselines        = [0.05, 0.01]
sigmas           = [0.04, 0.02]
n_components_list = [1, 2]
windows          = [4.0, 2.0]

results = []
count   = 0
total   = len(baselines) * len(sigmas) * len(n_components_list) * len(windows)

for baseline in baselines:
    for sigma in sigmas:
        for n_comp in n_components_list:
            for w in windows:
                count += 1
                print(f"[{count}/{total}] baseline={baseline}, sigma={sigma}, "
                      f"n_comp={n_comp}, win={w}")
                weights = gaussian_weight_matrix(abd_proc.shape[1], mat_peaks, fs,
                                                 sigma_sec=sigma, baseline=baseline)
                maternal_recon = adaptive_windowed_wsvd(abd_proc, weights, fs,
                                                        window_sec=w, overlap=0.5,
                                                        n_components=n_comp)
                residual = subtract_maternal(abd_proc, maternal_recon)

                best_cc = -1
                for ch in range(residual.shape[0]):
                    best_cc = max(best_cc,
                                  abs(pearsonr(residual[ch], dir_proc)[0]),
                                  abs(pearsonr(-residual[ch], dir_proc)[0]))

                ICs2, _ = run_ica(residual, n_components=4)
                best_ic_cc = -1
                for ic in ICs2:
                    best_ic_cc = max(best_ic_cc,
                                     abs(pearsonr(ic, dir_proc)[0]),
                                     abs(pearsonr(-ic, dir_proc)[0]))

                results.append({
                    "baseline"   : baseline,
                    "sigma"      : sigma,
                    "n_comp"     : n_comp,
                    "window"     : w,
                    "residual_cc": best_cc,
                    "ica2_cc"    : best_ic_cc,
                })
                print(f"  residual_cc={best_cc:.4f}, ica2_cc={best_ic_cc:.4f}")

best_by_res  = sorted(results, key=lambda x: x["residual_cc"], reverse=True)[:5]
best_by_ica2 = sorted(results, key=lambda x: x["ica2_cc"],    reverse=True)[:5]

print("\nTop configs by residual_cc:")
for r in best_by_res:
    print(r)

print("\nTop configs by ica2_cc:")
for r in best_by_ica2:
    print(r)
