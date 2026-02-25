"""
diagnostic_verify.py
Verify what is happening in each stage for all recordings.
Useful for debugging separation quality per recording.

FIX: Hardcoded Windows path replaced with a path relative to this script,
pointing to the bundled Datasets folder.
"""

import numpy as np
from pathlib import Path

from data.loader import load_edf
from preprocessing.filters import preprocess_channel, preprocess_multichannel
from separation.ica import run_ica, select_maternal_ic, get_ic_as_signal
from separation.wsvd import gaussian_weight_matrix, adaptive_windowed_wsvd, subtract_maternal
from preprocessing.qrs_detector import detect_reference_fetal_qrs, detect_maternal_qrs, compute_hr_stats

DATASET_PATH = Path(__file__).parent / "Datasets" / \
               "abdominal-and-direct-fetal-ecg-database-1.0.0"

recordings = ["r01.edf", "r04.edf", "r07.edf", "r08.edf", "r10.edf"]

for rec_file in recordings:
    fp  = DATASET_PATH / rec_file
    if not fp.exists():
        print(f"[SKIP] {rec_file} not found at {fp}")
        continue

    rec    = load_edf(str(fp))
    fs     = rec["fs"]
    abd    = rec["abdomen"]
    direct = rec["direct"]

    print(f"\n{'='*70}")
    print(f"  {rec_file}")
    print(f"{'='*70}")
    print(f"  Direct_1 raw: min={direct.min():.1f}, max={direct.max():.1f}, "
          f"std={direct.std():.1f}, mean={direct.mean():.1f}")

    # Preprocess
    abd_proc = preprocess_multichannel(abd, fs)
    dir_proc = preprocess_channel(direct, fs)

    # Stage 1: ICA1 (maternal)
    ICs1, _ = run_ica(abd_proc, n_components=4)
    mat_idx, scores = select_maternal_ic(ICs1, fs)
    mat_ic   = get_ic_as_signal(ICs1, mat_idx)
    mat_peaks = detect_maternal_qrs(mat_ic, fs)

    abd_std = np.std(abd_proc, axis=1)
    dir_std = np.std(dir_proc)
    print(f"  Abdominal channel stds: {abd_std}")
    print(f"  Direct_1 std (preprocessed): {dir_std:.4f}")

    mat_power_fraction = [
        np.corrcoef(abd_proc[ch], mat_ic)[0, 1]**2
        for ch in range(abd_proc.shape[0])
    ]
    print(f"  Maternal IC R^2 per channel: {[f'{x:.3f}' for x in mat_power_fraction]}")

    # Stage 2: QRS detection
    ref_peaks = detect_reference_fetal_qrs(dir_proc, fs)
    if len(ref_peaks) > 2:
        rr = np.diff(ref_peaks) / fs
        print(f"  RR intervals: min={rr.min()*1000:.0f}ms, "
              f"max={rr.max()*1000:.0f}ms, mean={rr.mean()*1000:.0f}ms")

    mat_stats = compute_hr_stats(mat_peaks, fs)
    ref_stats = compute_hr_stats(ref_peaks, fs)

    # Stage 3: WSVD
    weights = gaussian_weight_matrix(abd.shape[1], mat_peaks, fs)
    channel_r2 = np.array([
        float(np.corrcoef(abd_proc[ch], mat_ic)[0, 1] ** 2)
        for ch in range(abd_proc.shape[0])
    ])
    maternal_recon = adaptive_windowed_wsvd(abd_proc, weights, fs,
                                             mat_ic=mat_ic,
                                             channel_r2=channel_r2)
    residual = subtract_maternal(abd_proc, maternal_recon)

    # Stage 4: Check residual quality
    best_cc = 0.0
    for ch in range(residual.shape[0]):
        cc = np.corrcoef(
            residual[ch, :min(30*fs, residual.shape[1])],
            dir_proc[:min(30*fs, dir_proc.shape[0])]
        )[0, 1]
        best_cc = max(best_cc, cc)

    # Stage 5: ICA2
    ICs2, _ = run_ica(residual, n_components=4)
    best_ica2_cc = 0.0
    for ic_idx in range(4):
        ic = get_ic_as_signal(ICs2, ic_idx)
        cc = np.corrcoef(
            ic[:min(30*fs, len(ic))],
            dir_proc[:min(30*fs, dir_proc.shape[0])]
        )[0, 1]
        best_ica2_cc = max(best_ica2_cc, cc)

    # Summary
    print(f"  Maternal IC selection score  : {scores[mat_idx]:.4f}")
    print(f"  Maternal HR                  : {mat_stats['mean_hr']:.1f} BPM "
          f"({mat_stats['n_peaks']} peaks)")
    print(f"  Fetal HR (reference)         : {ref_stats['mean_hr']:.1f} BPM "
          f"({ref_stats['n_peaks']} peaks)")
    print(f"  Best residual-fetal CC       : {best_cc:.4f}")
    print(f"  Best ICA2-fetal CC           : {best_ica2_cc:.4f}")

    if best_cc < 0.15:
        print(f"  WARNING: Residual has almost no fetal content!")
        print(f"     Maternal IC score = {scores[mat_idx]:.4f}")
        if scores[mat_idx] < 0.5:
            print(f"     ACTION: Maternal IC score is LOW; ICA1 may have picked wrong IC")
    elif best_cc < 0.25:
        print(f"  WARNING: Residual has weak fetal signal")
        print(f"     WSVD parameters may need tuning (window size, n_components)")
    else:
        print(f"  GOOD: Residual has strong fetal signal")
