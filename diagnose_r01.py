"""
diagnose_r01.py
Deep diagnostic of each pipeline stage for r01.
Run this to understand exactly where fetal signal is lost.

Usage:
    python diagnose_r01.py
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent))

from data.loader import load_edf
from preprocessing.filters import preprocess_multichannel, preprocess_channel
from preprocessing.qrs_detector import (
    detect_maternal_qrs, detect_fetal_qrs,
    detect_reference_fetal_qrs, compute_hr_stats, pan_tompkins
)
from separation.ica import run_ica, select_maternal_ic, get_ic_as_signal

Path("figures/diagnostics").mkdir(parents=True, exist_ok=True)

DATASET = Path(__file__).parent / "Datasets" / \
          "abdominal-and-direct-fetal-ecg-database-1.0.0"

rec = load_edf(str(DATASET / "r01.edf"))
fs  = rec["fs"]
abd = rec["abdomen"]
direct = rec["direct"]

abd_proc = preprocess_multichannel(abd, fs)
dir_proc = preprocess_channel(direct, fs)

print("\n" + "="*60)
print("STAGE 0: Reference signal quality")
print("="*60)
ref_peaks = detect_reference_fetal_qrs(dir_proc, fs)
ref_stats = compute_hr_stats(ref_peaks, fs)
print(f"  Reference fetal peaks : {len(ref_peaks)}")
print(f"  Reference fetal HR    : {ref_stats['mean_hr']:.1f} BPM")
print(f"  dir_proc std          : {np.std(dir_proc):.4f}")
print(f"  dir_proc range        : [{dir_proc.min():.3f}, {dir_proc.max():.3f}]")

print("\n" + "="*60)
print("STAGE 1: Raw abdominal channels vs Direct_1")
print("="*60)
for ch in range(4):
    cc = pearsonr(abd_proc[ch], dir_proc)[0]
    print(f"  ch{ch+1} vs Direct_1: CC={cc:.4f}, std={np.std(abd_proc[ch]):.4f}")

print("\n" + "="*60)
print("STAGE 2: ICA1 components")
print("="*60)
ICs1, _ = run_ica(abd_proc, n_components=4)
mat_idx, _ = select_maternal_ic(ICs1, fs)
mat_ic = get_ic_as_signal(ICs1, mat_idx)
mat_peaks = detect_maternal_qrs(mat_ic, fs)
print(f"  Maternal IC: IC{mat_idx+1}, peaks={len(mat_peaks)}, "
      f"HR={compute_hr_stats(mat_peaks, fs)['mean_hr']:.1f} BPM")
for i, ic in enumerate(ICs1):
    cc = pearsonr(ic / (np.std(ic)+1e-10), dir_proc)[0]
    pks, _ = pan_tompkins(ic, fs, min_hr_bpm=100, max_hr_bpm=180)
    print(f"  IC{i+1}: |CC with Direct_1|={abs(cc):.4f}, "
          f"fetal-range peaks={len(pks)}")

print("\n" + "="*60)
print("STAGE 3: After maternal subtraction — residual channels")
print("="*60)
from separation.wsvd import gaussian_weight_matrix, adaptive_windowed_wsvd, subtract_maternal
weights    = gaussian_weight_matrix(abd_proc.shape[1], mat_peaks, fs)
channel_r2 = np.array([
    float(np.corrcoef(abd_proc[ch], mat_ic)[0,1]**2) for ch in range(4)
])
mat_recon = adaptive_windowed_wsvd(abd_proc, weights, fs,
                                    mat_ic=mat_ic, channel_r2=channel_r2)
residual  = subtract_maternal(abd_proc, mat_recon)

print(f"  Maternal recon std per channel: "
      f"{[f'{np.std(mat_recon[c]):.4f}' for c in range(4)]}")
print(f"  Residual std per channel      : "
      f"{[f'{np.std(residual[c]):.4f}' for c in range(4)]}")
for ch in range(4):
    cc_pos = pearsonr(residual[ch], dir_proc)[0]
    cc_neg = pearsonr(-residual[ch], dir_proc)[0]
    best_cc = max(abs(cc_pos), abs(cc_neg))
    pks_raw = detect_fetal_qrs(residual[ch], fs)
    print(f"  Residual ch{ch+1}: |CC with Direct_1|={best_cc:.4f}, "
          f"detectable peaks={len(pks_raw)}")

print("\n" + "="*60)
print("STAGE 4: ICA2 on residual")
print("="*60)
ICs2, _ = run_ica(residual, n_components=4)
for i, ic in enumerate(ICs2):
    cc = pearsonr(ic / (np.std(ic)+1e-10), dir_proc)[0]
    pks = detect_fetal_qrs(ic / (np.std(ic)+1e-10), fs)
    hr  = compute_hr_stats(pks, fs)['mean_hr']
    print(f"  IC{i+1}: |CC with Direct_1|={abs(cc):.4f}, "
          f"detected peaks={len(pks)}, mean HR={hr:.1f} BPM")

print("\n" + "="*60)
print("STAGE 5: Direct fetal signal check — can ANY approach find it?")
print("="*60)
# Check if the best residual channel directly has usable fetal peaks
best_ch_peaks = 0
best_ch = -1
for ch in range(4):
    sig_norm = residual[ch] / (np.std(residual[ch]) + 1e-10)
    pks = detect_fetal_qrs(sig_norm, fs)
    if len(pks) > best_ch_peaks:
        best_ch_peaks = len(pks)
        best_ch = ch
print(f"  Best residual channel: ch{best_ch+1} with {best_ch_peaks} fetal peaks")

# Try with lower thresholds on best channel
sig_norm = residual[best_ch] / (np.std(residual[best_ch]) + 1e-10)
nyq = 0.5 * fs
b, a = butter(1, [5/nyq, 15/nyq], btype='band')
filt = filtfilt(b, a, sig_norm)
diff = np.gradient(filt)
sq   = diff**2
win  = int(0.15 * fs)
intg = np.convolve(sq, np.ones(win)/win, mode='same')
min_dist = int(60/185*fs)
print(f"\n  Adaptive threshold scan on residual ch{best_ch+1}:")
for factor in [0.5, 0.3, 0.15, 0.08, 0.03, 0.01, 0.005, 0.001]:
    thr  = np.mean(intg) + factor * np.std(intg)
    pks, _ = find_peaks(intg, height=thr, distance=min_dist)
    hr_stats = compute_hr_stats(pks, fs)
    print(f"    factor={factor:.3f}: {len(pks):4d} peaks, "
          f"mean HR={hr_stats['mean_hr']:.1f} BPM")

# Also try on ALL abdominal channels without subtraction
print(f"\n  Scan on raw preprocessed channels (no subtraction):")
for ch in range(4):
    sig_norm = abd_proc[ch] / (np.std(abd_proc[ch]) + 1e-10)
    pks = detect_fetal_qrs(sig_norm, fs)
    cc  = abs(pearsonr(sig_norm, dir_proc)[0])
    print(f"    abd ch{ch+1}: {len(pks)} peaks, |CC|={cc:.4f}")

# ── SAVE DIAGNOSTIC FIGURE ────────────────────────────────────────────────────
print("\nSaving diagnostic figure...")
fig, axes = plt.subplots(6, 1, figsize=(16, 18), sharex=True)
t_30  = 30 * fs   # first 30 seconds
t_axis = np.arange(t_30) / fs

def norm(x):
    r = x.max() - x.min()
    return (x - x.min()) / (r + 1e-12)

axes[0].plot(t_axis, norm(dir_proc[:t_30]), 'purple', lw=0.7, label='Direct_1 (reference)')
ref_in = ref_peaks[ref_peaks < t_30]
axes[0].scatter(ref_in/fs, norm(dir_proc[ref_in]), color='red', s=15, zorder=5)
axes[0].set_title(f'Direct_1 reference — {len(ref_peaks)} fetal peaks detected')
axes[0].legend(fontsize=8)

for ch in range(4):
    axes[1].plot(t_axis, norm(abd_proc[ch, :t_30]) + ch, lw=0.5,
                 label=f'ch{ch+1}')
axes[1].set_title('Preprocessed abdominal channels (stacked)')
axes[1].legend(fontsize=8)

axes[2].plot(t_axis, norm(mat_ic[:t_30]), color='orange', lw=0.7, label=f'Maternal IC{mat_idx+1}')
mat_in = mat_peaks[mat_peaks < t_30]
axes[2].scatter(mat_in/fs, norm(mat_ic[mat_in]), color='red', s=15, zorder=5)
axes[2].set_title(f'Maternal IC — {len(mat_peaks)} peaks @ {compute_hr_stats(mat_peaks,fs)["mean_hr"]:.1f} BPM')
axes[2].legend(fontsize=8)

for ch in range(4):
    axes[3].plot(t_axis, norm(residual[ch, :t_30]) + ch, lw=0.5,
                 label=f'residual ch{ch+1}')
axes[3].set_title('Residual channels after maternal cancellation (stacked)')
axes[3].legend(fontsize=8)

best_ic2 = np.argmax([abs(pearsonr(ic/(np.std(ic)+1e-10), dir_proc)[0]) for ic in ICs2])
for i, ic in enumerate(ICs2):
    ic_n = ic / (np.std(ic)+1e-10)
    axes[4].plot(t_axis, norm(ic_n[:t_30]) + i, lw=0.5,
                 label=f'IC{i+1}{"*" if i==best_ic2 else ""}')
axes[4].set_title('ICA2 components on residual (* = highest |CC| with Direct_1)')
axes[4].legend(fontsize=8)

best_ic_sig = ICs2[best_ic2] / (np.std(ICs2[best_ic2])+1e-10)
axes[5].plot(t_axis, norm(dir_proc[:t_30]), 'purple', lw=0.8, alpha=0.7, label='Direct_1')
axes[5].plot(t_axis, norm(best_ic_sig[:t_30]), 'green', lw=0.6, alpha=0.8,
             label=f'Best ICA2 IC{best_ic2+1}')
axes[5].set_title('Direct_1 vs best ICA2 component overlay')
axes[5].legend(fontsize=8)
axes[5].set_xlabel('Time (s)')

plt.tight_layout()
out = 'figures/diagnostics/r01_stage_diagnostic.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"  Saved: {out}")
print("\nDiagnostic complete. Check figures/diagnostics/r01_stage_diagnostic.png")
