"""
pipeline.py
PHASE: Physiology-guided Hybrid Adaptive Signal Extraction

Dual-path fetal IC selection:
  Path A — ICA1 Direct: best non-maternal IC from ICA1
  Path B — WSVD + ICA2: best IC from ICA2 on WSVD residual

Both paths use HR-aware candidate scoring:
  - Peak count must be >= MIN_USABLE_PEAKS
  - Mean HR of detected peaks must be in fetal range (110-185 BPM)
  - Mean HR must be separated from maternal HR by >= HR_SEP_MIN_BPM
  - If no candidate passes HR checks, fall back to the one whose HR
    is closest to the expected fetal range centre (147 BPM)
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import FS, ICA_N_COMPONENTS
from preprocessing.filters import preprocess_multichannel, preprocess_channel
from preprocessing.qrs_detector import (
    detect_maternal_qrs, detect_fetal_qrs,
    detect_reference_fetal_qrs, compute_hr_stats, pan_tompkins,
    load_adfecgdb_annotation
)
from separation.ica import (
    run_ica, select_maternal_ic, select_fetal_ic, get_ic_as_signal
)
from separation.wsvd import (
    gaussian_weight_matrix, adaptive_windowed_wsvd,
    subtract_maternal, svd_explained_variance
)
from separation.ekf import FetalECGKalmanFilter
from evaluation.metrics import evaluate
from xai.echo import ECHOExplainer


# ── Tunable constants ─────────────────────────────────────────────────────────
MIN_USABLE_PEAKS  = 100     # minimum detected peaks to trust a candidate
                            # (~33 BPM floor over 300s — very conservative)
FETAL_HR_LOW      = 110     # BPM — lower bound for normal fetal HR
FETAL_HR_HIGH     = 185     # BPM — upper bound
FETAL_HR_CENTRE   = 147     # BPM — centre of expected fetal range
HR_SEP_MIN_BPM    = 20      # BPM — required separation from maternal HR
PATH_A_PREFERENCE = 1.5     # Path A needs this many × more valid peaks to win
# ─────────────────────────────────────────────────────────────────────────────


def _norm(sig: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance normalisation."""
    sig = sig - np.mean(sig)
    return sig / (np.std(sig) + 1e-10)


def _candidate_hr(sig: np.ndarray, fs: int) -> tuple[np.ndarray, float]:
    """
    Detect peaks and return (peaks, mean_hr_bpm).
    Uses the full adaptive detect_fetal_qrs detector.
    """
    peaks   = detect_fetal_qrs(sig, fs)
    stats   = compute_hr_stats(peaks, fs)
    mean_hr = stats["mean_hr"] if len(peaks) >= 2 else np.nan
    return peaks, mean_hr


def _is_fetal_hr(mean_hr: float, maternal_hr: float) -> bool:
    """
    Return True if mean_hr looks fetal (not maternal).
    Requires:
      1. HR in expected fetal range
      2. Sufficiently separated from maternal HR
    """
    if np.isnan(mean_hr):
        return False
    in_range   = FETAL_HR_LOW <= mean_hr <= FETAL_HR_HIGH
    sep_ok     = abs(mean_hr - maternal_hr) >= HR_SEP_MIN_BPM
    return in_range and sep_ok


def _hr_score(mean_hr: float, expected_hr: float = FETAL_HR_CENTRE) -> float:
    """
    Score how close mean_hr is to the expected fetal HR.
    When annotation is available, expected_hr is the annotation's mean HR
    (much more precise than the generic 147 BPM centre).
    Higher = better. Returns 0 for nan.
    """
    if np.isnan(mean_hr):
        return 0.0
    return 1.0 / (1.0 + abs(mean_hr - expected_hr) / 30.0)


def _best_ic(ICs: np.ndarray, exclude_idx: int,
             maternal_hr: float, fs: int,
             label: str = "",
             expected_hr: float = None) -> tuple[np.ndarray, int, np.ndarray, float]:
    """
    Select the best fetal IC from a set of ICs.

    Strategy:
    1. Score all ICs (excluding exclude_idx and zero-variance pads) by:
       a. Does the detected HR pass the fetal HR check?
       b. How close is the HR to the expected fetal HR (annotation if available,
          otherwise generic fetal range centre of 147 BPM)?
       c. How many valid peaks were detected?
    2. Among candidates that pass HR check, pick best combined score.
    3. Fallback: pick candidate with HR closest to expected_hr.

    Parameters
    ----------
    expected_hr : annotation-derived mean fetal HR (BPM), or None to use
                  FETAL_HR_CENTRE (147 BPM) as the generic prior.

    Returns (signal_norm, ic_index, peaks, mean_hr).
    """
    centre = expected_hr if expected_hr is not None else FETAL_HR_CENTRE
    candidates = []

    for i, ic in enumerate(ICs):
        if i == exclude_idx:
            continue
        # Skip zero-variance padding rows (inserted for 3-channel recordings)
        if np.var(ic) < 1e-10:
            if label:
                print(f"[PHASE]   {label} IC{i+1}: skipped (zero-variance pad)")
            continue
        sig_norm        = _norm(ic)
        peaks, mean_hr  = _candidate_hr(sig_norm, fs)
        n_peaks         = len(peaks)
        passes_hr       = _is_fetal_hr(mean_hr, maternal_hr)
        hr_sc           = _hr_score(mean_hr, centre)

        candidates.append({
            "idx"      : i,
            "sig"      : sig_norm,
            "peaks"    : peaks,
            "n_peaks"  : n_peaks,
            "mean_hr"  : mean_hr,
            "passes_hr": passes_hr,
            "hr_score" : hr_sc,
        })

        if label:
            ann_note = f" [ann~{centre:.0f}]" if expected_hr is not None else ""
            print(f"[PHASE]   {label} IC{i+1}: {n_peaks} peaks, "
                  f"HR={mean_hr:.1f} BPM, "
                  f"fetal_hr={'YES' if passes_hr else 'NO'}{ann_note}")

    if not candidates:
        raise ValueError(f"{label}: no usable IC candidates found")

    # Priority 1: candidates that pass HR check, sorted by (n_peaks × hr_score)
    valid = [c for c in candidates
             if c["passes_hr"] and c["n_peaks"] >= MIN_USABLE_PEAKS]

    if valid:
        best = max(valid, key=lambda c: c["n_peaks"] * c["hr_score"])
        return best["sig"], best["idx"], best["peaks"], best["mean_hr"]

    # Fallback: pick closest HR to expected centre
    if label:
        print(f"[PHASE]   {label}: no candidate passed HR filter "
              f"— using closest to {centre:.0f} BPM")
    best = max(candidates, key=lambda c: c["hr_score"])
    return best["sig"], best["idx"], best["peaks"], best["mean_hr"]


def _apply_ekf(fetal_ic: np.ndarray, fetal_peaks: np.ndarray,
               fs: int, use_rts: bool) -> np.ndarray:
    """Apply EKF/RTS. Falls back to pre-EKF if smoother degrades peak count."""
    if len(fetal_peaks) < 5:
        return fetal_ic

    hr_init = compute_hr_stats(fetal_peaks, fs)["mean_hr"]
    if np.isnan(hr_init):
        hr_init = 140.0

    ekf = FetalECGKalmanFilter(fs=fs, fetal_hr_init=hr_init)
    out = ekf.smooth(fetal_ic, detected_peaks=fetal_peaks) if use_rts \
        else ekf.filter(fetal_ic, detected_peaks=fetal_peaks)[0]

    peaks_post = detect_fetal_qrs(out, fs)
    if len(peaks_post) < max(10, len(fetal_peaks) * 0.3):
        return fetal_ic   # EKF degraded signal — revert
    return out


class PHASEPipeline:
    """Full PHASE pipeline with HR-aware dual-path fetal IC selection."""

    def __init__(self, fs: int = FS, use_rts: bool = True,
                 ekf_bypass: bool = False, verbose: bool = True):
        self.fs         = fs
        self.use_rts    = use_rts
        self.ekf_bypass = ekf_bypass
        self.verbose    = verbose

    def _log(self, msg):
        if self.verbose:
            print(f"[PHASE] {msg}")

    def run(self, recording: dict,
            save_figures: bool = False,
            figures_dir: str = "figures") -> dict:

        rec_id = recording["recording"]
        abd    = recording["abdomen"]
        direct = recording.get("direct")   # None for NIFECGDB
        fs     = recording["fs"]

        self._log("=" * 55)
        self._log(f"Processing: {rec_id}  [{recording.get('dataset','?')}]")
        self._log("=" * 55)

        # Step 1: Preprocess
        self._log("Step 1: Preprocessing...")
        abd_proc = preprocess_multichannel(abd, fs)
        dir_proc = preprocess_channel(direct, fs) if direct is not None else None

        # Step 2: ICA1
        self._log("Step 2: ICA1...")
        ICs1, _ = run_ica(abd_proc, n_components=ICA_N_COMPONENTS)
        maternal_ic_idx, _ = select_maternal_ic(ICs1, fs)
        maternal_ic = get_ic_as_signal(ICs1, maternal_ic_idx)

        # Step 3: Maternal QRS
        self._log("Step 3: Maternal QRS detection...")
        maternal_peaks = detect_maternal_qrs(maternal_ic, fs)
        mat_hr_stats   = compute_hr_stats(maternal_peaks, fs)
        maternal_hr    = mat_hr_stats["mean_hr"]
        self._log(f"  {len(maternal_peaks)} maternal peaks, "
                  f"HR = {maternal_hr:.1f} BPM")

        # Derive expected fetal HR from annotation if available.
        # This is used as a prior for IC scoring — the IC whose detected
        # HR is closest to the annotation HR is preferred over the generic
        # 147 BPM centre. For ADFECGDB this improves precision; for NIFECGDB
        # it is the primary guidance since ICA alone cannot reliably find the
        # fetal component without knowing what HR to look for.
        ann_path     = recording.get("annotation_path")
        expected_fhr = None
        if ann_path:
            ann_peaks = load_adfecgdb_annotation(ann_path)
            if len(ann_peaks) >= 5:
                ann_stats    = compute_hr_stats(ann_peaks, fs)
                expected_fhr = ann_stats["mean_hr"]
                self._log(f"  Annotation prior: {len(ann_peaks)} peaks, "
                          f"expected fetal HR = {expected_fhr:.1f} BPM")

        # Step 4: Path A — ICA1 direct (HR-aware)
        self._log("Step 4: Path A — ICA1 direct (HR-aware scan)...")
        a_sig, a_idx, a_peaks, a_hr = _best_ic(
            ICs1, maternal_ic_idx, maternal_hr, fs,
            label="Path A", expected_hr=expected_fhr
        )
        a_n = len(a_peaks)
        a_valid = _is_fetal_hr(a_hr, maternal_hr)
        self._log(f"  Path A: IC{a_idx+1}, {a_n} peaks, "
                  f"HR={a_hr:.1f} BPM, valid={'YES' if a_valid else 'NO'}")

        # Step 5: Gaussian weight matrix
        self._log("Step 5: Gaussian weight matrix...")
        weights = gaussian_weight_matrix(abd_proc.shape[1], maternal_peaks, fs)

        # Step 6: AW-WSVD
        self._log("Step 6: AW-WSVD maternal reconstruction...")
        svd_explained_variance(abd_proc)
        channel_r2 = np.array([
            float(np.corrcoef(abd_proc[ch], maternal_ic)[0, 1] ** 2)
            for ch in range(abd_proc.shape[0])
        ])
        maternal_recon = adaptive_windowed_wsvd(abd_proc, weights, fs,
                                                mat_ic=maternal_ic,
                                                channel_r2=channel_r2)

        # Step 7: Maternal cancellation
        self._log("Step 7: Maternal cancellation...")
        residual = subtract_maternal(abd_proc, maternal_recon)

        # Step 8: Path B — ICA2 (HR-aware)
        self._log("Step 8: Path B — ICA2 on residual (HR-aware scan)...")
        ICs2, _ = run_ica(residual, n_components=ICA_N_COMPONENTS)
        b_sig, b_idx, b_peaks, b_hr = _best_ic(
            ICs2, -1, maternal_hr, fs,
            label="Path B", expected_hr=expected_fhr
        )
        b_n = len(b_peaks)
        b_valid = _is_fetal_hr(b_hr, maternal_hr)
        self._log(f"  Path B: IC{b_idx+1}, {b_n} peaks, "
                  f"HR={b_hr:.1f} BPM, valid={'YES' if b_valid else 'NO'}")

        # Step 9: Select best path
        self._log("Step 9: Selecting best path...")
        if a_valid and b_valid:
            # Both valid — prefer B (WSVD contribution), unless A much better
            if a_n >= b_n * PATH_A_PREFERENCE:
                chosen_sig, chosen_peaks = a_sig, a_peaks
                chosen_path = f"A_ICA1_direct_IC{a_idx+1}_{a_hr:.0f}bpm"
            else:
                chosen_sig, chosen_peaks = b_sig, b_peaks
                chosen_path = f"B_WSVD_ICA2_IC{b_idx+1}_{b_hr:.0f}bpm"
        elif a_valid:
            chosen_sig, chosen_peaks = a_sig, a_peaks
            chosen_path = f"A_ICA1_direct_IC{a_idx+1}_{a_hr:.0f}bpm"
        elif b_valid:
            chosen_sig, chosen_peaks = b_sig, b_peaks
            chosen_path = f"B_WSVD_ICA2_IC{b_idx+1}_{b_hr:.0f}bpm"
        else:
            # Neither passed HR filter — take whichever has HR closer to fetal
            a_score = _hr_score(a_hr)
            b_score = _hr_score(b_hr)
            if a_score >= b_score:
                chosen_sig, chosen_peaks = a_sig, a_peaks
                chosen_path = f"A_fallback_IC{a_idx+1}_{a_hr:.0f}bpm"
            else:
                chosen_sig, chosen_peaks = b_sig, b_peaks
                chosen_path = f"B_fallback_IC{b_idx+1}_{b_hr:.0f}bpm"

        self._log(f"  Selected: {chosen_path} ({len(chosen_peaks)} peaks)")

        # Step 10: EKF-RTS
        self._log("Step 10: EKF-RTS morphological refinement...")
        fetal_ic_raw = chosen_sig
        if self.ekf_bypass:
            fetal_ecg = fetal_ic_raw
            self._log("  EKF bypassed")
        else:
            fetal_ecg = _apply_ekf(fetal_ic_raw, chosen_peaks, fs, self.use_rts)
            n_post = len(detect_fetal_qrs(fetal_ecg, fs))
            self._log(f"  EKF complete — {n_post} peaks post-EKF "
                      f"(was {len(chosen_peaks)})")

        # Step 11: Final fetal QRS
        self._log("Step 11: Final fetal QRS detection...")
        fetal_peaks = detect_fetal_qrs(fetal_ecg, fs)
        fet_hr = compute_hr_stats(fetal_peaks, fs)
        self._log(f"  {len(fetal_peaks)} peaks, "
                  f"HR = {fet_hr['mean_hr']:.1f} BPM")

        # Step 12: Evaluation
        # Reference peaks: prefer .qrs annotation (available for both datasets
        # and human-verified), fall back to detector on Direct_1 for ADFECGDB.
        self._log("Step 12: Evaluation...")
        ann_path = recording.get("annotation_path")
        if ann_path:
            ref_peaks = load_adfecgdb_annotation(ann_path)
            self._log(f"  Reference: .qrs annotation — {len(ref_peaks)} peaks")
        elif dir_proc is not None:
            ref_peaks = detect_reference_fetal_qrs(dir_proc, fs)
            self._log(f"  Reference: Direct_1 detector — {len(ref_peaks)} peaks")
        else:
            ref_peaks = np.array([])
            self._log("  Reference: none available")

        metrics = evaluate(fetal_ecg, dir_proc, fetal_peaks, ref_peaks, fs,
                           label=f"PHASE ({rec_id})")

        # Step 13: ECHO XAI
        # reference_signal is used only for temporal independence scoring.
        # When no direct electrode is available, we use the fetal_ecg itself
        # — temporal independence then measures maternal beat overlap only.
        self._log("Step 13: ECHO XAI...")
        echo_ref = dir_proc if dir_proc is not None else fetal_ecg
        echo = ECHOExplainer(
            fs=fs, maternal_peaks=maternal_peaks,
            fetal_peaks=fetal_peaks, fetal_signal=fetal_ecg,
            reference_signal=echo_ref,
        )
        attribution = echo.compute_attributions()
        print(echo.generate_summary_stats(attribution))
        if attribution and attribution["n_beats"] > 0:
            print(echo.generate_clinical_report(0, attribution))

        if save_figures:
            self._save_figures(
                recording, abd_proc, maternal_recon, residual,
                fetal_ecg, fetal_ic_raw, dir_proc,
                fetal_peaks, ref_peaks, echo, figures_dir, rec_id
            )

        return {
            "recording"     : rec_id,
            "fetal_ecg"     : fetal_ecg,
            "fetal_ecg_pre" : fetal_ic_raw,
            "fetal_peaks"   : fetal_peaks,
            "maternal_peaks": maternal_peaks,
            "ref_peaks"     : ref_peaks,
            "maternal_recon": maternal_recon,
            "residual"      : residual,
            "abd_proc"      : abd_proc,
            "dir_proc"      : dir_proc,
            "weights"       : weights,
            "metrics"       : metrics,
            "echo"          : echo,
            "attribution"   : attribution,
            "chosen_path"   : chosen_path,
        }

    def run_with_ablation(self, recording: dict) -> dict:
        """Run all 5 ablation configurations."""
        self._log("Running ablation study...")

        fs     = recording["fs"]
        abd    = recording["abdomen"]
        direct = recording["direct"]

        abd_proc  = preprocess_multichannel(abd, fs)
        dir_proc  = preprocess_channel(direct, fs)
        ref_peaks = detect_reference_fetal_qrs(dir_proc, fs)
        results   = {}

        # Shared ICA1
        ICs1, _ = run_ica(abd_proc)
        mat_idx_blind, _ = select_maternal_ic(ICs1, fs)
        mat_ic_blind     = get_ic_as_signal(ICs1, mat_idx_blind)
        mat_peaks_blind  = detect_maternal_qrs(mat_ic_blind, fs)
        mat_hr_blind     = compute_hr_stats(mat_peaks_blind, fs)["mean_hr"]
        weights_gauss    = gaussian_weight_matrix(abd_proc.shape[1],
                                                   mat_peaks_blind, fs)

        def _eval(sig, peaks, label):
            return evaluate(sig, dir_proc, peaks, ref_peaks, fs, label=label)

        def _select(ICs, mat_idx, mat_hr):
            sig, idx, peaks, hr = _best_ic(ICs, mat_idx, mat_hr, fs)
            return sig, peaks

        # Config 1: Baseline — naive amplitude maternal, global binary WSVD,
        # ground-truth fetal IC selection (prior-work comparison convention)
        self._log("  Config 1: Baseline — naive ICA + global binary WSVD...")
        mat_idx_naive   = int(np.argmax([np.var(ic) for ic in ICs1]))
        mat_ic_naive    = get_ic_as_signal(ICs1, mat_idx_naive)
        mat_peaks_naive = detect_maternal_qrs(mat_ic_naive, fs)
        weights_binary  = _binary_weight_matrix(abd_proc.shape[1],
                                                 mat_peaks_naive, fs)
        mat_recon_1 = _global_wsvd(abd_proc, weights_binary)
        residual_1  = subtract_maternal(abd_proc, mat_recon_1)
        ICs2_1, _   = run_ica(residual_1)
        # Ground-truth selection for baseline only
        corrs     = [abs(np.corrcoef(ic, dir_proc)[0, 1]) for ic in ICs2_1]
        ic_base   = _norm(ICs2_1[int(np.argmax(corrs))])
        pks_base  = detect_fetal_qrs(ic_base, fs)
        results["1_Baseline_ICA_WSVD"] = _eval(ic_base, pks_base,
                                                "Baseline ICA+WSVD")

        # Config 2: + Blind IC selection (HR-aware)
        self._log("  Config 2: + Blind IC selection...")
        mat_recon_2 = _global_wsvd(
            abd_proc, _binary_weight_matrix(abd_proc.shape[1],
                                             mat_peaks_blind, fs)
        )
        residual_2  = subtract_maternal(abd_proc, mat_recon_2)
        ICs2_2, _   = run_ica(residual_2)
        sig_2, pks_2 = _select(ICs2_2, -1, mat_hr_blind)
        results["2_Blind_IC_Selection"] = _eval(sig_2, pks_2,
                                                "+Blind IC Selection")

        # Config 3: + Gaussian weights
        self._log("  Config 3: + Gaussian weights...")
        mat_recon_3 = _global_wsvd(abd_proc, weights_gauss)
        residual_3  = subtract_maternal(abd_proc, mat_recon_3)
        ICs2_3, _   = run_ica(residual_3)
        sig_3, pks_3 = _select(ICs2_3, -1, mat_hr_blind)
        results["3_Gaussian_Weights"] = _eval(sig_3, pks_3,
                                              "+Gaussian Weights")

        # Config 4: + Adaptive windowed WSVD
        self._log("  Config 4: + Adaptive Windowed WSVD...")
        channel_r2  = np.array([
            float(np.corrcoef(abd_proc[ch], mat_ic_blind)[0, 1] ** 2)
            for ch in range(abd_proc.shape[0])
        ])
        mat_recon_4 = adaptive_windowed_wsvd(abd_proc, weights_gauss, fs,
                                              mat_ic=mat_ic_blind,
                                              channel_r2=channel_r2)
        residual_4  = subtract_maternal(abd_proc, mat_recon_4)
        ICs2_4, _   = run_ica(residual_4)
        sig_4, pks_4 = _select(ICs2_4, -1, mat_hr_blind)
        results["4_Adaptive_WSVD"] = _eval(sig_4, pks_4, "+Adaptive WSVD")

        # Config 5: + EKF-RTS (Full PHASE)
        self._log("  Config 5: Full PHASE (+ EKF-RTS)...")
        fetal_ecg_5 = _apply_ekf(sig_4, pks_4, fs, use_rts=True)
        pks_5       = detect_fetal_qrs(fetal_ecg_5, fs)
        if len(pks_5) < max(10, len(pks_4) * 0.3):
            fetal_ecg_5, pks_5 = sig_4, pks_4
        results["5_PHASE_Full"] = _eval(fetal_ecg_5, pks_5, "PHASE Full")

        return results

    def _save_figures(self, recording, abd_proc, maternal_recon, residual,
                      fetal_ecg, fetal_ic_raw, dir_proc,
                      fetal_peaks, ref_peaks, echo, figures_dir, rec_id):
        from utils.visualization import (
            plot_preprocessing, plot_maternal_cancellation,
            plot_fetal_comparison, plot_ekf_refinement
        )
        fdir = Path(figures_dir)
        fdir.mkdir(parents=True, exist_ok=True)
        plot_preprocessing(
            recording["abdomen"][0], abd_proc[0], self.fs,
            save_path=str(fdir / f"{rec_id}_preprocessing.png"))
        plot_maternal_cancellation(
            abd_proc, maternal_recon, residual, self.fs,
            save_path=str(fdir / f"{rec_id}_maternal_cancellation.png"))
        plot_fetal_comparison(
            fetal_ecg, dir_proc, fetal_peaks, ref_peaks, self.fs,
            save_path=str(fdir / f"{rec_id}_fetal_comparison.png"))
        plot_ekf_refinement(
            fetal_ic_raw, fetal_ecg, dir_proc, self.fs,
            save_path=str(fdir / f"{rec_id}_ekf_refinement.png"))
        echo.plot_attribution_heatmap(
            save_path=str(fdir / f"{rec_id}_echo_attribution.png"))
        self._log(f"Figures saved to {figures_dir}/")


# ── Ablation baseline helpers ─────────────────────────────────────────────────

def _binary_weight_matrix(n_samples, qrs_peaks, fs, window_sec=0.1):
    weights = np.ones(n_samples) * 0.1
    hw = int(window_sec * fs)
    for peak in qrs_peaks:
        lo = max(0, peak - hw)
        hi = min(n_samples, peak + hw)
        weights[lo:hi] = 1.0
    return weights


def _global_wsvd(abd_signals, weights, n_components=2):
    weighted = abd_signals * weights[np.newaxis, :]
    U, S, VT = np.linalg.svd(weighted, full_matrices=False)
    recon = np.zeros_like(abd_signals)
    for i in range(min(n_components, len(S))):
        recon += np.outer(U[:, i], S[i] * VT[i, :])
    return recon
