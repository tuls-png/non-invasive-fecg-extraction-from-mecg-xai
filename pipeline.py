"""
pipeline.py
PHASE: Physiology-guided Hybrid Adaptive Signal Extraction

Dual-path fetal IC selection:
  Path A -- ICA1 Direct: best non-maternal IC from ICA1
  Path B -- WSVD + ICA2: best IC from ICA2 on WSVD residual

CHANGES FROM ORIGINAL:
  [FIX-1] Path B ICA2: exclude ICA2 components correlated with maternal IC
          (|corr| > MATERNAL_ICA2_CORR_THRESH). Previously exclude_idx=-1
          meant no exclusion, allowing residual-maternal ICs to be selected.
  [FIX-2] min_usable_peaks is now recording-length-adaptive (was hardcoded 100).
  [FIX-3] ECHO has_reference passed explicitly; None passed for NIFECGDB so
          morphology score is disabled rather than self-referential.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config_loader import get_config
from preprocessing.filters import preprocess_multichannel, preprocess_channel
from preprocessing.qrs_detector import (
    detect_maternal_qrs, detect_fetal_qrs,
    detect_reference_fetal_qrs, compute_hr_stats, pan_tompkins,
    load_adfecgdb_annotation
)
from separation.ica import (
    run_ica, run_ica_best_contrast, select_maternal_ic, select_fetal_ic, get_ic_as_signal
)
from separation.wsvd import (
    gaussian_weight_matrix, adaptive_windowed_wsvd,
    subtract_maternal, svd_explained_variance
)
from separation.ekf import FetalECGKalmanFilter
from evaluation.metrics import evaluate
from xai.echo import ECHOExplainer
from preprocessing.qrs_detector import load_wfdb_annotation


def _min_usable_peaks(duration_sec: float, cfg, dataset: str = "ADFECGDB") -> int:
    if dataset == "NIFECGDB":
        expected = duration_sec * cfg.FETAL_HR_MIN / 60.0
        return max(20, int(expected * 0.15))
    else:
        expected = duration_sec * cfg.FETAL_HR_LOW / 60.0
        return max(30, int(expected * 0.5))


def _norm(sig):
    sig = sig - np.mean(sig)
    return sig / (np.std(sig) + 1e-10)


def _candidate_hr(sig, fs, cfg):
    peaks   = detect_fetal_qrs(sig, fs, cfg=cfg)
    stats   = compute_hr_stats(peaks, fs)
    mean_hr = stats["mean_hr"] if len(peaks) >= 2 else np.nan
    return peaks, mean_hr


def _is_fetal_hr(mean_hr: float, maternal_hr: float, cfg) -> bool:
    """
    Check if a candidate HR is in the fetal range and sufficiently
    separated from maternal HR.

    FIX: Separation threshold is now adaptive — when maternal HR is high
    (>85 BPM, common in active labor), the minimum separation is relaxed
    from HR_SEP_MIN_BPM to HR_SEP_MIN_BPM * 0.7 to avoid rejecting valid
    fetal ICs that happen to be in the lower fetal range.
    """
    print("HR_SEP_MIN_BPM", cfg.HR_SEP_MIN_BPM )
    if np.isnan(mean_hr):
        return False
    in_range = cfg.FETAL_HR_LOW <= mean_hr <= cfg.FETAL_HR_HIGH
    if not np.isnan(maternal_hr) and maternal_hr > 85:
        sep_threshold = cfg.HR_SEP_MIN_BPM * 0.7
    else:
        sep_threshold = cfg.HR_SEP_MIN_BPM
    sep_ok = abs(mean_hr - maternal_hr) >= sep_threshold
    return in_range and sep_ok


def _hr_score(mean_hr, cfg, expected_hr=None):
    centre = expected_hr if expected_hr is not None else cfg.FETAL_HR_CENTRE
    if np.isnan(mean_hr):
        return 0.0
    return 1.0 / (1.0 + abs(mean_hr - centre) / 30.0)


def _find_maternal_residual_idx(ICs, maternal_ic, cfg):
    """
    [FIX-1] Find ICA2 component most correlated with maternal IC.
    Returns the index to exclude, or -1 if none exceed threshold.
    """
    best_idx  = -1
    best_corr = cfg.MATERNAL_ICA2_CORR_THRESH
    for i, ic in enumerate(ICs):
        if np.var(ic) < 1e-10:
            continue
        try:
            corr = abs(float(np.corrcoef(ic, maternal_ic)[0, 1]))
        except Exception:
            continue
        if corr > best_corr:
            best_corr = corr
            best_idx  = i
    if best_idx >= 0:
        print(f"[PHASE] Path B: excluding IC{best_idx+1} "
              f"(|corr| with maternal IC = {best_corr:.3f} "
              f"> threshold {cfg.MATERNAL_ICA2_CORR_THRESH})")
    return best_idx


def _best_ic(ICs, exclude_idx, maternal_hr, fs, cfg,
             label="", expected_hr=None, min_peaks=100):
    centre     = expected_hr if expected_hr is not None else cfg.FETAL_HR_CENTRE
    candidates = []
    for i, ic in enumerate(ICs):
        if i == exclude_idx:
            continue
        if np.var(ic) < 1e-10:
            if label:
                print(f"[PHASE]   {label} IC{i+1}: skipped (zero-variance pad)")
            continue
        sig_norm       = _norm(ic)
        peaks, mean_hr = _candidate_hr(sig_norm, fs, cfg)
        n_peaks        = len(peaks)
        passes_hr      = _is_fetal_hr(mean_hr, maternal_hr, cfg)
        hr_sc          = _hr_score(mean_hr, cfg, expected_hr)
        candidates.append({
            "idx": i, "sig": sig_norm, "peaks": peaks,
            "n_peaks": n_peaks, "mean_hr": mean_hr,
            "passes_hr": passes_hr, "hr_score": hr_sc,
        })
        if label:
            ann_note = f" [ann~{centre:.0f}]" if expected_hr is not None else ""
            print(f"[PHASE]   {label} IC{i+1}: {n_peaks} peaks, "
                  f"HR={mean_hr:.1f} BPM, "
                  f"fetal_hr={'YES' if passes_hr else 'NO'}{ann_note}")

    if not candidates:
        raise ValueError(f"{label}: no usable IC candidates found")

    valid = [c for c in candidates
             if c["passes_hr"] and c["n_peaks"] >= min_peaks]
    if valid:
        best = max(valid, key=lambda c: c["n_peaks"] * c["hr_score"])
        return best["sig"], best["idx"], best["peaks"], best["mean_hr"]

    if label:
        print(f"[PHASE]   {label}: no candidate passed HR filter "
              f"-- using closest to {centre:.0f} BPM")
    best = max(candidates, key=lambda c: c["hr_score"])
    return best["sig"], best["idx"], best["peaks"], best["mean_hr"]

def _refine_peaks_on_smoothed(smoothed, rough_peaks, fs, search_radius_ms=40.0):
    radius  = int(search_radius_ms * fs / 1000)
    refined = []
    for p in rough_peaks:
        lo  = max(0, p - radius)
        hi  = min(len(smoothed), p + radius)
        window = smoothed[lo:hi]
        # Use sign of the original peak location to avoid polarity confusion
        if smoothed[p] >= 0:
            local_max = lo + int(np.argmax(window))
        else:
            local_max = lo + int(np.argmin(window))
        refined.append(local_max)
    return np.array(refined, dtype=int)

def _apply_ekf(fetal_ic, fetal_peaks, fs, use_rts, cfg=None):
    if len(fetal_peaks) < 5:
        return fetal_ic
    hr_init = compute_hr_stats(fetal_peaks, fs)["mean_hr"]
    if np.isnan(hr_init):
        hr_init = 140.0
    ekf = FetalECGKalmanFilter(fs=fs, fetal_hr_init=hr_init)
    out = (ekf.smooth(fetal_ic, detected_peaks=fetal_peaks) if use_rts
           else ekf.filter(fetal_ic, detected_peaks=fetal_peaks)[0])
    peaks_post = detect_fetal_qrs(out, fs, cfg=cfg)
    if len(peaks_post) < max(10, len(fetal_peaks) * 0.3):
        return fetal_ic
    return out

class PHASEPipeline:
    def __init__(self, fs=None, use_rts=True, ekf_bypass=False, verbose=True,
                 dataset=None):
        self.cfg        = get_config(dataset)
        self.fs         = fs if fs is not None else self.cfg.FS
        self.use_rts    = use_rts
        self.ekf_bypass = ekf_bypass
        self.verbose    = verbose

    def _log(self, msg):
        if self.verbose:
            print(f"[PHASE] {msg}")

    def run(self, recording, save_figures=False, figures_dir="figures"):
        cfg      = self.cfg 
        dataset  = recording.get("dataset", "ADFECGDB")
        rec_id   = recording["recording"]
        abd      = recording["abdomen"]
        direct   = recording.get("direct")
        fs       = recording["fs"]
        duration = recording.get("duration_sec", abd.shape[1] / fs)
        min_peaks = _min_usable_peaks(duration, cfg, dataset)

        self._log("=" * 55)
        self._log(f"Processing: {rec_id}  [{recording.get('dataset','?')}]")
        self._log(f"Duration: {duration:.1f}s  |  min_usable_peaks: {min_peaks}")
        self._log("=" * 55)

        # Step 1: Preprocess
        self._log("Step 1: Preprocessing...")
        abd_proc = preprocess_multichannel(abd, fs)
        dir_proc = preprocess_channel(direct, fs) if direct is not None else None

        # Step 2: ICA1
        self._log("Step 2: ICA1...")
        use_dual = getattr(cfg, 'ICA_DUAL_CONTRAST', False)
        if use_dual:
            ICs1, _, _ica1_winner = run_ica_best_contrast(
                abd_proc, n_components=cfg.ICA_N_COMPONENTS, fs=fs)
        else:
            ICs1, _ = run_ica(abd_proc, n_components=cfg.ICA_N_COMPONENTS)
        maternal_ic_idx, _ = select_maternal_ic(ICs1, fs)
        maternal_ic        = get_ic_as_signal(ICs1, maternal_ic_idx)

        # Step 3: Maternal QRS
        self._log("Step 3: Maternal QRS detection...")
        maternal_peaks = detect_maternal_qrs(maternal_ic, fs)
        mat_hr_stats   = compute_hr_stats(maternal_peaks, fs)
        maternal_hr    = mat_hr_stats["mean_hr"]
        self._log(f"  {len(maternal_peaks)} maternal peaks, HR = {maternal_hr:.1f} BPM")

        ann_path      = recording.get("annotation_path")
        ann_ext       = recording.get("annotation_ext", "qrs")
        ann_is_fetal  = recording.get("annotation_is_fetal", False)
        expected_fhr  = None

        if ann_path and ann_is_fetal:
            from preprocessing.qrs_detector import load_wfdb_annotation
            ann_peaks = load_wfdb_annotation(ann_path, ann_ext)
            if len(ann_peaks) >= 5:
                ann_stats    = compute_hr_stats(ann_peaks, fs)
                expected_fhr = ann_stats["mean_hr"]
                self._log(f"  Annotation prior: {len(ann_peaks)} peaks, "
                        f"expected fetal HR = {expected_fhr:.1f} BPM")
        elif ann_path and not ann_is_fetal:
            self._log("  Annotation skipped (not fetal ground truth)")

        # Step 4: Path A
        self._log("Step 4: Path A -- ICA1 direct (HR-aware scan)...")
        a_sig, a_idx, a_peaks, a_hr = _best_ic(
            ICs1, maternal_ic_idx, maternal_hr, fs, cfg,
            label="Path A", expected_hr=expected_fhr, min_peaks=min_peaks)
        a_n     = len(a_peaks)
        a_valid = _is_fetal_hr(a_hr, maternal_hr, cfg)
        self._log(f"  Path A: IC{a_idx+1}, {a_n} peaks, "
                  f"HR={a_hr:.1f} BPM, valid={'YES' if a_valid else 'NO'}")

        # Step 5: Gaussian weights
        self._log("Step 5: Gaussian weight matrix...")
        weights = gaussian_weight_matrix(abd_proc.shape[1], maternal_peaks, fs)

        # Step 6: AW-WSVD
        self._log("Step 6: AW-WSVD maternal reconstruction...")
        svd_explained_variance(abd_proc)
        channel_r2 = np.array([
            float(np.corrcoef(abd_proc[ch], maternal_ic)[0, 1] ** 2)
            for ch in range(abd_proc.shape[0])
        ])
        maternal_recon = adaptive_windowed_wsvd(
            abd_proc, weights, fs, mat_ic=maternal_ic, channel_r2=channel_r2)

        # Step 7: Maternal cancellation
        self._log("Step 7: Maternal cancellation...")
        residual = subtract_maternal(abd_proc, maternal_recon)

        # Step 8: Path B -- ICA2 with [FIX-1] maternal residual exclusion
        self._log("Step 8: Path B -- ICA2 on residual (HR-aware scan)...")
        ICs2, _          = run_ica(residual, n_components=cfg.ICA_N_COMPONENTS)
        mat_residual_idx = _find_maternal_residual_idx(ICs2, maternal_ic, cfg)
        b_sig, b_idx, b_peaks, b_hr = _best_ic(
            ICs2, mat_residual_idx, maternal_hr, fs, cfg,
            label="Path B", expected_hr=expected_fhr, min_peaks=min_peaks)
        b_n     = len(b_peaks)
        b_valid = _is_fetal_hr(b_hr, maternal_hr, cfg)
        self._log(f"  Path B: IC{b_idx+1}, {b_n} peaks, "
                  f"HR={b_hr:.1f} BPM, valid={'YES' if b_valid else 'NO'}")

        # Step 9: Select best path
        self._log("Step 9: Selecting best path...")
        if a_valid and b_valid:
            if a_n >= b_n * cfg.PATH_A_PREFERENCE:
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
            fetal_ecg = _apply_ekf(fetal_ic_raw, chosen_peaks, fs, self.use_rts, cfg=cfg)
            n_post = len(detect_fetal_qrs(fetal_ecg, fs, cfg=cfg))
            self._log(f"  EKF complete -- {n_post} peaks post-EKF (was {len(chosen_peaks)})")

        # Step 11: Final QRS
        self._log("Step 11: Final fetal QRS detection...")
        fetal_peaks = detect_fetal_qrs(fetal_ecg, fs, cfg=cfg)
        fet_hr = compute_hr_stats(fetal_peaks, fs)
        self._log(f"  {len(fetal_peaks)} peaks, HR = {fet_hr['mean_hr']:.1f} BPM")

        # Step 12: Evaluation
        self._log("Step 12: Evaluation...")
        if ann_path and ann_is_fetal:
            ref_peaks = load_wfdb_annotation(ann_path, ann_ext)
            self._log(f"  Reference: .{ann_ext} annotation — {len(ref_peaks)} peaks")
        elif dir_proc is not None:
            ref_peaks = detect_reference_fetal_qrs(dir_proc, fs)
            self._log(f"  Reference: Direct_1 detector — {len(ref_peaks)} peaks")
        else:
            ref_peaks = np.array([])
            self._log("  Reference: none available")
        metrics = evaluate(
                fetal_ecg, dir_proc, fetal_peaks, ref_peaks, fs,
                label=f"PHASE ({rec_id})",
                tolerance_ms=cfg.EVAL_TOLERANCE_MS   # pass from config
            )

        # Step 13: ECHO XAI -- [FIX-3] explicit has_reference flag
        self._log("Step 13: ECHO XAI...")
        has_ref  = dir_proc is not None
        echo_ref = dir_proc if has_ref else None
        echo = ECHOExplainer(
            fs=fs, maternal_peaks=maternal_peaks,
            fetal_peaks=fetal_peaks, fetal_signal=fetal_ecg,
            reference_signal=echo_ref, has_reference=has_ref)
        attribution = echo.compute_attributions()
        print(echo.generate_summary_stats(attribution))
        if attribution and attribution["n_beats"] > 0:
            print(echo.generate_clinical_report(0, attribution))

        if save_figures:
            self._save_figures(
                recording, abd_proc, maternal_recon, residual,
                fetal_ecg, fetal_ic_raw, dir_proc,
                fetal_peaks, ref_peaks, echo, figures_dir, rec_id)

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

    def run_sequential(self, recording, save_figures=False, figures_dir="figures"):
        """
        Sequential Path A→B pipeline (PHASE-SEQ).

        Instead of choosing between Path A and Path B, this method uses Path A's
        maternal IC estimate to produce a *cleaner* multichannel residual before
        running Path B's WSVD+ICA2 stage.

        Pipeline:
          Step 1-3  : Same as PHASE (preprocess, ICA1, maternal QRS)
          Step 4    : Path A — select best fetal IC from ICA1 (same as before)
          Step 5-6  : Reproject Path A's maternal IC onto each channel and
                      subtract it, producing an improved multichannel residual.
                      This is *additional* maternal cancellation on top of the
                      raw signal, before WSVD is even applied.
          Step 7    : Gaussian weights (same)
          Step 8    : AW-WSVD on the Path-A-cleaned residual (instead of raw)
          Step 9    : Subtract WSVD maternal reconstruction from Path-A residual
          Step 10   : ICA2 on the doubly-cleaned residual → Path B (sequential)
          Step 11   : EKF-RTS on the sequential Path B output
          Step 12-13: Evaluation + ECHO XAI
        """
        cfg      = self.cfg
        dataset  = recording.get("dataset", "ADFECGDB")
        rec_id   = recording["recording"]
        abd      = recording["abdomen"]
        direct   = recording.get("direct")
        fs       = recording["fs"]
        duration = recording.get("duration_sec", abd.shape[1] / fs)
        min_peaks = _min_usable_peaks(duration, cfg, dataset)

        self._log("=" * 55)
        self._log(f"[SEQ] Processing: {rec_id}  [{recording.get('dataset','?')}]")
        self._log(f"Duration: {duration:.1f}s  |  min_usable_peaks: {min_peaks}")
        self._log("=" * 55)

        # Step 1: Preprocess
        self._log("Step 1: Preprocessing...")
        abd_proc = preprocess_multichannel(abd, fs)
        dir_proc = preprocess_channel(direct, fs) if direct is not None else None

        # Step 2: ICA1
        self._log("Step 2: ICA1...")
        use_dual = getattr(cfg, 'ICA_DUAL_CONTRAST', False)
        if use_dual:
            ICs1, _, _ica1_winner = run_ica_best_contrast(
                abd_proc, n_components=cfg.ICA_N_COMPONENTS, fs=fs)
        else:
            ICs1, _ = run_ica(abd_proc, n_components=cfg.ICA_N_COMPONENTS)
        maternal_ic_idx, _ = select_maternal_ic(ICs1, fs)
        maternal_ic        = get_ic_as_signal(ICs1, maternal_ic_idx)

        # Step 3: Maternal QRS
        self._log("Step 3: Maternal QRS detection...")
        maternal_peaks = detect_maternal_qrs(maternal_ic, fs)
        mat_hr_stats   = compute_hr_stats(maternal_peaks, fs)
        maternal_hr    = mat_hr_stats["mean_hr"]
        self._log(f"  {len(maternal_peaks)} maternal peaks, HR = {maternal_hr:.1f} BPM")

        ann_path     = recording.get("annotation_path")
        ann_ext      = recording.get("annotation_ext", "qrs")
        ann_is_fetal = recording.get("annotation_is_fetal", False)
        expected_fhr = None

        if ann_path and ann_is_fetal:
            from preprocessing.qrs_detector import load_wfdb_annotation
            ann_peaks = load_wfdb_annotation(ann_path, ann_ext)
            if len(ann_peaks) >= 5:
                ann_stats    = compute_hr_stats(ann_peaks, fs)
                expected_fhr = ann_stats["mean_hr"]
                self._log(f"  Annotation prior: expected fetal HR = {expected_fhr:.1f} BPM")

        # Step 4: Path A — ICA1 direct (same as original)
        self._log("Step 4: Path A -- ICA1 direct (HR-aware scan)...")
        a_sig, a_idx, a_peaks, a_hr = _best_ic(
            ICs1, maternal_ic_idx, maternal_hr, fs, cfg,
            label="Path A", expected_hr=expected_fhr, min_peaks=min_peaks)
        self._log(f"  Path A: IC{a_idx+1}, {len(a_peaks)} peaks, HR={a_hr:.1f} BPM")

        # Step 5: Build Path-A-cleaned multichannel residual.
        #
        # Project the maternal IC back onto each channel using least-squares,
        # then subtract. This removes what ICA1 identified as the maternal
        # component from all channels simultaneously — better than subtracting
        # a single template because it respects each channel's mixing coefficient.
        self._log("Step 5: Path A maternal IC subtraction from all channels...")
        mat_ic_norm = maternal_ic - np.mean(maternal_ic)
        denom = np.dot(mat_ic_norm, mat_ic_norm) + 1e-10
        path_a_residual = np.zeros_like(abd_proc)
        for ch in range(abd_proc.shape[0]):
            alpha = np.dot(abd_proc[ch], mat_ic_norm) / denom
            path_a_residual[ch] = abd_proc[ch] - alpha * mat_ic_norm
        self._log(f"  Projected and subtracted maternal IC from {abd_proc.shape[0]} channels")

        # Step 6: Gaussian weights (based on maternal peaks, same as original)
        self._log("Step 6: Gaussian weight matrix...")
        weights = gaussian_weight_matrix(abd_proc.shape[1], maternal_peaks, fs)

        # Step 7: AW-WSVD on Path-A-cleaned residual (key difference from original)
        self._log("Step 7: AW-WSVD on Path-A residual (sequential stage)...")
        svd_explained_variance(path_a_residual)
        channel_r2 = np.array([
            float(np.corrcoef(path_a_residual[ch], maternal_ic)[0, 1] ** 2)
            for ch in range(path_a_residual.shape[0])
        ])
        maternal_recon_seq = adaptive_windowed_wsvd(
            path_a_residual, weights, fs,
            mat_ic=maternal_ic, channel_r2=channel_r2)

        # Step 8: Subtract WSVD reconstruction from Path-A residual
        self._log("Step 8: Second maternal cancellation (WSVD subtraction)...")
        residual_seq = subtract_maternal(path_a_residual, maternal_recon_seq)

        # Step 9: ICA2 on doubly-cleaned residual → sequential Path B
        self._log("Step 9: ICA2 on doubly-cleaned residual (Path B sequential)...")
        ICs2_seq, _      = run_ica(residual_seq, n_components=cfg.ICA_N_COMPONENTS)
        mat_residual_idx = _find_maternal_residual_idx(ICs2_seq, maternal_ic, cfg)
        seq_sig, seq_idx, seq_peaks, seq_hr = _best_ic(
            ICs2_seq, mat_residual_idx, maternal_hr, fs, cfg,
            label="Path B (sequential)", expected_hr=expected_fhr, min_peaks=min_peaks)
        self._log(f"  Sequential output: IC{seq_idx+1}, {len(seq_peaks)} peaks, "
                  f"HR={seq_hr:.1f} BPM")

        chosen_sig    = seq_sig
        chosen_peaks  = seq_peaks
        chosen_path   = f"SEQ_A_then_B_IC{seq_idx+1}_{seq_hr:.0f}bpm"

        # Step 10: EKF-RTS morphological refinement
        self._log("Step 10: EKF-RTS morphological refinement...")
        fetal_ic_raw = chosen_sig
        if self.ekf_bypass:
            fetal_ecg = fetal_ic_raw
            self._log("  EKF bypassed")
        else:
            fetal_ecg = _apply_ekf(fetal_ic_raw, chosen_peaks, fs, self.use_rts, cfg=cfg)
            n_post = len(detect_fetal_qrs(fetal_ecg, fs, cfg=cfg))
            self._log(f"  EKF complete -- {n_post} peaks post-EKF (was {len(chosen_peaks)})")

        # Step 11: Final QRS detection
        self._log("Step 11: Final fetal QRS detection...")
        fetal_peaks = detect_fetal_qrs(fetal_ecg, fs, cfg=cfg)
        fet_hr = compute_hr_stats(fetal_peaks, fs)
        self._log(f"  {len(fetal_peaks)} peaks, HR = {fet_hr['mean_hr']:.1f} BPM")

        # Step 12: Evaluation
        self._log("Step 12: Evaluation...")
        if ann_path and ann_is_fetal:
            ref_peaks = load_wfdb_annotation(ann_path, ann_ext)
            self._log(f"  Reference: .{ann_ext} annotation — {len(ref_peaks)} peaks")
        elif dir_proc is not None:
            ref_peaks = detect_reference_fetal_qrs(dir_proc, fs)
            self._log(f"  Reference: Direct_1 detector — {len(ref_peaks)} peaks")
        else:
            ref_peaks = np.array([])
            self._log("  Reference: none available")
        metrics = evaluate(
            fetal_ecg, dir_proc, fetal_peaks, ref_peaks, fs,
            label=f"PHASE-SEQ ({rec_id})",
            tolerance_ms=cfg.EVAL_TOLERANCE_MS)

        # Step 13: ECHO XAI
        self._log("Step 13: ECHO XAI...")
        has_ref  = dir_proc is not None
        echo_ref = dir_proc if has_ref else None
        echo = ECHOExplainer(
            fs=fs, maternal_peaks=maternal_peaks,
            fetal_peaks=fetal_peaks, fetal_signal=fetal_ecg,
            reference_signal=echo_ref, has_reference=has_ref)
        attribution = echo.compute_attributions()
        print(echo.generate_summary_stats(attribution))
        if attribution and attribution["n_beats"] > 0:
            print(echo.generate_clinical_report(0, attribution))

        if save_figures:
            self._save_figures(
                recording, abd_proc, maternal_recon_seq, residual_seq,
                fetal_ecg, fetal_ic_raw, dir_proc,
                fetal_peaks, ref_peaks, echo, figures_dir, rec_id)

        return {
            "recording"     : rec_id,
            "fetal_ecg"     : fetal_ecg,
            "fetal_ecg_pre" : fetal_ic_raw,
            "fetal_peaks"   : fetal_peaks,
            "maternal_peaks": maternal_peaks,
            "ref_peaks"     : ref_peaks,
            "maternal_recon": maternal_recon_seq,
            "residual"      : residual_seq,
            "abd_proc"      : abd_proc,
            "dir_proc"      : dir_proc,
            "weights"       : weights,
            "metrics"       : metrics,
            "echo"          : echo,
            "attribution"   : attribution,
            "chosen_path"   : chosen_path,
        }

    # ------------------------------------------------------------------
    # Shared helper: run all shared steps (1-8) and return intermediates
    # so fusion methods don't duplicate boilerplate.
    # ------------------------------------------------------------------
    def _run_shared_steps(self, recording):
        """
        Run steps 1-8 (preprocessing through Path B) and return all
        intermediates needed by the fusion methods.
        """
        cfg      = self.cfg
        dataset  = recording.get("dataset", "ADFECGDB")
        rec_id   = recording["recording"]
        abd      = recording["abdomen"]
        direct   = recording.get("direct")
        fs       = recording["fs"]
        duration = recording.get("duration_sec", abd.shape[1] / fs)
        min_peaks = _min_usable_peaks(duration, cfg, dataset)

        # Step 1
        abd_proc = preprocess_multichannel(abd, fs)
        dir_proc = preprocess_channel(direct, fs) if direct is not None else None

        # Step 2
        use_dual = getattr(cfg, 'ICA_DUAL_CONTRAST', False)
        if use_dual:
            ICs1, _, _ica1_winner = run_ica_best_contrast(
                abd_proc, n_components=cfg.ICA_N_COMPONENTS, fs=fs)
        else:
            ICs1, _ = run_ica(abd_proc, n_components=cfg.ICA_N_COMPONENTS)
        maternal_ic_idx, _ = select_maternal_ic(ICs1, fs)
        maternal_ic        = get_ic_as_signal(ICs1, maternal_ic_idx)

        # Step 3
        maternal_peaks = detect_maternal_qrs(maternal_ic, fs)
        mat_hr_stats   = compute_hr_stats(maternal_peaks, fs)
        maternal_hr    = mat_hr_stats["mean_hr"]

        ann_path     = recording.get("annotation_path")
        ann_ext      = recording.get("annotation_ext", "qrs")
        ann_is_fetal = recording.get("annotation_is_fetal", False)
        expected_fhr = None
        if ann_path and ann_is_fetal:
            ann_peaks = load_wfdb_annotation(ann_path, ann_ext)
            if len(ann_peaks) >= 5:
                expected_fhr = compute_hr_stats(ann_peaks, fs)["mean_hr"]

        # Step 4: Path A
        a_sig, a_idx, a_peaks, a_hr = _best_ic(
            ICs1, maternal_ic_idx, maternal_hr, fs, cfg,
            label="Path A", expected_hr=expected_fhr, min_peaks=min_peaks)
        a_valid = _is_fetal_hr(a_hr, maternal_hr, cfg)

        # Step 5-7: WSVD path
        weights    = gaussian_weight_matrix(abd_proc.shape[1], maternal_peaks, fs)
        svd_explained_variance(abd_proc)
        channel_r2 = np.array([
            float(np.corrcoef(abd_proc[ch], maternal_ic)[0, 1] ** 2)
            for ch in range(abd_proc.shape[0])
        ])
        maternal_recon = adaptive_windowed_wsvd(
            abd_proc, weights, fs, mat_ic=maternal_ic, channel_r2=channel_r2)
        residual = subtract_maternal(abd_proc, maternal_recon)

        # Step 8: Path B
        ICs2, _          = run_ica(residual, n_components=cfg.ICA_N_COMPONENTS)
        mat_residual_idx = _find_maternal_residual_idx(ICs2, maternal_ic, cfg)
        b_sig, b_idx, b_peaks, b_hr = _best_ic(
            ICs2, mat_residual_idx, maternal_hr, fs, cfg,
            label="Path B", expected_hr=expected_fhr, min_peaks=min_peaks)
        b_valid = _is_fetal_hr(b_hr, maternal_hr, cfg)

        return dict(
            cfg=cfg, rec_id=rec_id, fs=fs, abd_proc=abd_proc,
            dir_proc=dir_proc, maternal_ic=maternal_ic,
            maternal_peaks=maternal_peaks, maternal_hr=maternal_hr,
            maternal_recon=maternal_recon, residual=residual, weights=weights,
            ann_path=ann_path, ann_ext=ann_ext, ann_is_fetal=ann_is_fetal,
            expected_fhr=expected_fhr,
            a_sig=a_sig, a_idx=a_idx, a_peaks=a_peaks, a_hr=a_hr, a_valid=a_valid,
            b_sig=b_sig, b_idx=b_idx, b_peaks=b_peaks, b_hr=b_hr, b_valid=b_valid,
        )

    def _run_tail(self, shared, chosen_sig, chosen_peaks, chosen_path,
                  method_label, save_figures=False, figures_dir="figures"):
        """
        Run steps 10-13 (EKF → QRS → eval → ECHO) on the fused signal.
        Returns the standard result dict.
        """
        cfg            = shared["cfg"]
        fs             = shared["fs"]
        rec_id         = shared["rec_id"]
        abd_proc       = shared["abd_proc"]
        dir_proc       = shared["dir_proc"]
        maternal_peaks = shared["maternal_peaks"]
        maternal_recon = shared["maternal_recon"]
        residual       = shared["residual"]
        weights        = shared["weights"]
        ann_path       = shared["ann_path"]
        ann_ext        = shared["ann_ext"]
        ann_is_fetal   = shared["ann_is_fetal"]

        # EKF
        fetal_ic_raw = chosen_sig
        if self.ekf_bypass:
            fetal_ecg = fetal_ic_raw
        else:
            fetal_ecg = _apply_ekf(fetal_ic_raw, chosen_peaks, fs,
                                   self.use_rts, cfg=cfg)

        # Final QRS
        fetal_peaks = detect_fetal_qrs(fetal_ecg, fs, cfg=cfg)

        # Reference peaks
        if ann_path and ann_is_fetal:
            ref_peaks = load_wfdb_annotation(ann_path, ann_ext)
        elif dir_proc is not None:
            ref_peaks = detect_reference_fetal_qrs(dir_proc, fs)
        else:
            ref_peaks = np.array([])

        metrics = evaluate(
            fetal_ecg, dir_proc, fetal_peaks, ref_peaks, fs,
            label=f"{method_label} ({rec_id})",
            tolerance_ms=cfg.EVAL_TOLERANCE_MS)

        # ECHO XAI
        has_ref = dir_proc is not None
        echo = ECHOExplainer(
            fs=fs, maternal_peaks=maternal_peaks,
            fetal_peaks=fetal_peaks, fetal_signal=fetal_ecg,
            reference_signal=dir_proc if has_ref else None,
            has_reference=has_ref)
        attribution = echo.compute_attributions()
        print(echo.generate_summary_stats(attribution))

        if save_figures:
            self._save_figures(
                {"recording": rec_id}, abd_proc, maternal_recon, residual,
                fetal_ecg, fetal_ic_raw, dir_proc,
                fetal_peaks, ref_peaks, echo, figures_dir, rec_id)

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

    # ------------------------------------------------------------------
    # Option 1: Simple Ensemble — equal-weight average of A and B signals
    # ------------------------------------------------------------------
    def run_ensemble_simple(self, recording, save_figures=False, figures_dir="figures"):
        """
        PHASE-ENS-SIMPLE: average Path A and Path B fetal ECG signals
        with equal (50/50) weight before EKF and QRS detection.
        Both paths always contribute regardless of individual quality.
        """
        self._log("=" * 55)
        self._log(f"[ENS-SIMPLE] {recording['recording']}")
        self._log("=" * 55)

        s = self._run_shared_steps(recording)
        cfg = s["cfg"]

        # Equal-weight average in signal space
        fused_sig   = 0.5 * s["a_sig"] + 0.5 * s["b_sig"]
        fused_peaks = detect_fetal_qrs(fused_sig, s["fs"], cfg=cfg)

        # If fused detection collapses, fall back to whichever path had more peaks
        if len(fused_peaks) < 5:
            fused_sig   = s["a_sig"] if len(s["a_peaks"]) >= len(s["b_peaks"]) else s["b_sig"]
            fused_peaks = s["a_peaks"] if len(s["a_peaks"]) >= len(s["b_peaks"]) else s["b_peaks"]

        self._log(f"  A: {len(s['a_peaks'])} peaks @ {s['a_hr']:.1f} BPM | "
                  f"B: {len(s['b_peaks'])} peaks @ {s['b_hr']:.1f} BPM | "
                  f"Fused: {len(fused_peaks)} peaks")

        return self._run_tail(s, fused_sig, fused_peaks,
                              "ENS_SIMPLE", "PHASE-ENS-SIMPLE",
                              save_figures, figures_dir)

    # ------------------------------------------------------------------
    # Option 2: Weighted Ensemble — confidence-weighted average
    # ------------------------------------------------------------------
    def run_ensemble_weighted(self, recording, save_figures=False, figures_dir="figures"):
        """
        PHASE-ENS-WEIGHTED: weight each path's signal by its confidence
        score (HR score × normalised peak count) before averaging.
        A path that found more peaks at a more plausible fetal HR
        contributes more to the fused signal.
        """
        self._log("=" * 55)
        self._log(f"[ENS-WEIGHTED] {recording['recording']}")
        self._log("=" * 55)

        s   = self._run_shared_steps(recording)
        cfg = s["cfg"]
        fs  = s["fs"]

        # Confidence = HR score × peak count (normalised so weights sum to 1)
        a_hr_score = _hr_score(s["a_hr"], cfg, s["expected_fhr"])
        b_hr_score = _hr_score(s["b_hr"], cfg, s["expected_fhr"])
        a_conf = a_hr_score * len(s["a_peaks"])
        b_conf = b_hr_score * len(s["b_peaks"])
        total  = a_conf + b_conf + 1e-10
        w_a    = a_conf / total
        w_b    = b_conf / total

        self._log(f"  Confidence weights — A: {w_a:.3f}  B: {w_b:.3f}")
        self._log(f"  A: {len(s['a_peaks'])} peaks @ {s['a_hr']:.1f} BPM | "
                  f"B: {len(s['b_peaks'])} peaks @ {s['b_hr']:.1f} BPM")

        fused_sig   = w_a * s["a_sig"] + w_b * s["b_sig"]
        fused_peaks = detect_fetal_qrs(fused_sig, fs, cfg=cfg)

        if len(fused_peaks) < 5:
            fused_sig   = s["a_sig"] if w_a >= w_b else s["b_sig"]
            fused_peaks = s["a_peaks"] if w_a >= w_b else s["b_peaks"]

        chosen_path = f"ENS_WEIGHTED_wA={w_a:.2f}_wB={w_b:.2f}"
        return self._run_tail(s, fused_sig, fused_peaks,
                              chosen_path, "PHASE-ENS-WEIGHTED",
                              save_figures, figures_dir)

    # ------------------------------------------------------------------
    # Option 3: Rescue — Path B only activates when Path A is weak
    # ------------------------------------------------------------------
    def run_rescue(self, recording, save_figures=False, figures_dir="figures"):
        """
        PHASE-RESCUE: use Path A when it is confident; fall back to
        Path B only when Path A fails the HR filter OR finds too few peaks.

        Confidence threshold: Path A must pass the fetal HR filter AND
        find at least cfg.PATH_A_MIN_PEAKS_RESCUE peaks (default = 60% of
        min_usable_peaks). If it does, Path B is skipped entirely.
        """
        self._log("=" * 55)
        self._log(f"[RESCUE] {recording['recording']}")
        self._log("=" * 55)

        s   = self._run_shared_steps(recording)
        cfg = s["cfg"]
        fs  = s["fs"]

        duration  = recording.get("duration_sec", recording["abdomen"].shape[1] / fs)
        min_peaks = _min_usable_peaks(duration, cfg, recording.get("dataset", "ADFECGDB"))
        rescue_threshold = int(min_peaks * getattr(cfg, "RESCUE_THRESHOLD_FRAC", 0.85))
        ibi_cv_max = getattr(cfg, "RESCUE_IBI_CV_MAX", 0.30)

        # Check 1: Path A passed HR filter and found enough peaks
        a_enough_peaks = s["a_valid"] and len(s["a_peaks"]) >= rescue_threshold

        # Check 2: Path A peaks are rhythmically regular (low IBI coefficient
        # of variation). Spurious peaks from noise have high IBI variance.
        if len(s["a_peaks"]) >= 4:
            ibis   = np.diff(s["a_peaks"]).astype(float)
            ibi_cv = float(np.std(ibis) / (np.mean(ibis) + 1e-10))
        else:
            ibi_cv = float("inf")
        a_regular = ibi_cv <= ibi_cv_max

        a_strong = a_enough_peaks and a_regular

        self._log(f"  Path A: {len(s['a_peaks'])} peaks (need {rescue_threshold}), "
                  f"valid={s['a_valid']}, IBI_CV={ibi_cv:.3f} (max {ibi_cv_max})")

        if a_strong:
            chosen_sig, chosen_peaks = s["a_sig"], s["a_peaks"]
            chosen_path = f"RESCUE_used_A_IC{s['a_idx']+1}_{s['a_hr']:.0f}bpm"
            self._log(f"  Path A strong → using Path A, Path B skipped")
        else:
            reason = []
            if not s["a_valid"]:    reason.append("HR invalid")
            if not a_enough_peaks:  reason.append(f"too few peaks ({len(s['a_peaks'])} < {rescue_threshold})")
            if not a_regular:       reason.append(f"irregular IBI (CV={ibi_cv:.3f} > {ibi_cv_max})")
            chosen_sig, chosen_peaks = s["b_sig"], s["b_peaks"]
            chosen_path = f"RESCUE_used_B_IC{s['b_idx']+1}_{s['b_hr']:.0f}bpm"
            self._log(f"  Path A weak [{', '.join(reason)}] → Path B rescue activated")

        return self._run_tail(s, chosen_sig, chosen_peaks,
                              chosen_path, "PHASE-RESCUE",
                              save_figures, figures_dir)

    # ------------------------------------------------------------------
    # Option 4: Peak-level fusion — merge QRS peak lists from A and B
    # ------------------------------------------------------------------
    def run_peak_fusion(self, recording, save_figures=False, figures_dir="figures"):
        """
        PHASE-PEAK-FUSION: run both paths fully through to QRS detection,
        then merge their peak lists.

        Fusion rules (applied in order):
          1. Peaks present in BOTH lists (within tolerance window) → keep,
             use the one with higher local signal amplitude.
          2. Peaks present in only one list → keep if no conflicting peak
             from the other list is within the tolerance window.
          3. Conflicting single-path peaks (within window but not matched)
             → keep the one with higher local amplitude.

        The fused peak list is used for evaluation directly.
        The EKF runs on the confidence-weighted signal (same as ENS-WEIGHTED)
        purely for waveform quality, but QRS detection uses the fused list.
        """
        self._log("=" * 55)
        self._log(f"[PEAK-FUSION] {recording['recording']}")
        self._log("=" * 55)

        s   = self._run_shared_steps(recording)
        cfg = s["cfg"]
        fs  = s["fs"]

        # Get per-path peak lists (run QRS on each path's signal separately)
        a_peaks_det = detect_fetal_qrs(s["a_sig"], fs, cfg=cfg)
        b_peaks_det = detect_fetal_qrs(s["b_sig"], fs, cfg=cfg)
        self._log(f"  Path A peaks: {len(a_peaks_det)} | Path B peaks: {len(b_peaks_det)}")

        # Strict merge — amplitude + IBI guards filter false positives
        tol_samples = int(getattr(cfg, "EVAL_TOLERANCE_MS", 50) * fs / 1000)
        fused_peaks = self._merge_peak_lists(
            s["a_sig"], s["b_sig"], a_peaks_det, b_peaks_det, tol_samples,
            amp_thresh_frac=getattr(cfg, "PEAK_FUSION_AMP_THRESH", 0.4),
            fs=fs)
        self._log(f"  Path A: {len(a_peaks_det)} peaks | "
                  f"Path B: {len(b_peaks_det)} peaks | "
                  f"Fused (strict): {len(fused_peaks)} peaks")

        # Confidence-weighted signal for EKF waveform (not for QRS detection)
        a_hr_score = _hr_score(s["a_hr"], cfg, s["expected_fhr"])
        b_hr_score = _hr_score(s["b_hr"], cfg, s["expected_fhr"])
        a_conf = a_hr_score * len(a_peaks_det) + 1e-10
        b_conf = b_hr_score * len(b_peaks_det) + 1e-10
        total  = a_conf + b_conf
        fused_sig = (a_conf / total) * s["a_sig"] + (b_conf / total) * s["b_sig"]

        # EKF on fused signal using fused peaks
        fetal_ic_raw = fused_sig
        if self.ekf_bypass:
            fetal_ecg = fetal_ic_raw
        else:
            fetal_ecg = _apply_ekf(fetal_ic_raw, fused_peaks, fs,
                                   self.use_rts, cfg=cfg)

        # Use fused peaks directly for evaluation (override EKF re-detection)
        fused_peaks_final = fused_peaks if len(fused_peaks) >= 5 \
            else detect_fetal_qrs(fetal_ecg, fs, cfg=cfg)

        ann_path     = s["ann_path"]
        ann_ext      = s["ann_ext"]
        ann_is_fetal = s["ann_is_fetal"]
        dir_proc     = s["dir_proc"]

        if ann_path and ann_is_fetal:
            ref_peaks = load_wfdb_annotation(ann_path, ann_ext)
        elif dir_proc is not None:
            ref_peaks = detect_reference_fetal_qrs(dir_proc, fs)
        else:
            ref_peaks = np.array([])

        metrics = evaluate(
            fetal_ecg, dir_proc, fused_peaks_final, ref_peaks, fs,
            label=f"PHASE-PEAK-FUSION ({s['rec_id']})",
            tolerance_ms=cfg.EVAL_TOLERANCE_MS)

        has_ref = dir_proc is not None
        echo = ECHOExplainer(
            fs=fs, maternal_peaks=s["maternal_peaks"],
            fetal_peaks=fused_peaks_final, fetal_signal=fetal_ecg,
            reference_signal=dir_proc if has_ref else None,
            has_reference=has_ref)
        attribution = echo.compute_attributions()
        print(echo.generate_summary_stats(attribution))

        return {
            "recording"     : s["rec_id"],
            "fetal_ecg"     : fetal_ecg,
            "fetal_ecg_pre" : fetal_ic_raw,
            "fetal_peaks"   : fused_peaks_final,
            "maternal_peaks": s["maternal_peaks"],
            "ref_peaks"     : ref_peaks,
            "maternal_recon": s["maternal_recon"],
            "residual"      : s["residual"],
            "abd_proc"      : s["abd_proc"],
            "dir_proc"      : dir_proc,
            "weights"       : s["weights"],
            "metrics"       : metrics,
            "echo"          : echo,
            "attribution"   : attribution,
            "chosen_path"   : f"PEAK_FUSION_A{len(a_peaks_det)}_B{len(b_peaks_det)}_F{len(fused_peaks_final)}",
        }

    @staticmethod
    def _merge_peak_lists(sig_a, sig_b, peaks_a, peaks_b, tol_samples,
                          amp_thresh_frac=0.4, min_ibi_samples=None, fs=None):
        """
        Strict peak-list merge.

        MATCHED peaks (both paths agree within tol_samples):
          → Always kept. Use the one with higher local amplitude.

        UNMATCHED peaks (only one path detected it):
          → Kept ONLY if they pass both guards:
            1. Amplitude guard: local amplitude >= amp_thresh_frac * median
               amplitude of all matched peaks. Weak unmatched peaks are
               almost always ICA noise, not real fetal beats.
            2. IBI guard: the peak must not create a beat-to-beat interval
               shorter than min_ibi_samples (default: 60s/220bpm * fs).
               Physiologically impossible intervals are false positives.

        This directly fixes the over-detection problem where Path B adds
        many spurious peaks on recordings where Path A is already correct.
        On hard recordings where Path B finds real beats that Path A missed,
        those peaks have strong amplitude and valid IBIs — they pass both
        guards and are kept, preserving the fusion benefit.

        Parameters
        ----------
        amp_thresh_frac : float
            Unmatched peak amplitude must be >= this fraction of the median
            matched-peak amplitude. Default 0.4 (40%).
        min_ibi_samples : int or None
            Minimum allowed gap between consecutive fused peaks in samples.
            If None, derived from fs at 220 BPM (physiological ceiling).
        fs : float or None
            Sampling frequency. Used only to derive min_ibi_samples if not
            provided explicitly.
        """
        if len(peaks_a) == 0:
            return peaks_b.copy() if len(peaks_b) > 0 else np.array([], dtype=int)
        if len(peaks_b) == 0:
            return peaks_a.copy()

        # Minimum IBI: beats cannot be closer than 60/220 BPM seconds apart
        if min_ibi_samples is None:
            min_ibi_samples = int((60.0 / 220.0) * fs) if fs else 10

        def amp(sig, idx):
            lo = max(0, idx - 5)
            hi = min(len(sig), idx + 6)
            return float(np.max(np.abs(sig[lo:hi])))

        # --- Pass 1: match peaks across paths within tolerance ---------------
        matched_b   = set()
        matched_a   = set()
        fused       = []
        matched_amps = []

        for i, pa in enumerate(peaks_a):
            dists = np.abs(peaks_b - pa)
            nearest_idx = int(np.argmin(dists))
            if dists[nearest_idx] <= tol_samples:
                pb = peaks_b[nearest_idx]
                matched_b.add(nearest_idx)
                matched_a.add(i)
                # Keep the one with higher local amplitude
                amp_a = amp(sig_a, pa)
                amp_b = amp(sig_b, pb)
                keep  = pa if amp_a >= amp_b else pb
                fused.append(keep)
                matched_amps.append(max(amp_a, amp_b))

        # Amplitude reference: median of matched peaks (robust to outliers)
        if matched_amps:
            amp_ref = float(np.median(matched_amps))
        else:
            # No matched peaks at all — fall back to median of all Path A amps
            amp_ref = float(np.median([amp(sig_a, p) for p in peaks_a])) + 1e-10

        amp_threshold = amp_thresh_frac * amp_ref

        # --- Pass 2: add unmatched Path A peaks (with guards) ---------------
        for i, pa in enumerate(peaks_a):
            if i in matched_a:
                continue
            if amp(sig_a, pa) >= amp_threshold:
                fused.append(pa)
            # else: drop — weak unmatched A peak, likely noise

        # --- Pass 3: add unmatched Path B peaks (with guards) ---------------
        for j, pb in enumerate(peaks_b):
            if j in matched_b:
                continue
            if amp(sig_b, pb) >= amp_threshold:
                fused.append(int(pb))
            # else: drop — this is the main source of false positives

        # --- Sort and apply IBI guard ----------------------------------------
        fused = np.array(sorted(set(fused)), dtype=int)

        if len(fused) < 2:
            return fused

        # Remove peaks that create physiologically impossible short intervals
        keep_mask = np.ones(len(fused), dtype=bool)
        for i in range(1, len(fused)):
            if not keep_mask[i - 1]:
                # Find last kept peak before i
                prev_kept = -1
                for k in range(i - 1, -1, -1):
                    if keep_mask[k]:
                        prev_kept = fused[k]
                        break
                if prev_kept < 0:
                    continue
                ibi = fused[i] - prev_kept
            else:
                ibi = fused[i] - fused[i - 1]

            if ibi < min_ibi_samples:
                # Keep whichever of the two has higher amplitude (on sig_a)
                if amp(sig_a, fused[i]) > amp(sig_a, fused[i - 1]):
                    keep_mask[i - 1] = False
                else:
                    keep_mask[i] = False

        return fused[keep_mask]

    def run_with_ablation(self, recording):
        self._log("Running ablation study...")
        fs       = recording["fs"]
        abd      = recording["abdomen"]
        direct   = recording["direct"]
        duration = recording.get("duration_sec", abd.shape[1] / fs)
        min_peaks = _min_usable_peaks(duration)

        abd_proc  = preprocess_multichannel(abd, fs)
        dir_proc  = preprocess_channel(direct, fs)
        ref_peaks = detect_reference_fetal_qrs(dir_proc, fs)
        results   = {}

        ICs1, _          = run_ica(abd_proc)
        mat_idx_blind, _ = select_maternal_ic(ICs1, fs)
        mat_ic_blind     = get_ic_as_signal(ICs1, mat_idx_blind)
        mat_peaks_blind  = detect_maternal_qrs(mat_ic_blind, fs)
        mat_hr_blind     = compute_hr_stats(mat_peaks_blind, fs)["mean_hr"]
        weights_gauss    = gaussian_weight_matrix(abd_proc.shape[1], mat_peaks_blind, fs)

        def _eval(sig, peaks, label):
            return evaluate(sig, dir_proc, peaks, ref_peaks, fs, label=label)

        def _select(ICs, excl, mat_hr):
            sig, idx, peaks, hr = _best_ic(ICs, excl, mat_hr, fs, min_peaks=min_peaks)
            return sig, peaks

        # Config 1: Baseline
        self._log("  Config 1: Baseline -- naive ICA + global binary WSVD...")
        mat_idx_naive   = int(np.argmax([np.var(ic) for ic in ICs1]))
        mat_ic_naive    = get_ic_as_signal(ICs1, mat_idx_naive)
        mat_peaks_naive = detect_maternal_qrs(mat_ic_naive, fs)
        weights_binary  = _binary_weight_matrix(abd_proc.shape[1], mat_peaks_naive, fs)
        mat_recon_1 = _global_wsvd(abd_proc, weights_binary)
        residual_1  = subtract_maternal(abd_proc, mat_recon_1)
        ICs2_1, _   = run_ica(residual_1)
        corrs       = [abs(np.corrcoef(ic, dir_proc)[0, 1]) for ic in ICs2_1]
        ic_base     = _norm(ICs2_1[int(np.argmax(corrs))])
        pks_base    = detect_fetal_qrs(ic_base, fs, cfg=self.cfg)
        results["1_Baseline_ICA_WSVD"] = _eval(ic_base, pks_base, "Baseline ICA+WSVD")

        # Config 2: + Blind IC selection
        self._log("  Config 2: + Blind IC selection...")
        mat_recon_2 = _global_wsvd(abd_proc, _binary_weight_matrix(abd_proc.shape[1], mat_peaks_blind, fs))
        residual_2  = subtract_maternal(abd_proc, mat_recon_2)
        ICs2_2, _   = run_ica(residual_2)
        excl_2      = _find_maternal_residual_idx(ICs2_2, mat_ic_blind)
        sig_2, pks_2 = _select(ICs2_2, excl_2, mat_hr_blind)
        results["2_Blind_IC_Selection"] = _eval(sig_2, pks_2, "+Blind IC Selection")

        # Config 3: + Gaussian weights
        self._log("  Config 3: + Gaussian weights...")
        mat_recon_3 = _global_wsvd(abd_proc, weights_gauss)
        residual_3  = subtract_maternal(abd_proc, mat_recon_3)
        ICs2_3, _   = run_ica(residual_3)
        excl_3      = _find_maternal_residual_idx(ICs2_3, mat_ic_blind)
        sig_3, pks_3 = _select(ICs2_3, excl_3, mat_hr_blind)
        results["3_Gaussian_Weights"] = _eval(sig_3, pks_3, "+Gaussian Weights")

        # Config 4: + Adaptive windowed WSVD
        self._log("  Config 4: + Adaptive Windowed WSVD...")
        channel_r2  = np.array([float(np.corrcoef(abd_proc[ch], mat_ic_blind)[0, 1] ** 2)
                                 for ch in range(abd_proc.shape[0])])
        mat_recon_4 = adaptive_windowed_wsvd(abd_proc, weights_gauss, fs,
                                              mat_ic=mat_ic_blind, channel_r2=channel_r2)
        residual_4  = subtract_maternal(abd_proc, mat_recon_4)
        ICs2_4, _   = run_ica(residual_4)
        excl_4      = _find_maternal_residual_idx(ICs2_4, mat_ic_blind)
        sig_4, pks_4 = _select(ICs2_4, excl_4, mat_hr_blind)
        results["4_Adaptive_WSVD"] = _eval(sig_4, pks_4, "+Adaptive WSVD")

        # Config 5: + EKF-RTS
        self._log("  Config 5: Full PHASE (+ EKF-RTS)...")
        fetal_ecg_5 = _apply_ekf(sig_4, pks_4, fs, use_rts=True, cfg=self.cfg)
        pks_5       = detect_fetal_qrs(fetal_ecg_5, fs, cfg=self.cfg)
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

        # Pass None for reference if no direct electrode available
        has_direct = dir_proc is not None and hasattr(dir_proc, '__len__') and len(dir_proc) > 0
        plot_fetal_comparison(
            fetal_ecg,
            dir_proc if has_direct else None,
            fetal_peaks,
            ref_peaks if has_direct else None,
            self.fs,
            save_path=str(fdir / f"{rec_id}_fetal_comparison.png"))

        plot_ekf_refinement(
            fetal_ic_raw, fetal_ecg,
            dir_proc if has_direct else None,
            self.fs,
            save_path=str(fdir / f"{rec_id}_ekf_refinement.png"))

        echo.plot_attribution_heatmap(
            save_path=str(fdir / f"{rec_id}_echo_attribution.png"))
        self._log(f"Figures saved to {figures_dir}/")


# -- Ablation helpers --------------------------------------------------------

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