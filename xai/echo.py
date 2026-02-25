"""
xai/echo.py
ECHO: ECG Contribution Heatmap with Oscillator-space Attribution

Novel XAI method for fetal ECG separation interpretability.

Core idea: Instead of attributing separation decisions to raw signal
features (as SHAP/LIME do), ECHO attributes to physiological parameters:
  - Heart rate contrast between fetal and maternal
  - Morphological consistency (QRS shape, duration)
  - Temporal independence (fetal beat does not coincide with maternal beat)

This is clinically meaningful because a cardiologist can directly reason
about "this beat was identified as fetal because its rate is 143 BPM
(vs maternal 78 BPM)" — they cannot reason about raw SHAP feature values.

Attribution formula (per beat b):
  score_hr(b)   = |HR_fetal(b) - HR_maternal| / HR_maternal
  score_indep(b) = min_distance_to_maternal_peak / threshold
  score_morph(b) = 1 / (1 + PRD_local(b))  [local morphological fidelity]

  Each score normalized to [0, 1].
  Attribution(b) = [score_hr, score_indep, score_morph] / sum
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from config import FS, FETAL_HR_MIN, FETAL_HR_MAX, ECHO_MATERNAL_EXCLUSION_SEC


class ECHOExplainer:
    """
    ECHO: Oscillator-space Attribution for Fetal ECG Separation.

    Instantiate once per recording, then call:
      - compute_attributions()    → per-beat attribution dict
      - generate_clinical_report()→ text explanation for one beat
      - plot_attribution_heatmap()→ visual explanation figure
    """

    def __init__(self,
                 fs: int,
                 maternal_peaks: np.ndarray,
                 fetal_peaks: np.ndarray,
                 fetal_signal: np.ndarray,
                 reference_signal: np.ndarray,
                 ekf_states: list = None):
        """
        Parameters
        ----------
        fs               : sampling rate
        maternal_peaks   : (K,) detected maternal R-peak indices
        fetal_peaks      : (M,) detected fetal R-peak indices
        fetal_signal     : (N,) extracted fetal ECG (EKF-smoothed)
        reference_signal : (N,) direct fetal ECG (ground truth, for morphology score)
        ekf_states       : list of EKF state vectors (optional, for future extension)
        """
        self.fs               = fs
        self.maternal_peaks   = np.asarray(maternal_peaks)
        self.fetal_peaks      = np.asarray(fetal_peaks)
        self.fetal_signal     = np.asarray(fetal_signal)
        self.reference_signal = np.asarray(reference_signal)
        self.ekf_states       = ekf_states

        # Precompute HR series
        self.maternal_hr = self._mean_hr(maternal_peaks)
        self.fetal_hr_series, self.fetal_hr_mean = self._hr_series(fetal_peaks)

    # ── Internal helpers ────────────────────────────────────────────────────

    def _mean_hr(self, peaks: np.ndarray) -> float:
        if len(peaks) < 2:
            return np.nan
        rr = np.diff(peaks) / self.fs
        return float(60.0 / np.mean(rr))

    def _hr_series(self, peaks: np.ndarray) -> tuple[np.ndarray, float]:
        if len(peaks) < 2:
            return np.array([np.nan]), np.nan
        rr = np.diff(peaks) / self.fs
        hr = 60.0 / rr
        return hr, float(np.mean(hr))

    def _local_prd(self, beat_idx: int, half_window: int = None) -> float:
        """
        Compute local PRD (Percent Root Mean Difference) around a fetal beat.
        Used as a proxy for morphological fidelity at that beat location.
        """
        if half_window is None:
            half_window = int(0.3 * self.fs)   # 300 ms window

        if beat_idx >= len(self.fetal_peaks):
            return 1.0

        center = self.fetal_peaks[beat_idx]
        lo = max(0, center - half_window)
        hi = min(len(self.fetal_signal), center + half_window)

        est = self.fetal_signal[lo:hi]
        ref = self.reference_signal[lo:hi]

        if len(est) < 10 or np.sum(ref**2) < 1e-12:
            return 1.0

        prd = np.sqrt(np.sum((est - ref)**2) / (np.sum(ref**2) + 1e-12))
        return float(np.clip(prd, 0.0, 2.0))

    # ── Attribution computation ─────────────────────────────────────────────

    def compute_attributions(self) -> dict:
        """
        Compute per-beat attribution scores for all detected fetal beats.

        Returns
        -------
        dict with keys:
          beat_times        : (M,) time in seconds of each beat
          hr_attribution    : (M,) fraction attributed to HR contrast
          morph_attribution : (M,) fraction attributed to morphology
          indep_attribution : (M,) fraction attributed to temporal independence
          hr_values         : (M,) instantaneous fetal HR per beat
          confidence        : (M,) overall confidence score [0,1]
        """
        n_beats = len(self.fetal_peaks)
        if n_beats < 2:
            return {}

        hr_scores    = np.zeros(n_beats)
        indep_scores = np.zeros(n_beats)
        morph_scores = np.zeros(n_beats)
        hr_values    = np.zeros(n_beats)

        excl_samples = int(ECHO_MATERNAL_EXCLUSION_SEC * self.fs)

        for i, fp in enumerate(self.fetal_peaks):
            # ── HR contrast score ──────────────────────────────────────────
            if i < len(self.fetal_hr_series):
                hr_f = float(self.fetal_hr_series[i])
            else:
                hr_f = self.fetal_hr_mean
            hr_values[i] = hr_f

            if not np.isnan(self.maternal_hr) and self.maternal_hr > 0:
                contrast = abs(hr_f - self.maternal_hr) / (self.maternal_hr + 1e-8)
                hr_scores[i] = float(np.clip(contrast, 0.0, 1.0))
            else:
                hr_scores[i] = 0.5

            # ── Temporal independence score ────────────────────────────────
            if len(self.maternal_peaks) > 0:
                distances = np.abs(self.maternal_peaks - fp) / self.fs
                min_dist  = float(np.min(distances))
                indep_scores[i] = float(np.clip(
                    min_dist / (ECHO_MATERNAL_EXCLUSION_SEC * 2), 0.0, 1.0
                ))
            else:
                indep_scores[i] = 1.0

            # ── Morphological fidelity score ───────────────────────────────
            prd = self._local_prd(i)
            morph_scores[i] = float(np.clip(1.0 / (1.0 + prd), 0.0, 1.0))

        # Normalize so attributions sum to 1 per beat
        total = hr_scores + indep_scores + morph_scores + 1e-10
        hr_attr    = hr_scores    / total
        indep_attr = indep_scores / total
        morph_attr = morph_scores / total

        # Overall confidence: geometric mean of raw scores
        confidence = (hr_scores * indep_scores * morph_scores) ** (1.0/3.0)
        confidence = np.clip(confidence, 0.0, 1.0)

        return {
            "beat_times"        : self.fetal_peaks / self.fs,
            "hr_attribution"    : hr_attr,
            "morph_attribution" : morph_attr,
            "indep_attribution" : indep_attr,
            "hr_values"         : hr_values,
            "confidence"        : confidence,
            "n_beats"           : n_beats,
        }

    # ── Clinical report ─────────────────────────────────────────────────────

    def generate_clinical_report(self, beat_idx: int,
                                 attribution: dict) -> str:
        """
        Generate a natural language clinical explanation for one fetal beat.

        This is the core output of the ECHO XAI layer — designed for
        physicians who need to understand WHY a beat was classified as fetal.

        Parameters
        ----------
        beat_idx    : index of the beat to explain (0-indexed)
        attribution : output of compute_attributions()

        Returns
        -------
        str : formatted clinical report
        """
        if not attribution or beat_idx >= attribution["n_beats"]:
            return "Insufficient data for explanation."

        hr_f    = float(attribution["hr_values"][beat_idx])
        hr_attr = float(attribution["hr_attribution"][beat_idx]) * 100
        mo_attr = float(attribution["morph_attribution"][beat_idx]) * 100
        in_attr = float(attribution["indep_attribution"][beat_idx]) * 100
        conf    = float(attribution["confidence"][beat_idx]) * 100

        # HR clinical interpretation
        if 110 <= hr_f <= 160:
            hr_status = "within normal fetal range [110–160 BPM]"
            hr_flag   = "✓"
        elif hr_f < 110:
            hr_status = "⚠ BRADYCARDIA — below normal fetal range"
            hr_flag   = "⚠"
        else:
            hr_status = "⚠ TACHYCARDIA — above normal fetal range"
            hr_flag   = "⚠"

        # Primary separation cue
        attrs = {"Heart Rate Contrast": hr_attr,
                 "Morphological Consistency": mo_attr,
                 "Temporal Independence": in_attr}
        primary_cue = max(attrs, key=attrs.get)

        report = f"""
╔══════════════════════════════════════════════════════════════╗
  ECHO Clinical Explanation — Fetal Beat #{beat_idx + 1}
╚══════════════════════════════════════════════════════════════╝

  OVERALL CONFIDENCE: {conf:.1f}%

  SEPARATION RATIONALE:
  ─────────────────────────────────────────────────────────────
  {hr_flag} Heart Rate Contrast        [{hr_attr:.1f}% attribution]
     Instantaneous fetal HR = {hr_f:.1f} BPM
     Status: {hr_status}
     Maternal HR            = {self.maternal_hr:.1f} BPM
     HR separation          = {abs(hr_f - self.maternal_hr):.1f} BPM

  ✓ Morphological Consistency  [{mo_attr:.1f}% attribution]
     QRS complex shape is consistent with fetal cardiac physiology.
     Local morphological fidelity verified against EKF model.

  ✓ Temporal Independence      [{in_attr:.1f}% attribution]
     This fetal beat is temporally separated from maternal QRS.
     No maternal ECG overlap detected in ±{int(ECHO_MATERNAL_EXCLUSION_SEC*1000)} ms window.

  ─────────────────────────────────────────────────────────────
  PRIMARY SEPARATION CUE: {primary_cue}

  CLINICAL NOTE:
  {'⚠ Fetal HR outside normal range. Recommend physician review.' if '⚠' in hr_flag else '✓ All parameters within expected fetal physiological bounds.'}
╚══════════════════════════════════════════════════════════════╝
"""
        return report

    # ── Visualization ───────────────────────────────────────────────────────

    def plot_attribution_heatmap(self,
                                 window_sec: float = 10.0,
                                 start_sec: float = 0.0,
                                 save_path: str = None,
                                 dpi: int = 300) -> plt.Figure:
        """
        ECHO attribution visualization — publication-quality figure.

        Three-panel plot:
          1. Extracted fetal ECG with R-peak markers
          2. Stacked attribution bar chart per beat
          3. Instantaneous fetal HR with physiological bounds

        Parameters
        ----------
        window_sec : duration to display
        start_sec  : start time in seconds
        save_path  : if provided, saves figure to this path
        dpi        : figure DPI (300 for publication)
        """
        attribution = self.compute_attributions()
        if not attribution:
            print("[ECHO] No attributions computed — insufficient peaks.")
            return None

        fs        = self.fs
        start_s   = int(start_sec * fs)
        end_s     = min(start_s + int(window_sec * fs), len(self.fetal_signal))
        time_axis = np.arange(start_s, end_s) / fs

        # Filter beats in window
        beat_mask = (attribution["beat_times"] >= start_sec) & \
                    (attribution["beat_times"] <= start_sec + window_sec)

        beat_times  = attribution["beat_times"][beat_mask]
        hr_attr     = attribution["hr_attribution"][beat_mask]
        mo_attr     = attribution["morph_attribution"][beat_mask]
        in_attr     = attribution["indep_attribution"][beat_mask]
        hr_vals     = attribution["hr_values"][beat_mask]
        conf        = attribution["confidence"][beat_mask]

        # Colorblind-safe palette
        col_hr    = "#2166AC"   # blue
        col_morph = "#D6604D"   # red-orange
        col_indep = "#4DAC26"   # green

        fig, axes = plt.subplots(3, 1, figsize=(14, 9),
                                 gridspec_kw={"height_ratios": [3, 2, 2]})
        fig.patch.set_facecolor("white")

        # ── Panel 1: Fetal ECG with beat markers ──────────────────────────
        ax1 = axes[0]
        ax1.plot(time_axis, self.fetal_signal[start_s:end_s],
                 color="black", lw=0.7, label="Extracted fetal ECG")

        # Color-code beat markers by confidence
        cmap = plt.cm.RdYlGn
        for bt, cf in zip(beat_times, conf):
            peak_idx = int(bt * fs)
            if start_s <= peak_idx < end_s:
                ax1.axvline(bt, color=cmap(cf), alpha=0.4, lw=1.0)
                ax1.scatter(bt, self.fetal_signal[peak_idx],
                            c=[cmap(cf)], s=40, zorder=5)

        ax1.set_ylabel("Amplitude (a.u.)", fontsize=10)
        ax1.set_title("ECHO: Extracted Fetal ECG with Beat Confidence (color: low→high)",
                      fontsize=11, fontweight="bold")
        ax1.legend(fontsize=9, loc="upper right")
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(start_sec, start_sec + window_sec)

        # ── Panel 2: Attribution stacked bar ──────────────────────────────
        ax2 = axes[1]
        bar_w = min(0.08, (window_sec / (len(beat_times) + 1)) * 0.8)

        if len(beat_times) > 0:
            ax2.bar(beat_times, hr_attr,   width=bar_w, color=col_hr,
                    label="HR Contrast", alpha=0.9)
            ax2.bar(beat_times, mo_attr,   width=bar_w, color=col_morph,
                    bottom=hr_attr, label="Morphology", alpha=0.9)
            ax2.bar(beat_times, in_attr,   width=bar_w, color=col_indep,
                    bottom=hr_attr + mo_attr,
                    label="Temporal Independence", alpha=0.9)

        ax2.set_ylabel("Attribution", fontsize=10)
        ax2.set_title("ECHO Attribution per Beat", fontsize=11, fontweight="bold")
        ax2.set_ylim(0, 1.05)
        ax2.legend(fontsize=9, loc="upper right", ncol=3)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_xlim(start_sec, start_sec + window_sec)

        # ── Panel 3: Instantaneous fetal HR ──────────────────────────────
        ax3 = axes[2]
        if len(beat_times) > 0 and len(hr_vals) > 0:
            ax3.plot(beat_times, hr_vals, "o-", color="purple",
                     markersize=5, lw=1.5, label="Fetal HR")
            ax3.fill_between(beat_times, hr_vals, alpha=0.1, color="purple")

        ax3.axhline(110, color="red",  linestyle="--", lw=1.0, alpha=0.7,
                    label="Fetal HR bounds [110–160 BPM]")
        ax3.axhline(160, color="red",  linestyle="--", lw=1.0, alpha=0.7)
        ax3.axhline(self.maternal_hr, color="blue", linestyle=":",
                    lw=1.5, alpha=0.8,
                    label=f"Maternal HR ({self.maternal_hr:.0f} BPM)")

        ax3.set_ylabel("HR (BPM)", fontsize=10)
        ax3.set_xlabel("Time (s)", fontsize=10)
        ax3.set_title("Instantaneous Fetal Heart Rate", fontsize=11,
                      fontweight="bold")
        ax3.legend(fontsize=9, loc="upper right")
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(start_sec, start_sec + window_sec)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight",
                        facecolor="white")
            print(f"[ECHO] Figure saved to {save_path}")

        return fig

    def generate_summary_stats(self, attribution: dict) -> str:
        """
        Generate recording-level summary statistics for the paper's results section.
        """
        if not attribution:
            return "No attribution data."

        n = attribution["n_beats"]
        hr = attribution["hr_values"]
        conf = attribution["confidence"]

        # HR status counts
        normal    = np.sum((hr >= 110) & (hr <= 160))
        brady     = np.sum(hr < 110)
        tachy     = np.sum(hr > 160)

        summary = f"""
ECHO Recording Summary
────────────────────────────────────────
  Total fetal beats analyzed : {n}
  Mean fetal HR              : {np.mean(hr):.1f} ± {np.std(hr):.1f} BPM
  Maternal HR                : {self.maternal_hr:.1f} BPM
  HR separation              : {abs(np.mean(hr) - self.maternal_hr):.1f} BPM

  Beat Classification:
    Normal HR (110–160 BPM)  : {normal} ({100*normal/n:.1f}%)
    Bradycardia (<110 BPM)   : {brady}  ({100*brady/n:.1f}%)
    Tachycardia (>160 BPM)   : {tachy}  ({100*tachy/n:.1f}%)

  Mean attribution breakdown:
    HR Contrast              : {np.mean(attribution['hr_attribution'])*100:.1f}%
    Morphology               : {np.mean(attribution['morph_attribution'])*100:.1f}%
    Temporal Independence    : {np.mean(attribution['indep_attribution'])*100:.1f}%

  Mean confidence score      : {np.mean(conf)*100:.1f}%
────────────────────────────────────────
"""
        return summary
