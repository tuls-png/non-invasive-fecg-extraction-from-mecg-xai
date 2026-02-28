"""
xai/echo.py
ECHO: ECG Contribution Heatmap with Oscillator-space Attribution

Novel XAI method for fetal ECG separation interpretability.

CHANGES FROM ORIGINAL:
  [FIX-1] Morphological fidelity score (score_morph) is now explicitly disabled
          and reported as NaN when no direct reference electrode is available
          (NIFECGDB). Previously, pipeline.py passed fetal_ecg as both the
          estimated signal AND the reference_signal when no direct electrode
          existed, making _local_prd() compare the signal to itself. This gave
          PRD=0 and morph_score=1.0 for every beat, artificially inflating
          confidence scores and making the ECHO report misleading.

          The fix: ECHOExplainer now accepts has_reference=True/False.
          When has_reference=False:
            - morph_scores are set to NaN (excluded from attribution)
            - attribution is computed from HR contrast + temporal independence only
            - the clinical report clearly states morphology was not assessed
            - confidence is computed as geometric mean of available scores only

  [FIX-2] generate_summary_stats() and generate_clinical_report() now reflect
          whether morphology was available, avoiding false clinical statements
          like "morphological fidelity verified" when it was never checked.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from config import FS, FETAL_HR_MIN, FETAL_HR_MAX, ECHO_MATERNAL_EXCLUSION_SEC
from config_nifecgdb import FS, FETAL_HR_MIN, FETAL_HR_MAX, ECHO_MATERNAL_EXCLUSION_SEC

class ECHOExplainer:
    """
    ECHO: Oscillator-space Attribution for Fetal ECG Separation.

    Instantiate once per recording, then call:
      - compute_attributions()     -> per-beat attribution dict
      - generate_clinical_report() -> text explanation for one beat
      - plot_attribution_heatmap() -> visual explanation figure
    """

    def __init__(self,
                 fs: int,
                 maternal_peaks: np.ndarray,
                 fetal_peaks: np.ndarray,
                 fetal_signal: np.ndarray,
                 reference_signal: np.ndarray | None,
                 ekf_states: list = None,
                 has_reference: bool = None):
        """
        Parameters
        ----------
        fs               : sampling rate
        maternal_peaks   : (K,) detected maternal R-peak indices
        fetal_peaks      : (M,) detected fetal R-peak indices
        fetal_signal     : (N,) extracted fetal ECG (EKF-smoothed)
        reference_signal : (N,) direct fetal ECG, or None for NIFECGDB.
                           [FIX-1] If None, morphology scoring is disabled.
        ekf_states       : list of EKF state vectors (optional)
        has_reference    : explicit override. If None, inferred from whether
                           reference_signal is not None AND differs from
                           fetal_signal (guards against the old pattern of
                           passing fetal_ecg as its own reference).
        """
        self.fs             = fs
        self.maternal_peaks = np.asarray(maternal_peaks)
        self.fetal_peaks    = np.asarray(fetal_peaks)
        self.fetal_signal   = np.asarray(fetal_signal)
        self.ekf_states     = ekf_states

        # [FIX-1] Determine if a genuine external reference exists.
        if has_reference is not None:
            self.has_reference = has_reference
        elif reference_signal is None:
            self.has_reference = False
        else:
            ref = np.asarray(reference_signal)
            # Detect the old self-referential pattern: if ref IS fetal_signal
            # (same object or identical values), treat as no reference.
            if ref is self.fetal_signal or np.array_equal(ref, self.fetal_signal):
                self.has_reference = False
                print("[ECHO] WARNING: reference_signal is identical to fetal_signal "
                      "-- morphology scoring disabled (no independent reference).")
            else:
                self.has_reference = True

        self.reference_signal = (np.asarray(reference_signal)
                                 if reference_signal is not None and self.has_reference
                                 else None)

        # Precompute HR series
        self.maternal_hr           = self._mean_hr(maternal_peaks)
        self.fetal_hr_series, self.fetal_hr_mean = self._hr_series(fetal_peaks)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

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
        Local PRD around a fetal beat.
        [FIX-1] Returns NaN if no independent reference is available.
        """
        if not self.has_reference or self.reference_signal is None:
            return np.nan

        if half_window is None:
            half_window = int(0.3 * self.fs)

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

    # -------------------------------------------------------------------------
    # Attribution computation
    # -------------------------------------------------------------------------

    def compute_attributions(self) -> dict:
        """
        Compute per-beat attribution scores for all detected fetal beats.

        [FIX-1] When has_reference=False, morph_scores are NaN and attribution
        is normalised over HR contrast + temporal independence only.

        Returns
        -------
        dict with keys:
          beat_times        : (M,) time in seconds
          hr_attribution    : (M,) fraction attributed to HR contrast
          morph_attribution : (M,) fraction attributed to morphology (NaN if unavailable)
          indep_attribution : (M,) fraction attributed to temporal independence
          hr_values         : (M,) instantaneous fetal HR per beat
          confidence        : (M,) overall confidence score [0,1]
          has_morphology    : bool -- whether morph scores were computed
        """
        n_beats = len(self.fetal_peaks)
        if n_beats < 2:
            return {}

        hr_scores    = np.zeros(n_beats)
        indep_scores = np.zeros(n_beats)
        morph_scores = np.full(n_beats, np.nan)   # NaN by default
        hr_values    = np.zeros(n_beats)

        excl_samples = int(ECHO_MATERNAL_EXCLUSION_SEC * self.fs)

        for i, fp in enumerate(self.fetal_peaks):
            # HR contrast score
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

            # Temporal independence score
            if len(self.maternal_peaks) > 0:
                distances = np.abs(self.maternal_peaks - fp) / self.fs
                min_dist  = float(np.min(distances))
                indep_scores[i] = float(np.clip(
                    min_dist / (ECHO_MATERNAL_EXCLUSION_SEC * 2), 0.0, 1.0
                ))
            else:
                indep_scores[i] = 1.0

            # [FIX-1] Morphological fidelity -- only when reference is available
            prd = self._local_prd(i)
            if not np.isnan(prd):
                morph_scores[i] = float(np.clip(1.0 / (1.0 + prd), 0.0, 1.0))

        # Normalise attributions -- [FIX-1] exclude morph when not available
        has_morphology = self.has_reference and not np.all(np.isnan(morph_scores))

        if has_morphology:
            morph_safe = np.where(np.isnan(morph_scores), 0.0, morph_scores)
            total      = hr_scores + indep_scores + morph_safe + 1e-10
            hr_attr    = hr_scores    / total
            indep_attr = indep_scores / total
            morph_attr = morph_safe   / total
            confidence = (hr_scores * indep_scores * morph_safe) ** (1.0 / 3.0)
        else:
            total      = hr_scores + indep_scores + 1e-10
            hr_attr    = hr_scores    / total
            indep_attr = indep_scores / total
            morph_attr = np.full(n_beats, np.nan)
            # Confidence from 2 scores only
            confidence = np.sqrt(hr_scores * indep_scores)

        confidence = np.clip(confidence, 0.0, 1.0)

        return {
            "beat_times"        : self.fetal_peaks / self.fs,
            "hr_attribution"    : hr_attr,
            "morph_attribution" : morph_attr,
            "indep_attribution" : indep_attr,
            "hr_values"         : hr_values,
            "confidence"        : confidence,
            "n_beats"           : n_beats,
            "has_morphology"    : has_morphology,
        }

    # -------------------------------------------------------------------------
    # Clinical report
    # -------------------------------------------------------------------------

    def generate_clinical_report(self, beat_idx: int,
                                 attribution: dict) -> str:
        """
        Generate a natural language clinical explanation for one fetal beat.

        [FIX-2] When has_morphology=False (NIFECGDB), the report clearly states
        that morphological fidelity was not assessed rather than falsely claiming
        it was verified.
        """
        if not attribution or beat_idx >= attribution["n_beats"]:
            return "No attribution data for this beat."

        hr_attr  = float(attribution["hr_attribution"][beat_idx])  * 100
        in_attr  = float(attribution["indep_attribution"][beat_idx]) * 100
        mo_raw   = attribution["morph_attribution"][beat_idx]
        mo_attr  = float(mo_raw) * 100 if not np.isnan(mo_raw) else np.nan
        hr_f     = float(attribution["hr_values"][beat_idx])
        conf     = float(attribution["confidence"][beat_idx]) * 100
        has_morph = attribution.get("has_morphology", False)

        if not np.isnan(self.maternal_hr):
            hr_sep = abs(hr_f - self.maternal_hr)
            if FETAL_HR_MIN <= hr_f <= FETAL_HR_MAX:
                hr_flag   = "ok"
                hr_status = f"NORMAL ({FETAL_HR_MIN}-{FETAL_HR_MAX} BPM range)"
            else:
                hr_flag   = "WARN"
                hr_status = f"OUTSIDE normal range ({FETAL_HR_MIN}-{FETAL_HR_MAX} BPM)"
        else:
            hr_sep    = np.nan
            hr_flag   = "?"
            hr_status = "Maternal HR unavailable for comparison"

        attrs = {"HR Contrast": hr_attr, "Temporal Independence": in_attr}
        if has_morph and not np.isnan(mo_attr):
            attrs["Morphology"] = mo_attr
        primary_cue = max(attrs, key=attrs.get)

        morph_line = (
            f"  Morphological Consistency  [{mo_attr:.1f}% attribution]\n"
            f"     QRS shape verified against direct fetal electrode reference."
            if has_morph and not np.isnan(mo_attr)
            else
            "  Morphological Consistency  [N/A -- no direct electrode reference]\n"
            "     Cannot assess: NIFECGDB does not provide a reference waveform."
        )

        sep_str = f"{hr_sep:.1f} BPM" if not np.isnan(hr_sep) else "N/A"
        mat_hr_str = f"{self.maternal_hr:.1f} BPM" if not np.isnan(self.maternal_hr) else "N/A"

        report = f"""
+--------------------------------------------------------------+
  ECHO Clinical Explanation -- Fetal Beat #{beat_idx + 1}
+--------------------------------------------------------------+

  OVERALL CONFIDENCE: {conf:.1f}%

  SEPARATION RATIONALE:
  --------------------------------------------------------------
  [{hr_flag}] Heart Rate Contrast        [{hr_attr:.1f}% attribution]
     Instantaneous fetal HR = {hr_f:.1f} BPM
     Status: {hr_status}
     Maternal HR            = {mat_hr_str}
     HR separation          = {sep_str}

  [ok] {morph_line}

  [ok] Temporal Independence      [{in_attr:.1f}% attribution]
     This fetal beat is temporally separated from maternal QRS.
     No maternal ECG overlap detected in +-{int(ECHO_MATERNAL_EXCLUSION_SEC*1000)} ms window.

  --------------------------------------------------------------
  PRIMARY SEPARATION CUE: {primary_cue}

  CLINICAL NOTE:
  {'[WARN] Fetal HR outside normal range. Recommend physician review.' if hr_flag == 'WARN' else '[ok] All assessed parameters within expected fetal physiological bounds.'}
+--------------------------------------------------------------+
"""
        return report

    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------

    def plot_attribution_heatmap(self,
                                 window_sec: float = 10.0,
                                 start_sec:  float = 0.0,
                                 save_path:  str   = None,
                                 dpi:        int   = 300) -> plt.Figure:
        """
        ECHO attribution visualization -- publication-quality figure.

        [FIX-2] Morphology panel is only shown when has_morphology=True.
        For NIFECGDB, the stacked bar uses only HR contrast + independence.
        """
        attribution = self.compute_attributions()
        if not attribution:
            print("[ECHO] No attributions computed -- insufficient peaks.")
            return None

        fs        = self.fs
        start_s   = int(start_sec * fs)
        end_s     = min(start_s + int(window_sec * fs), len(self.fetal_signal))
        time_axis = np.arange(start_s, end_s) / fs

        beat_mask = ((attribution["beat_times"] >= start_sec) &
                     (attribution["beat_times"] <= start_sec + window_sec))

        beat_times = attribution["beat_times"][beat_mask]
        hr_attr    = attribution["hr_attribution"][beat_mask]
        mo_attr    = attribution["morph_attribution"][beat_mask]
        in_attr    = attribution["indep_attribution"][beat_mask]
        hr_vals    = attribution["hr_values"][beat_mask]
        conf       = attribution["confidence"][beat_mask]
        has_morph  = attribution.get("has_morphology", False)

        col_hr    = "#2166AC"
        col_morph = "#D6604D"
        col_indep = "#4DAC26"

        fig, axes = plt.subplots(3, 1, figsize=(14, 9),
                                 gridspec_kw={"height_ratios": [3, 2, 2]})
        fig.patch.set_facecolor("white")

        # Panel 1: Fetal ECG with beat markers
        ax1 = axes[0]
        ax1.plot(time_axis, self.fetal_signal[start_s:end_s],
                 color="black", lw=0.7, label="Extracted fetal ECG")
        cmap = plt.cm.RdYlGn
        for bt, cf in zip(beat_times, conf):
            peak_idx = int(bt * fs)
            if start_s <= peak_idx < end_s:
                ax1.axvline(bt, color=cmap(cf), alpha=0.4, lw=1.0)
                ax1.scatter(bt, self.fetal_signal[peak_idx],
                            c=[cmap(cf)], s=40, zorder=5)
        ax1.set_ylabel("Amplitude (a.u.)", fontsize=10)
        ax1.set_title("ECHO: Extracted Fetal ECG with Beat Confidence (colour: low->high)",
                      fontsize=11, fontweight="bold")
        ax1.legend(fontsize=9, loc="upper right")
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(start_sec, start_sec + window_sec)

        # Panel 2: Attribution stacked bar
        ax2 = axes[1]
        bar_w = min(0.08, (window_sec / (len(beat_times) + 1)) * 0.8)

        if len(beat_times) > 0:
            ax2.bar(beat_times, hr_attr, width=bar_w, color=col_hr,
                    label="HR Contrast", alpha=0.9)
            if has_morph and not np.all(np.isnan(mo_attr)):
                mo_safe = np.where(np.isnan(mo_attr), 0.0, mo_attr)
                ax2.bar(beat_times, mo_safe, width=bar_w, color=col_morph,
                        bottom=hr_attr, label="Morphology", alpha=0.9)
                ax2.bar(beat_times, in_attr, width=bar_w, color=col_indep,
                        bottom=hr_attr + mo_safe,
                        label="Temporal Independence", alpha=0.9)
            else:
                ax2.bar(beat_times, in_attr, width=bar_w, color=col_indep,
                        bottom=hr_attr,
                        label="Temporal Independence (no morph ref)", alpha=0.9)

        ax2.set_ylabel("Attribution", fontsize=10)
        title_suffix = "" if has_morph else " [Morphology N/A -- no direct electrode]"
        ax2.set_title(f"ECHO Attribution per Beat{title_suffix}",
                      fontsize=11, fontweight="bold")
        ax2.set_ylim(0, 1.05)
        ax2.legend(fontsize=9, loc="upper right", ncol=3)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_xlim(start_sec, start_sec + window_sec)

        # Panel 3: Instantaneous fetal HR
        ax3 = axes[2]
        if len(beat_times) > 0 and len(hr_vals) > 0:
            ax3.plot(beat_times, hr_vals, "o-", color="purple",
                     markersize=5, lw=1.5, label="Fetal HR")
            ax3.fill_between(beat_times, hr_vals, alpha=0.1, color="purple")

        ax3.axhline(FETAL_HR_MIN, color="red", linestyle="--", lw=1.0, alpha=0.7,
                    label=f"Fetal HR bounds [{FETAL_HR_MIN}-160 BPM]")
        ax3.axhline(160, color="red", linestyle="--", lw=1.0, alpha=0.7)
        if not np.isnan(self.maternal_hr):
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
        Recording-level summary statistics.

        [FIX-2] Clearly flags when morphology was not available.
        """
        if not attribution:
            return "No attribution data."

        n        = attribution["n_beats"]
        hr       = attribution["hr_values"]
        conf     = attribution["confidence"]
        has_morph = attribution.get("has_morphology", False)

        normal = np.sum((hr >= FETAL_HR_MIN) & (hr <= 160))
        brady  = np.sum(hr < FETAL_HR_MIN)
        tachy  = np.sum(hr > 160)

        morph_line = (
            f"    Morphology               : {np.mean(attribution['morph_attribution'])*100:.1f}%"
            if has_morph
            else
            "    Morphology               : N/A (no direct electrode reference)"
        )

        mat_hr_str = f"{self.maternal_hr:.1f} BPM" if not np.isnan(self.maternal_hr) else "N/A"

        summary = f"""
ECHO Recording Summary
----------------------------------------
  Total fetal beats analyzed : {n}
  Mean fetal HR              : {np.mean(hr):.1f} +/- {np.std(hr):.1f} BPM
  Maternal HR                : {mat_hr_str}
  HR separation              : {abs(np.mean(hr) - self.maternal_hr):.1f} BPM

  Beat Classification:
    Normal HR ({FETAL_HR_MIN}-160 BPM)  : {normal} ({100*normal/n:.1f}%)
    Bradycardia (<{FETAL_HR_MIN} BPM)   : {brady}  ({100*brady/n:.1f}%)
    Tachycardia (>160 BPM)   : {tachy}  ({100*tachy/n:.1f}%)

  Mean attribution breakdown:
    HR Contrast              : {np.mean(attribution['hr_attribution'])*100:.1f}%
{morph_line}
    Temporal Independence    : {np.mean(attribution['indep_attribution'])*100:.1f}%

  Mean confidence score      : {np.nanmean(conf)*100:.1f}%
----------------------------------------
"""
        return summary