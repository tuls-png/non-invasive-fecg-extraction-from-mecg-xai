"""
utils/visualization.py
Publication-quality figures for the paper.

IEEE Access formatting: 300 DPI, colorblind-safe palette, consistent fonts.

FIX: plot_ablation_results() previously tried to read error bars from
ablation_data.get("_std", {}) which was never populated by run_ablation().
Error bars are now passed as a separate optional std_data dict.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

COLORS = {
    "raw"       : "#5B7FA6",
    "filtered"  : "#D6604D",
    "maternal"  : "#F4A460",
    "fetal_est" : "#4DAC26",
    "fetal_ref" : "#9467BD",
    "residual"  : "#8C564B",
    "peaks_ref" : "#D62728",
    "peaks_est" : "#2CA02C",
}

FIG_DPI    = 300
FONT_LABEL = 11
FONT_TITLE = 12
FONT_TICK  = 9


def _apply_style():
    plt.rcParams.update({
        "font.family"      : "sans-serif",
        "font.size"        : FONT_TICK,
        "axes.labelsize"   : FONT_LABEL,
        "axes.titlesize"   : FONT_TITLE,
        "figure.dpi"       : 100,
        "savefig.dpi"      : FIG_DPI,
        "axes.spines.top"  : False,
        "axes.spines.right": False,
        "axes.grid"        : True,
        "grid.alpha"       : 0.3,
        "grid.linestyle"   : "--",
    })


def plot_preprocessing(raw: np.ndarray, processed: np.ndarray,
                        fs: int, start_sec: float = 5, duration_sec: float = 5,
                        save_path: str = None) -> plt.Figure:
    """Figure 1: Raw vs preprocessed abdominal signal."""
    _apply_style()
    s = int(start_sec * fs)
    e = s + int(duration_sec * fs)
    t = np.arange(s, e) / fs

    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].plot(t, raw[s:e], color=COLORS["raw"], lw=0.8)
    axes[0].set_ylabel("Amplitude (a.u.)")
    axes[0].set_title("(a) Raw abdominal signal (baseline drift visible)")
    axes[1].plot(t, processed[s:e], color=COLORS["filtered"], lw=0.8)
    axes[1].set_ylabel("Amplitude (a.u.)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title("(b) Preprocessed signal (bandpass 1-45 Hz, notch 50 Hz)")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    return fig


def plot_maternal_cancellation(abdomen: np.ndarray,
                                maternal_recon: np.ndarray,
                                residual: np.ndarray,
                                fs: int,
                                start_sec: float = 0,
                                duration_sec: float = 8,
                                channel: int = 0,
                                save_path: str = None) -> plt.Figure:
    """Figure 2: Three-panel showing abdominal -> maternal recon -> residual."""
    _apply_style()
    s = int(start_sec * fs)
    e = s + int(duration_sec * fs)
    t = np.arange(s, e) / fs

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(t, abdomen[channel, s:e], color=COLORS["raw"], lw=0.7)
    axes[0].set_ylabel("Amplitude (a.u.)")
    axes[0].set_title(f"(a) Abdominal signal — Channel {channel+1} (maternal + fetal + noise)")
    axes[1].plot(t, maternal_recon[channel, s:e], color=COLORS["maternal"], lw=0.7)
    axes[1].set_ylabel("Amplitude (a.u.)")
    axes[1].set_title("(b) Maternal ECG reconstruction (AW-WSVD)")
    axes[2].plot(t, residual[channel, s:e], color=COLORS["fetal_est"], lw=0.7)
    axes[2].set_ylabel("Amplitude (a.u.)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("(c) Residual after maternal cancellation (fetal ECG estimate)")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    return fig


def plot_fetal_comparison(fetal_estimated: np.ndarray,
                           fetal_reference: np.ndarray,
                           peaks_est: np.ndarray,
                           peaks_ref: np.ndarray,
                           fs: int,
                           start_sec: float = 0,
                           duration_sec: float = 10,
                           save_path: str = None) -> plt.Figure:
    """Figure 3: Estimated vs reference fetal ECG with QRS markers."""
    _apply_style()
    s = int(start_sec * fs)
    e = s + int(duration_sec * fs)
    t = np.arange(s, e) / fs

    cc = np.corrcoef(fetal_estimated, fetal_reference)[0, 1]
    sign = -1.0 if cc < 0 else 1.0
    fetal_est_plot = sign * fetal_estimated

    def norm(x):
        r = np.max(x) - np.min(x)
        return (x - np.min(x)) / (r + 1e-12)

    est_n = norm(fetal_est_plot[s:e])
    ref_n = norm(fetal_reference[s:e])

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, ref_n, color=COLORS["fetal_ref"], lw=1.0,
            label="Direct fetal ECG (reference)", alpha=0.85)
    ax.plot(t, est_n, color=COLORS["fetal_est"], lw=0.8,
            label="PHASE extracted fetal ECG", alpha=0.85, linestyle="--")

    ref_in = [p for p in peaks_ref if s <= p < e]
    est_in = [p for p in peaks_est if s <= p < e]

    ax.scatter(np.array(ref_in)/fs, ref_n[np.array(ref_in)-s],
               color=COLORS["peaks_ref"], marker="o", s=30, zorder=5,
               label="Reference QRS peaks")
    ax.scatter(np.array(est_in)/fs, est_n[np.array(est_in)-s],
               color=COLORS["peaks_est"], marker="x", s=40, zorder=5,
               label="Detected QRS peaks")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized amplitude")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(start_sec, start_sec + duration_sec)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    return fig


def plot_ablation_results(ablation_data: dict,
                           metric: str = "F1",
                           std_data: dict = None,
                           save_path: str = None) -> plt.Figure:
    """
    Figure 4: Ablation study bar chart with optional error bars.

    FIX: Error bars are now passed as a separate std_data dict instead of
    being silently looked up from ablation_data["_std"] (which was never
    populated by run_ablation(), so error bars were never shown).

    Parameters
    ----------
    ablation_data : dict mapping configuration name to mean metric value
    metric        : metric name for y-axis label
    std_data      : optional dict mapping configuration name to std value
                    (same keys as ablation_data). If None, no error bars.
    """
    _apply_style()

    labels = list(ablation_data.keys())
    values = list(ablation_data.values())
    n      = len(labels)
    colors = plt.cm.Blues(np.linspace(0.35, 0.85, n))

    fig, ax = plt.subplots(figsize=(10, 5))

    if std_data is not None:
        stds = [std_data.get(k, 0.0) for k in labels]
        bars = ax.bar(range(n), values, color=colors, edgecolor="white",
                      linewidth=0.5, width=0.6,
                      yerr=stds, capsize=4, error_kw={"elinewidth": 1.5})
    else:
        bars = ax.bar(range(n), values, color=colors, edgecolor="white",
                      linewidth=0.5, width=0.6)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(f"{metric} Score (%)" if "F1" in metric or "Se" in metric else metric)
    ax.set_ylim(0, max(values) * 1.15)
    ax.set_title(f"Ablation Study — {metric}")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    return fig


def plot_sota_comparison(methods: dict,
                          metrics: list = None,
                          save_path: str = None) -> plt.Figure:
    """Figure 5: Multi-metric bar chart comparing PHASE against baselines."""
    _apply_style()
    if metrics is None:
        metrics = ["Se", "PPV", "F1"]

    method_names = list(methods.keys())
    n_methods    = len(method_names)
    n_metrics    = len(metrics)
    x            = np.arange(n_metrics)
    w            = 0.8 / n_methods
    palette      = ["#5B7FA6", "#D6604D", "#4DAC26", "#9467BD", "#F4A460"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (name, vals) in enumerate(methods.items()):
        offsets  = x + (i - n_methods/2 + 0.5) * w
        bar_vals = [vals.get(m, 0) for m in metrics]
        ax.bar(offsets, bar_vals, width=w * 0.9,
               color=palette[i % len(palette)],
               label=name, alpha=0.9, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title("Method Comparison")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    return fig


def plot_ekf_refinement(before_ekf: np.ndarray,
                         after_ekf: np.ndarray,
                         reference: np.ndarray,
                         fs: int,
                         start_sec: float = 0,
                         duration_sec: float = 5,
                         save_path: str = None) -> plt.Figure:
    """Figure 6: Effect of EKF refinement on morphological detail."""
    _apply_style()
    s = int(start_sec * fs)
    e = s + int(duration_sec * fs)
    t = np.arange(s, e) / fs

    def norm(x):
        r = np.max(x) - np.min(x)
        return (x - np.min(x)) / (r + 1e-12)

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(t, norm(reference[s:e]),   color=COLORS["fetal_ref"], lw=0.9)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("(a) Reference direct fetal ECG")
    axes[1].plot(t, norm(before_ekf[s:e]),  color=COLORS["fetal_est"], lw=0.9)
    axes[1].set_ylabel("Amplitude")
    axes[1].set_title("(b) Fetal ECG estimate before EKF refinement (ICA output)")
    axes[2].plot(t, norm(after_ekf[s:e]),   color=COLORS["maternal"],  lw=0.9)
    axes[2].set_ylabel("Amplitude")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("(c) Fetal ECG after EKF-RTS refinement (improved morphology)")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    return fig
