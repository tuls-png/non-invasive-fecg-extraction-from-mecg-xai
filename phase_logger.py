"""
phase_logger.py
Structured per-recording diagnostic logger for PHASE pipeline.

Place this file in the root of your repo (same level as pipeline.py).

Usage in run_experiment_new.py:
    from phase_logger import PhaseLogger
    logger = PhaseLogger(log_dir="logs")

    # After pipeline.run() returns result:
    logger.log_recording(result["recording"], result)

    # After all recordings:
    logger.close()

The logger writes two files per run:
  logs/phase_debug_<timestamp>.jsonl   -- one JSON line per recording
  logs/phase_debug_<timestamp>.txt     -- human-readable summary

CHANGES FROM ORIGINAL:
  [FIX-1] _format_entry() now uses a safe helper _f() that handles missing
          keys and '?' fallbacks without crashing on :.1f / :.5f / :.4f /
          :.3f / :.2f format specs. The root cause was:
              e.get('key', '?')  →  returns str '?'
              f"{...:.1f}"       →  ValueError on str
          Every numeric field is now formatted through _f() which returns
          a plain string, so the outer f-string needs no format spec.

  [FIX-2] log_recording() now accepts the full pipeline result dict
          (not a separate debug_log sub-dict). It extracts all fields it
          needs from result["metadata"], result["metrics"], and the top-level
          result keys. This means no changes to pipeline.py are required —
          the logger works with the result dict returned by pipeline.run()
          as-is.

  [FIX-3] Added safe float conversion throughout so numpy scalars, Python
          floats, ints, NaN, and None all render cleanly.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime


class PhaseLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.jsonl_path = self.log_dir / f"phase_debug_{ts}.jsonl"
        self.txt_path   = self.log_dir / f"phase_debug_{ts}.txt"
        self._jsonl = open(self.jsonl_path, "w")
        self._txt   = open(self.txt_path,  "w")
        self._txt.write(f"PHASE Debug Log — {datetime.now().isoformat()}\n")
        self._txt.write("=" * 80 + "\n\n")
        print(f"[PhaseLogger] Writing to:\n  {self.jsonl_path}\n  {self.txt_path}")

    def log_recording(self, rec_id: str, result: dict):
        """
        Call once per recording with the full pipeline result dict.

        Accepts either:
          - result dict returned directly by pipeline.run()
          - a flat debug_log dict (backward compatible)

        Fields are extracted from result["metadata"] and result["metrics"]
        if present, otherwise read from the flat dict directly.
        """
        entry = _extract_entry(rec_id, result)
        entry["recording"] = rec_id
        entry["timestamp"] = datetime.now().isoformat()

        self._jsonl.write(json.dumps(entry, default=_json_default) + "\n")
        self._jsonl.flush()
        self._txt.write(_format_entry(rec_id, entry))
        self._txt.flush()

    def close(self):
        self._jsonl.close()
        self._txt.close()
        print(f"[PhaseLogger] Closed. Files written to {self.log_dir}/")


# ── Field extraction ──────────────────────────────────────────────────────────

def _extract_entry(rec_id: str, result: dict) -> dict:
    """
    Extract a flat diagnostic dict from the pipeline result.

    Works whether result is:
      (a) the full pipeline.run() return dict (has "metadata" and "metrics" keys)
      (b) a flat dict already (old-style debug_log, or empty {})
    """
    meta    = result.get("metadata", {}) or {}
    metrics = result.get("metrics",  {}) or {}

    # Maternal
    mat_peaks = result.get("maternal_peaks", np.array([]))
    mat_hr    = _safe_mean_hr(mat_peaks, result.get("fs", 1000))

    # Path A — from metadata
    a_idx      = meta.get("path_a_selected_ic_index",    result.get("a_ic_idx"))
    a_n        = meta.get("path_a_selected_ic_peak_count", result.get("a_n_peaks"))
    a_hr       = meta.get("path_a_selected_ic_hr_bpm",   result.get("a_hr"))
    a_valid    = meta.get("path_a_selected_ic_is_valid",  result.get("a_valid"))
    a_score    = meta.get("path_a_selected_ic_score",     result.get("a_score"))
    a_stab     = meta.get("path_a_selected_ic_stability", result.get("a_stability"))

    # Path B — from metadata
    b_idx      = meta.get("path_b_selected_ic_index",    result.get("b_ic_idx"))
    b_n        = meta.get("path_b_selected_ic_peak_count", result.get("b_n_peaks"))
    b_hr       = meta.get("path_b_selected_ic_hr_bpm",   result.get("b_hr"))
    b_valid    = meta.get("path_b_selected_ic_is_valid",  result.get("b_valid"))
    b_score    = meta.get("path_b_selected_ic_score",     result.get("b_score"))
    b_stab     = meta.get("path_b_selected_ic_stability", result.get("b_stability"))

    chosen     = meta.get("chosen_path_description",      result.get("chosen_path"))
    chosen_sc  = meta.get("chosen_ic_selection_score",    result.get("a_score"))
    low_conf   = meta.get("low_confidence",               False)
    sparse_ann = meta.get("sparse_annotation",            False)

    # EKF
    fetal_peaks_pre  = result.get("fetal_ecg_pre",  None)
    fetal_peaks_post = result.get("fetal_peaks",    np.array([]))
    fs               = result.get("fs", 1000)
    n_pre  = result.get("ekf_n_pre",  None)
    n_post = len(fetal_peaks_post) if hasattr(fetal_peaks_post, '__len__') else None

    # Final metrics
    f1      = metrics.get("F1",          result.get("f1"))
    se      = metrics.get("Se",          result.get("se"))
    ppv     = metrics.get("PPV",         result.get("ppv"))
    fhr_mae = metrics.get("FHR_MAE_bpm", result.get("fhr_mae"))
    n_det   = metrics.get("n_detected",  result.get("final_n_peaks"))
    n_ref   = metrics.get("n_reference", None)

    final_hr = _safe_mean_hr(fetal_peaks_post, fs)

    return {
        # maternal
        "n_maternal_peaks"  : _safe_int(len(mat_peaks) if hasattr(mat_peaks, '__len__') else None),
        "maternal_hr"       : _safe_float(mat_hr),
        # path a
        "a_ic_idx"          : _safe_int(a_idx),
        "a_n_peaks"         : _safe_int(a_n),
        "a_hr"              : _safe_float(a_hr),
        "a_valid"           : a_valid,
        "a_score"           : _safe_float(a_score),
        "a_stability"       : _safe_float(a_stab),
        "a_spectral"        : _safe_float(result.get("a_spectral")),
        "a_robust_hr"       : _safe_float(result.get("a_robust_hr")),
        "a_used_spectral_fallback": result.get("a_used_spectral_fallback"),
        "a_combined_score"  : _safe_float(a_score),
        # path b
        "b_ic_idx"          : _safe_int(b_idx),
        "b_n_peaks"         : _safe_int(b_n),
        "b_hr"              : _safe_float(b_hr),
        "b_valid"           : b_valid,
        "b_score"           : _safe_float(b_score),
        "b_stability"       : _safe_float(b_stab),
        "b_spectral"        : _safe_float(result.get("b_spectral")),
        "b_robust_hr"       : _safe_float(result.get("b_robust_hr")),
        "b_used_spectral_fallback": result.get("b_used_spectral_fallback"),
        "b_combined_score"  : _safe_float(b_score),
        # path selection
        "chosen_path"       : chosen,
        "chosen_score"      : _safe_float(chosen_sc),
        "low_confidence"    : low_conf,
        "sparse_annotation" : sparse_ann,
        # ekf
        "ekf_n_pre"         : _safe_int(n_pre),
        "ekf_n_post"        : _safe_int(n_post),
        "ekf_peak_ratio"    : _safe_float(result.get("ekf_peak_ratio")),
        "ekf_cc"            : _safe_float(result.get("ekf_cc")),
        "ekf_rr_shift_ms"   : _safe_float(result.get("ekf_rr_shift_ms")),
        "ekf_gate_passed"   : result.get("ekf_gate_passed"),
        "ekf_used"          : result.get("ekf_used"),
        # final
        "final_n_peaks"     : _safe_int(n_det),
        "n_reference"       : _safe_int(n_ref),
        "final_hr"          : _safe_float(final_hr),
        "f1"                : _safe_float(f1),
        "se"                : _safe_float(se),
        "ppv"               : _safe_float(ppv),
        "fhr_mae"           : _safe_float(fhr_mae),
    }


# ── Safe type helpers ─────────────────────────────────────────────────────────

def _safe_float(v) -> float | None:
    """Convert any numeric-ish value to Python float, or None if not available."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if (f != f) else f   # NaN → None
    except (TypeError, ValueError):
        return None


def _safe_int(v) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _safe_mean_hr(peaks, fs: int = 1000) -> float | None:
    """Compute mean HR from peak array, returns None if not possible."""
    if peaks is None:
        return None
    arr = np.asarray(peaks) if not isinstance(peaks, np.ndarray) else peaks
    if len(arr) < 2:
        return None
    rr = np.diff(arr) / fs
    rr = rr[rr > 0]
    if len(rr) == 0:
        return None
    return float(np.mean(60.0 / rr))


# ── Formatting ────────────────────────────────────────────────────────────────

def _f(value, fmt: str = ".1f", missing: str = "N/A") -> str:
    """
    [FIX-1] Safe numeric formatter.

    Converts value to float and formats it, returning `missing` for None,
    NaN, or anything that cannot be converted. This prevents the
    ValueError that occurred when '?' was passed to :.1f.
    """
    if value is None:
        return missing
    try:
        f = float(value)
        if f != f:   # NaN
            return missing
        return format(f, fmt)
    except (TypeError, ValueError):
        return str(value)


def _b(value) -> str:
    """Format a boolean or None as YES / NO / N/A."""
    if value is None:
        return "N/A"
    return "YES" if value else "NO"


def _format_entry(rec_id: str, e: dict) -> str:
    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"Recording : {rec_id}")
    lines.append(f"{'='*60}")

    # Maternal
    lines.append(f"  Maternal peaks : {e.get('n_maternal_peaks', 'N/A')}")
    lines.append(f"  Maternal HR    : {_f(e.get('maternal_hr'))} BPM")

    # Path A
    lines.append("")
    lines.append("  --- Path A (ICA1 direct) ---")
    lines.append(f"  IC index       : {e.get('a_ic_idx', 'N/A')}")
    lines.append(f"  Peaks          : {e.get('a_n_peaks', 'N/A')}")
    lines.append(f"  HR (mean)      : {_f(e.get('a_hr'))} BPM")
    lines.append(f"  HR (robust)    : {_f(e.get('a_robust_hr'))} BPM")
    lines.append(f"  Valid HR gate  : {_b(e.get('a_valid'))}")
    lines.append(f"  Unified score  : {_f(e.get('a_score'), '.5f')}")
    lines.append(f"  Spectral score : {_f(e.get('a_spectral'), '.4f')}")
    lines.append(f"  Stability      : {_f(e.get('a_stability'), '.3f')}")
    lines.append(f"  Spectral fb    : {_b(e.get('a_used_spectral_fallback'))}")

    # Path B
    lines.append("")
    lines.append("  --- Path B (WSVD + ICA2) ---")
    lines.append(f"  IC index       : {e.get('b_ic_idx', 'N/A')}")
    lines.append(f"  Peaks          : {e.get('b_n_peaks', 'N/A')}")
    lines.append(f"  HR (mean)      : {_f(e.get('b_hr'))} BPM")
    lines.append(f"  HR (robust)    : {_f(e.get('b_robust_hr'))} BPM")
    lines.append(f"  Valid HR gate  : {_b(e.get('b_valid'))}")
    lines.append(f"  Unified score  : {_f(e.get('b_score'), '.5f')}")
    lines.append(f"  Spectral score : {_f(e.get('b_spectral'), '.4f')}")
    lines.append(f"  Stability      : {_f(e.get('b_stability'), '.3f')}")
    lines.append(f"  Spectral fb    : {_b(e.get('b_used_spectral_fallback'))}")

    # Path selection
    lines.append("")
    lines.append("  --- Path Selection ---")
    lines.append(f"  Chosen         : {e.get('chosen_path', 'N/A')}")
    lines.append(f"  Combined A     : {_f(e.get('a_combined_score'), '.5f')}")
    lines.append(f"  Combined B     : {_f(e.get('b_combined_score'), '.5f')}")
    lines.append(f"  Chosen score   : {_f(e.get('chosen_score'), '.5f')}")
    lines.append(f"  Low confidence : {_b(e.get('low_confidence'))}")
    lines.append(f"  Sparse annot.  : {_b(e.get('sparse_annotation'))}")

    # EKF
    lines.append("")
    lines.append("  --- EKF ---")
    lines.append(f"  Pre-EKF peaks  : {e.get('ekf_n_pre', 'N/A')}")
    lines.append(f"  Post-EKF peaks : {e.get('ekf_n_post', 'N/A')}")
    lines.append(f"  Peak ratio     : {_f(e.get('ekf_peak_ratio'), '.3f')} (gate: >=0.70)")
    lines.append(f"  CC             : {_f(e.get('ekf_cc'), '.4f')} (gate: >=0.60)")
    lines.append(f"  RR shift (ms)  : {_f(e.get('ekf_rr_shift_ms'), '.2f')} (gate: <=15ms)")
    lines.append(f"  Gate passed    : {_b(e.get('ekf_gate_passed'))}")
    lines.append(f"  EKF used       : {_b(e.get('ekf_used'))}")

    # Final
    lines.append("")
    lines.append("  --- Final ---")
    lines.append(f"  Final peaks    : {e.get('final_n_peaks', 'N/A')}")
    lines.append(f"  Reference peaks: {e.get('n_reference', 'N/A')}")
    lines.append(f"  Final HR       : {_f(e.get('final_hr'))} BPM")
    lines.append(f"  F1             : {_f(e.get('f1'), '.2f')}")
    lines.append(f"  Se             : {_f(e.get('se'), '.2f')}")
    lines.append(f"  PPV            : {_f(e.get('ppv'), '.2f')}")
    lines.append(f"  FHR MAE        : {_f(e.get('fhr_mae'), '.2f')} BPM")
    lines.append("")

    return "\n".join(lines) + "\n"


# ── JSON serialisation ────────────────────────────────────────────────────────

def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64, np.floating)):
        f = float(obj)
        return None if (f != f) else f
    if isinstance(obj, (np.int32, np.int64, np.integer)):
        return int(obj)
    return str(obj)