"""
data/loader.py
Load ADFECGDB and NIFECGDB EDF recordings.

REFACTORED: Now uses the new dataset_handlers module for dataset handling.
Old functions (load_edf, load_all_recordings, load_nifecgdb_edf, etc.) are
kept for backward compatibility but delegate to the new dataset handlers.

Recording dict schema (both datasets)
--------------------------------------
  recording     : str        — filename stem (e.g. 'r01' or 'ecgca444')
  dataset       : str        — 'ADFECGDB' or 'NIFECGDB'
  fs            : int        — sampling rate (Hz)
  duration_sec  : float
  abdomen       : ndarray (4, N) — abdominal channels (zero-padded if <4)
  direct        : ndarray (N,) | None — direct fetal ECG (ADFECGDB only)
  labels        : list[str]
  annotation_path : str | None — path to .edf.qrs file if present

For new code, use the dataset_handlers module directly:
    from dataset_handlers import get_dataset
    handler = get_dataset("adfecgdb")
    recordings = handler.load_all_recordings("/path/to/data")
"""

from dataset_handlers import get_dataset


# ── ADFECGDB — Backward compatibility wrappers ─────────────────────────────

def load_edf(filepath: str) -> dict:
    """
    Load one ADFECGDB EDF file.
    
    BACKWARD COMPATIBILITY: Use dataset_handlers.get_dataset("adfecgdb") instead.
    """
    handler = get_dataset("adfecgdb")
    return handler.load_single_recording(filepath)


def load_all_recordings(data_dir: str, max_recordings: int = None) -> list:
    """
    Load all ADFECGDB recordings from a directory.
    
    BACKWARD COMPATIBILITY: Use dataset_handlers.get_dataset("adfecgdb") instead.
    """
    handler = get_dataset("adfecgdb")
    return handler.load_all_recordings(data_dir, max_recordings=max_recordings)


# ── NIFECGDB — Backward compatibility wrappers ─────────────────────────────

def load_nifecgdb_edf(filepath: str) -> dict:
    """
    Load one NIFECGDB EDF file.
    
    BACKWARD COMPATIBILITY: Use dataset_handlers.get_dataset("nifecgdb") instead.
    """
    handler = get_dataset("nifecgdb")
    return handler.load_single_recording(filepath)


def load_all_nifecgdb(data_dir: str, max_recordings: int = None) -> list:
    """
    Load all NIFECGDB recordings from a directory.
    
    BACKWARD COMPATIBILITY: Use dataset_handlers.get_dataset("nifecgdb") instead.
    """
    handler = get_dataset("nifecgdb")
    return handler.load_all_recordings(data_dir, max_recordings=max_recordings)


# ── Shared helpers ──────────────────────────────────────────────────────────

def print_recording_summary(rec: dict) -> None:
    """
    Print a human-readable summary of a loaded recording.
    
    BACKWARD COMPATIBILITY: Use handler.print_recording_summary() instead.
    """
    handler = get_dataset(rec.get("dataset", "adfecgdb"))
    handler.print_recording_summary(rec)

