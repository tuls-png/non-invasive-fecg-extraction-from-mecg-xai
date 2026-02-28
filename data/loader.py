"""
data/loader.py
Load ADFECGDB and NIFECGDB EDF recordings.

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
"""

import pyedflib
import numpy as np
from pathlib import Path
from config import (
    ADFECGDB_ABDOMEN_CHANNELS, ADFECGDB_DIRECT_CHANNEL,
    NIFECGDB_ABDOMINAL_PREFIX, NIFECGDB_THORACIC_PREFIX,
    NIFECGDB_MAX_ABD_CHANNELS, FS
)
from config_nifecgdb import (
    ADFECGDB_ABDOMEN_CHANNELS, ADFECGDB_DIRECT_CHANNEL,
    NIFECGDB_ABDOMINAL_PREFIX, NIFECGDB_THORACIC_PREFIX,
    NIFECGDB_MAX_ABD_CHANNELS, FS
)


# ── ADFECGDB ──────────────────────────────────────────────────────────────────

def load_edf(filepath: str) -> dict:
    """
    Load one ADFECGDB EDF file.
    Expects channels: Direct_1, Abdomen_1..4.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"EDF file not found: {filepath}")

    f       = pyedflib.EdfReader(str(filepath))
    labels  = f.getSignalLabels()
    fs_list = [int(f.getSampleFrequency(i)) for i in range(f.signals_in_file)]
    signals = np.vstack([f.readSignal(i) for i in range(f.signals_in_file)])
    f._close(); del f

    for ch in ADFECGDB_ABDOMEN_CHANNELS + [ADFECGDB_DIRECT_CHANNEL]:
        if ch not in labels:
            raise ValueError(
                f"Expected channel '{ch}' not found in {filepath.name}. "
                f"Available: {labels}"
            )

    abd_indices = [labels.index(ch) for ch in ADFECGDB_ABDOMEN_CHANNELS]
    direct_idx  = labels.index(ADFECGDB_DIRECT_CHANNEL)
    fs_rec      = fs_list[0]
    abdomen     = signals[abd_indices]   # (4, N)
    direct      = signals[direct_idx]   # (N,)
    N           = abdomen.shape[1]

    print(f"[Loader] {filepath.name} — {N} samples @ {fs_rec} Hz "
          f"({N/fs_rec:.1f} s), {len(labels)} channels")

    # Annotation file alongside EDF
    ann_path = filepath.parent / (filepath.name + ".qrs")

    return {
        "recording"      : filepath.stem,
        "dataset"        : "ADFECGDB",
        "labels"         : labels,
        "fs"             : fs_rec,
        "duration_sec"   : N / fs_rec,
        "abdomen"        : abdomen,
        "direct"         : direct,
        "annotation_path": str(ann_path) if ann_path.exists() else None,
    }


def load_all_recordings(data_dir: str) -> list[dict]:
    """Load every ADFECGDB EDF in a directory."""
    data_dir  = Path(data_dir)
    edf_files = sorted(data_dir.glob("*.edf"))
    if not edf_files:
        raise FileNotFoundError(f"No EDF files found in {data_dir}")

    recordings = []
    for f in edf_files:
        try:
            recordings.append(load_edf(str(f)))
        except Exception as e:
            print(f"[Loader] WARNING: Skipping {f.name} — {e}")

    print(f"[Loader] Loaded {len(recordings)} recordings from {data_dir}")
    return recordings


# ── NIFECGDB ──────────────────────────────────────────────────────────────────

def load_nifecgdb_edf(filepath: str) -> dict:
    """
    Load one NIFECGDB EDF file.

    Channel layout varies per record:
      - 2 thoracic channels (maternal ECG, labelled 'thoracic_*') — excluded
      - 3 or 4 abdominal channels (labelled 'channel_*') — used for ICA

    No direct fetal electrode exists. The .edf.qrs annotation file is
    the only source of ground-truth fetal peak timings.

    Returns a recording dict compatible with PHASEPipeline.run().
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"EDF file not found: {filepath}")

    f       = pyedflib.EdfReader(str(filepath))
    labels  = f.getSignalLabels()
    fs_list = [int(f.getSampleFrequency(i)) for i in range(f.signals_in_file)]
    signals = np.vstack([f.readSignal(i) for i in range(f.signals_in_file)])
    f._close(); del f

    # Identify abdominal vs thoracic channels.
    # Normalise labels to lowercase with underscores for robust matching.
    # Confirmed NIFECGDB labels: 'Thorax_1', 'Thorax_2', 'Abdomen_1' .. 'Abdomen_4'
    labels_lower = [l.lower().replace(' ', '_') for l in labels]

    abd_indices = [
        i for i, l in enumerate(labels_lower)
        if l.startswith(NIFECGDB_ABDOMINAL_PREFIX.lower())
    ]
    thoracic_indices = [
        i for i, l in enumerate(labels_lower)
        if l.startswith(NIFECGDB_THORACIC_PREFIX.lower())
    ]

    if not abd_indices:
        # Hard fallback: explicitly exclude known thoracic channels,
        # use everything else. Log a warning so the problem is visible.
        abd_indices = [i for i in range(len(labels)) if i not in thoracic_indices]
        print(f"[Loader] WARNING: no 'abdomen_*' channels found in {filepath.name}. "
              f"Labels: {labels}. Using non-thoracic channels: "
              f"{[labels[i] for i in abd_indices]}")

    fs_rec = fs_list[0]
    N      = signals.shape[1]

    # Build (4, N) abdominal array — zero-pad if fewer than 4 channels
    n_abd    = min(len(abd_indices), NIFECGDB_MAX_ABD_CHANNELS)
    abdomen  = np.zeros((NIFECGDB_MAX_ABD_CHANNELS, N), dtype=np.float64)
    for k, idx in enumerate(abd_indices[:n_abd]):
        abdomen[k] = signals[idx]

    print(f"[Loader] {filepath.name} — {N} samples @ {fs_rec} Hz "
          f"({N/fs_rec:.1f} s), {n_abd} abdominal / "
          f"{len(thoracic_indices)} thoracic channels")

    ann_path = filepath.parent / (filepath.name + ".qrs")

    return {
        "recording"      : filepath.stem,
        "dataset"        : "NIFECGDB",
        "labels"         : labels,
        "fs"             : fs_rec,
        "duration_sec"   : N / fs_rec,
        "abdomen"        : abdomen,
        "direct"         : None,            # no direct electrode
        "annotation_path": str(ann_path) if ann_path.exists() else None,
        "n_abd_channels" : n_abd,
    }


def load_all_nifecgdb(data_dir: str,
                      max_recordings: int = None) -> list[dict]:
    """
    Load NIFECGDB EDF files from a directory.

    Parameters
    ----------
    data_dir        : path to NIFECGDB folder
    max_recordings  : if set, load at most this many (useful for quick tests)
    """
    data_dir  = Path(data_dir)
    edf_files = sorted(data_dir.glob("*.edf"))
    if not edf_files:
        raise FileNotFoundError(f"No EDF files found in {data_dir}")

    if max_recordings:
        edf_files = edf_files[:max_recordings]

    recordings = []
    for f in edf_files:
        try:
            recordings.append(load_nifecgdb_edf(str(f)))
        except Exception as e:
            print(f"[Loader] WARNING: Skipping {f.name} — {e}")

    print(f"[Loader] Loaded {len(recordings)} NIFECGDB recordings from {data_dir}")
    return recordings


# ── Shared helpers ─────────────────────────────────────────────────────────────

def print_recording_summary(rec: dict) -> None:
    """Print a human-readable summary of a loaded recording."""
    has_direct = rec.get("direct") is not None
    has_ann    = rec.get("annotation_path") is not None
    print(f"\n{'='*50}")
    print(f"Recording : {rec['recording']}  [{rec.get('dataset','?')}]")
    print(f"Duration  : {rec['duration_sec']:.2f} s")
    print(f"Fs        : {rec['fs']} Hz")
    print(f"Abdomen   : shape {rec['abdomen'].shape}")
    print(f"Direct    : {'shape ' + str(rec['direct'].shape) if has_direct else 'N/A'}")
    print(f"Annotation: {'yes — ' + Path(rec['annotation_path']).name if has_ann else 'none'}")
    print(f"Channels  : {rec['labels']}")
    print(f"{'='*50}\n")
