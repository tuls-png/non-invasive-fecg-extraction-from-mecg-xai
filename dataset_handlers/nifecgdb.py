"""
datasets/nifecgdb.py
NIFECGDB dataset handler.

Loads Non-Invasive Fetal ECG Database recordings.
Characteristics:
- Variable number of abdominal channels (3-4, zero-padded to 4)
- 2 thoracic channels (maternal ECG, excluded from analysis)
- No direct fetal electrode
- .edf.qrs annotation files for ground truth
"""

import pyedflib
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

from .base import AbstractDatasetHandler


class NIFECGDBHandler(AbstractDatasetHandler):
    """Handler for NIFECGDB dataset."""

    def __init__(self, abdominal_prefix=None, thoracic_prefix=None, max_abd_channels=4):
        """
        Initialize NIFECGDB handler with channel name patterns.
        
        Parameters
        ----------
        abdominal_prefix : str, optional
            Prefix for abdominal channels. Defaults to 'abdomen_'.
        thoracic_prefix : str, optional
            Prefix for thoracic channels. Defaults to 'thorax_'.
        max_abd_channels : int, optional
            Maximum abdominal channels to keep (default 4).
        """
        self.abdominal_prefix = (abdominal_prefix or 'abdomen_').lower()
        self.thoracic_prefix = (thoracic_prefix or 'thorax_').lower()
        self.max_abd_channels = max_abd_channels

    @property
    def name(self) -> str:
        """Return dataset identifier."""
        return "NIFECGDB"

    def load_single_recording(self, filepath: str) -> Dict[str, Any]:
        """
        Load one NIFECGDB EDF file.
        
        Channel layout varies per record:
        - 2 thoracic channels (maternal ECG) — excluded
        - 3-4 abdominal channels — used for analysis
        
        Parameters
        ----------
        filepath : str
            Path to .edf file.
        
        Returns
        -------
        dict
            Recording dictionary. Note: 'direct' is None (no direct electrode).
        
        Raises
        ------
        FileNotFoundError
            If file does not exist.
        ValueError
            If no abdominal channels found and fallback fails.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"EDF file not found: {filepath}")

        # Read EDF file
        f = pyedflib.EdfReader(str(filepath))
        labels = f.getSignalLabels()
        fs_list = [int(f.getSampleFrequency(i)) for i in range(f.signals_in_file)]
        signals = np.vstack([f.readSignal(i) for i in range(f.signals_in_file)])
        f._close()
        del f

        # Identify abdominal vs thoracic channels
        # Normalize labels to lowercase with underscores for robust matching
        labels_lower = [l.lower().replace(' ', '_') for l in labels]

        abd_indices = [
            i for i, l in enumerate(labels_lower)
            if l.startswith(self.abdominal_prefix)
        ]
        thoracic_indices = [
            i for i, l in enumerate(labels_lower)
            if l.startswith(self.thoracic_prefix)
        ]

        if not abd_indices:
            # Fallback: exclude known thoracic channels, use everything else
            abd_indices = [i for i in range(len(labels)) if i not in thoracic_indices]
            print(
                f"[Loader] WARNING: no '{self.abdominal_prefix}*' channels found "
                f"in {filepath.name}. Labels: {labels}. "
                f"Using non-thoracic channels: {[labels[i] for i in abd_indices]}"
            )

        fs_rec = fs_list[0]
        N = signals.shape[1]

        # Build (max_abd_channels, N) abdominal array — zero-pad if fewer channels
        n_abd = min(len(abd_indices), self.max_abd_channels)
        abdomen = np.zeros((self.max_abd_channels, N), dtype=np.float64)
        for k, idx in enumerate(abd_indices[:n_abd]):
            abdomen[k] = signals[idx]

        print(
            f"[Loader] {filepath.name} — {N} samples @ {fs_rec} Hz "
            f"({N/fs_rec:.1f} s), {n_abd} abdominal / "
            f"{len(thoracic_indices)} thoracic channels"
        )

        # Find annotation file
        ann_path = filepath.parent / (filepath.name + ".qrs")

        return {
            "recording": filepath.stem,
            "dataset": self.name,
            "labels": labels,
            "fs": fs_rec,
            "duration_sec": N / fs_rec,
            "abdomen": abdomen,
            "direct": None,  # No direct electrode for NIFECGDB
            "annotation_path": str(ann_path) if ann_path.exists() else None,
            "n_abd_channels": n_abd,
        }

    def load_all_recordings(self, directory: str,
                           max_recordings: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load all NIFECGDB recordings from a directory.
        
        Parameters
        ----------
        directory : str
            Path to NIFECGDB folder.
        max_recordings : int, optional
            If set, load at most this many recordings.
        
        Returns
        -------
        list of dict
            List of recording dictionaries.
        
        Raises
        ------
        FileNotFoundError
            If directory doesn't exist or contains no EDF files.
        """
        directory = Path(directory)
        edf_files = sorted(directory.glob("*.edf"))
        
        if not edf_files:
            raise FileNotFoundError(f"No EDF files found in {directory}")

        if max_recordings:
            edf_files = edf_files[:max_recordings]

        recordings = []
        for f in edf_files:
            try:
                rec = self.load_single_recording(str(f))
                recordings.append(rec)
            except Exception as e:
                print(f"[Loader] WARNING: Skipping {f.name} — {e}")

        print(f"[Loader] Loaded {len(recordings)} NIFECGDB recordings from {directory}")
        return recordings
