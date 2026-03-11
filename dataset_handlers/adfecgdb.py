"""
datasets/adfecgdb.py
ADFECGDB dataset handler.

Loads Abdominal and Direct Fetal ECG Database recordings.
Every recording has:
- 4 abdominal channels (Abdomen_1..4)
- 1 direct fetal ECG channel (Direct_1)
- .edf.qrs annotation files for ground truth
"""

import pyedflib
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

from .base import AbstractDatasetHandler


class ADFECGDBHandler(AbstractDatasetHandler):
    """Handler for ADFECGDB dataset."""

    def __init__(self, abdomen_channels=None, direct_channel=None):
        """
        Initialize ADFECGDB handler with channel names.
        
        Parameters
        ----------
        abdomen_channels : list[str], optional
            Names of abdominal channels. Defaults to standard ADFECGDB naming.
        direct_channel : str, optional
            Name of direct fetal ECG channel. Defaults to 'Direct_1'.
        """
        self.abdomen_channels = abdomen_channels or [
            'Abdomen_1', 'Abdomen_2', 'Abdomen_3', 'Abdomen_4'
        ]
        self.direct_channel = direct_channel or 'Direct_1'

    @property
    def name(self) -> str:
        """Return dataset identifier."""
        return "ADFECGDB"

    def load_single_recording(self, filepath: str) -> Dict[str, Any]:
        """
        Load one ADFECGDB EDF file.
        
        Parameters
        ----------
        filepath : str
            Path to .edf file.
        
        Returns
        -------
        dict
            Recording dictionary with keys: recording, dataset, fs, duration_sec,
            abdomen (4, N), direct (N,), labels, annotation_path.
        
        Raises
        ------
        FileNotFoundError
            If file does not exist.
        ValueError
            If required channels are missing.
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

        # Validate required channels
        for ch in self.abdomen_channels + [self.direct_channel]:
            if ch not in labels:
                raise ValueError(
                    f"Expected channel '{ch}' not found in {filepath.name}. "
                    f"Available: {labels}"
                )

        # Extract channels
        abd_indices = [labels.index(ch) for ch in self.abdomen_channels]
        direct_idx = labels.index(self.direct_channel)
        fs_rec = fs_list[0]
        abdomen = signals[abd_indices]  # (4, N)
        direct = signals[direct_idx]    # (N,)
        N = abdomen.shape[1]

        print(
            f"[Loader] {filepath.name} — {N} samples @ {fs_rec} Hz "
            f"({N/fs_rec:.1f} s), {len(labels)} channels"
        )

        # Find annotation file
        ann_path = filepath.parent / (filepath.name + ".qrs")

        return {
            "recording"          : filepath.stem,
            "dataset"            : self.name,
            "labels"             : labels,
            "fs"                 : fs_rec,
            "duration_sec"       : N / fs_rec,
            "abdomen"            : abdomen,
            "direct"             : direct,
            "annotation_path"    : str(filepath.parent / filepath.name) if ann_path.exists() else None,
            "annotation_ext"     : "qrs",
            "annotation_is_fetal": True,
        }

    def load_all_recordings(self, directory: str,
                           max_recordings: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load all ADFECGDB recordings from a directory.
        
        Parameters
        ----------
        directory : str
            Path to ADFECGDB folder.
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

        print(f"[Loader] Loaded {len(recordings)} ADFECGDB recordings from {directory}")
        return recordings
