"""
dataset_handlers/cinc2013.py
CinC 2013 Challenge Set A handler.

Key differences from ADFECGDB/NIFECGDB:
  - WFDB binary format (.dat + .hea), NOT EDF
  - Annotations in .fqrs files — these ARE fetal QRS (crowd-sourced
    with reference to direct fetal ECG). Opposite of NIFECGDB .qrs
    which are maternal.
  - 1-minute recordings (vs 5-min ADFECGDB, variable NIFECGDB)
  - No direct fetal electrode in released data
  - Invalid samples stored as -32768 digital / NaN physical — zeroed on load
  - 4 abdominal channels, 1000 Hz
"""

import numpy as np
import wfdb
from pathlib import Path
from typing import List, Dict, Any, Optional
from .base import AbstractDatasetHandler


class CinC2013Handler(AbstractDatasetHandler):

    @property
    def name(self) -> str:
        return "CINC2013"

    def load_single_recording(self, filepath: str) -> Dict[str, Any]:
        """
        Load one CinC 2013 WFDB record.

        Parameters
        ----------
        filepath : str
            Path to the .hea file (e.g. '/path/to/set-a/a01.hea')
            OR the record stem (e.g. '/path/to/set-a/a01').
            Both are accepted.
        """
        filepath = Path(filepath)

        # Accept either .hea path or bare stem
        if filepath.suffix == '.hea':
            record_stem = str(filepath.with_suffix(''))
        else:
            record_stem = str(filepath)

        rec_id = Path(record_stem).name  # e.g. 'a01'

        # Read WFDB record — p_signal returns physical units (mV), NaN for invalid
        record  = wfdb.rdrecord(record_stem)
        fs      = int(record.fs)          # 1000 Hz
        N       = record.sig_len
        signals = record.p_signal         # shape (N, n_channels)

        # Replace NaN (invalid samples) with 0
        signals = np.where(np.isnan(signals), 0.0, signals)

        n_channels = signals.shape[1]

        # CinC2013 has exactly 4 abdominal channels, no direct electrode
        # Transpose to (n_channels, N) to match pipeline convention
        abdomen = np.zeros((4, N), dtype=np.float64)
        for ch in range(min(n_channels, 4)):
            abdomen[ch] = signals[:, ch]

        # Channel labels from header
        labels = list(record.sig_name) if record.sig_name else \
                 [f"ch{i+1}" for i in range(n_channels)]

        # Annotation: record_stem + '.fqrs'
        ann_file   = Path(record_stem + '.fqrs')
        ann_exists = ann_file.exists()

        print(
            f"[Loader] {rec_id}.hea — {N} samples @ {fs} Hz "
            f"({N/fs:.1f}s), {n_channels} channels, "
            f"fqrs={'yes' if ann_exists else 'none'}"
        )

        return {
            "recording"         : rec_id,
            "dataset"           : self.name,
            "labels"            : labels,
            "fs"                : fs,
            "duration_sec"      : N / fs,
            "abdomen"           : abdomen,          # (4, N)
            "direct"            : None,             # not in released data
            "annotation_path"   : record_stem if ann_exists else None,
            "annotation_ext"    : "fqrs",           # wfdb extension
            "annotation_is_fetal": True,            # fqrs = fetal QRS
        }

    def load_all_recordings(self, directory: str,
                            max_recordings: Optional[int] = None
                            ) -> List[Dict[str, Any]]:
        """
        Load all CinC 2013 records from a directory.

        Parameters
        ----------
        directory : str
            Path to set-a folder (contains a01.hea, a01.dat, a01.fqrs, etc.)
        max_recordings : int, optional
            Cap on number of recordings to load.
        """
        directory = Path(directory)
        hea_files = sorted(directory.glob("*.hea"))

        if not hea_files:
            raise FileNotFoundError(
                f"No .hea files found in {directory}. "
                f"Ensure you are pointing to the set-a folder."
            )

        if max_recordings:
            hea_files = hea_files[:max_recordings]

        recordings = []
        for hf in hea_files:
            try:
                recordings.append(self.load_single_recording(str(hf)))
            except Exception as e:
                print(f"[Loader] WARNING: Skipping {hf.name} — {e}")

        print(
            f"[Loader] Loaded {len(recordings)} CinC2013 recordings "
            f"from {directory}"
        )
        return recordings