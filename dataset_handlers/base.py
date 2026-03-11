"""
datasets/base.py
Abstract base class for dataset handlers.

All dataset implementations (ADFECGDB, NIFECGDB, new datasets) inherit from
AbstractDatasetHandler and implement:
- load_single_recording() : Load one recording from disk
- load_all_recordings() : Load all recordings from a directory
- get_dataset_name() : Return the dataset identifier
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Optional, Any


class AbstractDatasetHandler(ABC):
    """
    Abstract base class for dataset handlers.
    
    Subclasses must implement load_single_recording() and load_all_recordings().
    Recording dicts conform to this schema:
    
    {
        "recording"      : str        — filename stem
        "dataset"        : str        — dataset name (e.g., 'ADFECGDB')
        "fs"             : int        — sampling rate (Hz)
        "duration_sec"   : float      — duration in seconds
        "abdomen"        : ndarray (4, N) — abdominal channels
        "direct"         : ndarray (N,) | None — direct fetal ECG (if available)
        "labels"         : list[str]  — channel labels
        "annotation_path": str | None — path to annotation file (.qrs)
    }
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the dataset identifier (e.g., 'ADFECGDB', 'NIFECGDB')."""
        pass

    @abstractmethod
    def load_single_recording(self, filepath: str) -> Dict[str, Any]:
        """
        Load a single recording from disk.
        
        Parameters
        ----------
        filepath : str
            Path to the recording file (typically .edf).
        
        Returns
        -------
        dict
            Recording dictionary matching the schema above.
        
        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file format is invalid or required channels are missing.
        """
        pass

    @abstractmethod
    def load_all_recordings(self, directory: str, 
                           max_recordings: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load all recordings from a directory.
        
        Parameters
        ----------
        directory : str
        directory_number : string_number, 
            Path to the dataset directory.
        max_recordings : int, optional
            If set, load at most this many recordings (useful for testing).
        
        Returns
        -------
        list of dict
            List of recording dictionaries.
        
        Raises
        ------
        FileNotFoundError
            If the directory does not exist or contains no valid recordings.
        """
        pass

    def validate_recording(self, rec: Dict[str, Any]) -> bool:
        """
        Validate a recording dict conforms to the expected schema.
        
        Parameters
        ----------
        rec : dict
            Recording dictionary to validate.
        
        Returns
        -------
        bool
            True if valid, False otherwise.
        """
        required_keys = {"recording", "dataset", "fs", "duration_sec", "abdomen", "labels"}
        return all(k in rec for k in required_keys)

    def print_recording_summary(self, rec: Dict[str, Any]) -> None:
        """Print a human-readable summary of a loaded recording."""
        has_direct = rec.get("direct") is not None
        has_ann = rec.get("annotation_path") is not None
        
        print(f"\n{'='*50}")
        print(f"Recording : {rec['recording']}  [{rec.get('dataset', '?')}]")
        print(f"Duration  : {rec['duration_sec']:.2f} s")
        print(f"Fs        : {rec['fs']} Hz")
        print(f"Abdomen   : shape {rec['abdomen'].shape}")
        print(f"Direct    : {'shape ' + str(rec['direct'].shape) if has_direct else 'N/A'}")
        if has_ann:
            ann_name = Path(rec['annotation_path']).name
            print(f"Annotation: yes — {ann_name}")
        else:
            print(f"Annotation: none")
        print(f"Channels  : {rec['labels']}")
        print(f"{'='*50}\n")
