"""
dataset_handlers/__init__.py
Public API for dataset handling.

Usage:
    from dataset_handlers import get_dataset
    handler = get_dataset("adfecgdb")
    recordings = handler.load_all_recordings("/path/to/data")
"""

from .base import AbstractDatasetHandler
from .adfecgdb import ADFECGDBHandler
from .nifecgdb import NIFECGDBHandler
from .cinc2013 import CinC2013Handler
from .registry import (
    DatasetRegistry,
    get_dataset,
    register_dataset,
    list_datasets,
)

__all__ = [
    "AbstractDatasetHandler",
    "ADFECGDBHandler",
    "NIFECGDBHandler",
    "CinC2013Handler",
    "DatasetRegistry",
    "get_dataset",
    "register_dataset",
    "list_datasets",
]
