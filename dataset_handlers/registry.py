"""
datasets/registry.py
DatasetRegistry: Factory pattern for dataset handlers.

Usage:
    from datasets import get_dataset
    
    handler = get_dataset("adfecgdb")
    recordings = handler.load_all_recordings("/path/to/adfecgdb")
    
    # Or register a new dataset handler
    from datasets import register_dataset
    from datasets.adfecgdb import ADFECGDBHandler
    
    register_dataset("custom_dataset", CustomDatasetHandler)
"""

from typing import Dict, Type, List, Optional, Any
from .base import AbstractDatasetHandler


class DatasetRegistry:
    """
    Registry for dataset handlers.
    
    Implements factory pattern for creating and managing dataset handlers.
    """

    _handlers: Dict[str, Type[AbstractDatasetHandler]] = {}

    @classmethod
    def register(cls, name: str, handler_class: Type[AbstractDatasetHandler]) -> None:
        """
        Register a dataset handler class.
        
        Parameters
        ----------
        name : str
            Unique identifier for the dataset (e.g., 'adfecgdb').
        handler_class : type
            Handler class (must inherit from AbstractDatasetHandler).
        
        Raises
        ------
        ValueError
            If handler_class is not a subclass of AbstractDatasetHandler.
        """
        if not issubclass(handler_class, AbstractDatasetHandler):
            raise ValueError(
                f"{handler_class} must inherit from AbstractDatasetHandler"
            )
        cls._handlers[name.lower()] = handler_class

    @classmethod
    def get(cls, name: str, **kwargs) -> AbstractDatasetHandler:
        """
        Get an instance of a registered dataset handler.
        
        Parameters
        ----------
        name : str
            Dataset identifier (case-insensitive).
        **kwargs
            Keyword arguments passed to the handler's __init__.
        
        Returns
        -------
        AbstractDatasetHandler
            Instantiated handler.
        
        Raises
        ------
        KeyError
            If dataset is not registered.
        """
        name_lower = name.lower()
        if name_lower not in cls._handlers:
            raise KeyError(
                f"Dataset '{name}' not registered. Available: {list(cls._handlers.keys())}"
            )
        handler_class = cls._handlers[name_lower]
        return handler_class(**kwargs)

    @classmethod
    def list_datasets(cls) -> List[str]:
        """Return list of registered dataset identifiers."""
        return list(cls._handlers.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a dataset is registered."""
        return name.lower() in cls._handlers


# ── Registry initialization ────────────────────────────────────────────────

def _initialize_registry():
    """Initialize registry with built-in datasets."""
    from .adfecgdb import ADFECGDBHandler
    from .nifecgdb import NIFECGDBHandler
    from .cinc2013 import CinC2013Handler
    
    DatasetRegistry.register("adfecgdb", ADFECGDBHandler)
    DatasetRegistry.register("nifecgdb", NIFECGDBHandler)
    DatasetRegistry.register("cinc2013", CinC2013Handler)


# Initialize on module load
_initialize_registry()


# ── Convenience functions ──────────────────────────────────────────────────

def get_dataset(name: str, **kwargs) -> AbstractDatasetHandler:
    """
    Convenience function to get a dataset handler.
    
    Parameters
    ----------
    name : str
        Dataset identifier ('adfecgdb', 'nifecgdb', or custom).
    **kwargs
        Keyword arguments for the handler.
    
    Returns
    -------
    AbstractDatasetHandler
        Instantiated handler.
    """
    return DatasetRegistry.get(name, **kwargs)


def register_dataset(name: str, handler_class: Type[AbstractDatasetHandler]) -> None:
    """
    Register a custom dataset handler.
    
    Parameters
    ----------
    name : str
        Unique dataset identifier.
    handler_class : type
        Handler class (must inherit from AbstractDatasetHandler).
    """
    DatasetRegistry.register(name, handler_class)


def list_datasets() -> List[str]:
    """Return list of all registered datasets."""
    return DatasetRegistry.list_datasets()
