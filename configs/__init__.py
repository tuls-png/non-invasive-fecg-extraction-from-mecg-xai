"""
configs/__init__.py
Unified configuration module for all datasets.

Usage:
    from configs import get_config
    
    config = get_config("adfecgdb")  # Returns BaseConfig with ADFECGDB overrides
    config = get_config("nifecgdb")  # Returns BaseConfig with NIFECGDB overrides
"""

from .base import BaseConfig


def get_config(dataset: str = "adfecgdb") -> BaseConfig:
    """
    Get a config instance for the specified dataset.
    
    Parameters
    ----------
    dataset : str
        Dataset name ('adfecgdb' or 'nifecgdb'). Case-insensitive.
    
    Returns
    -------
    BaseConfig
        Configuration object with dataset-specific overrides applied.
    """
    return BaseConfig(dataset=dataset)


__all__ = ["BaseConfig", "get_config"]
