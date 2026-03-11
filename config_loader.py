"""
config_loader.py
Returns the correct config object based on dataset name.

DEPRECATED: Use `from configs import get_config` instead.

This module is kept for backward compatibility. New code should use:
    from configs import get_config
    config = get_config("adfecgdb")  # Returns BaseConfig instance
"""

def get_config(dataset: str = "adfecgdb"):
    """
    Get config for a dataset.
    
    BACKWARD COMPATIBILITY WRAPPER: Now returns BaseConfig instance from
    the new configs module instead of the old module-based approach.
    
    Parameters
    ----------
    dataset : str
        Dataset name ('adfecgdb' or 'nifecgdb' or 'cinc2013'). Case-insensitive.
    
    Returns
    -------
    BaseConfig
        Configuration object with dataset-specific settings.
    """
    from configs import get_config as get_new_config
    return get_new_config(dataset)