"""
config_loader.py
Returns the correct config module based on dataset name.
"""

def get_config(dataset: str = "adfecgdb"):
    if dataset == "nifecgdb":
        import config_nifecgdb as cfg
    else:
        import config as cfg
    return cfg