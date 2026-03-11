"""
STREAMLINED PIPELINE SETUP GUIDE
=================================

This project now uses a modular architecture for handling multiple datasets.
Below is a complete guide for using and extending the new structure.

╔═══════════════════════════════════════════════════════════════════════════╗
│  KEY COMPONENTS                                                           │
╚═══════════════════════════════════════════════════════════════════════════╝

1. CONFIGS MODULE (configs/)
   ├── base.py              — BaseConfig class (unified configuration)
   ├── adfecgdb.yaml        — ADFECGDB-specific overrides
   ├── nifecgdb.yaml        — NIFECGDB-specific overrides
   └── __init__.py          — Public API (get_config)

2. DATASETS MODULE (dataset_handlers/)
   ├── base.py              — AbstractDatasetHandler interface
   ├── adfecgdb.py          — ADFECGDB implementation
   ├── nifecgdb.py          — NIFECGDB implementation
   ├── registry.py          — Factory pattern registry
   └── __init__.py          — Public API (get_dataset, register_dataset)

3. BACKWARD COMPATIBILITY
   ├── config_loader.py     — Legacy wrapper (now uses configs module)
   └── data/loader.py       — Legacy wrapper (now uses dataset_handlers module)


╔═══════════════════════════════════════════════════════════════════════════╗
│  QUICK START                                                              │
╚═══════════════════════════════════════════════════════════════════════════╝

OPTION A: Old way (backward compatible)
──────────────────────────────────────
    from config_loader import get_config
    from data.loader import load_all_recordings
    
    config = get_config("adfecgdb")       # Returns BaseConfig instance
    recordings = load_all_recordings("/path/to/adfecgdb")
    

OPTION B: New way (recommended)
───────────────────────────────
    from configs import get_config
    from dataset_handlers import get_dataset
    
    config = get_config("adfecgdb")                    # Returns BaseConfig
    handler = get_dataset("adfecgdb")
    recordings = handler.load_all_recordings("/path/to/adfecgdb")


╔═══════════════════════════════════════════════════════════════════════════╗
│  USING DIFFERENT DATASETS                                                 │
╚═══════════════════════════════════════════════════════════════════════════╝

ADFECGDB:
─────────
    from configs import get_config
    from dataset_handlers import get_dataset
    from pipeline import PHASEPipeline
    
    config = get_config("adfecgdb")
    handler = get_dataset("adfecgdb")
    
    recordings = handler.load_all_recordings(
        "/path/to/abdominal-and-direct-fetal-ecg-database-1.0.0"
    )
    
    pipe = PHASEPipeline(verbose=True, dataset="adfecgdb")
    for rec in recordings:
        result = pipe.run(rec, save_figures=True, figures_dir="figures")
        print(f"F1: {result['metrics']['F1']:.4f}")


NIFECGDB:
─────────
    from configs import get_config
    from dataset_handlers import get_dataset
    from pipeline import PHASEPipeline
    
    config = get_config("nifecgdb")  # Applies NIFECGDB overrides
    handler = get_dataset("nifecgdb")
    
    recordings = handler.load_all_recordings(
        "/path/to/non-invasive-fetal-ecg-database-1.0.0"
    )
    
    pipe = PHASEPipeline(verbose=True, dataset="nifecgdb")
    for rec in recordings:
        result = pipe.run(rec, save_figures=False)
        print(f"F1: {result['metrics']['F1']:.4f}")


╔═══════════════════════════════════════════════════════════════════════════╗
│  ACCESSING CONFIGURATION                                                  │
╚═══════════════════════════════════════════════════════════════════════════╝

All config values are accessed the same way:

    from configs import get_config
    
    cfg = get_config("adfecgdb")
    
    # Access any parameter as an attribute
    print(cfg.FETAL_HR_MIN)              # 100
    print(cfg.MATERNAL_HR_MAX)           # 110
    print(cfg.ICA_N_COMPONENTS)          # 4
    print(cfg.PT_FETAL_BANDPASS_HIGH)    # 40.0
    
    # Check which dataset this config is for
    print(cfg.dataset)                   # 'adfecgdb'


╔═══════════════════════════════════════════════════════════════════════════╗
│  ADDING A NEW DATASET                                                     │
╚═══════════════════════════════════════════════════════════════════════════╝

Step 1: Create handler class
──────────────────────────

    # File: datasets/mydataset.py
    
    from datasets import AbstractDatasetHandler
    import pyedflib
    import numpy as np
    from pathlib import Path
    
    class MyDatasetHandler(AbstractDatasetHandler):
        @property
        def name(self) -> str:
            return "MYDATASET"
        
        def load_single_recording(self, filepath: str) -> dict:
            # Implement your loading logic
            filepath = Path(filepath)
            
            # Load EDF, extract channels, etc.
            # Return dict with required keys: recording, dataset, fs, 
            # duration_sec, abdomen, direct, labels, annotation_path
            
            return {
                "recording": filepath.stem,
                "dataset": self.name,
                "fs": 1000,
                "duration_sec": 100.0,
                "abdomen": np.zeros((4, 100000)),
                "direct": None,
                "labels": [],
                "annotation_path": None,
            }
        
        def load_all_recordings(self, directory: str, max_recordings=None):
            # Implement directory loading
            directory = Path(directory)
            recordings = []
            # ... load all files ...
            return recordings


Step 2: Register the dataset
────────────────────────────

    # File: dataset_handlers/__init__.py (add these imports)
    
    from .mydataset import MyDatasetHandler
    
    # At module initialization (after _initialize_registry):
    register_dataset("mydataset", MyDatasetHandler)


Step 3: Use the dataset
──────────────────────

    from configs import get_config
    from dataset_handlers import get_dataset
    
    config = get_config("mydataset")  # Will load mydataset.yaml if it exists
    handler = get_dataset("mydataset")
    recordings = handler.load_all_recordings("/path/to/mydataset")


Step 4 (Optional): Create configuration overrides
──────────────────────────────────────────────────

    # File: configs/mydataset.yaml
    
    # Overrides to BaseConfig for MYDATASET
    # Only include parameters that differ from base
    
    maternal_hr_min: 60
    maternal_hr_max: 120
    ekf_fetal_hr_init: 145


╔═══════════════════════════════════════════════════════════════════════════╗
│  DATASET-SPECIFIC PIPELINE VARIATIONS                                      │
╚═══════════════════════════════════════════════════════════════════════════╝

Since each dataset has its own handler, you can customize preprocessing
and pipeline logic by dataset without polluting the core code:

    from dataset_handlers import get_dataset
    from configs import get_config
    
    def run_analysis(dataset_name, data_dir):
        config = get_config(dataset_name)
        handler = get_dataset(dataset_name)
        
        # All dataset-specific logic is isolated in the handler
        recordings = handler.load_all_recordings(data_dir)
        
        # Config automatically has dataset-specific values
        print(f"Fetal HR range for {dataset_name}: "
              f"{config.FETAL_HR_LOW}-{config.FETAL_HR_HIGH}")
        
        for rec in recordings:
            # Handler provides validation
            if not handler.validate_recording(rec):
                print(f"Skipping invalid recording: {rec['recording']}")
                continue
            
            # Print summary using handler method
            handler.print_recording_summary(rec)


╔═══════════════════════════════════════════════════════════════════════════╗
│  BENEFITS OF THIS STRUCTURE                                               │
╚═══════════════════════════════════════════════════════════════════════════╝

✓ Single source of truth for configuration (configs/base.py)
✓ No duplicate config files anymore
✓ Dataset-specific overrides via YAML files
✓ Easy to add new datasets (just inherit AbstractDatasetHandler)
✓ Clear separation of concerns (each dataset isolated)
✓ Backward compatible (old code still works)
✓ Plugin architecture (datasets can be registered dynamically)
✓ Type hints for better IDE support
✓ Testable (mock datasets for testing)


╔═══════════════════════════════════════════════════════════════════════════╗
│  MIGRATION FROM OLD CODE                                                  │
╚═══════════════════════════════════════════════════════════════════════════╝

OLD (still works):
──────────────────
    from config_loader import get_config
    cfg = get_config("adfecgdb")


NEW (recommended):
──────────────────
    from configs import get_config
    cfg = get_config("adfecgdb")


OLD:
────
    from data.loader import load_all_recordings
    recs = load_all_recordings("/path")


NEW:
────
    from datasets import get_dataset
    handler = get_dataset("adfecgdb")
    recs = handler.load_all_recordings("/path")


The old functions are thin wrappers around the new system, so you can
migrate at your own pace. New code should prefer the new API.


╔═══════════════════════════════════════════════════════════════════════════╗
│  FILE STRUCTURE REFERENCE                                                 │
╚═══════════════════════════════════════════════════════════════════════════╝

.
├── configs/
│   ├── __init__.py           — Exports: get_config()
│   ├── base.py               — BaseConfig class (all parameters)
│   ├── adfecgdb.yaml         — ADFECGDB overrides
│   └── nifecgdb.yaml         — NIFECGDB overrides
│
├── dataset_handlers/
│   ├── __init__.py           — Exports: get_dataset(), register_dataset()
│   ├── base.py               — AbstractDatasetHandler interface
│   ├── registry.py           — DatasetRegistry (factory pattern)
│   ├── adfecgdb.py           — ADFECGDBHandler
│   └── nifecgdb.py           — NIFECGDBHandler
│
├── data/
│   └── loader.py             — Backward compatibility wrappers
│
├── pipeline.py               — PHASEPipeline (works with any config)
├── run_experiment.py         — Main entry point
└── config_loader.py          — Backward compatibility wrapper


For questions or to extend this, see:
- configs/base.py for all available config parameters
- dataset_handlers/base.py for AbstractDatasetHandler interface
- dataset_handlers/adfecgdb.py for example implementation
"""
