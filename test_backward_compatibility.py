#!/usr/bin/env python
"""Backward compatibility test - verify old code still works."""

import sys
from pathlib import Path

print("=" * 70)
print("BACKWARD COMPATIBILITY TEST")
print("=" * 70 + "\n")

# Test 1: Old config_loader API
print("[TEST 1] Old config_loader API")
print("-" * 70)
try:
    from config_loader import get_config
    
    cfg = get_config("adfecgdb")
    assert hasattr(cfg, 'FETAL_HR_MAX'), "Config missing FETAL_HR_MAX"
    assert cfg.FETAL_HR_MAX == 185, f"Wrong value: {cfg.FETAL_HR_MAX}"
    assert cfg.dataset == "adfecgdb", f"Wrong dataset: {cfg.dataset}"
    
    print(f"✓ config_loader.get_config('adfecgdb') works")
    print(f"  - Returns: {type(cfg).__name__}")
    print(f"  - FETAL_HR_MAX: {cfg.FETAL_HR_MAX}")
    
    cfg_nif = get_config("nifecgdb")
    assert cfg_nif.FETAL_HR_HIGH == 200, f"Wrong NIFECGDB value: {cfg_nif.FETAL_HR_HIGH}"
    print(f"✓ config_loader.get_config('nifecgdb') works")
    print(f"  - FETAL_HR_HIGH: {cfg_nif.FETAL_HR_HIGH}")
    print()
except AssertionError as e:
    print(f"✗ FAILED: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Old data.loader API
print("[TEST 2] Old data.loader API")
print("-" * 70)
try:
    from data.loader import load_edf, load_all_recordings, load_nifecgdb_edf, load_all_nifecgdb, print_recording_summary
    
    print(f"✓ Imported: load_edf")
    print(f"✓ Imported: load_all_recordings")
    print(f"✓ Imported: load_nifecgdb_edf")
    print(f"✓ Imported: load_all_nifecgdb")
    print(f"✓ Imported: print_recording_summary")
    print()
    
    # Note: we can't actually load files without the data,
    # but we can verify the functions are callable
    assert callable(load_edf), "load_edf is not callable"
    assert callable(load_all_recordings), "load_all_recordings is not callable"
    assert callable(load_nifecgdb_edf), "load_nifecgdb_edf is not callable"
    assert callable(load_all_nifecgdb), "load_all_nifecgdb is not callable"
    assert callable(print_recording_summary), "print_recording_summary is not callable"
    
    print("✓ All legacy loader functions are callable")
    print()
except AssertionError as e:
    print(f"✗ FAILED: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: New API is available
print("[TEST 3] New API Availability")
print("-" * 70)
try:
    from configs import get_config as new_get_config
    from dataset_handlers import get_dataset, register_dataset, list_datasets
    
    print(f"✓ Imported: configs.get_config")
    print(f"✓ Imported: dataset_handlers.get_dataset")
    print(f"✓ Imported: dataset_handlers.register_dataset")
    print(f"✓ Imported: dataset_handlers.list_datasets")
    
    datasets = list_datasets()
    print(f"✓ Available datasets: {', '.join(datasets)}")
    print()
except Exception as e:
    print(f"✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("=" * 70)
print("✅ BACKWARD COMPATIBILITY VERIFIED")
print("=" * 70)
print("\nOld API (still works):")
print("  ✓ from config_loader import get_config")
print("  ✓ from data.loader import load_* functions")
print("\nNew API (recommended for new code):")
print("  ✓ from configs import get_config")
print("  ✓ from dataset_handlers import get_dataset")
print("\n✅ You can migrate gradually - old code works as-is!")
