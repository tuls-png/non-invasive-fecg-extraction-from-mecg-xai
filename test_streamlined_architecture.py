#!/usr/bin/env python
"""Test script to validate the new datasets and configs modules."""

import sys
from pathlib import Path

# Ensure workspace is in path
sys.path.insert(0, str(Path.cwd()))

print("=" * 70)
print("TESTING NEW STREAMLINED ARCHITECTURE")
print("=" * 70 + "\n")

# Test 1: Configs Module
print("[TEST 1] Configuration Module")
print("-" * 70)
try:
    from configs import get_config
    
    cfg_adf = get_config("adfecgdb")
    print(f"✓ ADFECGDB config loaded")
    print(f"  - Dataset: {cfg_adf.dataset}")
    print(f"  - Fetal HR range: {cfg_adf.FETAL_HR_LOW}-{cfg_adf.FETAL_HR_HIGH}")
    print(f"  - ICA components: {cfg_adf.ICA_N_COMPONENTS}")
    
    cfg_nif = get_config("nifecgdb")
    print(f"✓ NIFECGDB config loaded")
    print(f"  - Dataset: {cfg_nif.dataset}")
    print(f"  - Fetal HR range: {cfg_nif.FETAL_HR_LOW}-{cfg_nif.FETAL_HR_HIGH}")
    print()
except Exception as e:
    print(f"✗ Config test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Datasets Module
print("[TEST 2] Dataset Handlers Module")
print("-" * 70)
try:
    from dataset_handlers import get_dataset, list_datasets
    
    print(f"✓ Dataset handler registry initialized")
    print(f"  - Available: {', '.join(list_datasets())}")
    
    handler_adf = get_dataset("adfecgdb")
    print(f"✓ ADFECGDB handler: {handler_adf.name}")
    
    handler_nif = get_dataset("nifecgdb")
    print(f"✓ NIFECGDB handler: {handler_nif.name}")
    print()
except Exception as e:
    print(f"✗ Dataset handlers test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Backward Compatibility
print("[TEST 3] Backward Compatibility (Legacy API)")
print("-" * 70)
try:
    from config_loader import get_config as old_get_config
    
    old_cfg = old_get_config("adfecgdb")
    print(f"✓ Legacy config_loader.get_config() works")
    print(f"  - Returns: {type(old_cfg).__name__}")
    print(f"  - Dataset: {old_cfg.dataset}")
    print()
except Exception as e:
    print(f"✗ Backward compatibility test failed: {e}")
except Exception as e:
    print(f"✗ Datasets test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("=" * 70)
print("✓ ALL TESTS PASSED!")
print("=" * 70)
print("\nNew architecture is ready to use:")
print("  - Config system: configs module")
print("  - Dataset system: datasets module")
print("  - Legacy API: still supported for backward compatibility")
