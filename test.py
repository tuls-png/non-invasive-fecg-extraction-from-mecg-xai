"""
test.py
Quick sanity check — verifies all imports work and one EDF loads correctly.

FIX: Hardcoded Windows path replaced with relative path to bundled Datasets.
"""

import sys
from pathlib import Path

print(f"Python: {sys.version}")
print(f"Working dir: {Path(__file__).parent}")

try:
    import numpy as np;      print("numpy OK")
    import scipy;            print("scipy OK")
    import sklearn;          print("sklearn OK")
    import pyedflib;         print("pyedflib OK")
    import matplotlib;       print("matplotlib OK")
except ImportError as e:
    print(f"MISSING PACKAGE: {e}")
    sys.exit(1)

try:
    sys.path.insert(0, str(Path(__file__).parent))
    from configs import BaseConfig; _cfg = BaseConfig(); FS = _cfg.FS; print(f"config OK — FS={FS}")
    from data.loader import load_edf;                    print("loader OK")
    from preprocessing.filters import preprocess_channel; print("filters OK")
    from preprocessing.qrs_detector import detect_fetal_qrs; print("qrs_detector OK")
    from separation.ica import run_ica;                  print("ica OK")
    from separation.wsvd import adaptive_windowed_wsvd;  print("wsvd OK")
    from separation.ekf import FetalECGKalmanFilter;     print("ekf OK")
    from evaluation.metrics import evaluate;             print("metrics OK")
    from xai.echo import ECHOExplainer;                  print("echo OK")
except Exception as e:
    import traceback
    print(f"\nIMPORT ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\nAll imports OK. Testing EDF load...")
DATASET_PATH = Path(__file__).parent / "Datasets" / \
               "abdominal-and-direct-fetal-ecg-database-1.0.0"

edfs = sorted(DATASET_PATH.glob("*.edf"))
if not edfs:
    print(f"No EDF files found in {DATASET_PATH} — skipping load test")
else:
    rec = load_edf(str(edfs[0]))
    print(f"Loaded: {rec['recording']}, shape: {rec['abdomen'].shape}")
    print("\nAll systems OK.")
