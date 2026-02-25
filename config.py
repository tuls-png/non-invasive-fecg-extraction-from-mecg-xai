"""
PHASE Framework Configuration
Physiology-guided Hybrid Adaptive Signal Extraction for Fetal ECG Separation
"""

import numpy as np

# ── Sampling ────────────────────────────────────────────────────────────────
FS = 1000  # Hz — ADFECGDB native sampling rate

# ── Preprocessing ───────────────────────────────────────────────────────────
BANDPASS_LOW  = 1.0    # Hz
BANDPASS_HIGH = 45.0   # Hz
BANDPASS_ORDER = 4
NOTCH_FREQ    = 50.0   # Hz (power line)
NOTCH_QUALITY = 30.0
MEDFILT_KERNEL = 3

# ── Physiological Constraints ────────────────────────────────────────────────
# Maternal HR (BPM)
MATERNAL_HR_MIN = 55
MATERNAL_HR_MAX = 110

# Fetal HR (BPM)
FETAL_HR_MIN = 100
FETAL_HR_MAX = 180

# QRS duration for Gaussian weight sigma
QRS_SIGMA_SEC   = 0.04   # 40 ms — typical QRS duration
QRS_BASELINE_WEIGHT = 0.05

# ── ICA ─────────────────────────────────────────────────────────────────────
ICA_N_COMPONENTS = 4
ICA_MAX_ITER     = 2000
ICA_RANDOM_STATE = 42
ICA_TOL          = 1e-6

# ── Adaptive Windowed WSVD ───────────────────────────────────────────────────
WSVD_WINDOW_SEC            = 2.0   # seconds per window
WSVD_OVERLAP               = 0.5   # 50% overlap
WSVD_N_COMPONENTS          = 1     # SVD components for maternal reconstruction
WSVD_COMPONENT_CORR_THRESH = 0.40  # min |correlation| to accept component as maternal
WSVD_MAX_ENERGY_REMOVAL    = 0.70  # don't subtract if removes >70% of window energy
WSVD_CHANNEL_R2_MIN        = 0.15  # only subtract from channels where maternal R²≥this

# ── Pan-Tompkins QRS Detector ────────────────────────────────────────────────
PT_BANDPASS_LOW  = 5.0   # Hz
PT_BANDPASS_HIGH = 15.0  # Hz
PT_INTEGRATION_WINDOW_SEC = 0.15   # 150 ms
PT_MIN_PEAK_DISTANCE_SEC  = 0.35   # 350 ms (~170 BPM max)
PT_THRESHOLD_FACTOR = 0.5

# ── EKF ─────────────────────────────────────────────────────────────────────
EKF_FETAL_HR_INIT  = 140   # BPM initial estimate
EKF_PROCESS_NOISE  = [0.01, 0.01, 0.1]   # increased for real data scale
EKF_OBSERVE_NOISE  = 1.0                  # increased — trust observations more
EKF_STATE_COV_INIT = 1.0                  # increased initial uncertainty

# McSharry fetal ECG PQRST Gaussian parameters [alpha, b, theta]
# Tuned for fetal cardiac physiology (shorter intervals than adult)
EKF_PQRST_PARAMS = np.array([
    [ 0.30,  0.10, -np.pi / 3   ],   # P wave
    [-0.50,  0.05, -np.pi / 12  ],   # Q wave
    [ 1.50,  0.10,  0.0         ],   # R wave
    [-0.50,  0.05,  np.pi / 12  ],   # S wave
    [ 0.30,  0.20,  np.pi / 2   ],   # T wave
])

# ── Evaluation ───────────────────────────────────────────────────────────────
EVAL_TOLERANCE_MS  = 50   # ms — R-peak match tolerance
EVAL_MIN_PEAK_HEIGHT = 0.35
EVAL_MIN_PEAK_DISTANCE_SEC = 0.33

# ── ECHO XAI ─────────────────────────────────────────────────────────────────
ECHO_MATERNAL_EXCLUSION_SEC = 0.08   # 80 ms exclusion window around maternal peaks

# ── Dataset ──────────────────────────────────────────────────────────────────
ADFECGDB_ABDOMEN_CHANNELS = ['Abdomen_1', 'Abdomen_2', 'Abdomen_3', 'Abdomen_4']
ADFECGDB_DIRECT_CHANNEL   = 'Direct_1'

# NIFECGDB has no direct fetal electrode.
# Actual channel labels (confirmed from dataset): 'Thorax_1', 'Thorax_2',
# 'Abdomen_1' ... 'Abdomen_4'. Some records have only 3 abdominal channels.
NIFECGDB_ABDOMINAL_PREFIX = 'abdomen_'   # normalised lowercase prefix
NIFECGDB_THORACIC_PREFIX  = 'thorax_'    # maternal chest leads — exclude from ICA
NIFECGDB_MAX_ABD_CHANNELS = 4            # use up to 4; zero-pad if fewer

# ── Random seed (for reproducibility) ────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)