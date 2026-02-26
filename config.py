"""
PHASE Framework Configuration
Physiology-guided Hybrid Adaptive Signal Extraction for Fetal ECG Separation

CHANGES FROM ORIGINAL:
  [FIX-1] Split Pan-Tompkins bandpass into separate MATERNAL (5-15 Hz) and
          FETAL (10-40 Hz) configs. Shared 5-15 Hz preferred adult QRS energy.
  [FIX-2] PT_THRESHOLD_FACTOR raised from 0.5 to 1.0 (less noise false-positives).
  [FIX-3] WSVD_N_COMPONENTS raised from 1 to 2 (maternal ECG spans 2-3 SVs).
  [FIX-4] EKF_PROCESS_NOISE raised so filter trusts observations more.
  [FIX-5] Added MATERNAL_ICA2_CORR_THRESH for Path B maternal residual exclusion.
  [FIX-6] PT_INTEGRATION_WINDOW_SEC shortened from 150ms to 80ms.
"""

import numpy as np

# -- Sampling ----------------------------------------------------------------
FS = 1000  # Hz

# -- Preprocessing -----------------------------------------------------------
BANDPASS_LOW   = 1.0
BANDPASS_HIGH  = 45.0
BANDPASS_ORDER = 4
NOTCH_FREQ     = 50.0
NOTCH_QUALITY  = 30.0
MEDFILT_KERNEL = 3

# -- Physiological Constraints -----------------------------------------------
MATERNAL_HR_MIN = 55
MATERNAL_HR_MAX = 110

# [FIX-3] FETAL_HR_MAX aligned with pipeline constant (185)
FETAL_HR_MIN = 100
FETAL_HR_MAX = 185

QRS_SIGMA_SEC       = 0.04
QRS_BASELINE_WEIGHT = 0.05

# -- ICA ---------------------------------------------------------------------
ICA_N_COMPONENTS = 4
ICA_MAX_ITER     = 2000
ICA_RANDOM_STATE = 42
ICA_TOL          = 1e-6

# -- Adaptive Windowed WSVD --------------------------------------------------
WSVD_WINDOW_SEC            = 2.0
WSVD_OVERLAP               = 0.5
# [FIX-3] Raised from 1 -> 2
WSVD_N_COMPONENTS          = 3
WSVD_COMPONENT_CORR_THRESH = 0.30
WSVD_MAX_ENERGY_REMOVAL    = 0.70
WSVD_CHANNEL_R2_MIN        = 0.35

# -- Pan-Tompkins -- MATERNAL ------------------------------------------------
# [FIX-1] Dedicated maternal bandpass (5-15 Hz = standard adult QRS range)
PT_MATERNAL_BANDPASS_LOW   = 5.0
PT_MATERNAL_BANDPASS_HIGH  = 15.0
PT_MATERNAL_BANDPASS_ORDER = 2   # raised from 1 for better roll-off

# -- Pan-Tompkins -- FETAL --------------------------------------------------
# [FIX-1] Fetal QRS is narrower -> more energy above 15 Hz. 10-40 Hz captures
# fetal QRS energy while rejecting residual low-frequency maternal components.
PT_FETAL_BANDPASS_LOW   = 10.0
PT_FETAL_BANDPASS_HIGH  = 40.0
PT_FETAL_BANDPASS_ORDER = 2

# -- Pan-Tompkins shared -----------------------------------------------------
# [FIX-6] 80ms resolves adjacent fetal beats better than original 150ms
PT_INTEGRATION_WINDOW_SEC = 0.08
# [FIX-2] Raised from 0.5 -> 1.0
PT_THRESHOLD_FACTOR = 1.0

# -- EKF ---------------------------------------------------------------------
EKF_FETAL_HR_INIT  = 140
# [FIX-4] Raised from [0.01, 0.01, 0.1]. Low process noise made EKF over-trust
# its oscillator model and smooth away real fetal beats.
EKF_PROCESS_NOISE  = [0.1, 0.1, 1.0]
EKF_OBSERVE_NOISE  = 1.0
EKF_STATE_COV_INIT = 1.0

EKF_PQRST_PARAMS = np.array([
    [ 0.30,  0.10, -np.pi / 3   ],   # P wave
    [-0.50,  0.05, -np.pi / 12  ],   # Q wave
    [ 1.50,  0.10,  0.0         ],   # R wave
    [-0.50,  0.05,  np.pi / 12  ],   # S wave
    [ 0.30,  0.20,  np.pi / 2   ],   # T wave
])

# -- Path B ICA2 maternal residual exclusion ---------------------------------
# [FIX-5] ICA2 components with |corr| to maternal IC above this threshold are
# treated as residual-maternal and excluded from fetal IC selection in Path B.
MATERNAL_ICA2_CORR_THRESH = 0.30

# -- Evaluation --------------------------------------------------------------
EVAL_TOLERANCE_MS          = 50
EVAL_MIN_PEAK_HEIGHT       = 0.35
EVAL_MIN_PEAK_DISTANCE_SEC = 0.33

# -- ECHO XAI ----------------------------------------------------------------
ECHO_MATERNAL_EXCLUSION_SEC = 0.08

# -- Dataset -----------------------------------------------------------------
ADFECGDB_ABDOMEN_CHANNELS = ['Abdomen_1', 'Abdomen_2', 'Abdomen_3', 'Abdomen_4']
ADFECGDB_DIRECT_CHANNEL   = 'Direct_1'

NIFECGDB_ABDOMINAL_PREFIX = 'abdomen_'
NIFECGDB_THORACIC_PREFIX  = 'thorax_'
NIFECGDB_MAX_ABD_CHANNELS = 4

# -- Random seed -------------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)