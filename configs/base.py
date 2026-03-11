"""
configs/base.py
BaseConfig: Unified configuration class for all datasets.

This replaces the duplicated config.py and config_nifecgdb.py files.
Dataset-specific overrides are applied via YAML configuration files.

All parameters are stored as class attributes and can be accessed as:
    config = BaseConfig()
    value = config.PARAMETER_NAME
"""

import numpy as np


class BaseConfig:
    """
    Unified configuration for PHASE pipeline.
    
    Supports per-dataset overrides via YAML files. Initialize with:
        config = BaseConfig(dataset='adfecgdb')  # applies adfecgdb.yaml overrides
    """

    # -- Sampling ---------------------------------------------------------------
    FS = 1000  # Hz

    # -- Preprocessing ----------------------------------------------------------
    BANDPASS_LOW = 1.0
    BANDPASS_HIGH = 45.0
    BANDPASS_ORDER = 4
    NOTCH_FREQ = 50.0
    NOTCH_QUALITY = 30.0
    MEDFILT_KERNEL = 3

    # -- Physiological Constraints (ADFECGDB defaults) -------------------------
    MATERNAL_HR_MIN = 55
    MATERNAL_HR_MAX = 110
    FETAL_HR_MIN = 100
    FETAL_HR_MAX = 185
    FETAL_HR_LOW = 100
    FETAL_HR_HIGH = 185
    FETAL_HR_CENTRE = 130

    QRS_SIGMA_SEC = 0.04
    QRS_BASELINE_WEIGHT = 0.05

    # -- ICA ------------------------------------------------------------------
    ICA_N_COMPONENTS = 4
    ICA_MAX_ITER = 2000
    ICA_RANDOM_STATE = 42
    ICA_TOL = 1e-6

    # -- Adaptive Windowed WSVD -------------------------------------------------
    WSVD_WINDOW_SEC = 2.0
    WSVD_OVERLAP = 0.5
    WSVD_N_COMPONENTS = 3
    WSVD_COMPONENT_CORR_THRESH = 0.30
    WSVD_MAX_ENERGY_REMOVAL = 0.70
    WSVD_CHANNEL_R2_MIN = 0.35

    # -- Pan-Tompkins -- MATERNAL -----------------------------------------------
    PT_MATERNAL_BANDPASS_LOW = 5.0
    PT_MATERNAL_BANDPASS_HIGH = 15.0
    PT_MATERNAL_BANDPASS_ORDER = 2

    # -- Pan-Tompkins -- FETAL --------------------------------------------------
    PT_FETAL_BANDPASS_LOW = 10.0
    PT_FETAL_BANDPASS_HIGH = 40.0
    PT_FETAL_BANDPASS_ORDER = 2

    # -- Pan-Tompkins shared -----------------------------------------------------
    PT_INTEGRATION_WINDOW_SEC = 0.08
    PT_THRESHOLD_FACTOR = 1.0

    # -- EKF -------------------------------------------------------------------
    EKF_FETAL_HR_INIT = 150
    EKF_PROCESS_NOISE = [0.1, 0.1, 1.0]
    EKF_OBSERVE_NOISE = 1.0
    EKF_STATE_COV_INIT = 1.0

    EKF_PQRST_PARAMS = np.array([
        [0.30, 0.10, -np.pi / 3],       # P wave
        [-0.50, 0.05, -np.pi / 12],     # Q wave
        [1.50, 0.10, 0.0],              # R wave
        [-0.50, 0.05, np.pi / 12],      # S wave
        [0.30, 0.20, np.pi / 2],        # T wave
    ])

    # -- Path B ICA2 maternal residual exclusion --------------------------------
    MATERNAL_ICA2_CORR_THRESH = 0.30

    # -- Evaluation ---------------------------------------------------------------
    EVAL_TOLERANCE_MS = 50
    EVAL_MIN_PEAK_HEIGHT = 0.35
    EVAL_MIN_PEAK_DISTANCE_SEC = 0.33

    # -- ECHO XAI -----------------------------------------------------------------
    ECHO_MATERNAL_EXCLUSION_SEC = 0.08

    # -- Dataset Metadata (ADFECGDB defaults) -----------------------------------
    ADFECGDB_ABDOMEN_CHANNELS = ['Abdomen_1', 'Abdomen_2', 'Abdomen_3', 'Abdomen_4']
    ADFECGDB_DIRECT_CHANNEL = 'Direct_1'

    NIFECGDB_ABDOMINAL_PREFIX = 'abdomen_'
    NIFECGDB_THORACIC_PREFIX = 'thorax_'
    NIFECGDB_MAX_ABD_CHANNELS = 4

    # -- Random seed ----------------------------------------------------------
    RANDOM_SEED = 42
    HR_SEP_MIN_BPM = 15
    PATH_A_PREFERENCE = 1.5

    def __init__(self, dataset: str = "adfecgdb"):
        """
        Initialize BaseConfig. Optionally apply dataset-specific YAML overrides.
        
        Parameters
        ----------
        dataset : str
            Dataset name ('adfecgdb' or 'nifecgdb'). Used to load YAML overrides.
        """
        self.dataset = dataset.lower()
        self._load_overrides()
        np.random.seed(self.RANDOM_SEED)

    def _load_overrides(self):
        """Load and apply dataset-specific YAML overrides if they exist."""
        from pathlib import Path
        import yaml

        override_file = Path(__file__).parent / f"{self.dataset}.yaml"
        if override_file.exists():
            with open(override_file, 'r') as f:
                overrides = yaml.safe_load(f) or {}
            
            # Apply overrides to this config instance
            for key, value in overrides.items():
                setattr(self, key.upper(), value)

    def __repr__(self):
        return f"<BaseConfig dataset={self.dataset}>"
