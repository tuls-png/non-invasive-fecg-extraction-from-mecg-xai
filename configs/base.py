"""
configs/base.py
BaseConfig: Unified configuration class for all datasets.

This replaces the duplicated config.py and config_nifecgdb.py files.
Dataset-specific overrides are applied via YAML configuration files.

All parameters are stored as class attributes and can be accessed as:
    config = BaseConfig()
    value = config.PARAMETER_NAME

DISSERTATION MODIFICATION [Literature Review / Enhancement Roadmap]:
  All new hyperparameters introduced by Rank 1-6 of the enhancement
  roadmap are added below as class-level defaults, each individually
  overridable via the per-dataset YAML files (configs/<dataset>.yaml)
  using the same lower-case-key -> UPPER_CASE-attribute convention as
  every existing parameter. Every new feature is also individually
  switchable via an *_ENABLED boolean so it can be turned off for
  ablation studies without touching pipeline.py.
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

    # [Rank 5] EKF forward-backward (RTS) smoothing default. Exposed here so
    # it is a dataset-tunable/ablatable hyperparameter instead of only a
    # PHASEPipeline constructor argument. PHASEPipeline(use_rts=None) will
    # fall back to this value; explicit True/False still overrides it.
    EKF_USE_RTS_DEFAULT = True

    # -- Path B ICA2 maternal residual exclusion --------------------------------
    MATERNAL_ICA2_CORR_THRESH = 0.30

    # -- Evaluation ---------------------------------------------------------------
    EVAL_TOLERANCE_MS = 50
    EVAL_MIN_PEAK_HEIGHT = 0.35
    EVAL_MIN_PEAK_DISTANCE_SEC = None  # Computed dynamically in __init__

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

    # -- Path selection -------------------------------------------------------
    # Used as SCORE multiplier in Step 9:
    #   Path A chosen when a_score >= b_score * PATH_A_PREFERENCE
    # Default 1.5 for ADFECGDB. cinc2013.yaml overrides to 1.1.
    PATH_A_PREFERENCE = 1.0

    # -- Confidence gate ------------------------------------------------------
    # chosen_ic_selection_score < threshold → low_confidence=True in metadata.
    CONFIDENCE_GATE_THRESHOLD = 0.05

    # =====================================================================
    # [Rank 1] Path C — Adaptive Windowed Weighted SVD, epoch domain
    # =====================================================================
    # Path B (separation/wsvd.py) performs adaptive windowed weighted SVD
    # along the channel x time axis. Path C performs the same underlying
    # operation -- adaptive, windowed, weighted low-rank decomposition for
    # maternal-component estimation -- along the beat-epoch x within-beat-
    # sample axis instead (see separation/template_subtraction.py module
    # docstring for the full two-axis framing). Estimates a maternal PQRST
    # template from a local window of aligned beats and subtracts it
    # directly at each maternal beat location, at a least-squares-optimal
    # per-beat amplitude scale.
    PATH_C_ENABLED = True
    TEMPLATE_HALF_WINDOW_SEC = 0.15      # +/- window around each maternal R-peak
    TEMPLATE_UPDATE_EVERY_BEATS = 20     # re-estimate template every N beats
    TEMPLATE_CONTEXT_BEATS = 15          # beats on each side used to build a local template
    TEMPLATE_MIN_BEATS = 5               # minimum maternal beats required to run Path C
    # "median": robust (outlier-beat-resistant) estimate of the recurring
    #   beat shape across the local beat-epoch matrix. Default.
    # "svd": explicit weighted SVD of the beat-epoch matrix (top
    #   `TEMPLATE_SVD_N_COMPONENTS` singular modes), the direct epoch-
    #   domain analogue of Path B's channel-time weighted SVD, and the
    #   formulation matching Kanjilal, Palit & Saha (1997) TS_SVD.
    # Compare both via scripts/compare_template_estimators.py before
    # relying on either in the dissertation write-up.
    TEMPLATE_ESTIMATOR = "median"
    TEMPLATE_SVD_N_COMPONENTS = 1        # used only when TEMPLATE_ESTIMATOR="svd"

    # =====================================================================
    # [Rank 2] SQI-weighted fusion across Path A / B / C
    # =====================================================================
    # Promotes evaluation/sqi.py from a diagnostic role to a first-class
    # trust-weighting input in path selection (pipeline.py Step 9) and in
    # IC candidate scoring inside _best_ic_ensemble().
    SQI_FUSION_ENABLED = True
    SQI_FUSION_WEIGHT = 0.35             # blend weight of SQI vs raw unified score, in [0, 1]
    SQI_KURTOSIS_NORM = 20.0             # normalisation constant for the kurtosis sub-score
    SQI_MIN_QUALITY_THRESH = 0.15        # used by evaluation.sqi.select_best_channels()

    # =====================================================================
    # [Rank 3] Periodicity-constrained IC scoring
    # =====================================================================
    # Adds an autocorrelation-peak-sharpness bonus term to score_fetal_ic()
    # / score_maternal_ic() in separation/ica.py, following Sameni's
    # periodic-component-analysis lineage and Kotas et al. (2024).
    PERIODICITY_SCORE_ENABLED = True
    PERIODICITY_SCORE_WEIGHT = 1.0       # scales the periodicity bonus contribution

    # =====================================================================
    # [Rank 4] Adaptive filter (RLS/NLMS) residual cleanup
    # =====================================================================
    # Runs after subtract_maternal() and before ICA2, using the AW-WSVD
    # maternal reconstruction as the adaptive-filter reference signal, to
    # mop up residual periodic maternal leakage before Path B's ICA stage.
    ADAPTIVE_FILTER_ENABLED = True
    ADAPTIVE_FILTER_METHOD = "rls"       # "rls" or "nlms"
    ADAPTIVE_FILTER_N_TAPS = 5
    ADAPTIVE_FILTER_FORGETTING = 0.995   # RLS forgetting factor (lambda)
    ADAPTIVE_FILTER_DELTA = 1.0          # RLS inverse-correlation init constant
    ADAPTIVE_FILTER_STEP_SIZE = 0.02     # NLMS step size (mu)
    ADAPTIVE_FILTER_EPS = 1e-6           # NLMS division-by-zero guard

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
        
        # Compute EVAL_MIN_PEAK_DISTANCE_SEC dynamically from FETAL_HR_MAX
        # Minimum peak distance = minimum RR interval = 60 sec / max HR
        if self.EVAL_MIN_PEAK_DISTANCE_SEC is None:
            self.EVAL_MIN_PEAK_DISTANCE_SEC = 60.0 / self.FETAL_HR_MAX

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
