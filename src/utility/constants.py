"""
Shared constants for the synthetic data evaluation pipeline.

This module centralizes:
- reproducibility defaults
- experiment scope
- artifact naming conventions
- directory layout
- result schema conventions
- serialization defaults
"""

from pathlib import Path

# ── Reproducibility ───────────────────────────────────────────────────────────

RANDOM_STATE = 42

# ── W&B ───────────────────────────────────────────────────────────────────────

WANDB_PROJECT = "synthetic-data-eval"

# ── Experiment scope ──────────────────────────────────────────────────────────

SYNTHESIZERS = {"gaussian_copula", "ctgan", "tvae"}

DP_SYNTHESIZERS = {"dpctgan", "patectgan"}

# Epsilon values spanning strong to progressively weaker privacy guarantees.
#
# epsilon=0.1:
#     very strong privacy guarantee with typically substantial utility loss
#
# epsilon=0.5:
#     intermediate strong-privacy setting to better evaluate the transition
#     region below epsilon=1
#
# epsilon=1.0:
#     commonly cited reference threshold in DP literature
#
# epsilon=5.0 and epsilon=10.0:
#     progressively weaker privacy guarantees with improved utility
DP_EPSILONS = [0.1, 0.5, 1.0, 5.0, 10.0]

DP_PREPROCESSOR_EPS_FRACTION = 0.1

CLASSIFIERS = [
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
]

# ── Dataset split filenames ───────────────────────────────────────────────────

TRAIN_FILENAME = "train.csv"
VALIDATION_FILENAME = "validation.csv"
TEST_FILENAME = "test.csv"

SYNTHETIC_TRAIN_FILENAME = "synthetic_train.csv"

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
CLEANED_DATA_DIR = DATA_DIR / "cleaned"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"

MODELS_DIR = BASE_DIR / "models"
SYNTHESIZER_MODELS_DIR = MODELS_DIR / "synthesizers"

CONFIG_DIR = BASE_DIR / "config"

RESULTS_DIR = BASE_DIR / "results"

# ── Result schema ─────────────────────────────────────────────────────────────

RESULTS_SCHEMA_VERSION = "1.0"

RESULTS_KEY_SCHEMA_VERSION = "schema_version"
RESULTS_KEY_TIMESTAMP = "timestamp"
RESULTS_KEY_RUN_NAME = "run_name"
RESULTS_KEY_SCRIPT = "script"
RESULTS_KEY_PARAMETERS = "parameters"
RESULTS_KEY_RESULTS = "results"

# ── Serialization ─────────────────────────────────────────────────────────────

DEFAULT_ENCODING = "utf-8"
JSON_INDENT = 2
