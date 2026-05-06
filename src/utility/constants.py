"""
constants.py

Shared constants for the synthetic data evaluation pipeline.

This module is the single source of truth for project-wide conventions that
all scripts must follow so the pipeline remains easy to orchestrate and
dashboard later. In particular, it centralises:

- reproducibility defaults
- fixed experiment choices
- directory and artifact naming conventions
- result envelope schema keys
- shared serialization defaults

W&B configuration (entity, project) is loaded from environment variables
so that personal credentials are never committed to the repository.
See .env.example for the required variables.

Usage:
    from src.utility.constants import (
        RANDOM_STATE,
        WANDB_PROJECT,
        SYNTHESIZERS,
        DP_SYNTHESIZERS,
        DP_EPSILONS,
        BASE_DIR,
        DATA_DIR,
        MODELS_DIR,
        SYNTHESIZER_MODELS_DIR,
        CONFIG_DIR,
        PROCESSED_DATA_DIR,
        SYNTHETIC_DATA_DIR,
        RESULTS_DIR,
    )
"""

from pathlib import Path

# ── Reproducibility ───────────────────────────────────────────────────────────

RANDOM_STATE = 42

# ── W&B ───────────────────────────────────────────────────────────────────────

# Override by setting WANDB_PROJECT in your environment or .env file.
WANDB_PROJECT = "synthetic-data-eval"

# ── Experiment choices (fixed project scope) ─────────────────────────────────

SYNTHESIZERS = {"gaussian_copula", "ctgan", "tvae"}

DP_SYNTHESIZERS = {"dpctgan", "patectgan"}

# Epsilon values following a logarithmic scale spanning the range commonly
# used in DP literature. epsilon=0.1 represents a strong privacy guarantee,
# epsilon=1 corresponds to the threshold cited in Dwork & Roth (2014),
# epsilon=5 and epsilon=10 represent progressively weaker guarantees.
DP_EPSILONS = [0.1, 1.0, 5.0, 10.0]

# Fraction of epsilon allocated to the preprocessor for inferring continuous
# column bounds. Standard practice to avoid consuming the full budget on
# preprocessing alone.
DP_PREPROCESSOR_EPS_FRACTION = 0.1

CLASSIFIERS = ["logistic_regression", "random_forest", "gradient_boosting"]

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"

CONFIG_DIR = BASE_DIR / "config"

MODELS_DIR = BASE_DIR / "models"
SYNTHESIZER_MODELS_DIR = MODELS_DIR / "synthesizers"

RESULTS_DIR = BASE_DIR / "results"

# ── Canonical artifact names ──────────────────────────────────────────────────

TRAIN_FILENAME = "train.csv"
VALIDATION_FILENAME = "validation.csv"
TEST_FILENAME = "test.csv"
SYNTHETIC_TRAIN_FILENAME = "synthetic_train.csv"

# ── Serialization defaults ────────────────────────────────────────────────────

DEFAULT_ENCODING = "utf-8"
JSON_INDENT = 2

# ── Result envelope schema ────────────────────────────────────────────────────

RESULTS_SCHEMA_VERSION = "1.0"

RESULTS_KEY_SCHEMA_VERSION = "schema_version"
RESULTS_KEY_SCRIPT = "script"
RESULTS_KEY_RUN_NAME = "run_name"
RESULTS_KEY_TIMESTAMP = "timestamp"
RESULTS_KEY_PARAMETERS = "parameters"
RESULTS_KEY_RESULTS = "results"
