"""
constants.py

Shared constants for the synthetic data evaluation pipeline.
All project-wide values are defined here so they can be changed in one place.

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
    )
"""

from pathlib import Path

# ── Reproducibility ───────────────────────────────────────────────────────────

RANDOM_STATE = 42

# ── W&B ───────────────────────────────────────────────────────────────────────

# Override by setting WANDB_PROJECT in your environment or .env file.
WANDB_PROJECT = "synthetic-data-eval"

# ── Synthesizers ──────────────────────────────────────────────────────────────

SYNTHESIZERS = ["gaussian_copula", "ctgan", "tvae"]

# ── DP Synthesizers ───────────────────────────────────────────────────────────

DP_SYNTHESIZERS = ["dpctgan", "patectgan"]

# Epsilon values following a logarithmic scale spanning the range commonly
# used in DP literature. epsilon=0.1 represents a strong privacy guarantee,
# epsilon=1 corresponds to the threshold cited in Dwork & Roth (2014),
# epsilon=5 and epsilon=10 represent progressively weaker guarantees.
DP_EPSILONS = [0.1, 1.0, 5.0, 10.0]

# Fraction of epsilon allocated to the preprocessor for inferring continuous
# column bounds. Standard practice to avoid consuming the full budget on
# preprocessing alone.
DP_PREPROCESSOR_EPS_FRACTION = 0.1

# ── Classifiers ───────────────────────────────────────────────────────────────

CLASSIFIERS = ["logistic_regression", "random_forest", "gradient_boosting"]

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
SYNTHESIZER_MODELS_DIR = MODELS_DIR / "synthesizers"
RESULTS_DIR = BASE_DIR / "results"
