"""
constants.py

Shared constants for the synthetic data evaluation pipeline.
All project-wide values are defined here so they can be changed in one place.

Usage:
    from src.utility.constants import (
        RANDOM_STATE,
        WANDB_PROJECT,
        WANDB_ENTITY,
        SYNTHESIZERS,
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

WANDB_PROJECT = "synthetic-data-eval"
WANDB_ENTITY = "baa_fs26_pm"  # TODO: replace with your W&B entity

# ── Synthesizers ──────────────────────────────────────────────────────────────

SYNTHESIZERS = ["gaussian_copula", "ctgan", "tvae"]

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
SYNTHESIZER_MODELS_DIR = MODELS_DIR / "synthesizers"
