# src/core/paths.py

"""
Canonical path helpers for the synthetic data evaluation pipeline.

These helpers centralize path construction while preserving the existing
directory structure, filenames, and result-file compatibility.
"""

from pathlib import Path

from src.core.data_source import REAL_DATA_SOURCE
from src.utility.constants import (
    CONFIG_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    SYNTHESIZER_MODELS_DIR,
    SYNTHETIC_DATA_DIR,
    SYNTHETIC_TRAIN_FILENAME,
)


def processed_split_path(filename: str) -> Path:
    """Return the canonical path to a processed split file."""
    return PROCESSED_DATA_DIR / filename


def synthetic_train_path(data_source: str) -> Path:
    """
    Return the canonical synthetic training-data path.

    Examples:
        ctgan -> data/synthetic/ctgan/default/synthetic_train.csv
        dpctgan/eps_1.0 -> data/synthetic/dpctgan/eps_1.0/synthetic_train.csv
    """
    if data_source == REAL_DATA_SOURCE:
        raise ValueError("Real data does not have a synthetic training-data path.")

    if "/" in data_source:
        return SYNTHETIC_DATA_DIR / data_source / SYNTHETIC_TRAIN_FILENAME

    return SYNTHETIC_DATA_DIR / data_source / "default" / SYNTHETIC_TRAIN_FILENAME


def synthetic_output_dir(synthesizer_name: str, mode: str = "default") -> Path:
    """Return the canonical output directory for generated synthetic data."""
    return SYNTHETIC_DATA_DIR / synthesizer_name / mode


def synthetic_output_path(synthesizer_name: str, mode: str = "default") -> Path:
    """Return the canonical output path for generated synthetic training data."""
    return synthetic_output_dir(synthesizer_name, mode) / SYNTHETIC_TRAIN_FILENAME


def synthesizer_metadata_path(synthesizer_name: str) -> Path:
    """Return the canonical saved SDV metadata path for a synthesizer."""
    return SYNTHESIZER_MODELS_DIR / f"{synthesizer_name}_metadata.json"


def synthesizer_model_path(synthesizer_name: str, mode: str = "default") -> Path:
    """Return the canonical saved synthesizer model path."""
    return SYNTHESIZER_MODELS_DIR / f"{synthesizer_name}_{mode}.pkl"


def classifier_model_path(
    classifier_name: str,
    data_source: str,
    model_type: str,
) -> Path:
    """Return the canonical saved classifier model path."""
    safe_source = data_source.replace("/", "_")
    return MODELS_DIR / f"{safe_source}_{classifier_name}_{model_type}.pkl"


def best_params_path(classifier_name: str, data_source: str) -> Path:
    """Return the canonical best-params config path."""
    safe_source = data_source.replace("/", "_")
    return CONFIG_DIR / f"best_{classifier_name}_{safe_source}.yaml"
