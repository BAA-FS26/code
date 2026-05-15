# src/core/paths.py

"""
Canonical path helpers for the synthetic data evaluation pipeline.

These helpers centralize path construction while preserving the existing
directory structure, filenames, and result-file compatibility.

Path conventions are intentionally stable because they are relied upon by:
- synthesis scripts
- classifier training
- evaluation scripts
- saved model loading
- dashboard result loading
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


def _safe_data_source_name(data_source: str) -> str:
    """
    Convert a canonical data-source key into a filename-safe identifier.

    DP data sources contain forward slashes:
        dpctgan/eps_1.0

    These are replaced with underscores when constructing filenames:
        dpctgan_eps_1.0

    This preserves compatibility across operating systems while keeping
    filenames readable and stable.
    """
    return data_source.replace("/", "_")


def processed_split_path(filename: str) -> Path:
    """
    Return the canonical path to a processed split file.

    Examples:
        train.csv
        validation.csv
        test.csv
    """
    return PROCESSED_DATA_DIR / filename


def synthetic_train_path(data_source: str) -> Path:
    """
    Return the canonical synthetic training-data path.

    Examples:
        ctgan
            -> data/synthetic/ctgan/default/synthetic_train.csv

        dpctgan/eps_1.0
            -> data/synthetic/dpctgan/eps_1.0/synthetic_train.csv

    Raises:
        ValueError:
            If called with the real-data source.
    """
    if data_source == REAL_DATA_SOURCE:
        raise ValueError("Real data does not have a synthetic training-data path.")

    if "/" in data_source:
        return SYNTHETIC_DATA_DIR / data_source / SYNTHETIC_TRAIN_FILENAME

    return SYNTHETIC_DATA_DIR / data_source / "default" / SYNTHETIC_TRAIN_FILENAME


def synthetic_output_dir(
    synthesizer_name: str,
    mode: str = "default",
) -> Path:
    """
    Return the canonical output directory for generated synthetic data.

    Examples:
        data/synthetic/ctgan/default/
        data/synthetic/dpctgan/eps_1.0/
    """
    return SYNTHETIC_DATA_DIR / synthesizer_name / mode


def synthetic_output_path(
    synthesizer_name: str,
    mode: str = "default",
) -> Path:
    """
    Return the canonical output path for generated synthetic training data.
    """
    return synthetic_output_dir(synthesizer_name, mode) / SYNTHETIC_TRAIN_FILENAME


def synthesizer_metadata_path(
    synthesizer_name: str,
) -> Path:
    """
    Return the canonical saved SDV metadata path.

    Example:
        models/synthesizers/ctgan_metadata.json
    """
    return SYNTHESIZER_MODELS_DIR / f"{synthesizer_name}_metadata.json"


def synthesizer_model_path(
    synthesizer_name: str,
    mode: str = "default",
) -> Path:
    """
    Return the canonical saved synthesizer model path.

    Examples:
        models/synthesizers/ctgan_default.pkl
        models/synthesizers/dpctgan_eps_1.0.pkl
    """
    return SYNTHESIZER_MODELS_DIR / f"{synthesizer_name}_{mode}.pkl"


def classifier_model_path(
    classifier_name: str,
    data_source: str,
    model_type: str,
) -> Path:
    """
    Return the canonical saved classifier model path.

    Examples:
        models/real_gradient_boosting_best.pkl
        models/ctgan_gradient_boosting_best.pkl
        models/dpctgan_eps_1.0_gradient_boosting_best.pkl
    """
    safe_source = _safe_data_source_name(data_source)

    return MODELS_DIR / f"{safe_source}_{classifier_name}_{model_type}.pkl"


def best_params_path(
    classifier_name: str,
    data_source: str,
) -> Path:
    """
    Return the canonical best-parameter config path.

    Examples:
        config/best_gradient_boosting_real.yaml
        config/best_gradient_boosting_ctgan.yaml
        config/best_gradient_boosting_dpctgan_eps_1.0.yaml
    """
    safe_source = _safe_data_source_name(data_source)

    return CONFIG_DIR / f"best_{classifier_name}_{safe_source}.yaml"
