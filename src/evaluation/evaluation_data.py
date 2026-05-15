"""
Shared data-loading helpers for evaluation scripts.

This module centralizes evaluation-specific data access so fidelity, privacy,
and utility evaluation use the same split loading and schema validation logic.
"""

from pathlib import Path

import pandas as pd

from src.core.io import load_csv, validate_matching_columns
from src.core.paths import processed_split_path, synthetic_train_path
from src.utility.constants import (
    TEST_FILENAME,
    TRAIN_FILENAME,
    VALIDATION_FILENAME,
)


def load_utility_test_dataset() -> tuple[pd.DataFrame, Path]:
    """
    Load the real held-out test split for final utility evaluation.

    Utility evaluation always tests on real data, regardless of whether the
    model was trained on real or synthetic data.
    """
    return _load_real_test()


def load_fidelity_datasets(
    data_source: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    """
    Load real training data and synthetic training data for fidelity evaluation.
    """
    real_train_df, real_train_path = _load_real_train()
    synthetic_df, synthetic_path = _load_synthetic_train(data_source)

    validate_matching_columns(
        reference_df=real_train_df,
        candidate_df=synthetic_df,
        candidate_name="Synthetic training data",
    )

    return (
        real_train_df,
        synthetic_df,
        {
            "real_train_path": real_train_path,
            "synthetic_path": synthetic_path,
        },
    )


def load_privacy_datasets(
    data_source: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    """
    Load train, holdout, and synthetic datasets for privacy evaluation.

    Holdout is validation + test. The test split is used only as privacy
    control data here, not for model training or hyperparameter tuning.
    """
    train_df, train_path = _load_real_train()
    validation_df, validation_path = _load_real_validation()
    test_df, test_path = _load_real_test()
    synthetic_df, synthetic_path = _load_synthetic_train(data_source)

    validate_matching_columns(train_df, validation_df, "Validation split")
    validate_matching_columns(train_df, test_df, "Test split")
    validate_matching_columns(train_df, synthetic_df, "Synthetic training data")

    holdout_df = pd.concat([validation_df, test_df], ignore_index=True)

    return (
        train_df,
        holdout_df,
        synthetic_df,
        {
            "train_path": train_path,
            "validation_path": validation_path,
            "test_path": test_path,
            "synthetic_path": synthetic_path,
        },
    )


def _load_real_train() -> tuple[pd.DataFrame, Path]:
    """Load the real training split."""
    path = processed_split_path(TRAIN_FILENAME)
    return load_csv(path, "Real training split"), path


def _load_real_validation() -> tuple[pd.DataFrame, Path]:
    """Load the real validation split."""
    path = processed_split_path(VALIDATION_FILENAME)
    return load_csv(path, "Validation split"), path


def _load_real_test() -> tuple[pd.DataFrame, Path]:
    """Load the real held-out test split."""
    path = processed_split_path(TEST_FILENAME)
    return load_csv(path, "Test split"), path


def _load_synthetic_train(data_source: str) -> tuple[pd.DataFrame, Path]:
    """
    Load synthetic training data for a canonical data source.

    Examples:
        ctgan
        dpctgan/eps_1.0
    """
    path = synthetic_train_path(data_source)
    return load_csv(path, "Synthetic training data"), path
