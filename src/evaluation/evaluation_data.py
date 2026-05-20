"""
Shared data-loading helpers for evaluation scripts.

Centralizes evaluation-specific data access so Fidelity, Privacy, and Utility
evaluation use the same split loading and schema validation logic.
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
    """Load the real held-out test split for final utility evaluation."""
    return _load_real_test()


def load_fidelity_datasets(
    data_source: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    """Load real and synthetic training data for Fidelity evaluation."""
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
    Load real train, real holdout, and synthetic data for Privacy evaluation.

    Holdout is validation + test. The test split is used here only as privacy
    control data, not for model training or hyperparameter tuning.
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
    return _load_processed_split(TRAIN_FILENAME, "Real training split")


def _load_real_validation() -> tuple[pd.DataFrame, Path]:
    """Load the real validation split."""
    return _load_processed_split(VALIDATION_FILENAME, "Validation split")


def _load_real_test() -> tuple[pd.DataFrame, Path]:
    """Load the real held-out test split."""
    return _load_processed_split(TEST_FILENAME, "Test split")


def _load_processed_split(
    filename: str,
    description: str,
) -> tuple[pd.DataFrame, Path]:
    """Load one processed real-data split."""
    path = processed_split_path(filename)
    return load_csv(path, description), path


def _load_synthetic_train(data_source: str) -> tuple[pd.DataFrame, Path]:
    """Load synthetic training data for a canonical data-source key."""
    path = synthetic_train_path(data_source)
    return load_csv(path, "Synthetic training data"), path
