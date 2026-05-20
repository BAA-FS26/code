"""
Data loading and preprocessing helpers for classifier training and evaluation.
"""

from typing import Any

import pandas as pd

from src.core.data_source import REAL_DATA_SOURCE
from src.core.io import load_csv, validate_columns
from src.core.paths import processed_split_path, synthetic_train_path
from src.dataset.feature_engineering import (
    build_preprocessor_logistic_regression,
    build_preprocessor_random_forest,
    prepare_data,
    prepare_data_gradient_boosting,
)
from src.utility.constants import TRAIN_FILENAME, VALIDATION_FILENAME

PreparedDataset = tuple[Any, Any]
PreparedSplits = tuple[Any, Any, Any, Any, Any]


def load_splits(data_source: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load training and validation splits for classifier training."""
    train_path = (
        processed_split_path(TRAIN_FILENAME)
        if data_source == REAL_DATA_SOURCE
        else synthetic_train_path(data_source)
    )
    val_path = processed_split_path(VALIDATION_FILENAME)

    train_df = load_csv(train_path, "Training split")
    val_df = load_csv(val_path, "Validation split")

    validate_columns(
        train_df,
        expected_columns=list(val_df.columns),
        dataframe_name=f"Training data for source '{data_source}'",
    )

    print(
        f"[classify] Loaded training data for source '{data_source}' "
        f"({len(train_df)} rows)"
    )
    print(f"[classify] Loaded validation data ({len(val_df)} rows)")

    return train_df, val_df


def prepare_splits(
    classifier_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> PreparedSplits:
    """Apply classifier-specific preprocessing to train and validation splits."""
    if classifier_name == "logistic_regression":
        preprocessor = build_preprocessor_logistic_regression(train_df)
        X_train, y_train = prepare_data(preprocessor, train_df)
        X_val, y_val = prepare_data(preprocessor, val_df)

    elif classifier_name == "random_forest":
        preprocessor = build_preprocessor_random_forest(train_df)
        X_train, y_train = prepare_data(preprocessor, train_df)
        X_val, y_val = prepare_data(preprocessor, val_df)

    elif classifier_name == "gradient_boosting":
        preprocessor = None
        X_train, y_train = prepare_data_gradient_boosting(train_df)
        X_val, y_val = prepare_data_gradient_boosting(val_df)

    else:
        raise ValueError(f"Unsupported classifier: {classifier_name}")

    return X_train, y_train, X_val, y_val, preprocessor


def prepare_single(
    classifier_name: str,
    df: pd.DataFrame,
    preprocessor: Any,
) -> PreparedDataset:
    """Prepare a single dataset using an already fitted preprocessor."""
    if classifier_name in {"logistic_regression", "random_forest"}:
        if preprocessor is None:
            raise ValueError(
                f"preprocessor is required for classifier '{classifier_name}'."
            )

        return prepare_data(preprocessor, df)

    if classifier_name == "gradient_boosting":
        return prepare_data_gradient_boosting(df)

    raise ValueError(f"Unsupported classifier: {classifier_name}")
