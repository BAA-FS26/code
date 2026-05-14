"""
Data loading and preprocessing helpers for classifier training and evaluation.

This module loads real or synthetic training data, loads real validation data,
and applies classifier-specific preprocessing. It returns fitted preprocessors
so saved models can be evaluated consistently later.
"""

import pandas as pd

from src.core.io import load_csv, validate_columns
from src.core.paths import processed_split_path, synthetic_train_path
from src.dataset.adult_census import TARGET_COL
from src.dataset.feature_engineering import (
    build_preprocessor_logistic_regression,
    build_preprocessor_random_forest,
    prepare_data,
    prepare_data_gradient_boosting,
)
from src.utility.constants import TRAIN_FILENAME, VALIDATION_FILENAME


def load_splits(data_source: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train and validation splits from disk.

    For real data, both train and val are loaded from processed data.
    For synthetic data (TSTR), the synthetic train split is loaded as
    training data while the real val split is used for evaluation.
    This ensures that hyperparameter tuning is always evaluated against
    real data distributions.

    The test split is intentionally excluded here. Final held-out test
    evaluation is handled by src.evaluation.evaluate_utility.

    Args:
        data_source: One of 'real', 'gaussian_copula', 'ctgan', 'tvae', or a
                     DP source in the form 'dpctgan/eps_1.0'.

    Returns:
        Tuple of (train_df, val_df) DataFrames.

    Raises:
        FileNotFoundError: If required input files are missing.
        ValueError: If loaded training data schema does not match validation schema.
    """
    if data_source == "real":
        train_path = processed_split_path(TRAIN_FILENAME)
    else:
        train_path = synthetic_train_path(data_source)

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
) -> tuple:
    """
    Apply classifier-specific preprocessing to train and val splits.

    The preprocessor is fitted on the training set only and applied
    consistently to the val split. Returns the fitted preprocessor
    so it can be reused on test or synthetic data.

    Args:
        classifier_name: One of 'logistic_regression', 'random_forest',
                         'gradient_boosting'.
        train_df: Training split DataFrame including target column.
        val_df: Validation split DataFrame including target column.

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, preprocessor).
        preprocessor is None for gradient_boosting.
    """
    preprocessor = _get_preprocessor(classifier_name, train_df)
    X_train, y_train = _apply_preprocessor(classifier_name, train_df, preprocessor)
    X_val, y_val = _apply_preprocessor(classifier_name, val_df, preprocessor)
    return X_train, y_train, X_val, y_val, preprocessor


def prepare_single(classifier_name: str, df: pd.DataFrame, preprocessor) -> tuple:
    """
    Apply classifier-specific preprocessing to a single DataFrame.

    Uses a previously fitted preprocessor to transform the data
    consistently. Used for test set evaluation.

    Args:
        classifier_name: One of 'logistic_regression', 'random_forest',
                         'gradient_boosting'.
        df: DataFrame including target column to transform.
        preprocessor: Fitted preprocessor from prepare_splits().
                      Pass None for gradient_boosting.

    Returns:
        Tuple of (X, y).
    """
    if classifier_name != "gradient_boosting" and preprocessor is None:
        raise ValueError(
            f"Classifier '{classifier_name}' requires a fitted preprocessor."
        )

    return _apply_preprocessor(classifier_name, df, preprocessor)


def _get_preprocessor(classifier_name: str, train_df: pd.DataFrame):
    """
    Build and fit the appropriate preprocessor for a given classifier.

    Returns None for gradient_boosting, which handles preprocessing natively.

    Args:
        classifier_name: One of 'logistic_regression', 'random_forest',
                         'gradient_boosting'.
        train_df: Training DataFrame including the target column.

    Returns:
        Fitted ColumnTransformer, or None for gradient_boosting.
    """
    builders = {
        "logistic_regression": build_preprocessor_logistic_regression,
        "random_forest": build_preprocessor_random_forest,
    }
    builder = builders.get(classifier_name)
    if builder is None:
        return None
    return builder(train_df.drop(columns=[TARGET_COL]))


def _apply_preprocessor(
    classifier_name: str,
    df: pd.DataFrame,
    preprocessor,
) -> tuple:
    """
    Apply classifier-specific preprocessing to a DataFrame.

    Args:
        classifier_name: One of 'logistic_regression', 'random_forest',
                         'gradient_boosting'.
        df: DataFrame including the target column to transform.
        preprocessor: Fitted preprocessor, or None for gradient_boosting.

    Returns:
        Tuple of (X, y).
    """
    if classifier_name == "gradient_boosting":
        return prepare_data_gradient_boosting(df)
    return prepare_data(preprocessor, df)
