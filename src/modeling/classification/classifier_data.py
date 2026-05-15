"""
Data loading and preprocessing helpers for classifier training and evaluation.

This module loads real or synthetic training data, loads real validation data,
and applies classifier-specific preprocessing. It returns fitted preprocessors
so saved models can be evaluated consistently later.
"""

from typing import Any

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

PreparedDataset = tuple[Any, Any]
PreparedSplits = tuple[Any, Any, Any, Any, Any]


def load_splits(data_source: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train and validation splits from disk.

    For real data, both train and validation splits are loaded from the
    processed real-data directory.

    For synthetic-data workflows (TSTR), the synthetic training split is
    loaded as training data while the real validation split is always used
    for evaluation. This ensures that hyperparameter tuning remains grounded
    in the real data distribution.

    The held-out test split is intentionally excluded here. Final test-set
    evaluation is handled separately in src.evaluation.evaluate_utility.

    Args:
        data_source:
            One of:
                - real
                - gaussian_copula
                - ctgan
                - tvae
                - dpctgan/eps_1.0
                - patectgan/eps_1.0

    Returns:
        Tuple of (train_df, val_df).

    Raises:
        FileNotFoundError:
            If required input files are missing.

        ValueError:
            If the training-data schema does not match the validation schema.
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
) -> PreparedSplits:
    """
    Apply classifier-specific preprocessing to train and validation splits.

    The preprocessor is fitted on the training split only and then reused
    consistently for validation transformation.

    The fitted preprocessor is returned so it can later be reused for
    held-out test evaluation or inference on additional synthetic datasets.

    Args:
        classifier_name:
            One of:
                - logistic_regression
                - random_forest
                - gradient_boosting

        train_df:
            Training DataFrame including the target column.

        val_df:
            Validation DataFrame including the target column.

    Returns:
        Tuple of:
            (X_train, y_train, X_val, y_val, preprocessor)

        For gradient_boosting, preprocessor is None because the classifier
        handles categorical features and missing values natively.
    """
    preprocessor = _get_preprocessor(classifier_name, train_df)

    X_train, y_train = _apply_preprocessor(
        classifier_name,
        train_df,
        preprocessor,
    )

    X_val, y_val = _apply_preprocessor(
        classifier_name,
        val_df,
        preprocessor,
    )

    return X_train, y_train, X_val, y_val, preprocessor


def prepare_single(
    classifier_name: str,
    df: pd.DataFrame,
    preprocessor: Any,
) -> PreparedDataset:
    """
    Apply classifier-specific preprocessing to a single dataset.

    Uses an already-fitted preprocessor to ensure transformation consistency
    between training, validation, and held-out test evaluation.

    Args:
        classifier_name:
            One of:
                - logistic_regression
                - random_forest
                - gradient_boosting

        df:
            DataFrame including the target column.

        preprocessor:
            Previously fitted preprocessor returned by prepare_splits().

            Must be None for gradient_boosting because preprocessing is
            handled natively by the classifier.

    Returns:
        Tuple of (X, y).

    Raises:
        ValueError:
            If a required preprocessor is missing.
    """
    if classifier_name != "gradient_boosting" and preprocessor is None:
        raise ValueError(
            f"Classifier '{classifier_name}' requires a fitted preprocessor."
        )

    return _apply_preprocessor(classifier_name, df, preprocessor)


def _get_preprocessor(
    classifier_name: str,
    train_df: pd.DataFrame,
) -> Any:
    """
    Build and fit the appropriate preprocessor for a classifier.

    Logistic Regression and Random Forest require explicit preprocessing.

    HistGradientBoostingClassifier handles categorical features and missing
    values natively and therefore does not use a fitted preprocessor.

    Args:
        classifier_name:
            One of:
                - logistic_regression
                - random_forest
                - gradient_boosting

        train_df:
            Training DataFrame including the target column.

    Returns:
        Fitted preprocessor object, or None for gradient_boosting.

    Raises:
        ValueError:
            If classifier_name is not recognised.
    """
    builders = {
        "logistic_regression": build_preprocessor_logistic_regression,
        "random_forest": build_preprocessor_random_forest,
    }

    if classifier_name == "gradient_boosting":
        return None

    builder = builders.get(classifier_name)

    if builder is None:
        available = sorted(list(builders.keys()) + ["gradient_boosting"])

        raise ValueError(
            f"Unknown classifier '{classifier_name}'. "
            f"Available classifiers: {available}"
        )

    return builder(train_df.drop(columns=[TARGET_COL]))


def _apply_preprocessor(
    classifier_name: str,
    df: pd.DataFrame,
    preprocessor: Any,
) -> PreparedDataset:
    """
    Apply classifier-specific preprocessing to a dataset.

    Args:
        classifier_name:
            One of:
                - logistic_regression
                - random_forest
                - gradient_boosting

        df:
            DataFrame including the target column.

        preprocessor:
            Fitted preprocessing object, or None for gradient_boosting.

    Returns:
        Tuple of (X, y).

    Raises:
        ValueError:
            If classifier_name is not recognised.
    """
    if classifier_name == "gradient_boosting":
        return prepare_data_gradient_boosting(df)

    if preprocessor is None:
        raise ValueError(
            f"Classifier '{classifier_name}' requires a fitted preprocessor."
        )

    return prepare_data(preprocessor, df)
