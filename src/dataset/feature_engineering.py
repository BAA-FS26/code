"""
feature_engineering.py

Feature engineering for the Adult Census Income dataset.

Provides classifier-specific preprocessing pipeline builders for Logistic
Regression, Random Forest, and HistGradientBoostingClassifier.

All Adult-specific dataset definitions are owned by src.dataset.adult_census.
This module consumes those definitions and provides reusable preprocessing
helpers for model training and evaluation.

All transformations are fitted on the training set only and applied
consistently to validation, test, and synthetic datasets to prevent data
leakage.

Usage:
    from src.dataset.feature_engineering import (
        build_preprocessor_logistic_regression,
        build_preprocessor_random_forest,
        prepare_data,
        prepare_data_gradient_boosting,
        encode_target,
    )
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.dataset.adult_census import (
    CATEGORICAL_COLS,
    NUMERICAL_COLS,
    TARGET_COL,
    TARGET_MAP,
)


# ── Validation helpers ────────────────────────────────────────────────────────


def _validate_target_column(df: pd.DataFrame, target_col: str = TARGET_COL) -> None:
    """
    Validate that the target column exists in the input DataFrame.

    Raises:
        KeyError: If the target column is missing.
    """
    if target_col not in df.columns:
        raise KeyError(
            f"Target column '{target_col}' not found in DataFrame. "
            f"Available columns: {df.columns.tolist()}"
        )


def _validate_feature_columns(
    df: pd.DataFrame,
    numerical_cols: List[str],
    categorical_cols: List[str],
) -> None:
    """
    Validate that all required feature columns exist in the input DataFrame.

    Raises:
        KeyError: If one or more required feature columns are missing.
    """
    required_cols = numerical_cols + categorical_cols
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(
            "Input DataFrame is missing required feature columns: "
            f"{missing_cols}. Available columns: {df.columns.tolist()}"
        )


# ── Target encoding ───────────────────────────────────────────────────────────


def encode_target(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.Series:
    """
    Encode the target column to binary integer labels using TARGET_MAP.

    Args:
        df: DataFrame containing the target column.
        target_col: Name of the target column. Defaults to TARGET_COL.

    Returns:
        Target column as an integer Series.

    Raises:
        KeyError: If the target column is missing.
        ValueError: If the target column contains values not in TARGET_MAP.
    """
    _validate_target_column(df, target_col)

    encoded = df[target_col].map(TARGET_MAP)
    unmapped = df[target_col][encoded.isna()].unique()
    if len(unmapped) > 0:
        raise ValueError(
            f"Target column '{target_col}' contains values not in TARGET_MAP: "
            f"{unmapped.tolist()}. Update TARGET_MAP to include all classes."
        )

    return encoded.astype(int)


def drop_target(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    """
    Drop the target column from a DataFrame.

    Args:
        df: DataFrame containing the target column.
        target_col: Name of the target column. Defaults to TARGET_COL.

    Returns:
        DataFrame without the target column.

    Raises:
        KeyError: If the target column is missing.
    """
    _validate_target_column(df, target_col)
    return df.drop(columns=[target_col])


# ── Pipeline builders ─────────────────────────────────────────────────────────


def build_preprocessor_logistic_regression(
    train: pd.DataFrame,
    numerical_cols: List[str] = NUMERICAL_COLS,
    categorical_cols: List[str] = CATEGORICAL_COLS,
) -> ColumnTransformer:
    """
    Build and fit a preprocessor for Logistic Regression on the training set.

    Applies the following transformations:
    - Numerical features: median imputation followed by standard scaling
    - Categorical features: most frequent imputation followed by one-hot
      encoding (unknown categories at transform time are ignored)

    The fitted preprocessor can be reused to transform validation, test,
    and synthetic datasets consistently.

    Args:
        train: Training split DataFrame excluding the target column.
        numerical_cols: List of numerical feature column names.
        categorical_cols: List of categorical feature column names.

    Returns:
        Fitted ColumnTransformer preprocessor.

    Raises:
        KeyError: If required feature columns are missing.
    """
    _validate_feature_columns(train, numerical_cols, categorical_cols)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numerical_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_cols,
            ),
        ],
        sparse_threshold=0,
    )
    preprocessor.fit(train)
    return preprocessor


def build_preprocessor_random_forest(
    train: pd.DataFrame,
    numerical_cols: List[str] = NUMERICAL_COLS,
    categorical_cols: List[str] = CATEGORICAL_COLS,
) -> ColumnTransformer:
    """
    Build and fit a preprocessor for Random Forest on the training set.

    Applies the following transformations:
    - Numerical features: median imputation, no scaling required
    - Categorical features: most frequent imputation followed by ordinal
      encoding

    No normalization is applied as Random Forest is invariant to feature
    scale. The fitted preprocessor can be reused to transform validation,
    test, and synthetic datasets consistently.

    Args:
        train: Training split DataFrame excluding the target column.
        numerical_cols: List of numerical feature column names.
        categorical_cols: List of categorical feature column names.

    Returns:
        Fitted ColumnTransformer preprocessor.

    Raises:
        KeyError: If required feature columns are missing.
    """
    _validate_feature_columns(train, numerical_cols, categorical_cols)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numerical_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )
    preprocessor.fit(train)
    return preprocessor


# ── Data preparation ──────────────────────────────────────────────────────────


def prepare_data(
    preprocessor: ColumnTransformer,
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
) -> Tuple[np.ndarray, pd.Series]:
    """
    Transform a dataset using a fitted preprocessor.

    Used for both Logistic Regression and Random Forest — the preprocessing
    steps differ (handled by the preprocessor itself) but the transform
    call is identical.

    Returns a dense NumPy feature matrix for consistency across supported
    classifiers in this pipeline.

    Args:
        preprocessor: Fitted ColumnTransformer, as returned by
                      build_preprocessor_logistic_regression() or
                      build_preprocessor_random_forest().
        df: DataFrame including the target column to transform.
        target_col: Name of the target column. Defaults to TARGET_COL.

    Returns:
        Tuple of (X, y) where X is the transformed dense feature matrix
        and y is the encoded target Series.

    Raises:
        KeyError: If required columns are missing.
    """
    _validate_target_column(df, target_col)
    feature_df = drop_target(df, target_col)
    y = encode_target(df, target_col)
    X = np.asarray(preprocessor.transform(feature_df))
    return X, y


def prepare_data_gradient_boosting(
    df: pd.DataFrame,
    categorical_cols: List[str] = CATEGORICAL_COLS,
    target_col: str = TARGET_COL,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare a dataset for HistGradientBoostingClassifier.

    HistGradientBoostingClassifier natively handles missing values and
    categorical features, requiring minimal preprocessing. Categorical
    columns are cast to the pandas Categorical dtype so the classifier
    can identify and process them correctly. No imputation, encoding,
    scaling, or fitted preprocessor is needed.

    Args:
        df: DataFrame including the target column.
        categorical_cols: List of categorical feature column names.
        target_col: Name of the target column. Defaults to TARGET_COL.

    Returns:
        Tuple of (X, y) where X is the prepared DataFrame and y is the
        encoded target Series.

    Raises:
        KeyError: If required columns are missing.
    """
    _validate_target_column(df, target_col)

    X = drop_target(df, target_col).copy()
    _validate_feature_columns(
        X,
        numerical_cols=[],
        categorical_cols=categorical_cols,
    )

    y = encode_target(df, target_col)

    for col in categorical_cols:
        X[col] = X[col].astype("category")

    return X, y
