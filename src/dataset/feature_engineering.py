"""
feature_engineering.py

Feature engineering for the Adult Census Income dataset.
Provides classifier-specific preprocessing pipelines for Logistic Regression,
Random Forest and HistGradientBoostingClassifier.

All transformations are fitted on the training set only and applied
consistently to validation, test and synthetic datasets to prevent data
leakage.

Usage:
    from feature_engineering import (
        build_preprocessor_logistic_regression,
        build_preprocessor_random_forest,
        prepare_data,
        prepare_data_gradient_boosting,
        encode_target,
    )
"""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


CATEGORICAL_COLS = [
    "workclass",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

NUMERICAL_COLS = [
    "age",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

TARGET_COL = "income"
TARGET_MAP = {"<=50K": 0, ">50K": 1}


def encode_target(df: pd.DataFrame) -> pd.Series:
    """
    Encode the target column to binary integer labels.

    Maps '<=50K' to 0 and '>50K' to 1.

    Args:
        df: DataFrame containing the target column.

    Returns:
        Target column as integer Series.
    """
    return df[TARGET_COL].map(TARGET_MAP)


def drop_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop the target column from a DataFrame.

    Args:
        df: DataFrame containing the target column.

    Returns:
        DataFrame without the target column.
    """
    return df.drop(columns=[TARGET_COL])


def build_preprocessor_logistic_regression(train: pd.DataFrame) -> ColumnTransformer:
    """
    Build and fit a preprocessor for Logistic Regression on the training set.

    Applies the following transformations:
    - Numerical features: median imputation followed by standard scaling
    - Categorical features: most frequent imputation followed by one-hot
      encoding (unknown categories are ignored)

    The fitted preprocessor can be reused to transform validation, test
    and synthetic datasets consistently.

    Args:
        train: Training split DataFrame excluding the target column.

    Returns:
        Fitted ColumnTransformer preprocessor.
    """
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
                NUMERICAL_COLS,
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
                CATEGORICAL_COLS,
            ),
        ],
        sparse_threshold=0,
    )
    preprocessor.fit(train)
    return preprocessor


def build_preprocessor_random_forest(train: pd.DataFrame) -> ColumnTransformer:
    """
    Build and fit a preprocessor for Random Forest on the training set.

    Applies the following transformations:
    - Numerical features: median imputation, no scaling required
    - Categorical features: most frequent imputation followed by ordinal
      encoding

    No normalization is applied as Random Forest is invariant to feature
    scale. The fitted preprocessor can be reused to transform validation,
    test and synthetic datasets consistently.

    Args:
        train: Training split DataFrame excluding the target column.

    Returns:
        Fitted ColumnTransformer preprocessor.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), NUMERICAL_COLS),
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
                CATEGORICAL_COLS,
            ),
        ]
    )
    preprocessor.fit(train)
    return preprocessor


def prepare_data(
    preprocessor: ColumnTransformer,
    df: pd.DataFrame,
) -> Tuple[np.ndarray, pd.Series]:
    """
    Transform a dataset using a fitted preprocessor.

    Used for both Logistic Regression and Random Forest — the preprocessing
    steps differ (handled by the preprocessor itself) but the transform
    call is identical.

    Args:
        preprocessor: Fitted ColumnTransformer, as returned by
                      build_preprocessor_logistic_regression() or
                      build_preprocessor_random_forest().
        df: DataFrame including the target column to transform.

    Returns:
        Tuple of (X, y) where X is the transformed feature matrix and y
        is the encoded target Series.
    """
    y = encode_target(df)
    X = np.asarray(preprocessor.transform(drop_target(df)))
    return X, y


def prepare_data_gradient_boosting(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare a dataset for HistGradientBoostingClassifier.

    HistGradientBoostingClassifier natively handles missing values and
    categorical features, requiring minimal preprocessing. Categorical
    columns are cast to the pandas Categorical dtype so that
    HistGradientBoostingClassifier can identify and process them correctly.
    No imputation, encoding, scaling or fitted preprocessor is needed.

    Args:
        df: DataFrame including the target column to transform.

    Returns:
        Tuple of (X, y) where X is the prepared DataFrame and y is the
        encoded target Series.
    """
    y = encode_target(df)
    X = drop_target(df).copy()
    for col in CATEGORICAL_COLS:
        X[col] = X[col].astype("category")
    return X, y
