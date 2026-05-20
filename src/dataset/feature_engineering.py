"""
Feature engineering for tabular classification datasets.

Provides classifier-specific preprocessing for:
- Logistic Regression
- Random Forest
- HistGradientBoostingClassifier

All transformations that require fitting are fitted on the training split only
and then reused for validation and held-out test evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.dataset.dataset_config import DatasetConfig, get_dataset_config

DEFAULT_CONFIG = get_dataset_config()


def _validate_target_column(df: pd.DataFrame, target_col: str) -> None:
    """Validate that the target column exists."""
    if target_col not in df.columns:
        raise KeyError(
            f"Target column '{target_col}' not found in DataFrame. "
            f"Available columns: {df.columns.tolist()}"
        )


def _validate_feature_columns(
    df: pd.DataFrame,
    numerical_cols: list[str],
    categorical_cols: list[str],
) -> None:
    """Validate that all required feature columns exist."""
    required_cols = numerical_cols + categorical_cols
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise KeyError(
            "Input DataFrame is missing required feature columns: "
            f"{missing_cols}. Available columns: {df.columns.tolist()}"
        )


def encode_target(
    df: pd.DataFrame,
    config: DatasetConfig = DEFAULT_CONFIG,
) -> pd.Series:
    """Encode the target column to integer labels using the dataset target map."""
    _validate_target_column(df, config.target_col)

    encoded = df[config.target_col].map(config.target_map)
    unmapped = df[config.target_col][encoded.isna()].unique()

    if len(unmapped) > 0:
        raise ValueError(
            f"Target column '{config.target_col}' contains values not in "
            f"target_map: {unmapped.tolist()}."
        )

    return encoded.astype(int)


def drop_target(
    df: pd.DataFrame,
    config: DatasetConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Return a copy of the DataFrame without the target column."""
    _validate_target_column(df, config.target_col)
    return df.drop(columns=[config.target_col])


def build_preprocessor_logistic_regression(
    train: pd.DataFrame,
    config: DatasetConfig = DEFAULT_CONFIG,
) -> ColumnTransformer:
    """Build and fit the Logistic Regression preprocessor."""
    _validate_feature_columns(
        train,
        config.numerical_cols,
        config.categorical_cols,
    )

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, config.numerical_cols),
            ("cat", categorical_pipeline, config.categorical_cols),
        ],
        sparse_threshold=0,
    )

    preprocessor.fit(train)
    return preprocessor


def build_preprocessor_random_forest(
    train: pd.DataFrame,
    config: DatasetConfig = DEFAULT_CONFIG,
) -> ColumnTransformer:
    """Build and fit the Random Forest preprocessor."""
    _validate_feature_columns(
        train,
        config.numerical_cols,
        config.categorical_cols,
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), config.numerical_cols),
            ("cat", categorical_pipeline, config.categorical_cols),
        ]
    )

    preprocessor.fit(train)
    return preprocessor


def prepare_data(
    preprocessor: ColumnTransformer,
    df: pd.DataFrame,
    config: DatasetConfig = DEFAULT_CONFIG,
) -> tuple[np.ndarray, pd.Series]:
    """Transform a dataset using a fitted preprocessor."""
    _validate_target_column(df, config.target_col)

    feature_df = drop_target(df, config)
    X = np.asarray(preprocessor.transform(feature_df))
    y = encode_target(df, config)

    return X, y


def prepare_data_gradient_boosting(
    df: pd.DataFrame,
    config: DatasetConfig = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare a dataset for HistGradientBoostingClassifier."""
    _validate_target_column(df, config.target_col)

    X = drop_target(df, config).copy()

    _validate_feature_columns(
        X,
        numerical_cols=[],
        categorical_cols=config.categorical_cols,
    )

    for col in config.categorical_cols:
        X[col] = X[col].astype("category")

    y = encode_target(df, config)

    return X, y
