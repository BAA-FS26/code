"""
Generic data pipeline utilities for tabular classification datasets.

Handles stratified train/validation/test splitting and split verification.
Dataset-specific logic belongs in dataset adapter modules.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utility.constants import (
    RANDOM_STATE,
    TEST_FILENAME,
    TRAIN_FILENAME,
    VALIDATION_FILENAME,
)


def _validate_split_inputs(df: pd.DataFrame, target_col: str) -> None:
    """Validate that a DataFrame is ready for stratified splitting."""
    if df.empty:
        raise ValueError("Cannot split an empty DataFrame.")

    if target_col not in df.columns:
        raise KeyError(
            f"Target column '{target_col}' not found in DataFrame. "
            f"Available columns: {df.columns.tolist()}"
        )

    if df[target_col].isna().any():
        raise ValueError(
            f"Target column '{target_col}' contains missing values. "
            "Please clean or impute the target column before splitting."
        )

    class_counts = df[target_col].value_counts(dropna=False)
    if (class_counts < 2).any():
        raise ValueError(
            "Stratified splitting requires at least 2 samples per class. "
            f"Classes with insufficient samples: "
            f"{class_counts[class_counts < 2].to_dict()}"
        )


def _save_split(split_df: pd.DataFrame, output_dir: Path, filename: str) -> None:
    """Persist one split as CSV using the existing filename convention."""
    split_df.to_csv(output_dir / filename, index=False)
    print(
        f"[data_pipeline] Saved {filename} ({len(split_df)} rows) "
        f"to {output_dir.resolve()}"
    )


def split_data(
    df: pd.DataFrame,
    target_col: str,
    output_dir: str | Path,
    random_state: int | None = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a cleaned dataset into stratified train, validation, and test sets."""
    _validate_split_inputs(df, target_col)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_val_df, test_df = train_test_split(
        df,
        test_size=0.20,
        stratify=df[target_col],
        random_state=random_state,
    )

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.25,
        stratify=train_val_df[target_col],
        random_state=random_state,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    _save_split(train_df, output_path, TRAIN_FILENAME)
    _save_split(val_df, output_path, VALIDATION_FILENAME)
    _save_split(test_df, output_path, TEST_FILENAME)

    return train_df, val_df, test_df


def verify_stratification(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
) -> pd.DataFrame:
    """Return target-class distributions for train, validation, and test splits."""
    splits = {
        "Train": train,
        "Val": val,
        "Test": test,
    }

    for split_name, split_df in splits.items():
        if target_col not in split_df.columns:
            raise KeyError(
                f"Target column '{target_col}' not found in {split_name} split. "
                f"Available columns: {split_df.columns.tolist()}"
            )

    summary = pd.DataFrame(
        {
            split_name: split_df[target_col]
            .value_counts(normalize=True)
            .mul(100)
            .round(2)
            for split_name, split_df in splits.items()
        }
    )

    return summary.sort_index()
