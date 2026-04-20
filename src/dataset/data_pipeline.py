"""
data_pipeline.py

Generic data pipeline utilities for tabular classification datasets.
Handles stratified train/validation/test splitting and split verification.

Dataset-specific logic (loading, cleaning, column definitions) belongs in
the dataset adapter module — see src/dataset/adult_census.py for the
Adult Census Income implementation.

Usage:
    from src.dataset.data_pipeline import split_data, verify_stratification
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utility.constants import RANDOM_STATE


def split_data(
    df: pd.DataFrame,
    target_col: str,
    output_dir: Union[str, Path],
    random_state: Optional[int] = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a cleaned dataset into stratified train, validation and test sets.

    Splits the dataset in a 60/20/20 ratio using stratified sampling to
    preserve the class distribution of the target variable across all three
    splits. Stratification is particularly important for imbalanced datasets.

    The splits are saved as train.csv, validation.csv and test.csv in
    output_dir and returned as DataFrames. The test set must remain
    untouched until final evaluation to prevent data leakage.

    Args:
        df: Cleaned DataFrame ready for splitting.
        target_col: Name of the target column to stratify on.
        output_dir: Directory where the CSV splits will be saved.
        random_state: Random seed for reproducibility. Defaults to 42.
                      Pass None for non-deterministic splits.

    Returns:
        Tuple of (train_df, val_df, test_df) DataFrames.

    Raises:
        KeyError: If target_col is not present in df.
    """
    if target_col not in df.columns:
        raise KeyError(
            f"Target column '{target_col}' not found in DataFrame. "
            f"Available columns: {df.columns.tolist()}"
        )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_val, test_df = train_test_split(
        df,
        test_size=0.20,
        stratify=df[target_col],
        random_state=random_state,
    )
    train_df, val_df = train_test_split(
        train_val,
        test_size=0.25,
        stratify=train_val[target_col],
        random_state=random_state,
    )

    for filename, split in [
        ("train.csv", train_df),
        ("validation.csv", val_df),
        ("test.csv", test_df),
    ]:
        split.to_csv(out_path / filename, index=False)
        print(f"Saved {filename} ({len(split)} rows) to {out_path.resolve()}")

    return train_df, val_df, test_df


def verify_stratification(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
) -> pd.DataFrame:
    """
    Verify that the target variable is consistently distributed across splits.

    Args:
        train: Training split DataFrame.
        val: Validation split DataFrame.
        test: Test split DataFrame.
        target_col: Name of the target column to check.

    Returns:
        DataFrame showing the class distribution (%) for each split side by
        side, useful for confirming stratification was applied correctly.
    """
    return pd.DataFrame(
        {
            name: df[target_col].value_counts(normalize=True).mul(100).round(2)
            for name, df in zip(["Train", "Val", "Test"], [train, val, test])
        }
    )
