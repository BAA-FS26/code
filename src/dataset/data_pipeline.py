"""
data_pipeline.py

Data pipeline for the Adult Census Income dataset.
Handles raw data cleaning and stratified train/validation/test splitting.

Usage:
    from data_pipeline import clean_data, split_data, verify_stratification
"""

import json
from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utility.constants import RANDOM_STATE


TARGET_COL = "income"

REQUIRED_COLS = {
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
}


def _save_education_map(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Save a mapping of education-num to education labels as a JSON file.

    This mapping is saved before the education column is dropped, serving
    as a reference for interpreting ordinal education values in downstream
    analysis.

    Args:
        df: DataFrame containing both 'education' and 'education-num' columns.
        out_dir: Directory where the education_map.json file will be saved.
    """
    education_map = (
        df[["education", "education-num"]]
        .drop_duplicates()
        .sort_values("education-num")
        .set_index("education-num")["education"]
        .to_dict()
    )
    with open(out_dir / "education_map.json", "w") as f:
        json.dump(education_map, f, indent=2)


def clean_data(
    data: pd.DataFrame,
    output_dir: Union[str, Path] = Path("../data/cleaned"),
) -> pd.DataFrame:
    """
    Clean the raw Adult Census Income dataset and save the result to disk.

    Performs the following cleaning steps:
    - Validates that all required columns are present
    - Replaces '?' placeholders with pandas NA
    - Strips whitespace and trailing dots from income labels
    - Saves an education-num to education label mapping as JSON
    - Drops redundant and uninformative columns (fnlwgt, education)
    - Saves the cleaned DataFrame as adult_cleaned.csv

    The following columns are dropped based on EDA findings:
    - fnlwgt: near-zero correlation with target, uninformative for synthesis
      and classification
    - education: redundant with education-num, which encodes the same
      information as an ordinal integer

    Args:
        data: Raw Adult Census Income DataFrame as loaded from the UCI
              repository.
        output_dir: Directory where the cleaned CSV and education map will
                    be saved. Defaults to '../data/cleaned'.

    Returns:
        Cleaned DataFrame with redundant columns removed and missing values
        encoded as pandas NA.

    Raises:
        TypeError: If data is not a pandas DataFrame.
        ValueError: If any required Adult Census columns are missing from
                    the input DataFrame.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(data).__name__}")

    missing = REQUIRED_COLS - set(data.columns)
    if missing:
        raise ValueError(
            f"Input DataFrame is missing required Adult Census columns: {missing}. "
            "Please ensure you are loading the correct dataset."
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = (
        data.copy()
        .replace("?", pd.NA)
        .assign(
            income=lambda d: d["income"].str.strip().str.replace(".", "", regex=False)
        )
    )

    _save_education_map(df, out_dir)
    df = df.drop(columns=["fnlwgt", "education"])

    out_path = out_dir / "adult_cleaned.csv"
    df.to_csv(out_path, index=False)
    print(f"File saved successfully to: {out_path.resolve()}")

    return df


def split_data(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    output_dir: Union[str, Path] = Path("../data/processed"),
    random_state: Optional[int] = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the cleaned dataset into stratified train, validation and test sets.

    The dataset is split into a 60/20/20 ratio using stratified sampling to
    preserve the class distribution of the target variable across all three
    splits. This is particularly important given the class imbalance in the
    Adult Census dataset (~76% <=50K, ~24% >50K).

    The splits are saved as CSV files to the specified output directory and
    returned as DataFrames. The test set must remain untouched until final
    evaluation to ensure unbiased assessment of model performance and privacy
    metrics.

    Args:
        df: Cleaned Adult Census DataFrame as returned by clean_data().
        target_col: Name of the target column to stratify on. Defaults to
                    'income'.
        output_dir: Directory where train.csv, validation.csv and test.csv
                    will be saved. Defaults to '../data/processed'.
        random_state: Random seed for reproducibility. Defaults to 42.
                      Pass None for non-deterministic splits.

    Returns:
        Tuple of (train_df, val_df, test_df) DataFrames.

    Raises:
        KeyError: If target_col is not present in the input DataFrame.
    """
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

    return train_df, val_df, test_df


def verify_stratification(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    target: str = TARGET_COL,
) -> pd.DataFrame:
    """
    Verify that the target variable is consistently distributed across splits.

    Args:
        train: Training split DataFrame.
        val: Validation split DataFrame.
        test: Test split DataFrame.
        target: Name of the target column to check. Defaults to 'income'.

    Returns:
        DataFrame showing the class distribution (%) for each split side by
        side.
    """
    return pd.DataFrame(
        {
            name: d[target].value_counts(normalize=True).mul(100).round(2)
            for name, d in zip(["Train", "Val", "Test"], [train, val, test])
        }
    )
