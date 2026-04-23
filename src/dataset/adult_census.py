"""
adult_census.py

Dataset adapter for the UCI Adult Census Income dataset.

Contains all Adult-specific logic for this project:
- downloading via ucimlrepo
- cleaning and schema validation
- dataset-specific column definitions

Nothing in this file should need to change when adapting the pipeline to a
different dataset — create a new adapter instead.

The full pipeline for a fresh environment is a single call:
    df = load_cleaned(output_dir)

Which is equivalent to:
    raw = download(output_dir)    # skipped if cleaned file already present
    df = clean(raw, output_dir)   # produces adult_cleaned.csv

Usage:
    from src.dataset.adult_census import (
        load_cleaned,
        TARGET_COL,
        CATEGORICAL_COLS,
        NUMERICAL_COLS,
    )

    df = load_cleaned(DATA_DIR / "cleaned")
"""

import json
from pathlib import Path
from typing import Union

import pandas as pd
from ucimlrepo import fetch_ucirepo

from src.utility.constants import DEFAULT_ENCODING, JSON_INDENT

# ── Dataset-specific column definitions ──────────────────────────────────────

TARGET_COL = "income"
TARGET_MAP = {"<=50K": 0, ">50K": 1}

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

# UCI ML Repository dataset ID for Adult Census Income.
_UCI_DATASET_ID = 2

# Columns that must be present before cleaning begins.
# fnlwgt and education are included here because they are present in the
# raw download and validated before being dropped in clean().
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
    TARGET_COL,
}

# Expected columns after cleaning completes.
CLEANED_REQUIRED_COLS = set(NUMERICAL_COLS + CATEGORICAL_COLS + [TARGET_COL])

# Columns dropped during cleaning, with rationale documented here.
# fnlwgt:    near-zero correlation with target; uninformative for synthesis
#            and classification.
# education: redundant with education-num, which encodes the same
#            information as an ordinal integer.
_COLS_TO_DROP = ["fnlwgt", "education"]

_RAW_FILENAME = "adult_raw.csv"
_CLEANED_FILENAME = "adult_cleaned.csv"
_EDUCATION_MAP_FILENAME = "education_map.json"


# ── Internal helpers ─────────────────────────────────────────────────────────


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names to lowercase with spaces replaced by hyphens.
    """
    normalized = df.copy()
    normalized.columns = [
        str(col).lower().replace(" ", "-") for col in normalized.columns
    ]
    return normalized


def _validate_raw_schema(df: pd.DataFrame) -> None:
    """
    Validate that the raw Adult dataset contains all required columns.

    Raises:
        ValueError: If required columns are missing.
    """
    missing = sorted(REQUIRED_COLS - set(df.columns))
    if missing:
        raise ValueError(
            "Input DataFrame is missing required Adult Census columns: "
            f"{missing}. Please ensure you are loading the correct dataset."
        )


def _validate_cleaned_schema(df: pd.DataFrame) -> None:
    """
    Validate the cleaned Adult dataset schema and dtypes.

    Raises:
        ValueError: If expected cleaned columns are missing, dropped columns
                    remain, or required numeric columns cannot be interpreted
                    as numeric.
    """
    actual_cols = set(df.columns)

    missing = sorted(CLEANED_REQUIRED_COLS - actual_cols)
    if missing:
        raise ValueError(
            "Cleaned Adult dataset is missing required columns: " f"{missing}."
        )

    unexpected_dropped_present = sorted(set(_COLS_TO_DROP) & actual_cols)
    if unexpected_dropped_present:
        raise ValueError(
            "Cleaned Adult dataset still contains columns that should have "
            f"been dropped: {unexpected_dropped_present}."
        )

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Cleaned Adult dataset must contain target column '{TARGET_COL}'."
        )

    for col in NUMERICAL_COLS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(
                f"Column '{col}' must be numeric after cleaning, but got "
                f"dtype '{df[col].dtype}'."
            )


def _normalize_cleaned_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize cleaned dataset dtypes for stable downstream processing.

    Numerical columns are coerced to numeric dtype. Categorical and target
    columns are stored using pandas' string dtype where possible, while
    preserving missing values.
    """
    normalized = df.copy()

    for col in NUMERICAL_COLS:
        normalized[col] = pd.to_numeric(normalized[col], errors="raise")

    for col in CATEGORICAL_COLS + [TARGET_COL]:
        normalized[col] = normalized[col].astype("string")

    return normalized


def _save_education_map(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Save a mapping of education-num to education labels as a JSON file.

    Saved before the education column is dropped so downstream analysis
    can still interpret ordinal education values.

    Existing files are overwritten intentionally so the mapping always
    matches the latest cleaned dataset.

    Args:
        df: DataFrame containing both 'education' and 'education-num'.
        out_dir: Directory where education_map.json will be written.
    """
    education_map = (
        df[["education", "education-num"]]
        .drop_duplicates()
        .sort_values("education-num")
        .set_index("education-num")["education"]
        .to_dict()
    )

    map_path = out_dir / _EDUCATION_MAP_FILENAME
    with open(map_path, "w", encoding=DEFAULT_ENCODING) as f:
        json.dump(education_map, f, indent=JSON_INDENT)

    print(f"[adult_census] Education map saved to {map_path.resolve()}")


# ── Download ──────────────────────────────────────────────────────────────────


def download(output_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Download the Adult Census Income dataset via the ucimlrepo package.

    Fetches the dataset using its UCI ML Repository ID, combines features
    and targets into a single DataFrame, normalizes column names, validates
    the raw schema, and saves the result as adult_raw.csv.

    This function always re-downloads. Use load_cleaned() to skip the
    download when the cleaned file is already present.

    Args:
        output_dir: Directory where adult_raw.csv will be saved.

    Returns:
        Combined raw DataFrame with all expected UCI columns.

    Raises:
        ConnectionError: If the UCI repository is unreachable or the dataset
                         cannot be fetched correctly.
        ValueError: If the downloaded dataset does not match the expected
                    Adult schema.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        "[adult_census] Downloading Adult Census Income dataset from UCI ML Repository..."
    )

    try:
        dataset = fetch_ucirepo(id=_UCI_DATASET_ID)
    except Exception as exc:
        raise ConnectionError(
            f"Failed to fetch Adult Census dataset (UCI ID {_UCI_DATASET_ID})."
        ) from exc

    if dataset is None or dataset.data is None:
        raise ConnectionError(
            f"Received an invalid response while fetching dataset ID {_UCI_DATASET_ID}."
        )

    features = dataset.data.features
    targets = dataset.data.targets

    if features is None or targets is None:
        raise ConnectionError(
            f"Fetched dataset ID {_UCI_DATASET_ID}, but features or targets were missing."
        )

    features = _normalize_column_names(features)
    targets = _normalize_column_names(targets)

    combined = pd.concat([features, targets], axis=1).reset_index(drop=True)
    _validate_raw_schema(combined)

    raw_path = out_dir / _RAW_FILENAME
    combined.to_csv(raw_path, index=False)
    print(
        f"[adult_census] Raw data saved to {raw_path.resolve()} ({len(combined)} rows)"
    )

    return combined


# ── Cleaning ──────────────────────────────────────────────────────────────────


def clean(
    data: pd.DataFrame,
    output_dir: Union[str, Path],
) -> pd.DataFrame:
    """
    Clean the raw Adult Census Income DataFrame and save the result.

    Cleaning steps applied:
    - validates that all required columns are present
    - normalizes column names
    - replaces '?' placeholders with pandas NA
    - strips whitespace from income labels and removes any trailing dots
    - saves an education-num → education label mapping as JSON
    - drops redundant and uninformative columns (see _COLS_TO_DROP)
    - normalizes cleaned dtypes
    - validates the cleaned schema
    - saves the cleaned DataFrame as adult_cleaned.csv

    Args:
        data: Raw Adult Census DataFrame as returned by download().
        output_dir: Directory where adult_cleaned.csv and education_map.json
                    will be saved.

    Returns:
        Cleaned DataFrame with redundant columns removed and missing values
        encoded as pandas NA.

    Raises:
        TypeError: If data is not a pandas DataFrame.
        ValueError: If required columns are missing or cleaning produces an
                    invalid schema.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(data).__name__}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    normalized = _normalize_column_names(data)
    _validate_raw_schema(normalized)

    df = (
        normalized.copy()
        .replace("?", pd.NA)
        .assign(
            income=lambda d: d[TARGET_COL].astype("string").str.strip().str.rstrip(".")
        )
    )

    _save_education_map(df, out_dir)

    df = df.drop(columns=_COLS_TO_DROP)
    df = _normalize_cleaned_dtypes(df)
    df = df.reset_index(drop=True)

    _validate_cleaned_schema(df)

    out_path = out_dir / _CLEANED_FILENAME
    df.to_csv(out_path, index=False)
    print(f"[adult_census] Cleaned data saved to {out_path.resolve()} ({len(df)} rows)")

    return df


# ── Convenience loader ────────────────────────────────────────────────────────


def load_cleaned(output_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Load the cleaned dataset from disk, downloading and cleaning it first
    if it is not already present.

    This is the recommended entry point for the dataset pipeline. It is safe
    to call repeatedly — download and cleaning are skipped when
    adult_cleaned.csv already exists in output_dir.

    Args:
        output_dir: Directory to check for adult_cleaned.csv and to save
                    it to if it needs to be created.

    Returns:
        Cleaned Adult Census DataFrame ready for splitting.
    """
    out_dir = Path(output_dir)
    cleaned_path = out_dir / _CLEANED_FILENAME

    if cleaned_path.exists():
        print(f"[adult_census] Found existing cleaned data at {cleaned_path.resolve()}")
        return pd.read_csv(cleaned_path)

    print("[adult_census] Cleaned data not found — downloading from UCI repository...")
    raw_df = download(out_dir)
    return clean(raw_df, out_dir)
