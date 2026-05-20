"""
Dataset adapter for the UCI Adult Census Income dataset.

Contains all Adult-specific logic:
- downloading via ucimlrepo
- raw and cleaned schema validation
- column-name normalization
- missing-value cleanup
- dataset-specific column definitions
"""

import json
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

from src.utility.constants import DEFAULT_ENCODING, JSON_INDENT

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

_UCI_DATASET_ID = 2

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

CLEANED_REQUIRED_COLS = set(NUMERICAL_COLS + CATEGORICAL_COLS + [TARGET_COL])

_COLS_TO_DROP = ["fnlwgt", "education"]

_RAW_DIRNAME = "raw"
_CLEANED_DIRNAME = "cleaned"
_RAW_FILENAME = "adult_raw.csv"
_CLEANED_FILENAME = "adult_cleaned.csv"
_EDUCATION_MAP_FILENAME = "education_map.json"


def _resolve_dataset_dirs(base_dir: str | Path) -> tuple[Path, Path]:
    """Resolve canonical raw and cleaned directories below the data directory."""
    base_path = Path(base_dir)
    return base_path / _RAW_DIRNAME, base_path / _CLEANED_DIRNAME


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lowercase with spaces replaced by hyphens."""
    normalized = df.copy()
    normalized.columns = [
        str(column).lower().replace(" ", "-") for column in normalized.columns
    ]
    return normalized


def _validate_raw_schema(df: pd.DataFrame) -> None:
    """Validate that the raw Adult dataset contains all required columns."""
    missing = sorted(REQUIRED_COLS - set(df.columns))
    if missing:
        raise ValueError(
            "Input DataFrame is missing required Adult Census columns: "
            f"{missing}. Please ensure you are loading the correct dataset."
        )


def _validate_cleaned_schema(df: pd.DataFrame) -> None:
    """Validate the cleaned Adult dataset schema and numeric dtypes."""
    actual_cols = set(df.columns)

    missing = sorted(CLEANED_REQUIRED_COLS - actual_cols)
    if missing:
        raise ValueError(f"Cleaned Adult dataset is missing columns: {missing}.")

    unexpected_dropped_present = sorted(set(_COLS_TO_DROP) & actual_cols)
    if unexpected_dropped_present:
        raise ValueError(
            "Cleaned Adult dataset still contains columns that should have "
            f"been dropped: {unexpected_dropped_present}."
        )

    for column in NUMERICAL_COLS:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(
                f"Column '{column}' must be numeric after cleaning, "
                f"but got dtype '{df[column].dtype}'."
            )


def _normalize_cleaned_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize cleaned dataset dtypes for stable downstream processing."""
    normalized = df.copy()

    for column in NUMERICAL_COLS:
        normalized[column] = pd.to_numeric(normalized[column], errors="raise")

    for column in CATEGORICAL_COLS + [TARGET_COL]:
        normalized[column] = normalized[column].astype("string")

    return normalized


def _save_education_map(df: pd.DataFrame, out_dir: Path) -> None:
    """Save the education-num to education label mapping as JSON."""
    education_map = (
        df[["education", "education-num"]]
        .drop_duplicates()
        .sort_values("education-num")
        .set_index("education-num")["education"]
        .to_dict()
    )

    map_path = out_dir / _EDUCATION_MAP_FILENAME
    with open(map_path, "w", encoding=DEFAULT_ENCODING) as file:
        json.dump(education_map, file, indent=JSON_INDENT)

    print(f"[adult_census] Education map saved to {map_path.resolve()}")


def download(output_dir: str | Path) -> pd.DataFrame:
    """Download the Adult Census Income dataset and save the raw CSV."""
    raw_dir = Path(output_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("[adult_census] Downloading Adult Census Income dataset...")

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

    combined = pd.concat(
        [_normalize_column_names(features), _normalize_column_names(targets)],
        axis=1,
    ).reset_index(drop=True)

    _validate_raw_schema(combined)

    raw_path = raw_dir / _RAW_FILENAME
    combined.to_csv(raw_path, index=False)

    print(
        f"[adult_census] Raw data saved to {raw_path.resolve()} ({len(combined)} rows)"
    )

    return combined


def clean(
    data: pd.DataFrame,
    cleaned_dir: str | Path,
    raw_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Clean the raw Adult Census Income DataFrame and save the result."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(data).__name__}")

    cleaned_path_dir = Path(cleaned_dir)
    cleaned_path_dir.mkdir(parents=True, exist_ok=True)

    education_map_dir = Path(raw_dir) if raw_dir is not None else cleaned_path_dir
    education_map_dir.mkdir(parents=True, exist_ok=True)

    normalized = _normalize_column_names(data)
    _validate_raw_schema(normalized)

    df = normalized.copy()
    df = df.replace("?", pd.NA)
    df[TARGET_COL] = df[TARGET_COL].astype("string").str.strip().str.rstrip(".")

    _save_education_map(df, education_map_dir)

    df = df.drop(columns=_COLS_TO_DROP)
    df = _normalize_cleaned_dtypes(df)
    df = df.reset_index(drop=True)

    _validate_cleaned_schema(df)

    cleaned_path = cleaned_path_dir / _CLEANED_FILENAME
    df.to_csv(cleaned_path, index=False)

    print(
        f"[adult_census] Cleaned data saved to {cleaned_path.resolve()} "
        f"({len(df)} rows)"
    )

    return df


def load_cleaned(output_dir: str | Path) -> pd.DataFrame:
    """Load cleaned data, or download and clean it if missing."""
    raw_dir, cleaned_dir = _resolve_dataset_dirs(output_dir)
    cleaned_path = cleaned_dir / _CLEANED_FILENAME

    if cleaned_path.exists():
        print(f"[adult_census] Found existing cleaned data at {cleaned_path.resolve()}")
        return pd.read_csv(cleaned_path)

    print("[adult_census] Cleaned data not found — downloading from UCI repository...")
    raw_df = download(raw_dir)
    return clean(raw_df, cleaned_dir=cleaned_dir, raw_dir=raw_dir)
