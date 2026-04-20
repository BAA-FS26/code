"""
adult_census.py

Dataset adapter for the UCI Adult Census Income dataset.

Contains all Adult-specific logic: downloading via ucimlrepo, cleaning,
column definitions and the education ordinal mapping. Nothing in this
file should need to change when adapting the pipeline to a different
dataset — create a new adapter instead.

The full pipeline for a fresh environment is a single call:
    df = load_cleaned(output_dir)

Which is equivalent to:
    raw = download(output_dir)    # skipped if cleaned file already present
    df  = clean(raw, output_dir)  # produces adult_cleaned.csv

Usage:
    from src.dataset.adult_census import load_cleaned, TARGET_COL

    df = load_cleaned(DATA_DIR / "cleaned")
    train, val, test = split_data(df, target_col=TARGET_COL, ...)
"""

import json
from pathlib import Path
from typing import Union

import pandas as pd
from ucimlrepo import fetch_ucirepo

# ── Dataset-specific column definitions ──────────────────────────────────────

TARGET_COL = "income"

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
    "income",
}

# Columns dropped during cleaning, with rationale documented here.
# fnlwgt:    near-zero correlation with target; uninformative for synthesis
#            and classification.
# education: redundant with education-num, which encodes the same
#            information as an ordinal integer.
_COLS_TO_DROP = ["fnlwgt", "education"]

_RAW_FILENAME = "adult_raw.csv"
_CLEANED_FILENAME = "adult_cleaned.csv"


# ── Download ──────────────────────────────────────────────────────────────────


def download(output_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Download the Adult Census Income dataset via the ucimlrepo package.

    Fetches the dataset using its UCI ML Repository ID, combines features
    and targets into a single DataFrame, and saves it as adult_raw.csv.
    Column names are normalised to lowercase with spaces replaced by
    hyphens to match the expected REQUIRED_COLS format.

    This function always re-downloads. Use load_or_download() to skip
    the download when the cleaned file is already present.

    Args:
        output_dir: Directory where adult_raw.csv will be saved.

    Returns:
        Combined raw DataFrame with all UCI columns.

    Raises:
        ConnectionError: If the UCI repository is unreachable.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading Adult Census Income dataset from UCI ML Repository...")
    dataset = fetch_ucirepo(id=_UCI_DATASET_ID)

    if dataset is None or dataset.data is None:
        raise ConnectionError(f"Failed to fetch dataset with ID {_UCI_DATASET_ID}")

    features = dataset.data.features
    targets = dataset.data.targets

    # ucimlrepo may return column names with spaces — normalise to hyphens
    # to match the pipeline's expected REQUIRED_COLS format.
    features.columns = [c.lower().replace(" ", "-") for c in features.columns]
    targets.columns = [c.lower().replace(" ", "-") for c in targets.columns]

    combined = pd.concat([features, targets], axis=1)

    raw_path = out_dir / _RAW_FILENAME
    combined.to_csv(raw_path, index=False)
    print(f"Raw data saved to {raw_path.resolve()} ({len(combined)} rows)")

    return combined


# ── Cleaning ──────────────────────────────────────────────────────────────────


def _save_education_map(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Save a mapping of education-num to education labels as a JSON file.

    Saved before the education column is dropped so downstream analysis
    can still interpret ordinal education values.

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
    map_path = out_dir / "education_map.json"
    with open(map_path, "w") as f:
        json.dump(education_map, f, indent=2)
    print(f"Education map saved to {map_path.resolve()}")


def clean(
    data: pd.DataFrame,
    output_dir: Union[str, Path],
) -> pd.DataFrame:
    """
    Clean the raw Adult Census Income DataFrame and save the result.

    Cleaning steps applied:
    - Validates that all required columns are present
    - Replaces '?' placeholders with pandas NA
    - Strips whitespace from income labels and removes any trailing dots
    - Saves an education-num → education label mapping as JSON
    - Drops redundant and uninformative columns (see _COLS_TO_DROP)
    - Saves the cleaned DataFrame as adult_cleaned.csv

    Args:
        data: Raw Adult Census DataFrame as returned by download().
        output_dir: Directory where adult_cleaned.csv and education_map.json
                    will be saved.

    Returns:
        Cleaned DataFrame with redundant columns removed and missing values
        encoded as pandas NA.

    Raises:
        TypeError: If data is not a pandas DataFrame.
        ValueError: If any required columns are missing.
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
    df = df.drop(columns=_COLS_TO_DROP)

    out_path = out_dir / _CLEANED_FILENAME
    df.to_csv(out_path, index=False)
    print(f"Cleaned data saved to {out_path.resolve()} ({len(df)} rows)")

    return df


# ── Convenience loader ────────────────────────────────────────────────────────


def load_cleaned(output_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Load the cleaned dataset from disk, downloading and cleaning it first
    if it is not already present.

    This is the recommended entry point for the pipeline. It is safe to
    call repeatedly — the download and cleaning steps are skipped when
    adult_cleaned.csv already exists in output_dsir.

    Args:
        output_dir: Directory to check for adult_cleaned.csv and to save
                    it to if it needs to be created.

    Returns:
        Cleaned Adult Census DataFrame ready for splitting.
    """
    out_dir = Path(output_dir)
    cleaned_path = out_dir / _CLEANED_FILENAME

    if cleaned_path.exists():
        print(f"Found existing cleaned data at {cleaned_path.resolve()}")
        return pd.read_csv(cleaned_path)

    print("Cleaned data not found — downloading from UCI repository...")
    raw_df = download(out_dir)
    return clean(raw_df, out_dir)
