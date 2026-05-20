"""
Shared IO and DataFrame validation helpers.

Centralizes common CSV loading and schema validation logic used across
dataset preparation, synthesis, classification, and evaluation stages.
"""

from pathlib import Path

import pandas as pd


def load_csv(path: Path, description: str) -> pd.DataFrame:
    """Load a CSV file from disk with a pipeline-oriented error message."""
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found at {path}. "
            "Run the required earlier pipeline step first."
        )

    return pd.read_csv(path)


def validate_columns(
    df: pd.DataFrame,
    expected_columns: list[str],
    dataframe_name: str,
) -> None:
    """Validate that a DataFrame matches an expected ordered column schema."""
    actual_columns = list(df.columns)

    if actual_columns == expected_columns:
        return

    missing_columns = [col for col in expected_columns if col not in actual_columns]
    unexpected_columns = [col for col in actual_columns if col not in expected_columns]

    raise ValueError(
        f"{dataframe_name} columns do not match the expected schema.\n"
        f"Missing columns: {missing_columns}\n"
        f"Unexpected columns: {unexpected_columns}\n"
        f"Expected order: {expected_columns}\n"
        f"Actual order:   {actual_columns}"
    )


def validate_matching_columns(
    reference_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    candidate_name: str,
) -> None:
    """Validate that two DataFrames share the same ordered schema."""
    validate_columns(
        candidate_df,
        expected_columns=list(reference_df.columns),
        dataframe_name=candidate_name,
    )


def validate_non_empty_dataframe(
    df: pd.DataFrame,
    dataframe_name: str,
) -> None:
    """Validate that a DataFrame contains at least one row."""
    if df.empty:
        raise ValueError(f"{dataframe_name} is empty.")
