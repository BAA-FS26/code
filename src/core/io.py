# src/core/io.py

"""
Shared IO and dataframe validation helpers.

Centralizes common CSV loading and schema validation logic used across
dataset preparation, synthesis, classification, and evaluation stages.
"""

from pathlib import Path

import pandas as pd


def load_csv(path: Path, description: str) -> pd.DataFrame:
    """
    Load a CSV file with a pipeline-friendly error message.

    Args:
        path: CSV file path.
        description: Human-readable dataset description.

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError:
            If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found at {path}. "
            "Run the required earlier pipeline step first."
        )

    return pd.read_csv(path)


def validate_dataframe_schema(
    reference_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    candidate_name: str,
) -> None:
    """
    Validate that two DataFrames share the same columns in the same order.

    Args:
        reference_df: Reference DataFrame defining the expected schema.
        candidate_df: DataFrame to validate.
        candidate_name: Human-readable name of the candidate DataFrame.

    Raises:
        ValueError:
            If columns differ in names or order.
    """
    reference_columns = list(reference_df.columns)
    candidate_columns = list(candidate_df.columns)

    if candidate_columns != reference_columns:
        raise ValueError(
            f"{candidate_name} columns do not match reference schema.\n"
            f"Expected: {reference_columns}\n"
            f"Actual:   {candidate_columns}"
        )


def validate_expected_columns(
    df: pd.DataFrame,
    expected_columns: list[str],
    dataframe_name: str,
) -> None:
    """
    Validate that a DataFrame matches an expected column list exactly.

    Args:
        df: DataFrame to validate.
        expected_columns: Expected ordered column list.
        dataframe_name: Human-readable dataframe name.

    Raises:
        ValueError:
            If columns differ in names or order.
    """
    actual_columns = list(df.columns)

    if actual_columns != expected_columns:
        raise ValueError(
            f"{dataframe_name} columns do not match expected schema.\n"
            f"Expected: {expected_columns}\n"
            f"Actual:   {actual_columns}"
        )
