"""
Shared IO and dataframe validation helpers.

Centralizes common CSV loading and schema validation logic used across
dataset preparation, synthesis, classification, and evaluation stages.
"""

from pathlib import Path

import pandas as pd


def load_csv(path: Path, description: str) -> pd.DataFrame:
    """
    Load a CSV file from disk.

    This helper centralizes file-existence validation and provides
    pipeline-oriented error messages so users know which earlier
    pipeline step may be missing.

    Args:
        path:
            Path to the CSV file.

        description:
            Human-readable dataset description used in error messages.

    Returns:
        Loaded pandas DataFrame.

    Raises:
        FileNotFoundError:
            If the CSV file does not exist.
    """
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
    """
    Validate that a DataFrame matches an expected ordered column schema.

    Both column names and column order must match exactly. This strict
    validation prevents subtle preprocessing and evaluation issues caused
    by inconsistent schemas between real and synthetic datasets.

    Args:
        df:
            DataFrame to validate.

        expected_columns:
            Expected ordered list of column names.

        dataframe_name:
            Human-readable DataFrame name for error reporting.

    Raises:
        ValueError:
            If column names or ordering differ from the expected schema.
    """
    actual_columns = list(df.columns)

    if actual_columns != expected_columns:
        missing_columns = [col for col in expected_columns if col not in actual_columns]

        unexpected_columns = [
            col for col in actual_columns if col not in expected_columns
        ]

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
    """
    Validate that two DataFrames share the same ordered schema.

    The reference DataFrame defines the expected schema. The candidate
    DataFrame must contain the same columns in the same order.

    This helper is commonly used to ensure that:
    - synthetic data matches real-data schemas
    - validation/test splits remain consistent
    - preprocessing pipelines receive stable feature layouts

    Args:
        reference_df:
            Reference DataFrame defining the expected schema.

        candidate_df:
            DataFrame to validate.

        candidate_name:
            Human-readable candidate DataFrame name for error reporting.
    """
    validate_columns(
        candidate_df,
        expected_columns=list(reference_df.columns),
        dataframe_name=candidate_name,
    )


def validate_non_empty_dataframe(
    df: pd.DataFrame,
    dataframe_name: str,
) -> None:
    """
    Validate that a DataFrame is not empty.

    Args:
        df:
            DataFrame to validate.

        dataframe_name:
            Human-readable DataFrame name for error reporting.

    Raises:
        ValueError:
            If the DataFrame contains zero rows.
    """
    if df.empty:
        raise ValueError(f"{dataframe_name} is empty.")
