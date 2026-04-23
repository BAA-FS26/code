"""
utils.py

Shared utilities for the synthetic data evaluation pipeline.

This module contains two kinds of helpers that are used across the project:
- reproducibility helpers
- metadata helpers for SDV / SDMetrics interoperability

Note:
    The metadata fallback builder in this module is intentionally Adult-dataset
    specific because the project scope is fixed to the Adult Census dataset.
    Its naming makes that explicit so the boundary stays clear.

Usage:
    from src.utility.utils import (
        set_random_seeds,
        build_adult_sdmetrics_metadata,
        build_sdmetrics_metadata,
        load_metadata,
    )
"""

import os
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from sdv.metadata import Metadata

from src.dataset.adult_census import TARGET_COL
from src.utility.constants import RANDOM_STATE

# ── Adult dataset metadata fallback configuration ─────────────────────────────

_ADULT_NUMERICAL_COLS = [
    "age",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

_ADULT_CATEGORICAL_COLS = [
    "workclass",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


# ── Reproducibility ───────────────────────────────────────────────────────────


def set_random_seeds(seed: int = RANDOM_STATE, strict: bool = True) -> None:
    """
    Seed common random number generators and configure PyTorch determinism.

    Covers Python's random module, NumPy, PyTorch (CPU/GPU), and sets
    PYTHONHASHSEED in the process environment for consistency in child
    processes spawned after this function is called.

    Determinism notes:
    - PYTHONHASHSEED affects hash randomization at interpreter startup, so
      setting it here does not retroactively change hash behavior for the
      current Python process. It is still useful for subprocesses launched
      afterwards.
    - When strict=True, this function enables PyTorch deterministic
      algorithms where supported. Some operations may then raise runtime
      errors if no deterministic implementation exists.
    - Full bitwise reproducibility is still not guaranteed in all cases.
      Remaining nondeterminism may come from GPU operations, third-party
      model internals, and library-specific training code.

    Args:
        seed: Integer seed value. Defaults to RANDOM_STATE.
        strict: Whether to request deterministic PyTorch algorithms when
                supported. Defaults to True.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if strict and hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as exc:
            raise RuntimeError(
                "Failed to enable deterministic PyTorch algorithms. "
                "This environment or workload may rely on nondeterministic "
                "operations."
            ) from exc


# ── Metadata helpers ──────────────────────────────────────────────────────────


def build_adult_sdmetrics_metadata() -> dict[str, dict[str, dict[str, str]]]:
    """
    Build an SDMetrics-compatible single-table metadata dictionary for the
    Adult Census Income dataset.

    This is used as a fallback when no saved SDV metadata JSON exists, for
    example with DP synthesizers trained via SmartNoise which do not produce
    SDV metadata files.

    Returns:
        Single-table metadata dictionary with a top-level 'columns' key,
        compatible with SDMetrics QualityReport, DiagnosticReport, and DCR
        metrics.

    Raises:
        ValueError: If the generated metadata is structurally invalid.
    """
    columns: dict[str, dict[str, str]] = {}

    for col in _ADULT_NUMERICAL_COLS:
        columns[col] = {"sdtype": "numerical"}

    for col in _ADULT_CATEGORICAL_COLS:
        columns[col] = {"sdtype": "categorical"}

    columns[TARGET_COL] = {"sdtype": "categorical"}

    metadata = {"columns": columns}

    if "columns" not in metadata or not metadata["columns"]:
        raise ValueError(
            "Fallback SDMetrics metadata must contain a non-empty 'columns' mapping."
        )

    required_cols = set(_ADULT_NUMERICAL_COLS + _ADULT_CATEGORICAL_COLS + [TARGET_COL])
    actual_cols = set(metadata["columns"].keys())
    missing_cols = sorted(required_cols - actual_cols)
    if missing_cols:
        raise ValueError(
            "Fallback SDMetrics metadata is missing required Adult columns: "
            f"{missing_cols}"
        )

    target_sdtype = metadata["columns"][TARGET_COL].get("sdtype")
    if target_sdtype != "categorical":
        raise ValueError(
            f"Fallback SDMetrics metadata must mark '{TARGET_COL}' as categorical."
        )

    return metadata


def build_sdmetrics_metadata() -> dict[str, dict[str, dict[str, str]]]:
    """
    Backward-compatible alias for the Adult-specific SDMetrics metadata builder.

    Returns:
        SDMetrics-compatible single-table metadata dictionary for the Adult
        Census dataset.
    """
    return build_adult_sdmetrics_metadata()


def load_metadata(
    models_dir: Path,
    synthesizer_name: str,
    fallback: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Load saved SDV metadata for a synthesizer and return a single-table
    metadata dictionary compatible with SDMetrics.

    SDV's Metadata class stores metadata in a multi-table structure.
    This project is explicitly single-table, so this helper extracts the
    single table definition and returns it in the format expected by
    SDMetrics.

    If no metadata file exists and a fallback is provided, the fallback is
    returned instead. This is used for DP synthesizers trained with
    SmartNoise, which do not produce SDV metadata files.

    Args:
        models_dir: Directory where synthesizer metadata JSON files are stored.
        synthesizer_name: Name of the synthesizer, used to construct the
                          expected metadata filename.
        fallback: Optional metadata dictionary to return if the metadata file
                  does not exist.

    Returns:
        Single-table metadata dictionary with 'columns' at top level.

    Raises:
        FileNotFoundError: If no metadata file exists and no fallback is
                           provided.
        RuntimeError: If the metadata file cannot be loaded or parsed.
        ValueError: If the metadata structure is invalid for this single-table
                    project.
    """
    metadata_path = models_dir / f"{synthesizer_name}_metadata.json"

    if not metadata_path.exists():
        if fallback is not None:
            return fallback
        raise FileNotFoundError(
            f"No metadata found at {metadata_path}. "
            "Run synthesize.py first to generate and save the metadata."
        )

    try:
        full_dict = Metadata.load_from_json(str(metadata_path)).to_dict()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load or parse metadata from {metadata_path}."
        ) from exc

    tables = full_dict.get("tables")
    if not isinstance(tables, dict) or not tables:
        raise ValueError(
            f"Metadata at {metadata_path} does not contain a valid non-empty 'tables' mapping."
        )

    if len(tables) != 1:
        raise ValueError(
            f"Expected exactly one table in metadata at {metadata_path}, "
            f"but found {len(tables)}. This pipeline supports single-table metadata only."
        )

    table_name, table_metadata = next(iter(tables.items()))
    if not isinstance(table_metadata, dict):
        raise ValueError(
            f"Table metadata for '{table_name}' in {metadata_path} is not a dictionary."
        )

    columns = table_metadata.get("columns")
    if not isinstance(columns, dict) or not columns:
        raise ValueError(
            f"Table '{table_name}' in {metadata_path} does not contain a valid non-empty 'columns' mapping."
        )

    return table_metadata
