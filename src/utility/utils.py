"""
Shared utilities for the synthetic data evaluation pipeline.

Contains:
- reproducibility helpers
- SDV / SDMetrics metadata helpers
"""

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sdv.metadata import Metadata

from src.dataset.adult_census import CATEGORICAL_COLS, NUMERICAL_COLS, TARGET_COL
from src.utility.constants import RANDOM_STATE

_ADULT_NUMERICAL_COLS = NUMERICAL_COLS
_ADULT_CATEGORICAL_COLS = CATEGORICAL_COLS


def set_random_seeds(seed: int = RANDOM_STATE, strict: bool = True) -> None:
    """Seed common random number generators and configure PyTorch determinism."""
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
                "This environment or workload may rely on nondeterministic operations."
            ) from exc


def build_adult_sdmetrics_metadata() -> dict[str, dict[str, dict[str, str]]]:
    """Build an SDMetrics-compatible metadata dictionary for Adult Census."""
    columns: dict[str, dict[str, str]] = {}

    for column in _ADULT_NUMERICAL_COLS:
        columns[column] = {"sdtype": "numerical"}

    for column in _ADULT_CATEGORICAL_COLS:
        columns[column] = {"sdtype": "categorical"}

    columns[TARGET_COL] = {"sdtype": "categorical"}

    metadata = {"columns": columns}
    _validate_adult_sdmetrics_metadata(metadata)

    return metadata


def build_sdmetrics_metadata() -> dict[str, dict[str, dict[str, str]]]:
    """Backward-compatible alias for Adult-specific SDMetrics metadata."""
    return build_adult_sdmetrics_metadata()


def _validate_adult_sdmetrics_metadata(
    metadata: dict[str, dict[str, dict[str, str]]],
) -> None:
    """Validate the Adult fallback SDMetrics metadata structure."""
    columns = metadata.get("columns")

    if not columns:
        raise ValueError(
            "Fallback SDMetrics metadata must contain a non-empty 'columns' mapping."
        )

    required_cols = set(_ADULT_NUMERICAL_COLS + _ADULT_CATEGORICAL_COLS + [TARGET_COL])
    actual_cols = set(columns.keys())
    missing_cols = sorted(required_cols - actual_cols)

    if missing_cols:
        raise ValueError(
            "Fallback SDMetrics metadata is missing required Adult columns: "
            f"{missing_cols}"
        )

    target_sdtype = columns[TARGET_COL].get("sdtype")
    if target_sdtype != "categorical":
        raise ValueError(
            f"Fallback SDMetrics metadata must mark '{TARGET_COL}' as categorical."
        )


def load_metadata(
    models_dir: Path,
    synthesizer_name: str,
    fallback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load saved SDV metadata or return a provided fallback metadata dict."""
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

    return _extract_single_table_metadata(full_dict, metadata_path)


def _extract_single_table_metadata(
    full_metadata: dict[str, Any],
    metadata_path: Path,
) -> dict[str, Any]:
    """Extract the single-table metadata structure expected by SDMetrics."""
    tables = full_metadata.get("tables")

    if not isinstance(tables, dict) or not tables:
        raise ValueError(
            f"Metadata at {metadata_path} does not contain a valid non-empty "
            "'tables' mapping."
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
            f"Table '{table_name}' in {metadata_path} does not contain a valid "
            "non-empty 'columns' mapping."
        )

    return table_metadata
