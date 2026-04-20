"""
utils.py

Shared utilities for the synthetic data evaluation pipeline.

Usage:
    from src.utility.utils import load_metadata, build_sdmetrics_metadata, set_random_seeds
"""

import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sdv.metadata import Metadata

from src.utility.constants import RANDOM_STATE
from src.dataset.feature_engineering import CATEGORICAL_COLS, NUMERICAL_COLS, TARGET_COL


def set_random_seeds(seed: int = RANDOM_STATE) -> None:
    """
    Seed all RNGs and configure backends for reproducible training.

    Covers Python's random, NumPy, PyTorch (CPU/GPU), and Python hash seeds.
    Should be called once before model initialisation or data synthesis
    (CTGAN, TVAE) to ensure consistent weights and sampling.

    Note: While this sets cuDNN to deterministic mode, bit-wise 100%
    reproducibility on GPU may still require the environment variable
    CUBLAS_WORKSPACE_CONFIG=:4096:8 and torch.use_deterministic_algorithms(True).

    Args:
        seed: Integer seed value. Defaults to RANDOM_STATE (42).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_sdmetrics_metadata() -> dict:
    """
    Build a SDMetrics-compatible metadata dictionary for the Adult Census
    Income dataset.

    Used as a fallback when no SDV metadata file exists, i.e. for DP
    synthesizers trained with SmartNoise which do not produce SDV metadata.

    Returns:
        Single-table metadata dictionary with 'columns' at top level,
        compatible with SDMetrics QualityReport, DiagnosticReport and
        DCR metrics.
    """
    columns = {}
    for col in NUMERICAL_COLS:
        columns[col] = {"sdtype": "numerical"}
    for col in CATEGORICAL_COLS:
        columns[col] = {"sdtype": "categorical"}
    columns[TARGET_COL] = {"sdtype": "categorical"}
    return {"columns": columns}


def load_metadata(
    models_dir: Path,
    synthesizer_name: str,
    fallback: Optional[dict] = None,
) -> dict:
    """
    Load the saved SDV metadata for a given synthesizer and return as
    a single-table metadata dictionary compatible with SDMetrics.

    The new SDV Metadata class uses a multi-table structure internally.
    SDMetrics expects a single-table dict with 'columns' at the top level,
    so the table-level dict is extracted from the full metadata.

    If no metadata file exists and a fallback is provided, the fallback
    is returned instead. This is used for DP synthesizers trained with
    SmartNoise, which do not produce SDV metadata files.

    Args:
        models_dir: Directory where synthesizer metadata JSON files are stored.
        synthesizer_name: One of 'gaussian_copula', 'ctgan', 'tvae'.
        fallback: Optional metadata dict to return if no file exists.
                  Use build_sdmetrics_metadata() to generate one.

    Returns:
        Single-table metadata dictionary with 'columns' at top level.

    Raises:
        FileNotFoundError: If no metadata file exists and no fallback is
                           provided.
        ValueError: If the metadata file contains no tables.
    """
    metadata_path = models_dir / f"{synthesizer_name}_metadata.json"
    if not metadata_path.exists():
        if fallback is not None:
            return fallback
        raise FileNotFoundError(
            f"No metadata found at {metadata_path}. "
            "Run synthesize.py first to generate and save the metadata."
        )
    full_dict = Metadata.load_from_json(str(metadata_path)).to_dict()
    tables = full_dict.get("tables", {})
    if not tables:
        raise ValueError("Metadata contains no tables.")
    table_name = next(iter(tables))
    return tables[table_name]
