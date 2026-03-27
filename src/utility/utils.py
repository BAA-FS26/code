"""
utils.py

Shared utilities for the synthetic data evaluation pipeline.

Usage:
    from src.utility.utils import load_metadata, set_random_seeds
"""

import os
import random
from pathlib import Path

import numpy as np
import torch
from sdv.metadata import Metadata

from src.utility.constants import RANDOM_STATE


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


def load_metadata(models_dir: Path, synthesizer_name: str) -> dict:
    """
    Load the saved SDV metadata for a given synthesizer and return as
    a single-table metadata dictionary compatible with SDMetrics.

    The new SDV Metadata class uses a multi-table structure internally.
    SDMetrics expects a single-table dict with 'columns' at the top level,
    so the table-level dict is extracted from the full metadata.

    Args:
        models_dir: Directory where synthesizer metadata JSON files are stored.
        synthesizer_name: One of 'gaussian_copula', 'ctgan', 'tvae'.

    Returns:
        Single-table metadata dictionary with 'columns' at top level.

    Raises:
        FileNotFoundError: If no metadata file exists for the given synthesizer.
        ValueError: If the metadata file contains no tables.
    """
    metadata_path = models_dir / f"{synthesizer_name}_metadata.json"
    if not metadata_path.exists():
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
