"""
synthesize.py

Synthesis pipeline for the Adult Census Income dataset.
Trains a synthesizer on the real training data and generates a synthetic
dataset of equal size.

Supported synthesizers:
  - gaussian_copula
  - ctgan
  - tvae

Usage:
    # Without W&B (default)
    python -m src.modeling.synthesize --synthesizer gaussian_copula
    python -m src.modeling.synthesize --synthesizer ctgan
    python -m src.modeling.synthesize --synthesizer tvae

    # With W&B logging
    python -m src.modeling.synthesize --synthesizer ctgan --wandb

    # With GPU acceleration (CTGAN and TVAE only)
    python -m src.modeling.synthesize --synthesizer ctgan --cuda
    python -m src.modeling.synthesize --synthesizer ctgan --cuda --wandb
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from sdv.metadata import Metadata
from sdv.single_table import (
    CTGANSynthesizer,
    GaussianCopulaSynthesizer,
    TVAESynthesizer,
)

from src.utility.constants import (
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    SYNTHESIZER_MODELS_DIR,
    SYNTHESIZERS,
    SYNTHETIC_DATA_DIR,
    SYNTHETIC_TRAIN_FILENAME,
    TRAIN_FILENAME,
)
from src.utility.logger import RunLogger
from src.utility.utils import set_random_seeds

# ── Constants ────────────────────────────────────────────────────────────────

SCRIPT_NAME = "synthesize.py"
MODELS_SUBDIR = SYNTHESIZER_MODELS_DIR


# ── Path helpers ──────────────────────────────────────────────────────────────


def _training_data_path() -> Path:
    """Return the canonical path to the real training split."""
    return PROCESSED_DATA_DIR / TRAIN_FILENAME


def _metadata_path(synthesizer_name: str) -> Path:
    """Return the canonical metadata path for a synthesizer."""
    return MODELS_SUBDIR / f"{synthesizer_name}_metadata.json"


def _model_path(synthesizer_name: str) -> Path:
    """Return the canonical saved synthesizer model path."""
    return MODELS_SUBDIR / f"{synthesizer_name}_default.pkl"


def _synthetic_output_dir(synthesizer_name: str) -> Path:
    """Return the canonical output directory for synthetic training data."""
    return SYNTHETIC_DATA_DIR / synthesizer_name / "default"


def _synthetic_output_path(synthesizer_name: str) -> Path:
    """Return the canonical synthetic training data output path."""
    return _synthetic_output_dir(synthesizer_name) / SYNTHETIC_TRAIN_FILENAME


# ── Data loading and validation ───────────────────────────────────────────────


def _load_training_data() -> pd.DataFrame:
    """
    Load the real training split from disk.

    Returns:
        Training DataFrame.

    Raises:
        FileNotFoundError: If the canonical training split does not exist.
    """
    train_path = _training_data_path()
    if not train_path.exists():
        raise FileNotFoundError(
            f"Training split not found at {train_path}. "
            "Run the dataset cleaning and splitting pipeline first."
        )

    train_df = pd.read_csv(train_path)
    print(
        f"[synthesize] Loaded training data from {train_path.resolve()} ({len(train_df)} rows)"
    )
    return train_df


def _validate_synthetic_output(
    train_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    expected_rows: int,
) -> None:
    """
    Validate the generated synthetic dataset before saving.

    Args:
        train_df: Real training DataFrame used to fit the synthesizer.
        synthetic_df: Generated synthetic DataFrame.
        expected_rows: Expected number of synthetic rows.

    Raises:
        ValueError: If row count or columns do not match expectations.
    """
    if len(synthetic_df) != expected_rows:
        raise ValueError(
            f"Synthetic data has {len(synthetic_df)} rows, expected {expected_rows}."
        )

    train_columns = list(train_df.columns)
    synthetic_columns = list(synthetic_df.columns)
    if synthetic_columns != train_columns:
        raise ValueError(
            "Synthetic data columns do not match training data columns.\n"
            f"Expected: {train_columns}\n"
            f"Actual:   {synthetic_columns}"
        )


# ── Metadata ─────────────────────────────────────────────────────────────────


def build_metadata(df: pd.DataFrame) -> Metadata:
    """
    Auto-detect and validate SDV metadata from a DataFrame.

    This function is the canonical source of SDV metadata generation for
    non-DP synthesizers in this pipeline. Evaluation scripts later reuse
    the metadata persisted by this script.

    Auto-detects column types from the DataFrame using the SDV Metadata
    class and applies a manual correction for the target column:
    - income: explicitly set to categorical to ensure correct handling

    Args:
        df: Training DataFrame to detect metadata from.

    Returns:
        Validated Metadata object.
    """
    metadata = Metadata.detect_from_dataframe(data=df, table_name="adult")
    metadata.update_column(
        table_name="adult",
        column_name="income",
        sdtype="categorical",
    )
    metadata.validate()
    print("[synthesize] Metadata validated successfully.")
    return metadata


# ── Model building ────────────────────────────────────────────────────────────


def build_synthesizer(synthesizer_name: str, metadata: Metadata, cuda: bool = False):
    """
    Build a synthesizer instance with default parameters.

    Args:
        synthesizer_name: One of 'gaussian_copula', 'ctgan', 'tvae'.
        metadata: Validated Metadata object.
        cuda: Whether to use GPU acceleration for CTGAN and TVAE.
              Has no effect on GaussianCopula. Defaults to False.

    Returns:
        Unfitted SDV synthesizer instance.

    Raises:
        ValueError: If synthesizer_name is not recognised.
    """
    if synthesizer_name == "gaussian_copula":
        return GaussianCopulaSynthesizer(metadata=metadata)

    synthesizer_cls = {
        "ctgan": CTGANSynthesizer,
        "tvae": TVAESynthesizer,
    }.get(synthesizer_name)

    if synthesizer_cls is None:
        raise ValueError(f"Unknown synthesizer: {synthesizer_name}")

    return synthesizer_cls(metadata=metadata, enable_gpu=cuda, verbose=True)


# ── GPU info ──────────────────────────────────────────────────────────────────


def print_gpu_info(synthesizer_name: str, cuda: bool) -> None:
    """
    Print GPU availability and actual usage status.

    Args:
        synthesizer_name: Synthesizer being used.
        cuda: Whether GPU was requested via --cuda flag.
    """
    if synthesizer_name not in ("ctgan", "tvae"):
        print("[synthesize] GPU acceleration: not applicable for GaussianCopula")
        return

    gpu_available = torch.cuda.is_available()
    gpu_in_use = cuda and gpu_available

    print(f"[synthesize] GPU available: {gpu_available}")
    print(f"[synthesize] GPU requested: {cuda}")
    print(f"[synthesize] GPU in use: {gpu_in_use}")

    if cuda and not gpu_available:
        print(
            "[synthesize] Warning: --cuda was set but no GPU is available. "
            "Falling back to CPU."
        )


# ── Loss logging ──────────────────────────────────────────────────────────────


def log_loss_values(synthesizer_name: str, synthesizer, logger: RunLogger) -> None:
    """
    Log per-epoch loss values via the run logger.

    CTGAN logs generator and discriminator loss per epoch.
    TVAE logs a single ELBO loss per epoch, averaged across batches.
    GaussianCopula has no iterative training loop and is skipped.

    Args:
        synthesizer_name: One of 'gaussian_copula', 'ctgan', 'tvae'.
        synthesizer: Fitted SDV synthesizer instance.
        logger: Active RunLogger instance.
    """
    if synthesizer_name not in ("ctgan", "tvae"):
        return

    loss_values = synthesizer.get_loss_values()

    if synthesizer_name == "ctgan":
        for _, row in loss_values.iterrows():
            logger.log(
                {
                    "epoch": int(row["Epoch"]),
                    "loss_generator": row["Generator Loss"],
                    "loss_discriminator": row["Discriminator Loss"],
                }
            )

    elif synthesizer_name == "tvae":
        epoch_loss = loss_values.groupby("Epoch")["Loss"].mean().reset_index()
        for _, row in epoch_loss.iterrows():
            logger.log({"epoch": int(row["Epoch"]), "loss": row["Loss"]})


# ── Train and generate ────────────────────────────────────────────────────────


def train_and_generate(
    synthesizer_name: str,
    cuda: bool = False,
    use_wandb: bool = False,
) -> None:
    """
    Train a synthesizer on real training data and generate synthetic data.

    Loads the real training split, fits the synthesizer, generates a
    synthetic dataset of equal size, and saves both the fitted synthesizer
    and the synthetic data to disk. Training time and run metadata are
    always saved locally. Loss curves and metrics are also logged to W&B
    if enabled.

    The synthesizer is trained on the training split only. Validation and
    test splits are never used during synthesis.

    Args:
        synthesizer_name: One of 'gaussian_copula', 'ctgan', 'tvae'.
        cuda: Whether to use GPU acceleration for CTGAN and TVAE.
              Has no effect on GaussianCopula. Defaults to False.
        use_wandb: Whether to log results to W&B. Defaults to False.
    """
    run_name = f"synthesizer_{synthesizer_name}_default"
    gpu_available = torch.cuda.is_available()
    gpu_in_use = cuda and gpu_available

    data_source = synthesizer_name
    parameters = {
        "pipeline_stage": "synthesis",
        "evaluation": None,
        "mode": "default",
        "data_source": data_source,
        "synthesizer": synthesizer_name,
        "epsilon": None,
        "classifier": None,
        "model_type": None,
        "params": {},
        "random_state": RANDOM_STATE,
        "use_wandb": use_wandb,
        "cuda_requested": cuda,
        "gpu_available": gpu_available,
        "gpu_in_use": gpu_in_use,
    }

    with RunLogger(
        run_name=run_name,
        script_name=SCRIPT_NAME,
        parameters=parameters,
        use_wandb=use_wandb,
        category="synthesis",
    ) as logger:
        train_df = _load_training_data()
        n_samples = len(train_df)

        metadata = build_metadata(train_df)
        set_random_seeds(RANDOM_STATE)

        MODELS_SUBDIR.mkdir(parents=True, exist_ok=True)

        metadata_path = _metadata_path(synthesizer_name)
        if metadata_path.exists():
            print(
                f"[synthesize] Metadata already exists at {metadata_path.resolve()}, "
                "skipping save."
            )
        else:
            metadata.save_to_json(str(metadata_path))
            print(f"[synthesize] Metadata saved to {metadata_path.resolve()}")

        synthesizer = build_synthesizer(synthesizer_name, metadata, cuda=cuda)
        print_gpu_info(synthesizer_name, cuda)

        print(f"[synthesize] Training {synthesizer_name}...")
        start_time = time.time()
        synthesizer.fit(train_df)
        training_time = time.time() - start_time
        print(f"[synthesize] Training complete in {training_time:.1f}s")

        log_loss_values(synthesizer_name, synthesizer, logger)

        print(f"[synthesize] Generating {n_samples} synthetic samples...")
        synthetic_df = synthesizer.sample(num_rows=n_samples)
        _validate_synthetic_output(train_df, synthetic_df, expected_rows=n_samples)

        model_path = _model_path(synthesizer_name)
        synthesizer.save(str(model_path))
        print(f"[synthesize] Synthesizer saved to {model_path.resolve()}")

        out_dir = _synthetic_output_dir(synthesizer_name)
        out_dir.mkdir(parents=True, exist_ok=True)

        synthetic_path = _synthetic_output_path(synthesizer_name)
        synthetic_df.to_csv(synthetic_path, index=False)
        print(f"[synthesize] Synthetic data saved to {synthetic_path.resolve()}")

        logger.log(
            {
                "training_time_seconds": training_time,
                "n_samples_train": n_samples,
                "n_samples_synthetic": len(synthetic_df),
                "n_features": len(train_df.columns),
                "metadata_path": metadata_path,
                "model_path": model_path,
                "synthetic_data_path": synthetic_path,
            }
        )


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a synthesizer and generate synthetic data."
    )
    parser.add_argument(
        "--synthesizer",
        choices=SYNTHESIZERS,
        required=True,
        help="Synthesizer to train.",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        default=False,
        help="Use GPU acceleration for CTGAN and TVAE. Defaults to False.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="Log results to Weights & Biases. Local JSON logging remains primary.",
    )

    args = parser.parse_args()
    train_and_generate(
        synthesizer_name=args.synthesizer,
        cuda=args.cuda,
        use_wandb=args.wandb,
    )


if __name__ == "__main__":
    main()
