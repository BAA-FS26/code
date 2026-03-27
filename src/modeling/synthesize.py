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
    # CPU (default)
    python synthesize.py --synthesizer gaussian_copula
    python synthesize.py --synthesizer ctgan
    python synthesize.py --synthesizer tvae

    # GPU
    python synthesize.py --synthesizer ctgan --cuda
    python synthesize.py --synthesizer tvae --cuda
"""

import argparse
import time

import pandas as pd
import torch
import wandb
from sdv.metadata import Metadata
from sdv.single_table import (
    CTGANSynthesizer,
    GaussianCopulaSynthesizer,
    TVAESynthesizer,
)

from src.utility.constants import (
    DATA_DIR,
    RANDOM_STATE,
    SYNTHESIZER_MODELS_DIR,
    SYNTHESIZERS,
    WANDB_ENTITY,
    WANDB_PROJECT,
)
from src.utility.utils import set_random_seeds

# ── Constants ────────────────────────────────────────────────────────────────

MODELS_DIR = SYNTHESIZER_MODELS_DIR


# ── Metadata ─────────────────────────────────────────────────────────────────


def build_metadata(df: pd.DataFrame) -> Metadata:
    """
    Auto-detect and validate SDV metadata from a DataFrame.

    Auto-detects column types from the DataFrame using the new SDV
    Metadata class and applies manual corrections for columns that
    are commonly misidentified:
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
    print("Metadata validated successfully.")
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

    gpu_kwargs = {"enable_gpu": cuda, "verbose": True}

    synthesizer_cls = {
        "ctgan": CTGANSynthesizer,
        "tvae": TVAESynthesizer,
    }.get(synthesizer_name)

    if synthesizer_cls is None:
        raise ValueError(f"Unknown synthesizer: {synthesizer_name}")

    return synthesizer_cls(metadata=metadata, **gpu_kwargs)


# ── GPU info ──────────────────────────────────────────────────────────────────


def print_gpu_info(synthesizer_name: str, cuda: bool) -> None:
    """
    Print GPU availability and actual usage status.

    Args:
        synthesizer_name: Synthesizer being used.
        cuda: Whether GPU was requested via --cuda flag.
    """
    if synthesizer_name not in ("ctgan", "tvae"):
        print("GPU acceleration: not applicable for GaussianCopula")
        return

    gpu_available = torch.cuda.is_available()
    gpu_in_use = cuda and gpu_available

    print(f"GPU available:  {gpu_available}")
    print(f"GPU requested:  {cuda}")
    print(f"GPU in use:     {gpu_in_use}")

    if cuda and not gpu_available:
        print("Warning: --cuda was set but no GPU is available. Falling back to CPU.")


# ── Loss logging ──────────────────────────────────────────────────────────────


def log_loss_values(synthesizer_name: str, synthesizer) -> None:
    """
    Log loss values to W&B for CTGAN and TVAE.

    CTGAN logs generator and discriminator loss per epoch.
    TVAE logs a single ELBO loss per epoch, averaged across batches.
    GaussianCopula has no iterative training loop and is skipped.

    Args:
        synthesizer_name: One of 'gaussian_copula', 'ctgan', 'tvae'.
        synthesizer: Fitted SDV synthesizer instance.
    """
    if synthesizer_name not in ("ctgan", "tvae"):
        return

    loss_values = synthesizer.get_loss_values()

    if synthesizer_name == "ctgan":
        for _, row in loss_values.iterrows():
            wandb.log(
                {
                    "epoch": int(row["Epoch"]),
                    "loss_generator": row["Generator Loss"],
                    "loss_discriminator": row["Discriminator Loss"],
                }
            )

    elif synthesizer_name == "tvae":
        epoch_loss = loss_values.groupby("Epoch")["Loss"].mean().reset_index()
        for _, row in epoch_loss.iterrows():
            wandb.log({"epoch": int(row["Epoch"]), "loss": row["Loss"]})


# ── Train and generate ────────────────────────────────────────────────────────


def train_and_generate(synthesizer_name: str, cuda: bool = False) -> None:
    """
    Train a synthesizer on real training data and generate synthetic data.

    Loads the real training split, fits the synthesizer, generates a
    synthetic dataset of equal size, and saves both the fitted synthesizer
    and the synthetic data to disk. Training time, loss curves and run
    metadata are logged to W&B.

    The synthesizer is trained on the training split only. Validation and
    test splits are never used during synthesis.

    Args:
        synthesizer_name: One of 'gaussian_copula', 'ctgan', 'tvae'.
        cuda: Whether to use GPU acceleration for CTGAN and TVAE.
              Has no effect on GaussianCopula. Defaults to False.
    """
    run_name = f"synthesizer_{synthesizer_name}_default"

    with wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=run_name,
        config={
            "synthesizer": synthesizer_name,
            "mode": "default",
            "cuda": cuda,
            "gpu_in_use": cuda and torch.cuda.is_available(),
        },
    ):
        train_df = pd.read_csv(DATA_DIR / "processed" / "train.csv")
        n_samples = len(train_df)
        print(f"Loaded training data: {n_samples} rows")

        metadata = build_metadata(train_df)
        set_random_seeds(RANDOM_STATE)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        metadata_path = MODELS_DIR / f"{synthesizer_name}_metadata.json"
        metadata.save_to_json(str(metadata_path))
        print(f"Metadata saved to {metadata_path.resolve()}")

        synthesizer = build_synthesizer(synthesizer_name, metadata, cuda=cuda)
        print_gpu_info(synthesizer_name, cuda)

        print(f"Training {synthesizer_name}...")
        start_time = time.time()
        synthesizer.fit(train_df)
        training_time = time.time() - start_time
        print(f"Training complete in {training_time:.1f}s")

        log_loss_values(synthesizer_name, synthesizer)

        print(f"Generating {n_samples} synthetic samples...")
        synthetic_df = synthesizer.sample(num_rows=n_samples)

        model_path = MODELS_DIR / f"{synthesizer_name}_default.pkl"
        synthesizer.save(str(model_path))
        print(f"Synthesizer saved to {model_path.resolve()}")

        out_dir = DATA_DIR / "synthetic" / synthesizer_name / "default"
        out_dir.mkdir(parents=True, exist_ok=True)
        synthetic_path = out_dir / "synthetic_train.csv"
        synthetic_df.to_csv(synthetic_path, index=False)
        print(f"Synthetic data saved to {synthetic_path.resolve()}")

        wandb.log(
            {
                "training_time_seconds": training_time,
                "n_samples_train": n_samples,
                "n_samples_synthetic": len(synthetic_df),
                "n_features": len(train_df.columns),
            }
        )


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
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

    args = parser.parse_args()
    train_and_generate(synthesizer_name=args.synthesizer, cuda=args.cuda)


if __name__ == "__main__":
    main()
