"""
synthesize.py

Synthesis pipeline for the Adult Census Income dataset.
Trains a non-DP synthesizer on the real training data and generates a
synthetic dataset of equal size.

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

import pandas as pd
import torch
from sdv.metadata import Metadata
from sdv.single_table import (
    CTGANSynthesizer,
    GaussianCopulaSynthesizer,
    TVAESynthesizer,
)

from src.core.io import load_csv, validate_matching_columns
from src.core.paths import (
    processed_split_path,
    synthesizer_metadata_path,
    synthesizer_model_path,
    synthetic_output_dir,
    synthetic_output_path,
)
from src.utility.constants import (
    RANDOM_STATE,
    SYNTHESIZERS,
    SYNTHESIZER_MODELS_DIR,
    TRAIN_FILENAME,
)
from src.utility.logger import RunLogger
from src.utility.utils import set_random_seeds

SCRIPT_NAME = "synthesize.py"


def load_training_data() -> pd.DataFrame:
    """Load the real training split from disk."""
    train_path = processed_split_path(TRAIN_FILENAME)
    train_df = load_csv(train_path, "Training split")

    print(
        f"[synthesize] Loaded training data from {train_path.resolve()} "
        f"({len(train_df)} rows)"
    )
    return train_df


def validate_synthetic_output(
    train_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    expected_rows: int,
) -> None:
    """Validate generated synthetic data before saving."""
    if len(synthetic_df) != expected_rows:
        raise ValueError(
            f"Synthetic data has {len(synthetic_df)} rows, expected {expected_rows}."
        )

    validate_matching_columns(
        reference_df=train_df,
        candidate_df=synthetic_df,
        candidate_name="Synthetic data",
    )


def build_metadata(df: pd.DataFrame) -> Metadata:
    """Build and validate SDV metadata for the Adult Census table."""
    metadata = Metadata.detect_from_dataframe(data=df, table_name="adult")
    metadata.update_column(
        table_name="adult",
        column_name="income",
        sdtype="categorical",
    )
    metadata.validate()

    print("[synthesize] Metadata validated successfully.")
    return metadata


def build_synthesizer(
    synthesizer_name: str,
    metadata: Metadata,
    cuda: bool = False,
):
    """Build a non-DP SDV synthesizer."""
    if synthesizer_name == "gaussian_copula":
        return GaussianCopulaSynthesizer(metadata=metadata)

    synthesizer_cls = {
        "ctgan": CTGANSynthesizer,
        "tvae": TVAESynthesizer,
    }.get(synthesizer_name)

    if synthesizer_cls is None:
        raise ValueError(f"Unknown synthesizer: {synthesizer_name}")

    return synthesizer_cls(metadata=metadata, enable_gpu=cuda, verbose=True)


def print_gpu_info(synthesizer_name: str, cuda: bool) -> None:
    """Print GPU availability and usage status."""
    if synthesizer_name not in {"ctgan", "tvae"}:
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


def log_loss_values(synthesizer_name: str, synthesizer, logger: RunLogger) -> None:
    """Log iterative training losses for CTGAN and TVAE."""
    if synthesizer_name not in {"ctgan", "tvae"}:
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


def train_and_generate(
    synthesizer_name: str,
    cuda: bool = False,
    use_wandb: bool = False,
) -> None:
    """Train a non-DP synthesizer and save generated synthetic data."""
    run_name = f"synthesizer_{synthesizer_name}_default"

    gpu_available = torch.cuda.is_available()
    gpu_in_use = cuda and gpu_available

    parameters = {
        "pipeline_stage": "synthesis",
        "evaluation": None,
        "mode": "default",
        "data_source": synthesizer_name,
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
        train_df = load_training_data()
        n_samples = len(train_df)

        metadata = build_metadata(train_df)
        set_random_seeds(RANDOM_STATE)

        SYNTHESIZER_MODELS_DIR.mkdir(parents=True, exist_ok=True)

        metadata_path = synthesizer_metadata_path(synthesizer_name)
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
        validate_synthetic_output(train_df, synthetic_df, expected_rows=n_samples)

        model_path = synthesizer_model_path(synthesizer_name, mode="default")
        synthesizer.save(str(model_path))
        print(f"[synthesize] Synthesizer saved to {model_path.resolve()}")

        out_dir = synthetic_output_dir(synthesizer_name, mode="default")
        out_dir.mkdir(parents=True, exist_ok=True)

        synthetic_path = synthetic_output_path(synthesizer_name, mode="default")
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a non-DP synthesizer and generate synthetic data."
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
