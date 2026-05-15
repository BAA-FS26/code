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
    python -m src.modeling.synthesizing.synthesize --synthesizer gaussian_copula
    python -m src.modeling.synthesizing.synthesize --synthesizer ctgan
    python -m src.modeling.synthesizing.synthesize --synthesizer tvae

    # With W&B logging
    python -m src.modeling.synthesizing.synthesize --synthesizer ctgan --wandb

    # With GPU acceleration (CTGAN and TVAE only)
    python -m src.modeling.synthesizing.synthesize --synthesizer ctgan --cuda
    python -m src.modeling.synthesizing.synthesize --synthesizer ctgan --cuda --wandb
"""

import argparse
import time
from pathlib import Path
from typing import Any

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


def _run_name(synthesizer_name: str) -> str:
    """Return a consistent run name for logging."""
    return f"synthesizer_{synthesizer_name}_default"


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
            f"Synthetic data has {len(synthetic_df)} rows, "
            f"expected {expected_rows}."
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


def _validate_synthesizer_name(synthesizer_name: str) -> None:
    """Validate synthesizer selection."""
    if synthesizer_name not in SYNTHESIZERS:
        raise ValueError(
            f"Unknown synthesizer '{synthesizer_name}'. "
            f"Choose from: {sorted(SYNTHESIZERS)}"
        )


def build_synthesizer(
    synthesizer_name: str,
    metadata: Metadata,
    cuda: bool = False,
):
    """Build a non-DP SDV synthesizer."""
    _validate_synthesizer_name(synthesizer_name)

    if synthesizer_name == "gaussian_copula":
        return GaussianCopulaSynthesizer(metadata=metadata)

    synthesizer_cls = {
        "ctgan": CTGANSynthesizer,
        "tvae": TVAESynthesizer,
    }[synthesizer_name]

    return synthesizer_cls(
        metadata=metadata,
        enable_gpu=cuda,
        verbose=True,
    )


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


def _build_run_parameters(
    synthesizer_name: str,
    cuda: bool,
    use_wandb: bool,
) -> dict[str, Any]:
    """
    Build logger metadata for synthesis runs.

    IMPORTANT:
    Keys must remain stable for dashboard/result compatibility.
    """
    return {
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
        "cuda": cuda,
    }


def _save_metadata(
    metadata: Metadata,
    synthesizer_name: str,
) -> Path:
    """Save SDV metadata to disk."""
    SYNTHESIZER_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    metadata_path = synthesizer_metadata_path(synthesizer_name)

    metadata.save_to_json(filepath=str(metadata_path))

    print(f"[synthesize] Metadata saved to {metadata_path.resolve()}")

    return metadata_path


def _save_synthesizer(
    synthesizer: Any,
    synthesizer_name: str,
) -> Path:
    """Save trained synthesizer model to disk."""
    SYNTHESIZER_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = synthesizer_model_path(
        synthesizer_name=synthesizer_name,
        mode="default",
    )

    synthesizer.save(filepath=str(model_path))

    print(f"[synthesize] Synthesizer saved to {model_path.resolve()}")

    return model_path


def _save_synthetic_data(
    synthetic_df: pd.DataFrame,
    synthesizer_name: str,
) -> Path:
    """Save generated synthetic data to disk."""
    output_dir = synthetic_output_dir(
        synthesizer_name=synthesizer_name,
        mode="default",
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = synthetic_output_path(
        synthesizer_name=synthesizer_name,
        mode="default",
    )

    synthetic_df.to_csv(output_path, index=False)

    print(f"[synthesize] Synthetic data saved to {output_path.resolve()}")

    return output_path


def log_loss_values(
    synthesizer_name: str,
    synthesizer,
    logger: RunLogger,
) -> None:
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
            logger.log(
                {
                    "epoch": int(row["Epoch"]),
                    "loss": row["Loss"],
                }
            )


def train_and_generate(
    synthesizer_name: str,
    cuda: bool = False,
    use_wandb: bool = False,
) -> None:
    """
    Train a non-DP synthesizer and save generated synthetic data.
    """
    set_random_seeds(RANDOM_STATE)

    run_name = _run_name(synthesizer_name)

    with RunLogger(
        run_name=run_name,
        script_name=SCRIPT_NAME,
        parameters=_build_run_parameters(
            synthesizer_name=synthesizer_name,
            cuda=cuda,
            use_wandb=use_wandb,
        ),
        use_wandb=use_wandb,
        category="synthesis",
    ) as logger:
        train_df = load_training_data()

        print_gpu_info(
            synthesizer_name=synthesizer_name,
            cuda=cuda,
        )

        metadata = build_metadata(train_df)

        synthesizer = build_synthesizer(
            synthesizer_name=synthesizer_name,
            metadata=metadata,
            cuda=cuda,
        )

        print(f"[synthesize] Training {synthesizer_name}...")

        start_time = time.perf_counter()

        synthesizer.fit(train_df)

        training_time_seconds = time.perf_counter() - start_time

        print(
            f"[synthesize] Training completed in "
            f"{training_time_seconds:.2f} seconds"
        )

        synthetic_df = synthesizer.sample(num_rows=len(train_df))

        validate_synthetic_output(
            train_df=train_df,
            synthetic_df=synthetic_df,
            expected_rows=len(train_df),
        )

        metadata_path = _save_metadata(
            metadata=metadata,
            synthesizer_name=synthesizer_name,
        )

        model_path = _save_synthesizer(
            synthesizer=synthesizer,
            synthesizer_name=synthesizer_name,
        )

        synthetic_data_path = _save_synthetic_data(
            synthetic_df=synthetic_df,
            synthesizer_name=synthesizer_name,
        )

        logger.log(
            {
                "training_time_seconds": training_time_seconds,
                "n_samples_train": len(train_df),
                "n_samples_synthetic": len(synthetic_df),
                "n_features": train_df.shape[1],
                "metadata_path": str(metadata_path),
                "model_path": str(model_path),
                "synthetic_data_path": str(synthetic_data_path),
            }
        )

        log_loss_values(
            synthesizer_name=synthesizer_name,
            synthesizer=synthesizer,
            logger=logger,
        )

        print("[synthesize] Run completed successfully.")


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
