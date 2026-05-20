"""
Synthesis pipeline with Synthetic Data Vault synthesizer.

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
    TRAIN_FILENAME,
)
from src.utility.logger import RunLogger
from src.dataset.dataset_config import DatasetConfig, get_dataset_config
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


def build_metadata(
    df: pd.DataFrame,
    dataset_config: DatasetConfig,
) -> Metadata:
    """Build and validate SDV metadata for one tabular dataset."""
    metadata = Metadata.detect_from_dataframe(
        data=df,
        table_name=dataset_config.name,
    )

    metadata.update_column(
        table_name=dataset_config.name,
        column_name=dataset_config.target_col,
        sdtype="categorical",
    )

    metadata.validate()

    print("[synthesize] Metadata validated successfully.")
    return metadata


def build_synthesizer(
    synthesizer_name: str,
    metadata: Metadata,
    cuda: bool = False,
) -> Any:
    """Build a non-DP SDV synthesizer."""
    if synthesizer_name == "gaussian_copula":
        return GaussianCopulaSynthesizer(metadata=metadata)

    synthesizer_cls = {
        "ctgan": CTGANSynthesizer,
        "tvae": TVAESynthesizer,
    }.get(synthesizer_name)

    if synthesizer_cls is None:
        raise ValueError(f"Unknown synthesizer: {synthesizer_name}")

    return synthesizer_cls(metadata=metadata, cuda=cuda)


def _build_run_parameters(
    synthesizer_name: str,
    dataset_name: str,
    use_cuda: bool,
    use_wandb: bool,
) -> dict[str, Any]:
    """Build stable logger metadata for a synthesis run."""
    return {
        "pipeline_stage": "synthesis",
        "synthesizer": synthesizer_name,
        "dataset": dataset_name,
        "mode": "default",
        "epsilon": None,
        "random_state": RANDOM_STATE,
        "cuda": use_cuda,
        "use_wandb": use_wandb,
    }


def run_synthesis(
    synthesizer_name: str,
    dataset_name: str = "adult_census",
    cuda: bool = False,
    use_wandb: bool = False,
) -> None:
    """Train a SDV synthesizer and save synthetic training data."""
    if synthesizer_name not in SYNTHESIZERS:
        raise ValueError(
            f"Unsupported synthesizer '{synthesizer_name}'. "
            f"Available synthesizers: {sorted(SYNTHESIZERS)}"
        )

    if cuda and synthesizer_name == "gaussian_copula":
        raise ValueError("CUDA is only supported for CTGAN and TVAE.")

    if cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available.")

    set_random_seeds(RANDOM_STATE)

    dataset_config = get_dataset_config(dataset_name)

    train_df = load_training_data()
    metadata = build_metadata(train_df, dataset_config)

    output_dir = synthetic_output_dir(synthesizer_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dir = synthesizer_model_path(synthesizer_name).parent
    model_dir.mkdir(parents=True, exist_ok=True)

    run_name = f"synthesizer_{synthesizer_name}_default"

    with RunLogger(
        run_name=run_name,
        script_name=SCRIPT_NAME,
        parameters=_build_run_parameters(
            synthesizer_name=synthesizer_name,
            dataset_name=dataset_name,
            use_cuda=cuda,
            use_wandb=use_wandb,
        ),
        use_wandb=use_wandb,
        category="synthesis",
    ) as logger:
        synthesizer = build_synthesizer(
            synthesizer_name=synthesizer_name,
            metadata=metadata,
            cuda=cuda,
        )

        print(f"[synthesize] Training {synthesizer_name}...")
        start_time = time.time()
        synthesizer.fit(train_df)
        training_time_seconds = time.time() - start_time

        print(f"[synthesize] Sampling {len(train_df)} synthetic rows...")
        synthetic_df = synthesizer.sample(num_rows=len(train_df))

        validate_synthetic_output(
            train_df=train_df,
            synthetic_df=synthetic_df,
            expected_rows=len(train_df),
        )

        output_path = synthetic_output_path(synthesizer_name)
        synthetic_df.to_csv(output_path, index=False)

        metadata_path = synthesizer_metadata_path(synthesizer_name)
        metadata.save_to_json(metadata_path)

        model_path = synthesizer_model_path(synthesizer_name)
        synthesizer.save(filepath=model_path)

        print(f"[synthesize] Synthetic data saved to {output_path.resolve()}")
        print(f"[synthesize] Metadata saved to {metadata_path.resolve()}")
        print(f"[synthesize] Synthesizer saved to {model_path.resolve()}")

        logger.log(
            {
                "n_train_rows": len(train_df),
                "n_synthetic_rows": len(synthetic_df),
                "training_time_seconds": training_time_seconds,
                "synthetic_path": str(output_path),
                "metadata_path": str(metadata_path),
                "model_path": str(model_path),
            }
        )


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data with SDV Synthesizer."
    )
    parser.add_argument(
        "--dataset",
        default="adult_census",
        help="Dataset configuration to use.",
    )
    parser.add_argument("--synthesizer", choices=sorted(SYNTHESIZERS), required=True)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_synthesis(
        synthesizer_name=args.synthesizer,
        dataset_name=args.dataset,
        cuda=args.cuda,
        use_wandb=args.wandb,
    )


if __name__ == "__main__":
    main()
