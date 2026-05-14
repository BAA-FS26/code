"""
synthesize_dp.py

Differentially private synthesis pipeline for the Adult Census Income dataset.
Trains a DP synthesizer on the real training data and generates a synthetic
dataset of equal size for a given epsilon value.

Supported synthesizers:
  - dpctgan:   DP-SGD-based conditional tabular GAN (Xu et al., 2019)
  - patectgan: PATE-based conditional tabular GAN (Jordon et al., 2019)

Usage:
    # Without W&B (default)
    python -m src.modeling.synthesize_dp --synthesizer dpctgan --epsilon 1.0
    python -m src.modeling.synthesize_dp --synthesizer patectgan --epsilon 1.0

    # All epsilon values
    python -m src.modeling.synthesize_dp --synthesizer dpctgan --epsilon 0.1
    python -m src.modeling.synthesize_dp --synthesizer dpctgan --epsilon 1.0
    python -m src.modeling.synthesize_dp --synthesizer dpctgan --epsilon 5.0
    python -m src.modeling.synthesize_dp --synthesizer dpctgan --epsilon 10.0

    # With GPU acceleration
    python -m src.modeling.synthesize_dp --synthesizer dpctgan --epsilon 1.0 --cuda

    # With W&B logging
    python -m src.modeling.synthesize_dp --synthesizer dpctgan --epsilon 1.0 --wandb
"""

import argparse
import time

import pandas as pd
import torch
from snsynth import Synthesizer

from src.core.data_source import build_data_source_key
from src.core.io import load_csv, validate_matching_columns
from src.core.paths import (
    processed_split_path,
    synthetic_output_dir,
    synthetic_output_path,
)

from src.dataset.adult_census import CATEGORICAL_COLS, NUMERICAL_COLS, TARGET_COL
from src.utility.constants import (
    DP_EPSILONS,
    DP_PREPROCESSOR_EPS_FRACTION,
    DP_SYNTHESIZERS,
    RANDOM_STATE,
    TRAIN_FILENAME,
)
from src.utility.logger import RunLogger
from src.utility.utils import set_random_seeds

SCRIPT_NAME = "synthesize_dp.py"

# SmartNoise uses these hints to select appropriate transformers per column.
# education-num is treated as ordinal since it encodes an ordered scale.
# All other numerical columns are continuous.
# TARGET_COL is included in categorical columns as SmartNoise synthesizes
# the full row including the target, unlike SDV which handles it via metadata.
ORDINAL_COLS = ["education-num"]
CONTINUOUS_COLS = [c for c in NUMERICAL_COLS if c not in ORDINAL_COLS]
CATEGORICAL_COLS_WITH_TARGET = CATEGORICAL_COLS + [TARGET_COL]


def _run_name(synthesizer_name: str, epsilon: float) -> str:
    """Return a consistent run name for logging."""
    return f"synthesizer_{synthesizer_name}_eps_{epsilon}"


def load_training_data() -> pd.DataFrame:
    """Load the real training split from disk."""
    train_path = processed_split_path(TRAIN_FILENAME)
    train_df = load_csv(train_path, "Training split")

    print(
        f"[synthesize_dp] Loaded training data from {train_path.resolve()} "
        f"({len(train_df)} rows)"
    )

    return train_df


def _validate_epsilon_settings(epsilon: float, preprocessor_eps: float) -> None:
    """
    Validate epsilon-related settings for the DP synthesis run.

    Raises:
        ValueError: If epsilon settings are invalid.
    """
    if epsilon <= 0:
        raise ValueError(f"Epsilon must be positive, got {epsilon}.")

    if preprocessor_eps <= 0:
        raise ValueError(
            f"Derived preprocessor_eps must be positive, got {preprocessor_eps}."
        )

    if preprocessor_eps > epsilon:
        raise ValueError(
            "Derived preprocessor_eps must be less than or equal to epsilon. "
            f"Got preprocessor_eps={preprocessor_eps}, epsilon={epsilon}."
        )


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


def build_dp_synthesizer(
    synthesizer_name: str,
    epsilon: float,
    cuda: bool = False,
):
    """
    Build a SmartNoise DP synthesizer instance.

    Args:
        synthesizer_name: One of 'dpctgan', 'patectgan'.
        epsilon: Privacy budget for the synthesizer.
        cuda: Whether to use GPU acceleration. Defaults to False.

    Returns:
        Unfitted SmartNoise Synthesizer instance.

    Raises:
        ValueError: If synthesizer_name is not recognised.
    """
    if synthesizer_name not in DP_SYNTHESIZERS:
        raise ValueError(
            f"Unknown DP synthesizer: '{synthesizer_name}'. "
            f"Choose from: {DP_SYNTHESIZERS}"
        )

    return Synthesizer.create(
        synthesizer_name,
        epsilon=epsilon,
        cuda=cuda,
        verbose=False,
    )


def train_and_generate(
    synthesizer_name: str,
    epsilon: float,
    cuda: bool = False,
    use_wandb: bool = False,
) -> None:
    """
    Train a DP synthesizer on real training data and generate synthetic data.

    Loads the real training split, fits the synthesizer with the given
    epsilon, generates a synthetic dataset of equal size, and saves it
    to disk. Training time and run metadata are always saved locally.
    W&B logging is optional.

    The preprocessor epsilon is set to DP_PREPROCESSOR_EPS_FRACTION of
    the total epsilon budget. This reserves a small portion of the privacy
    budget for inferring continuous column bounds, following standard
    practice. The remaining budget is used for synthesis.

    Missing values are handled via nullable=True, consistent with the
    cleaned Adult Census dataset which encodes missing values as pandas NA.

    The synthesizer is trained on the training split only. Validation and
    test splits are never used during synthesis.

    Args:
        synthesizer_name: One of 'dpctgan', 'patectgan'.
        epsilon: Privacy budget for the synthesizer.
        cuda: Whether to use GPU acceleration. Defaults to False.
        use_wandb: Whether to log results to W&B. Defaults to False.
    """
    run_name = _run_name(synthesizer_name, epsilon)
    preprocessor_eps = round(epsilon * DP_PREPROCESSOR_EPS_FRACTION, 6)
    _validate_epsilon_settings(epsilon, preprocessor_eps)

    gpu_available = torch.cuda.is_available()
    gpu_in_use = cuda and gpu_available

    data_source = build_data_source_key(
        synthesizer_name=synthesizer_name,
        epsilon=epsilon,
    )
    parameters = {
        "pipeline_stage": "synthesis",
        "evaluation": None,
        "mode": f"eps_{epsilon}",
        "data_source": data_source,
        "synthesizer": synthesizer_name,
        "epsilon": epsilon,
        "classifier": None,
        "model_type": None,
        "params": {},
        "random_state": RANDOM_STATE,
        "use_wandb": use_wandb,
        "cuda_requested": cuda,
        "gpu_available": gpu_available,
        "gpu_in_use": gpu_in_use,
        "preprocessor_eps": preprocessor_eps,
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

        print(
            f"[synthesize_dp] Synthesizer: {synthesizer_name} | "
            f"epsilon: {epsilon} | preprocessor_eps: {preprocessor_eps}"
        )
        print(f"[synthesize_dp] GPU available: {gpu_available}")
        print(f"[synthesize_dp] GPU requested: {cuda}")
        print(f"[synthesize_dp] GPU in use: {gpu_in_use}")

        if cuda and not gpu_available:
            print(
                "[synthesize_dp] Warning: --cuda was set but no GPU is available. "
                "Falling back to CPU."
            )

        set_random_seeds(RANDOM_STATE)
        synthesizer = build_dp_synthesizer(synthesizer_name, epsilon, cuda=cuda)

        print(f"[synthesize_dp] Training {synthesizer_name} (epsilon={epsilon})...")
        start_time = time.time()
        synthesizer.fit(
            train_df,
            categorical_columns=CATEGORICAL_COLS_WITH_TARGET,
            ordinal_columns=ORDINAL_COLS,
            continuous_columns=CONTINUOUS_COLS,
            preprocessor_eps=preprocessor_eps,
            nullable=True,
        )
        training_time = time.time() - start_time
        print(f"[synthesize_dp] Training complete in {training_time:.1f}s")

        print(f"[synthesize_dp] Generating {n_samples} synthetic samples...")
        synthetic_df = synthesizer.sample(n_samples)
        validate_synthetic_output(train_df, synthetic_df, expected_rows=n_samples)

        out_dir = synthetic_output_dir(
            synthesizer_name,
            mode=f"eps_{epsilon}",
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        synthetic_path = synthetic_output_path(
            synthesizer_name,
            mode=f"eps_{epsilon}",
        )
        synthetic_df.to_csv(synthetic_path, index=False)
        print(f"[synthesize_dp] Synthetic data saved to {synthetic_path.resolve()}")

        logger.log(
            {
                "training_time_seconds": training_time,
                "n_samples_train": n_samples,
                "n_samples_synthetic": len(synthetic_df),
                "n_features": len(train_df.columns),
                "synthetic_data_path": synthetic_path,
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a DP synthesizer and generate synthetic data."
    )
    parser.add_argument(
        "--synthesizer",
        choices=DP_SYNTHESIZERS,
        required=True,
        help="DP synthesizer to train.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        choices=DP_EPSILONS,
        required=True,
        help=f"Privacy budget (epsilon). Choose from: {DP_EPSILONS}",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        default=False,
        help="Use GPU acceleration. Defaults to False.",
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
        epsilon=args.epsilon,
        cuda=args.cuda,
        use_wandb=args.wandb,
    )


if __name__ == "__main__":
    main()
