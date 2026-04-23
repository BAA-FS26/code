"""
synthesize_dp.py

Differentially private synthesis pipeline for the Adult Census Income dataset.
Trains a DP synthesizer on the real training data and generates a synthetic
dataset of equal size for a given epsilon value.

Supported synthesizers:
  - dpctgan:   DP-SGD-based conditional tabular GAN (Xu et al., 2019)
  - patectgan: PATE-based conditional tabular GAN (Jordon et al., 2019)

Both synthesizers are provided by the SmartNoise Synth library and use the
same fit/sample API. DPCTGAN applies differential privacy via DP-SGD to the
discriminator gradients. PATECTGAN uses the Private Aggregation of Teacher
Ensembles (PATE) framework, which provides privacy guarantees through an
ensemble of teacher discriminators and a student generator.

Epsilon values are not directly comparable across the two mechanisms due to
their different privacy accounting approaches. Results should be interpreted
within each synthesizer separately when comparing across epsilon values.

Output is saved to:
    data/synthetic/{synthesizer}/eps_{epsilon}/synthetic_train.csv

This path structure ensures each (synthesizer, epsilon) combination has its
own isolated output directory, consistent with the non-DP pipeline layout
under data/synthetic/{synthesizer}/default/.

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
from pathlib import Path

import pandas as pd
import torch
from snsynth import Synthesizer

from src.dataset.adult_census import CATEGORICAL_COLS, NUMERICAL_COLS, TARGET_COL
from src.utility.constants import (
    DP_EPSILONS,
    DP_PREPROCESSOR_EPS_FRACTION,
    DP_SYNTHESIZERS,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    SYNTHETIC_DATA_DIR,
    SYNTHETIC_TRAIN_FILENAME,
    TRAIN_FILENAME,
)
from src.utility.logger import RunLogger
from src.utility.utils import set_random_seeds

# ── Constants ────────────────────────────────────────────────────────────────

SCRIPT_NAME = "synthesize_dp.py"

# SmartNoise uses these hints to select appropriate transformers per column.
# education-num is treated as ordinal since it encodes an ordered scale.
# All other numerical columns are continuous.
# TARGET_COL is included in categorical columns as SmartNoise synthesizes
# the full row including the target, unlike SDV which handles it via metadata.
ORDINAL_COLS = ["education-num"]
CONTINUOUS_COLS = [c for c in NUMERICAL_COLS if c not in ORDINAL_COLS]
CATEGORICAL_COLS_WITH_TARGET = CATEGORICAL_COLS + [TARGET_COL]


# ── Path helpers ──────────────────────────────────────────────────────────────


def _training_data_path() -> Path:
    """Return the canonical path to the real training split."""
    return PROCESSED_DATA_DIR / TRAIN_FILENAME


def _output_dir(synthesizer_name: str, epsilon: float) -> Path:
    """
    Return the output directory for a given synthesizer and epsilon value.

    Uses 'eps_{epsilon}' as the canonical subdirectory name to mirror the
    non-DP pipeline's 'default' subdirectory and keep each run isolated.

    Args:
        synthesizer_name: One of 'dpctgan', 'patectgan'.
        epsilon: Privacy budget value.

    Returns:
        Path to the output directory.
    """
    return SYNTHETIC_DATA_DIR / synthesizer_name / f"eps_{epsilon}"


def _synthetic_output_path(synthesizer_name: str, epsilon: float) -> Path:
    """
    Return the canonical synthetic output CSV path for a DP synthesizer run.
    """
    return _output_dir(synthesizer_name, epsilon) / SYNTHETIC_TRAIN_FILENAME


def _run_name(synthesizer_name: str, epsilon: float) -> str:
    """Return a consistent run name for logging."""
    return f"synthesizer_{synthesizer_name}_eps_{epsilon}"


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


# ── Synthesizer ───────────────────────────────────────────────────────────────


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


# ── Train and generate ────────────────────────────────────────────────────────


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

    parameters = {
        "synthesizer": synthesizer_name,
        "epsilon": epsilon,
        "preprocessor_eps": preprocessor_eps,
        "cuda_requested": cuda,
        "gpu_available": gpu_available,
        "gpu_in_use": gpu_in_use,
        "random_state": RANDOM_STATE,
        "use_wandb": use_wandb,
        "reproducibility_note": (
            "Random seeds and deterministic PyTorch settings are enabled where "
            "possible. Some GPU operations and third-party model internals may "
            "still remain nondeterministic."
        ),
    }

    with RunLogger(
        run_name=run_name,
        script_name=SCRIPT_NAME,
        parameters=parameters,
        use_wandb=use_wandb,
    ) as logger:
        train_df = _load_training_data()
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
        _validate_synthetic_output(train_df, synthetic_df, expected_rows=n_samples)

        out_dir = _output_dir(synthesizer_name, epsilon)
        out_dir.mkdir(parents=True, exist_ok=True)

        synthetic_path = _synthetic_output_path(synthesizer_name, epsilon)
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


# ── Entry point ───────────────────────────────────────────────────────────────


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
