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

import pandas as pd
from snsynth import Synthesizer

from src.dataset.feature_engineering import CATEGORICAL_COLS, NUMERICAL_COLS, TARGET_COL
from src.utility.constants import (
    DATA_DIR,
    DP_EPSILONS,
    DP_PREPROCESSOR_EPS_FRACTION,
    DP_SYNTHESIZERS,
    RANDOM_STATE,
)
from src.utility.logger import RunLogger
from src.utility.utils import set_random_seeds

# ── Column type hints ─────────────────────────────────────────────────────────
# SmartNoise uses these hints to select appropriate transformers per column.
# education-num is treated as ordinal since it encodes an ordered scale.
# All other numerical columns are continuous.
# TARGET_COL is included in categorical columns as SmartNoise synthesizes
# the full row including the target, unlike SDV which handles it via metadata.

ORDINAL_COLS = ["education-num"]
CONTINUOUS_COLS = [c for c in NUMERICAL_COLS if c not in ORDINAL_COLS]
CATEGORICAL_COLS_WITH_TARGET = CATEGORICAL_COLS + [TARGET_COL]


# ── Path helpers ──────────────────────────────────────────────────────────────


def _output_dir(synthesizer_name: str, epsilon: float):
    """
    Return the output directory for a given synthesizer and epsilon value.

    Uses 'eps_{epsilon}' as the subdirectory name to mirror the non-DP
    pipeline's 'default' subdirectory and keep each run isolated.

    Args:
        synthesizer_name: One of 'dpctgan', 'patectgan'.
        epsilon: Privacy budget value.

    Returns:
        Path to the output directory.
    """
    return DATA_DIR / "synthetic" / synthesizer_name / f"eps_{epsilon}"


def _run_name(synthesizer_name: str, epsilon: float) -> str:
    """Return a consistent run name for logging."""
    return f"synthesizer_{synthesizer_name}_eps_{epsilon}"


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

    config = {
        "synthesizer": synthesizer_name,
        "epsilon": epsilon,
        "preprocessor_eps": preprocessor_eps,
        "cuda": cuda,
    }

    with RunLogger(run_name=run_name, config=config, use_wandb=use_wandb) as logger:
        train_df = pd.read_csv(DATA_DIR / "processed" / "train.csv")
        n_samples = len(train_df)
        print(f"Loaded training data: {n_samples} rows")
        print(f"Synthesizer: {synthesizer_name} | epsilon: {epsilon} | preprocessor_eps: {preprocessor_eps}")

        set_random_seeds(RANDOM_STATE)
        synthesizer = build_dp_synthesizer(synthesizer_name, epsilon, cuda=cuda)

        print(f"Training {synthesizer_name} (epsilon={epsilon})...")
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
        print(f"Training complete in {training_time:.1f}s")

        print(f"Generating {n_samples} synthetic samples...")
        synthetic_df = synthesizer.sample(n_samples)

        out_dir = _output_dir(synthesizer_name, epsilon)
        out_dir.mkdir(parents=True, exist_ok=True)
        synthetic_path = out_dir / "synthetic_train.csv"
        synthetic_df.to_csv(synthetic_path, index=False)
        print(f"Synthetic data saved to {synthetic_path.resolve()}")

        logger.log(
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
        help="Log results to Weights & Biases. Requires WANDB_ENTITY to be set.",
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
