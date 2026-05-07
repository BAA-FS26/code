"""
evaluate_privacy.py

Privacy evaluation for synthetic data using Anonymeter and SDMetrics.

Runs four evaluations for each synthesizer:
  - Singling Out: can an attacker uniquely identify a real person using
    the synthetic data?
  - Linkability: can an attacker link two records to the same person?
  - Inference: can an attacker infer sensitive attributes (income, race,
    sex, relationship) from known attributes?
  - DCR Metrics: distance-based privacy metrics measuring whether synthetic
    data is memorizing training records (SDMetrics)

Results are always saved locally as JSON. W&B logging is optional.

Usage:
    # Without W&B (default)
    python -m src.evaluation.evaluate_privacy --synthesizer gaussian_copula
    python -m src.evaluation.evaluate_privacy --synthesizer ctgan
    python -m src.evaluation.evaluate_privacy --synthesizer tvae

    # With W&B logging
    python -m src.evaluation.evaluate_privacy --synthesizer ctgan --wandb

    # DP synthesizers
    python -m src.evaluation.evaluate_privacy --synthesizer dpctgan --epsilon 1.0
    python -m src.evaluation.evaluate_privacy --synthesizer patectgan --epsilon 1.0
"""

import argparse
from pathlib import Path

import pandas as pd
from anonymeter.evaluators import (
    InferenceEvaluator,
    LinkabilityEvaluator,
    SinglingOutEvaluator,
)
from sdmetrics.single_table import DCRBaselineProtection, DCROverfittingProtection

from src.utility.constants import (
    DP_EPSILONS,
    DP_SYNTHESIZERS,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    SYNTHESIZER_MODELS_DIR,
    SYNTHESIZERS,
    SYNTHETIC_DATA_DIR,
    SYNTHETIC_TRAIN_FILENAME,
    TEST_FILENAME,
    TRAIN_FILENAME,
    VALIDATION_FILENAME,
)
from src.utility.logger import RunLogger
from src.utility.utils import (
    build_adult_sdmetrics_metadata,
    load_metadata,
    set_random_seeds,
)

# ── Constants ────────────────────────────────────────────────────────────────

SCRIPT_NAME = "evaluate_privacy.py"
MODELS_DIR = SYNTHESIZER_MODELS_DIR
N_ATTACKS = 2000

# Based on EDA findings — inter-feature associations and data protection relevance
SENSITIVE_COLS = ["income", "occupation", "sex", "relationship"]


# ── Path and argument helpers ────────────────────────────────────────────────


def _processed_split_path(filename: str) -> Path:
    """Return the canonical path to a processed split file."""
    return PROCESSED_DATA_DIR / filename


def _data_source_key(synthesizer_name: str, epsilon: float | None) -> str:
    """
    Build the canonical synthetic data source key used in directory paths.

    Non-DP example:
        gaussian_copula

    DP example:
        dpctgan/eps_1.0
    """
    if synthesizer_name in DP_SYNTHESIZERS:
        if epsilon is None:
            raise ValueError(
                f"Epsilon is required for DP synthesizer '{synthesizer_name}'."
            )
        return f"{synthesizer_name}/eps_{epsilon}"
    return synthesizer_name


def _synthetic_train_path(synthesizer_name: str, epsilon: float | None) -> Path:
    """Return the canonical path to the synthetic training data."""
    data_source = _data_source_key(synthesizer_name, epsilon)
    if synthesizer_name in DP_SYNTHESIZERS:
        return SYNTHETIC_DATA_DIR / data_source / SYNTHETIC_TRAIN_FILENAME
    return SYNTHETIC_DATA_DIR / synthesizer_name / "default" / SYNTHETIC_TRAIN_FILENAME


def _metadata_key(synthesizer_name: str) -> str:
    """Return the synthesizer key used for metadata lookup."""
    return synthesizer_name


def _load_csv(path: Path, description: str) -> pd.DataFrame:
    """
    Load a CSV file with a clear pipeline-friendly error if it is missing.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found at {path}. "
            "Run the required earlier pipeline step first."
        )
    return pd.read_csv(path)


def _validate_matching_schema(
    reference_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    candidate_name: str,
) -> None:
    """
    Validate that a candidate dataframe has identical columns to a reference.

    Raises:
        ValueError: If columns do not match exactly in order.
    """
    reference_columns = list(reference_df.columns)
    candidate_columns = list(candidate_df.columns)
    if candidate_columns != reference_columns:
        raise ValueError(
            f"{candidate_name} columns do not match reference training data columns.\n"
            f"Expected: {reference_columns}\n"
            f"Actual:   {candidate_columns}"
        )


# ── Data loading ──────────────────────────────────────────────────────────────


def load_data(
    synthesizer_name: str,
    epsilon: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    """
    Load real training data, combined holdout data, and synthetic data.

    The holdout dataset combines validation and test splits. Neither was used
    to train the synthesizer, making both valid as Anonymeter control
    data. Combining them gives a larger holdout for more statistically robust
    privacy evaluation.

    Note: the combined holdout is only used for privacy evaluation.
    The test split remains reserved for TSTR utility evaluation in classify.py.

    Args:
        synthesizer_name: One of 'gaussian_copula', 'ctgan', 'tvae',
                          'dpctgan', or 'patectgan'.
        epsilon: Privacy budget for DP synthesizers. Must be None for
                 non-DP synthesizers.

    Returns:
        Tuple of (train_df, holdout_df, synthetic_df, paths).
    """
    train_path = _processed_split_path(TRAIN_FILENAME)
    validation_path = _processed_split_path(VALIDATION_FILENAME)
    test_path = _processed_split_path(TEST_FILENAME)
    synthetic_path = _synthetic_train_path(synthesizer_name, epsilon)

    train_df = _load_csv(train_path, "Training split")
    val_df = _load_csv(validation_path, "Validation split")
    test_df = _load_csv(test_path, "Test split")
    synthetic_df = _load_csv(synthetic_path, "Synthetic training data")

    _validate_matching_schema(train_df, val_df, "Validation split")
    _validate_matching_schema(train_df, test_df, "Test split")
    _validate_matching_schema(train_df, synthetic_df, "Synthetic training data")

    holdout_df = pd.concat([val_df, test_df], ignore_index=True)

    return (
        train_df,
        holdout_df,
        synthetic_df,
        {
            "train_path": train_path,
            "validation_path": validation_path,
            "test_path": test_path,
            "synthetic_path": synthetic_path,
        },
    )


# ── Anonymeter evaluations ────────────────────────────────────────────────────


def run_singling_out(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    logger: RunLogger,
) -> None:
    """
    Run Anonymeter SinglingOutEvaluator in both univariate and multivariate
    mode and log results via the run logger.

    Univariate mode tests single-attribute identification. Multivariate
    mode tests combined-attribute identification — a stronger and more
    realistic attack. Both are reported for completeness.

    A risk value near 0 indicates good privacy protection.
    """
    print(f"[evaluate_privacy] Running SinglingOutEvaluator (n_attacks={N_ATTACKS})...")

    for mode in ["univariate", "multivariate"]:
        evaluator = SinglingOutEvaluator(
            ori=train_df,
            syn=synthetic_df,
            control=holdout_df,
            n_attacks=N_ATTACKS,
        )
        evaluator.evaluate(mode=mode)
        risk = evaluator.risk()

        if risk is None:
            raise ValueError(
                f"SinglingOutEvaluator returned no risk result for mode '{mode}'."
            )

        print(
            f"[evaluate_privacy] Singling Out Risk ({mode}): {float(risk.value):.4f} "
            f"[{float(risk.ci[0]):.4f}, {float(risk.ci[1]):.4f}]"
        )

        logger.log(
            {
                f"singling_out_risk_{mode}": float(risk.value),
                f"singling_out_risk_{mode}_ci_lower": float(risk.ci[0]),
                f"singling_out_risk_{mode}_ci_upper": float(risk.ci[1]),
            }
        )


def run_linkability(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    logger: RunLogger,
) -> None:
    """
    Run Anonymeter LinkabilityEvaluator and log results via the run logger.

    Measures whether an attacker can link two records from different
    datasets to the same person using the synthetic data. The columns
    are split evenly into two auxiliary sets.

    A risk value near 0 indicates good privacy protection.
    """
    print(f"[evaluate_privacy] Running LinkabilityEvaluator (n_attacks={N_ATTACKS})...")

    cols = train_df.columns.tolist()
    mid = len(cols) // 2
    aux_cols = (cols[:mid], cols[mid:])

    evaluator = LinkabilityEvaluator(
        ori=train_df,
        syn=synthetic_df,
        control=holdout_df,
        n_attacks=N_ATTACKS,
        aux_cols=aux_cols,
    )
    evaluator.evaluate()
    risk = evaluator.risk()

    if risk is None:
        raise ValueError("LinkabilityEvaluator returned no risk result.")

    print(
        f"[evaluate_privacy] Linkability Risk: {float(risk.value):.4f} "
        f"[{float(risk.ci[0]):.4f}, {float(risk.ci[1]):.4f}]"
    )

    logger.log(
        {
            "linkability_risk": float(risk.value),
            "linkability_risk_ci_lower": float(risk.ci[0]),
            "linkability_risk_ci_upper": float(risk.ci[1]),
        }
    )


def run_inference(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    logger: RunLogger,
) -> None:
    """
    Run Anonymeter InferenceEvaluator for each sensitive column and log
    results via the run logger.

    Models an attacker who possesses all available attributes except the
    target attribute as auxiliary information, following the methodology
    of Giomi et al. (2023). Sensitive target attributes are defined in
    SENSITIVE_COLS and selected based on their inter-feature associations
    and relevance under applicable data protection regulations.

    A risk value near 0 indicates good privacy protection.
    """
    print(f"[evaluate_privacy] Running InferenceEvaluator (n_attacks={N_ATTACKS})...")

    for secret in SENSITIVE_COLS:
        evaluator = InferenceEvaluator(
            ori=train_df,
            syn=synthetic_df,
            control=holdout_df,
            n_attacks=N_ATTACKS,
            secret=secret,
            aux_cols=[c for c in train_df.columns if c != secret],
        )
        evaluator.evaluate()
        risk = evaluator.risk()

        if risk is None:
            raise ValueError(
                f"InferenceEvaluator returned no risk result for secret '{secret}'."
            )

        print(
            f"[evaluate_privacy] Inference Risk ({secret}): {float(risk.value):.4f} "
            f"[{float(risk.ci[0]):.4f}, {float(risk.ci[1]):.4f}]"
        )

        logger.log(
            {
                f"inference_risk_{secret}": float(risk.value),
                f"inference_risk_{secret}_ci_lower": float(risk.ci[0]),
                f"inference_risk_{secret}_ci_upper": float(risk.ci[1]),
            }
        )


# ── DCR metrics ───────────────────────────────────────────────────────────────


def run_dcr_metrics(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    metadata: dict,
    logger: RunLogger,
) -> None:
    """
    Run SDMetrics DCR-based privacy metrics and log results via the run logger.

    Computes two metrics:
    - DCRBaselineProtection: compares distances from synthetic to real
      data against a random baseline. A score near 1 indicates good
      protection.
    - DCROverfittingProtection: checks whether synthetic data is too
      close to training data compared to holdout data. A score near 1
      indicates no memorization.

    num_rows_subsample is used to keep computation time manageable for
    datasets of ~30K rows.
    """
    print("[evaluate_privacy] Running DCR metrics...")

    dcr_baseline_raw = DCRBaselineProtection.compute(
        real_data=train_df,
        synthetic_data=synthetic_df,
        metadata=metadata,
        num_rows_subsample=5000,
    )
    dcr_overfitting_raw = DCROverfittingProtection.compute(
        real_training_data=train_df,
        synthetic_data=synthetic_df,
        real_validation_data=holdout_df,
        metadata=metadata,
        num_rows_subsample=5000,
    )

    if dcr_baseline_raw is None:
        raise ValueError("DCRBaselineProtection.compute() returned no score.")

    if dcr_overfitting_raw is None:
        raise ValueError("DCROverfittingProtection.compute() returned no score.")

    dcr_baseline = float(dcr_baseline_raw)
    dcr_overfitting = float(dcr_overfitting_raw)

    print(f"[evaluate_privacy] DCR Baseline Protection: {dcr_baseline:.4f}")
    print("[evaluate_privacy] DCR Overfitting Protection: " f"{dcr_overfitting:.4f}")

    logger.log(
        {
            "dcr_baseline_protection": dcr_baseline,
            "dcr_overfitting_protection": dcr_overfitting,
        }
    )


# ── Main evaluation ───────────────────────────────────────────────────────────


def evaluate_privacy(
    synthesizer_name: str,
    epsilon: float | None = None,
    use_wandb: bool = False,
) -> None:
    """
    Run full privacy evaluation for a synthesizer.

    Runs Singling Out, Linkability, Inference, and DCR evaluations
    against the synthetic data generated by the given synthesizer.
    Results are always saved locally. W&B logging is optional.

    Args:
        synthesizer_name: One of 'gaussian_copula', 'ctgan', 'tvae',
                          'dpctgan', or 'patectgan'.
        epsilon: Privacy budget for DP synthesizers. Must be None for
                 non-DP synthesizers.
        use_wandb: Whether to log results to W&B. Defaults to False.
    """
    data_source = _data_source_key(synthesizer_name, epsilon)
    run_name = f"eval_privacy_{data_source.replace('/', '_')}"
    metadata_key = _metadata_key(synthesizer_name)

    parameters = {
        "pipeline_stage": "evaluation",
        "evaluation": "privacy",
        "mode": "default" if epsilon is None else f"eps_{epsilon}",
        "data_source": data_source,
        "synthesizer": synthesizer_name,
        "epsilon": epsilon,
        "classifier": None,
        "model_type": None,
        "params": {},
        "random_state": RANDOM_STATE,
        "use_wandb": use_wandb,
        "metadata_key": metadata_key,
        "n_attacks": N_ATTACKS,
        "sensitive_cols": SENSITIVE_COLS,
    }

    with RunLogger(
        run_name=run_name,
        script_name=SCRIPT_NAME,
        parameters=parameters,
        use_wandb=use_wandb,
        category="privacy",
    ) as logger:
        train_df, holdout_df, synthetic_df, paths = load_data(
            synthesizer_name=synthesizer_name,
            epsilon=epsilon,
        )
        metadata = load_metadata(
            MODELS_DIR,
            metadata_key,
            fallback=build_adult_sdmetrics_metadata(),
        )

        print(f"[evaluate_privacy] Real training data: {len(train_df)} rows")
        print(f"[evaluate_privacy] Holdout data: {len(holdout_df)} rows")
        print(f"[evaluate_privacy] Synthetic data: {len(synthetic_df)} rows")

        logger.log(
            {
                "real_train_data_path": paths["train_path"],
                "real_validation_data_path": paths["validation_path"],
                "real_test_data_path": paths["test_path"],
                "synthetic_data_path": paths["synthetic_path"],
                "n_rows_real_train": len(train_df),
                "n_rows_holdout": len(holdout_df),
                "n_rows_synthetic": len(synthetic_df),
            }
        )

        set_random_seeds(RANDOM_STATE)
        run_singling_out(train_df, holdout_df, synthetic_df, logger)
        run_linkability(train_df, holdout_df, synthetic_df, logger)
        run_inference(train_df, holdout_df, synthetic_df, logger)
        run_dcr_metrics(train_df, holdout_df, synthetic_df, metadata, logger)

        print("[evaluate_privacy] Privacy evaluation complete.")


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate privacy of synthetic data using Anonymeter and SDMetrics."
    )
    parser.add_argument(
        "--synthesizer",
        choices=SYNTHESIZERS | DP_SYNTHESIZERS,
        required=True,
        help="Synthesizer to evaluate.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        choices=DP_EPSILONS,
        default=None,
        help="Privacy budget for DP synthesizers. Required for dpctgan/patectgan.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="Log results to Weights & Biases. Local JSON logging remains primary.",
    )

    args = parser.parse_args()

    if args.synthesizer in DP_SYNTHESIZERS and args.epsilon is None:
        parser.error("--epsilon is required for DP synthesizers.")

    if args.synthesizer not in DP_SYNTHESIZERS and args.epsilon is not None:
        parser.error("--epsilon should only be used with DP synthesizers.")

    evaluate_privacy(
        synthesizer_name=args.synthesizer,
        epsilon=args.epsilon,
        use_wandb=args.wandb,
    )


if __name__ == "__main__":
    main()
