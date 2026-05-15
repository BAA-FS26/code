"""
Privacy evaluation for synthetic tabular data.

This module evaluates synthetic data against real training and holdout data
using privacy attack metrics and distance-based privacy metrics, then logs the
resulting privacy scores.

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
from typing import Any

import pandas as pd
from anonymeter.evaluators import (
    InferenceEvaluator,
    LinkabilityEvaluator,
    SinglingOutEvaluator,
)
from sdmetrics.single_table import DCRBaselineProtection, DCROverfittingProtection

from src.core.data_source import build_data_source_key
from src.evaluation.evaluation_data import load_privacy_datasets
from src.utility.constants import (
    DP_EPSILONS,
    DP_SYNTHESIZERS,
    RANDOM_STATE,
    SYNTHESIZER_MODELS_DIR,
    SYNTHESIZERS,
)
from src.utility.logger import RunLogger
from src.utility.utils import (
    build_adult_sdmetrics_metadata,
    load_metadata,
    set_random_seeds,
)

SCRIPT_NAME = "evaluate_privacy.py"
N_ATTACKS = 2000

# Based on EDA findings — inter-feature associations and data protection relevance.
SENSITIVE_COLS = ["income", "occupation", "sex", "relationship"]


def _risk_metrics(prefix: str, risk: Any) -> dict[str, float]:
    """
    Convert an Anonymeter risk object into stable logger metric keys.

    The key names are intentionally unchanged for result and dashboard
    compatibility.
    """
    return {
        f"{prefix}": float(risk.value),
        f"{prefix}_ci_lower": float(risk.ci[0]),
        f"{prefix}_ci_upper": float(risk.ci[1]),
    }


def _print_risk_summary(label: str, risk: Any) -> None:
    """Print an Anonymeter risk estimate with its confidence interval."""
    print(
        f"[evaluate_privacy] {label}: {float(risk.value):.4f} "
        f"[{float(risk.ci[0]):.4f}, {float(risk.ci[1]):.4f}]"
    )


def _require_risk(risk: Any, evaluator_name: str) -> Any:
    """Validate that an Anonymeter evaluator returned a risk result."""
    if risk is None:
        raise ValueError(f"{evaluator_name} returned no risk result.")

    return risk


def _build_run_parameters(
    synthesizer_name: str,
    epsilon: float | None,
    data_source: str,
    use_wandb: bool,
) -> dict[str, Any]:
    """
    Build logger metadata for privacy evaluation.

    Field names are intentionally stable for result compatibility.
    """
    return {
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
        "metadata_key": synthesizer_name,
        "n_attacks": N_ATTACKS,
        "sensitive_cols": SENSITIVE_COLS,
    }


def run_singling_out(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    logger: RunLogger,
) -> None:
    """
    Run Anonymeter SinglingOutEvaluator in univariate and multivariate mode.

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

        risk = _require_risk(
            evaluator.risk(),
            evaluator_name=f"SinglingOutEvaluator for mode '{mode}'",
        )

        _print_risk_summary(
            label=f"Singling Out Risk ({mode})",
            risk=risk,
        )

        logger.log(_risk_metrics(f"singling_out_risk_{mode}", risk))


def run_linkability(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    logger: RunLogger,
) -> None:
    """
    Run Anonymeter LinkabilityEvaluator and log linkability risk.

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

    risk = _require_risk(
        evaluator.risk(),
        evaluator_name="LinkabilityEvaluator",
    )

    _print_risk_summary("Linkability Risk", risk)

    logger.log(_risk_metrics("linkability_risk", risk))


def run_inference(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    logger: RunLogger,
) -> None:
    """
    Run Anonymeter InferenceEvaluator for each configured sensitive column.

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
            aux_cols=[col for col in train_df.columns if col != secret],
        )

        evaluator.evaluate()

        risk = _require_risk(
            evaluator.risk(),
            evaluator_name=f"InferenceEvaluator for secret '{secret}'",
        )

        _print_risk_summary(
            label=f"Inference Risk ({secret})",
            risk=risk,
        )

        logger.log(_risk_metrics(f"inference_risk_{secret}", risk))


def run_dcr_metrics(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    metadata: dict,
    logger: RunLogger,
) -> None:
    """
    Run SDMetrics DCR-based privacy metrics and log results.

    A score near 1 indicates better protection.
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
    print(f"[evaluate_privacy] DCR Overfitting Protection: {dcr_overfitting:.4f}")

    logger.log(
        {
            "dcr_baseline_protection": dcr_baseline,
            "dcr_overfitting_protection": dcr_overfitting,
        }
    )


def evaluate_privacy(
    synthesizer_name: str,
    epsilon: float | None = None,
    use_wandb: bool = False,
) -> None:
    """
    Run full privacy evaluation for a synthesizer.
    """
    data_source = build_data_source_key(synthesizer_name, epsilon)
    run_name = f"eval_privacy_{data_source.replace('/', '_')}"

    parameters = _build_run_parameters(
        synthesizer_name=synthesizer_name,
        epsilon=epsilon,
        data_source=data_source,
        use_wandb=use_wandb,
    )

    with RunLogger(
        run_name=run_name,
        script_name=SCRIPT_NAME,
        parameters=parameters,
        use_wandb=use_wandb,
        category="privacy",
    ) as logger:
        train_df, holdout_df, synthetic_df, paths = load_privacy_datasets(data_source)

        metadata = load_metadata(
            SYNTHESIZER_MODELS_DIR,
            synthesizer_name,
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
