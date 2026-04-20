"""
evaluate_privacy.py

Privacy evaluation for synthetic data using Anonymeter and SDMetrics.

Runs four evaluations for each synthesizer:
  - Singling Out: can an attacker uniquely identify a real person using
    the synthetic data?
  - Linkability: can an attacker link two records to the same person?
  - Inference: can an attacker infer sensitive attributes (income, race,
    sex) from known attributes?
  - DCR Metrics: distance-based privacy metrics measuring whether synthetic
    data is memorizing training records (SDMetrics)

All results are logged to W&B.

The val set is used as the Anonymeter control dataset — it consists of
real records that were not used to train the synthesizer, making it
suitable for separating general population patterns from training-specific
privacy leakage.

Usage:
    python evaluate_privacy.py --synthesizer gaussian_copula
    python evaluate_privacy.py --synthesizer ctgan
    python evaluate_privacy.py --synthesizer tvae
"""

import argparse

import pandas as pd
import wandb
from anonymeter.evaluators import (
    InferenceEvaluator,
    LinkabilityEvaluator,
    SinglingOutEvaluator,
)
from sdmetrics.single_table import DCRBaselineProtection, DCROverfittingProtection

from src.utility.constants import (
    DATA_DIR,
    SYNTHESIZER_MODELS_DIR,
    SYNTHESIZERS,
    WANDB_ENTITY,
    WANDB_PROJECT,
)
from src.utility.utils import load_metadata

# ── Constants ────────────────────────────────────────────────────────────────

MODELS_DIR = SYNTHESIZER_MODELS_DIR
N_ATTACKS = 2000

# based on EDA findings
SENSITIVE_COLS = ["income", "occupation", "sex"]


# ── Data loading ──────────────────────────────────────────────────────────────


def load_data(
    synthesizer_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load real training data, combined holdout data and synthetic data.

    The holdout dataset combines val and test splits. Neither was used
    to train the synthesizer, making both valid as Anonymeter control
    data. Combining them gives a larger holdout (~19,500 rows) for more
    statistically robust privacy evaluation.

    Note: the combined holdout is only used for privacy evaluation.
    The test split remains reserved for TSTR utility evaluation in
    classify.py.

    Args:
        synthesizer_name: One of 'gaussian_copula', 'ctgan', 'tvae'.

    Returns:
        Tuple of (train_df, holdout_df, synthetic_df).
    """
    train_df = pd.read_csv(DATA_DIR / "processed" / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "processed" / "validation.csv")
    test_df = pd.read_csv(DATA_DIR / "processed" / "test.csv")
    holdout_df = pd.concat([val_df, test_df], ignore_index=True)
    synthetic_df = pd.read_csv(
        DATA_DIR / "synthetic" / synthesizer_name / "default" / "synthetic_train.csv"
    )
    return train_df, holdout_df, synthetic_df


# ── Anonymeter evaluations ────────────────────────────────────────────────────


def run_singling_out(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
) -> None:
    """
    Run Anonymeter SinglingOutEvaluator in both univariate and multivariate
    mode and log results to W&B.

    Univariate mode tests single-attribute identification. Multivariate
    mode tests combined-attribute identification — a stronger and more
    realistic attack. Both are reported for completeness.

    A risk value near 0 indicates good privacy protection.

    Args:
        train_df: Real training DataFrame (ori in Anonymeter terms).
        holdout_df: Combined val and test DataFrame used as control.
        synthetic_df: Synthetic DataFrame.
    """
    print(f"Running SinglingOutEvaluator (n_attacks={N_ATTACKS})...")

    for mode in ["univariate", "multivariate"]:
        evaluator = SinglingOutEvaluator(
            ori=train_df,
            syn=synthetic_df,
            control=holdout_df,
            n_attacks=N_ATTACKS,
        )
        evaluator.evaluate(mode=mode)
        risk = evaluator.risk()

        print(
            f"  Singling Out Risk ({mode}): {risk.value:.4f} "
            f"[{risk.ci[0]:.4f}, {risk.ci[1]:.4f}]"
        )

        wandb.log(
            {
                f"singling_out_risk_{mode}": risk.value,
                f"singling_out_risk_{mode}_ci_lower": risk.ci[0],
                f"singling_out_risk_{mode}_ci_upper": risk.ci[1],
            }
        )


def run_linkability(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
) -> None:
    """
    Run Anonymeter LinkabilityEvaluator and log results to W&B.

    Measures whether an attacker can link two records from different
    datasets to the same person using the synthetic data. The columns
    are split evenly into two auxiliary sets.

    A risk value near 0 indicates good privacy protection.

    Args:
        train_df: Real training DataFrame (ori in Anonymeter terms).
        holdout_df: Combined val and test DataFrame used as control.
        synthetic_df: Synthetic DataFrame.
    """
    print(f"Running LinkabilityEvaluator (n_attacks={N_ATTACKS})...")

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

    print(
        f"  Linkability Risk: {risk.value:.4f} " f"[{risk.ci[0]:.4f}, {risk.ci[1]:.4f}]"
    )

    wandb.log(
        {
            "linkability_risk": risk.value,
            "linkability_risk_ci_lower": risk.ci[0],
            "linkability_risk_ci_upper": risk.ci[1],
        }
    )


def run_inference(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
) -> None:
    """
    Run Anonymeter InferenceEvaluator for each sensitive column and log
    results to W&B.

    Models an attacker who possesses all available attributes except the
    target attribute as auxiliary information, following the methodology
    of Giomi et al. (2023). Sensitive target attributes are defined in
    SENSITIVE_COLS and selected based on their inter-feature associations
    and relevance under applicable data protection regulations.

    A risk value near 0 indicates good privacy protection.

    Args:
        train_df: Real training DataFrame (ori in Anonymeter terms).
        holdout_df: Combined val and test DataFrame used as control.
        synthetic_df: Synthetic DataFrame.
    """
    print(f"Running InferenceEvaluator (n_attacks={N_ATTACKS})...")

    for secret in SENSITIVE_COLS:
        aux_cols = [c for c in train_df.columns if c != secret]
        evaluator = InferenceEvaluator(
            ori=train_df,
            syn=synthetic_df,
            control=holdout_df,
            n_attacks=N_ATTACKS,
            secret=secret,
            aux_cols=aux_cols,
        )
        evaluator.evaluate()
        risk = evaluator.risk()

        print(
            f"  Inference Risk ({secret}): {risk.value:.4f} "
            f"[{risk.ci[0]:.4f}, {risk.ci[1]:.4f}]"
        )

        wandb.log(
            {
                f"inference_risk_{secret}": risk.value,
                f"inference_risk_{secret}_ci_lower": risk.ci[0],
                f"inference_risk_{secret}_ci_upper": risk.ci[1],
            }
        )


# ── DCR metrics ───────────────────────────────────────────────────────────────


def run_dcr_metrics(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    metadata: dict,
) -> None:
    """
    Run SDMetrics DCR-based privacy metrics and log results to W&B.

    Computes two metrics:
    - DCRBaselineProtection: compares distances from synthetic to real
      data against a random baseline. A score near 1 indicates good
      protection.
    - DCROverfittingProtection: checks whether synthetic data is too
      close to training data compared to holdout data. A score near 1
      indicates no memorization.

    num_rows_subsample is used to keep computation time manageable for
    datasets of ~30K rows.

    Args:
        train_df: Real training DataFrame.
        holdout_df: Combined val and test DataFrame used as holdout.
        synthetic_df: Synthetic DataFrame.
        metadata: SDV metadata dictionary.
    """
    print("Running DCR metrics...")

    dcr_baseline = DCRBaselineProtection.compute(
        real_data=train_df,
        synthetic_data=synthetic_df,
        metadata=metadata,
        num_rows_subsample=5000,
    )

    dcr_overfitting = DCROverfittingProtection.compute(
        real_training_data=train_df,
        synthetic_data=synthetic_df,
        real_validation_data=holdout_df,
        metadata=metadata,
        num_rows_subsample=5000,
    )

    print(f"  DCR Baseline Protection:    {dcr_baseline:.4f}")
    print(f"  DCR Overfitting Protection: {dcr_overfitting:.4f}")

    wandb.log(
        {
            "dcr_baseline_protection": dcr_baseline,
            "dcr_overfitting_protection": dcr_overfitting,
        }
    )


# ── Main evaluation ───────────────────────────────────────────────────────────


def evaluate_privacy(synthesizer_name: str) -> None:
    """
    Run full privacy evaluation for a synthesizer and log to W&B.

    Runs Singling Out, Linkability, Inference and DCR evaluations
    against the synthetic data generated by the given synthesizer.

    Args:
        synthesizer_name: One of 'gaussian_copula', 'ctgan', 'tvae'.
    """
    run_name = f"eval_privacy_{synthesizer_name}_default"

    with wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=run_name,
        config={
            "synthesizer": synthesizer_name,
            "evaluation": "privacy",
            "n_attacks": N_ATTACKS,
            "sensitive_cols": SENSITIVE_COLS,
        },
    ):
        train_df, holdout_df, synthetic_df = load_data(synthesizer_name)
        metadata_dict = load_metadata(MODELS_DIR, synthesizer_name)

        print(f"Real training data:  {len(train_df)} rows")
        print(f"Holdout data:        {len(holdout_df)} rows")
        print(f"Synthetic data:      {len(synthetic_df)} rows")

        run_singling_out(train_df, holdout_df, synthetic_df)
        run_linkability(train_df, holdout_df, synthetic_df)
        run_inference(train_df, holdout_df, synthetic_df)
        run_dcr_metrics(train_df, holdout_df, synthetic_df, metadata_dict)

        print("Privacy evaluation complete.")


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate privacy of synthetic data using Anonymeter and SDMetrics."
    )
    parser.add_argument(
        "--synthesizer",
        choices=SYNTHESIZERS,
        required=True,
        help="Synthesizer to evaluate.",
    )

    args = parser.parse_args()
    evaluate_privacy(synthesizer_name=args.synthesizer)


if __name__ == "__main__":
    main()
