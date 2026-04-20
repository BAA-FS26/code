"""
evaluate_fidelity.py
 
Fidelity evaluation for synthetic data using standalone SDMetrics.
 
SDMetrics is model-agnostic and evaluates synthetic data independently
of how it was generated. This ensures the evaluation pipeline is
reproducible and applicable to any synthesizer, not just SDV models.
 
Runs two reports for each synthesizer:
  - QualityReport: measures statistical similarity between synthetic and
    real data across two dimensions:
      * Column Shapes: univariate distribution similarity per feature
      * Column Pair Trends: bivariate correlation similarity
  - DiagnosticReport: checks synthetic data validity, including
    out-of-range values and invalid categories
 
Results are always saved locally as JSON. W&B logging is optional.
Per-column scores are logged as W&B tables when W&B is enabled.
 
Usage:
    # Without W&B (default)
    python -m src.evaluation.evaluate_fidelity --synthesizer gaussian_copula
    python -m src.evaluation.evaluate_fidelity --synthesizer ctgan
    python -m src.evaluation.evaluate_fidelity --synthesizer tvae
 
    # With W&B logging
    python -m src.evaluation.evaluate_fidelity --synthesizer ctgan --wandb
"""

import argparse

import pandas as pd
from sdmetrics.reports.single_table import DiagnosticReport, QualityReport

from src.utility.constants import (
    DATA_DIR,
    SYNTHESIZER_MODELS_DIR,
    SYNTHESIZERS,
)
from src.utility.logger import RunLogger
from src.utility.utils import load_metadata

# ── Constants ────────────────────────────────────────────────────────────────

MODELS_DIR = SYNTHESIZER_MODELS_DIR


# ── Data loading ──────────────────────────────────────────────────────────────


def load_data(synthesizer_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load real training data and synthetic data for a given synthesizer.

    Args:
        synthesizer_name: One of 'gaussian_copula', 'ctgan', 'tvae'.

    Returns:
        Tuple of (real_train_df, synthetic_df).
    """
    real_df = pd.read_csv(DATA_DIR / "processed" / "train.csv")
    synthetic_df = pd.read_csv(
        DATA_DIR / "synthetic" / synthesizer_name / "default" / "synthetic_train.csv"
    )
    return real_df, synthetic_df


# ── Quality report ────────────────────────────────────────────────────────────


def run_quality_report(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    metadata: dict,
    logger: RunLogger,
) -> None:
    """
    Run SDMetrics QualityReport and log results via the run logger.

    Evaluates two dimensions:
    - Column Shapes: univariate distribution similarity per feature
    - Column Pair Trends: bivariate correlation similarity

    Overall and per-dimension scores are logged as summary metrics.
    Per-column scores are logged as W&B tables when W&B is enabled.

    Args:
        real_df: Real training DataFrame.
        synthetic_df: Synthetic DataFrame.
        metadata: SDV metadata dictionary.
        logger: Active RunLogger instance.
    """
    print("Running QualityReport...")
    report = QualityReport()
    report.generate(real_df, synthetic_df, metadata)

    overall_score = report.get_score()
    properties = report.get_properties()

    column_shapes_score = properties.loc[
        properties["Property"] == "Column Shapes", "Score"
    ].values[0]
    column_pair_trends_score = properties.loc[
        properties["Property"] == "Column Pair Trends", "Score"
    ].values[0]

    print(f"  Overall Quality Score:    {overall_score:.4f}")
    print(f"  Column Shapes Score:      {column_shapes_score:.4f}")
    print(f"  Column Pair Trends Score: {column_pair_trends_score:.4f}")

    logger.log(
        {
            "quality_overall": overall_score,
            "quality_column_shapes": column_shapes_score,
            "quality_column_pair_trends": column_pair_trends_score,
        }
    )

    for property_name in ["Column Shapes", "Column Pair Trends"]:
        try:
            details = report.get_details(property_name=property_name)
            table_key = f"quality_{property_name.lower().replace(' ', '_')}_details"
            logger.log_table(table_key, details)
        except Exception as e:
            print(f"  Could not log details for {property_name}: {e}")


# ── Diagnostic report ─────────────────────────────────────────────────────────


def run_diagnostic_report(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    metadata: dict,
    logger: RunLogger,
) -> None:
    """
    Run SDMetrics DiagnosticReport and log results via the run logger.

    Checks synthetic data validity including:
    - Data Validity: boundary adherence, range coverage, category coverage
    - Data Structure: whether synthetic data has the correct columns

    Args:
        real_df: Real training DataFrame.
        synthetic_df: Synthetic DataFrame.
        metadata: SDV metadata dictionary.
        logger: Active RunLogger instance.
    """
    print("Running DiagnosticReport...")
    report = DiagnosticReport()
    report.generate(real_df, synthetic_df, metadata)

    properties = report.get_properties()
    metrics = {}

    for _, row in properties.iterrows():
        key = f"diagnostic_{row['Property'].lower().replace(' ', '_')}"
        metrics[key] = row["Score"]
        print(f"  {row['Property']}: {row['Score']:.4f}")

    logger.log(metrics)

    try:
        details = report.get_details(property_name="Data Validity")
        logger.log_table("diagnostic_data_validity_details", details)
    except Exception as e:
        print(f"  Could not log diagnostic details: {e}")


# ── Main evaluation ───────────────────────────────────────────────────────────


def evaluate_fidelity(synthesizer_name: str, use_wandb: bool = False) -> None:
    """
    Run full fidelity evaluation for a synthesizer.

    Runs both QualityReport and DiagnosticReport against the synthetic
    data generated by the given synthesizer, comparing against real
    training data. Results are always saved locally. W&B logging is
    optional.

    Args:
        synthesizer_name: One of 'gaussian_copula', 'ctgan', 'tvae'.
        use_wandb: Whether to log results to W&B. Defaults to False.
    """
    run_name = f"eval_fidelity_{synthesizer_name}_default"
    config = {"synthesizer": synthesizer_name, "evaluation": "fidelity"}

    with RunLogger(run_name=run_name, config=config, use_wandb=use_wandb) as logger:
        real_df, synthetic_df = load_data(synthesizer_name)
        metadata = load_metadata(MODELS_DIR, synthesizer_name)

        print(f"Real training data: {len(real_df)} rows")
        print(f"Synthetic data:     {len(synthetic_df)} rows")

        run_quality_report(real_df, synthetic_df, metadata, logger)
        run_diagnostic_report(real_df, synthetic_df, metadata, logger)

        print("Fidelity evaluation complete.")


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fidelity of synthetic data using SDMetrics."
    )
    parser.add_argument(
        "--synthesizer",
        choices=SYNTHESIZERS,
        required=True,
        help="Synthesizer to evaluate.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="Log results to Weights & Biases. Requires WANDB_ENTITY to be set.",
    )

    args = parser.parse_args()
    evaluate_fidelity(synthesizer_name=args.synthesizer, use_wandb=args.wandb)


if __name__ == "__main__":
    main()
