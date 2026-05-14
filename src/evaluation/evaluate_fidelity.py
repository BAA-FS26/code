"""
Fidelity evaluation for synthetic tabular data.

This module compares synthetic data against real training data using SDMetrics
quality and diagnostic reports, then logs the resulting fidelity metrics.

Usage:
    # Without W&B (default)
    python -m src.evaluation.evaluate_fidelity --synthesizer gaussian_copula
    python -m src.evaluation.evaluate_fidelity --synthesizer ctgan
    python -m src.evaluation.evaluate_fidelity --synthesizer tvae

    # With W&B logging
    python -m src.evaluation.evaluate_fidelity --synthesizer ctgan --wandb

    # DP synthesizers
    python -m src.evaluation.evaluate_fidelity --synthesizer dpctgan --epsilon 1.0
    python -m src.evaluation.evaluate_fidelity --synthesizer patectgan --epsilon 1.0
"""

import argparse
from pathlib import Path

import pandas as pd
from sdmetrics.reports.single_table import DiagnosticReport, QualityReport

from src.utility.constants import (
    DP_EPSILONS,
    DP_SYNTHESIZERS,
    RANDOM_STATE,
    SYNTHESIZER_MODELS_DIR,
    SYNTHESIZERS,
    TRAIN_FILENAME,
)
from src.utility.logger import RunLogger
from src.utility.utils import build_adult_sdmetrics_metadata, load_metadata
from src.core.data_source import build_data_source_key
from src.core.io import load_csv, validate_matching_columns
from src.core.paths import processed_split_path, synthetic_train_path

SCRIPT_NAME = "evaluate_fidelity.py"
MODELS_DIR = SYNTHESIZER_MODELS_DIR


def _get_property_score(properties: pd.DataFrame, property_name: str) -> float:
    """
    Extract a single property score from an SDMetrics properties dataframe.

    Raises:
        ValueError: If the requested property is not present.
    """
    matches = properties.loc[properties["Property"] == property_name, "Score"]
    if matches.empty:
        available = properties["Property"].tolist() if "Property" in properties else []
        raise ValueError(
            f"Expected SDMetrics property '{property_name}' not found. "
            f"Available properties: {available}"
        )
    return float(matches.iloc[0])


def load_data(
    synthesizer_name: str,
    epsilon: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    """
    Load real training data and synthetic data for a given synthesizer.

    For non-DP synthesizers, data is loaded from:
        data/synthetic/{name}/default/synthetic_train.csv

    For DP synthesizers, data is loaded from:
        data/synthetic/{name}/eps_{epsilon}/synthetic_train.csv

    Args:
        synthesizer_name: One of 'gaussian_copula', 'ctgan', 'tvae',
                          'dpctgan', or 'patectgan'.
        epsilon: Privacy budget for DP synthesizers. Must be None for
                 non-DP synthesizers.

    Returns:
        Tuple of (real_train_df, synthetic_df, real_path, synthetic_path).
    """
    data_source = build_data_source_key(synthesizer_name, epsilon)

    real_path = processed_split_path(TRAIN_FILENAME)
    synthetic_path = synthetic_train_path(data_source)

    real_df = load_csv(real_path, "Real training split")
    synthetic_df = load_csv(synthetic_path, "Synthetic training data")

    validate_matching_columns(
        reference_df=real_df,
        candidate_df=synthetic_df,
        candidate_name="Synthetic training data",
    )

    return real_df, synthetic_df, real_path, synthetic_path


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
        metadata: SDV/SDMetrics metadata dictionary.
        logger: Active RunLogger instance.
    """
    print("[evaluate_fidelity] Running QualityReport...")
    report = QualityReport()
    report.generate(real_df, synthetic_df, metadata)

    overall_score_raw = report.get_score()
    if overall_score_raw is None:
        raise ValueError("QualityReport returned no overall score.")
    overall_score = float(overall_score_raw)

    properties = report.get_properties()

    column_shapes_score = _get_property_score(properties, "Column Shapes")
    column_pair_trends_score = _get_property_score(properties, "Column Pair Trends")

    print(f"[evaluate_fidelity] Overall Quality Score: {overall_score:.4f}")
    print(f"[evaluate_fidelity] Column Shapes Score: {column_shapes_score:.4f}")
    print(
        "[evaluate_fidelity] Column Pair Trends Score: "
        f"{column_pair_trends_score:.4f}"
    )

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
        except Exception as exc:
            print(
                f"[evaluate_fidelity] Could not log details for "
                f"{property_name}: {exc}"
            )


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
        metadata: SDV/SDMetrics metadata dictionary.
        logger: Active RunLogger instance.
    """
    print("[evaluate_fidelity] Running DiagnosticReport...")
    report = DiagnosticReport()
    report.generate(real_df, synthetic_df, metadata)

    metrics: dict[str, float] = {}

    diagnostic_overall_raw = report.get_score()
    if diagnostic_overall_raw is not None:
        metrics["diagnostic_overall"] = float(diagnostic_overall_raw)
        print(
            f"[evaluate_fidelity] Overall Diagnostic Score: "
            f"{metrics['diagnostic_overall']:.4f}"
        )

    properties = report.get_properties()
    for _, row in properties.iterrows():
        key = f"diagnostic_{row['Property'].lower().replace(' ', '_')}"
        metrics[key] = float(row["Score"])
        print(f"[evaluate_fidelity] {row['Property']}: {float(row['Score']):.4f}")

    logger.log(metrics)

    try:
        details = report.get_details(property_name="Data Validity")
        logger.log_table("diagnostic_data_validity_details", details)
    except Exception as exc:
        print(f"[evaluate_fidelity] Could not log diagnostic details: {exc}")


def evaluate_fidelity(
    synthesizer_name: str,
    epsilon: float | None = None,
    use_wandb: bool = False,
) -> None:
    """
    Run full fidelity evaluation for a synthesizer.

    Runs both QualityReport and DiagnosticReport against the synthetic
    data generated by the given synthesizer, comparing against real
    training data. Results are always saved locally. W&B logging is
    optional.

    Args:
        synthesizer_name: One of 'gaussian_copula', 'ctgan', 'tvae',
                          'dpctgan', or 'patectgan'.
        epsilon: Privacy budget for DP synthesizers. Must be None for
                 non-DP synthesizers.
        use_wandb: Whether to log results to W&B. Defaults to False.
    """
    data_source = build_data_source_key(synthesizer_name, epsilon)
    run_name = f"eval_fidelity_{data_source.replace('/', '_')}"

    parameters = {
        "pipeline_stage": "evaluation",
        "evaluation": "fidelity",
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
    }

    with RunLogger(
        run_name=run_name,
        script_name=SCRIPT_NAME,
        parameters=parameters,
        use_wandb=use_wandb,
        category="fidelity",
    ) as logger:
        real_df, synthetic_df, real_path, synthetic_path = load_data(
            synthesizer_name=synthesizer_name,
            epsilon=epsilon,
        )
        metadata = load_metadata(
            MODELS_DIR,
            synthesizer_name,
            fallback=build_adult_sdmetrics_metadata(),
        )

        print(f"[evaluate_fidelity] Real training data: {len(real_df)} rows")
        print(f"[evaluate_fidelity] Synthetic data: {len(synthetic_df)} rows")

        logger.log(
            {
                "real_data_path": real_path,
                "synthetic_data_path": synthetic_path,
                "n_rows_real_train": len(real_df),
                "n_rows_synthetic": len(synthetic_df),
            }
        )

        run_quality_report(real_df, synthetic_df, metadata, logger)
        run_diagnostic_report(real_df, synthetic_df, metadata, logger)

        print("[evaluate_fidelity] Fidelity evaluation complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate fidelity of synthetic data using SDMetrics."
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

    evaluate_fidelity(
        synthesizer_name=args.synthesizer,
        epsilon=args.epsilon,
        use_wandb=args.wandb,
    )


if __name__ == "__main__":
    main()
