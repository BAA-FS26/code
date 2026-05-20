"""
Fidelity evaluation for synthetic tabular data.

Compares synthetic data against real training data using SDMetrics quality and
diagnostic reports, then logs the resulting Fidelity metrics.

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
from typing import Any

import pandas as pd
from sdmetrics.reports.single_table import DiagnosticReport, QualityReport

from src.core.data_source import build_data_source_key
from src.dataset.dataset_config import get_dataset_config
from src.evaluation.evaluation_data import load_fidelity_datasets
from src.utility.constants import (
    DP_EPSILONS,
    DP_SYNTHESIZERS,
    RANDOM_STATE,
    SYNTHESIZER_MODELS_DIR,
    SYNTHESIZERS,
)
from src.utility.logger import RunLogger
from src.utility.utils import build_sdmetrics_metadata, load_metadata

SCRIPT_NAME = "evaluate_fidelity.py"
MODELS_DIR = SYNTHESIZER_MODELS_DIR


def _get_property_score(properties: pd.DataFrame, property_name: str) -> float:
    """Extract a single property score from an SDMetrics properties table."""
    matches = properties.loc[properties["Property"] == property_name, "Score"]

    if matches.empty:
        available = properties["Property"].tolist() if "Property" in properties else []
        raise ValueError(
            f"Expected SDMetrics property '{property_name}' not found. "
            f"Available properties: {available}"
        )

    return float(matches.iloc[0])


def _log_report_details(
    report: Any,
    property_name: str,
    table_key: str,
    logger: RunLogger,
    context: str,
) -> None:
    """Log detailed SDMetrics report output as a table artifact when available."""
    try:
        details = report.get_details(property_name=property_name)
        logger.log_table(table_key, details)
    except Exception as exc:
        print(
            f"[evaluate_fidelity] Could not log {context} details for "
            f"{property_name}: {exc}"
        )


def run_quality_report(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    metadata: dict[str, Any],
    logger: RunLogger,
) -> None:
    """Run SDMetrics QualityReport and log Fidelity quality scores."""
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

    _log_report_details(
        report=report,
        property_name="Column Shapes",
        table_key="quality_column_shapes_details",
        logger=logger,
        context="quality",
    )
    _log_report_details(
        report=report,
        property_name="Column Pair Trends",
        table_key="quality_column_pair_trends_details",
        logger=logger,
        context="quality",
    )


def run_diagnostic_report(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    metadata: dict[str, Any],
    logger: RunLogger,
) -> None:
    """Run SDMetrics DiagnosticReport and log diagnostic scores."""
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
        property_name = str(row["Property"])
        key = f"diagnostic_{property_name.lower().replace(' ', '_')}"
        score = float(row["Score"])

        metrics[key] = score
        print(f"[evaluate_fidelity] {property_name}: {score:.4f}")

    logger.log(metrics)

    _log_report_details(
        report=report,
        property_name="Data Validity",
        table_key="diagnostic_data_validity_details",
        logger=logger,
        context="diagnostic",
    )


def _build_run_parameters(
    synthesizer_name: str,
    dataset_name: str,
    epsilon: float | None,
    data_source: str,
    use_wandb: bool,
) -> dict[str, Any]:
    """Build stable logger metadata for Fidelity evaluation."""
    return {
        "pipeline_stage": "evaluation",
        "evaluation": "fidelity",
        "dataset": dataset_name,
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


def evaluate_fidelity(
    synthesizer_name: str,
    epsilon: float | None = None,
    dataset_name: str = "adult_census",
    use_wandb: bool = False,
) -> None:
    """Run full Fidelity evaluation for a synthesizer."""
    data_source = build_data_source_key(synthesizer_name, epsilon)
    run_name = f"eval_fidelity_{data_source.replace('/', '_')}"

    parameters = _build_run_parameters(
        synthesizer_name=synthesizer_name,
        epsilon=epsilon,
        data_source=data_source,
        use_wandb=use_wandb,
        dataset_name=dataset_name,
    )

    with RunLogger(
        run_name=run_name,
        script_name=SCRIPT_NAME,
        parameters=parameters,
        use_wandb=use_wandb,
        category="fidelity",
    ) as logger:
        real_df, synthetic_df, paths = load_fidelity_datasets(data_source)

        metadata = load_metadata(
            MODELS_DIR,
            synthesizer_name,
            fallback=build_sdmetrics_metadata(get_dataset_config(dataset_name)),
        )

        print(f"[evaluate_fidelity] Real training data: {len(real_df)} rows")
        print(f"[evaluate_fidelity] Synthetic data: {len(synthetic_df)} rows")

        logger.log(
            {
                "real_data_path": paths["real_train_path"],
                "synthetic_data_path": paths["synthetic_path"],
                "n_rows_real_train": len(real_df),
                "n_rows_synthetic": len(synthetic_df),
            }
        )

        run_quality_report(real_df, synthetic_df, metadata, logger)
        run_diagnostic_report(real_df, synthetic_df, metadata, logger)

        print("[evaluate_fidelity] Fidelity evaluation complete.")


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate fidelity of synthetic data using SDMetrics."
    )
    parser.add_argument(
        "--dataset",
        default="adult_census",
        help="Dataset configuration to use.",
    )
    parser.add_argument(
        "--synthesizer",
        choices=sorted(SYNTHESIZERS | DP_SYNTHESIZERS),
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
        help="Log results to W&B. Local JSON logging remains primary.",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.synthesizer in DP_SYNTHESIZERS and args.epsilon is None:
        raise SystemExit("--epsilon is required for DP synthesizers.")

    if args.synthesizer not in DP_SYNTHESIZERS and args.epsilon is not None:
        raise SystemExit("--epsilon should only be used with DP synthesizers.")

    evaluate_fidelity(
        synthesizer_name=args.synthesizer,
        epsilon=args.epsilon,
        dataset_name=args.dataset,
        use_wandb=args.wandb,
    )


if __name__ == "__main__":
    main()
