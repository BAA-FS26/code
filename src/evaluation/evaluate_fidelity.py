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
    PROCESSED_DATA_DIR,
    SYNTHESIZER_MODELS_DIR,
    SYNTHESIZERS,
    SYNTHETIC_DATA_DIR,
    SYNTHETIC_TRAIN_FILENAME,
    TRAIN_FILENAME,
)
from src.utility.logger import RunLogger
from src.utility.utils import build_adult_sdmetrics_metadata, load_metadata

# ── Constants ────────────────────────────────────────────────────────────────

SCRIPT_NAME = "evaluate_fidelity.py"
MODELS_DIR = SYNTHESIZER_MODELS_DIR


# ── Path and argument helpers ────────────────────────────────────────────────


def _real_train_path() -> Path:
    """Return the canonical path to the real training split."""
    return PROCESSED_DATA_DIR / TRAIN_FILENAME


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
    """
    Return the synthesizer key used for metadata lookup.

    DP synthesizers use fallback metadata keyed by synthesizer name.
    Non-DP synthesizers use saved SDV metadata keyed by synthesizer name.
    """
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
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
) -> None:
    """
    Validate that real and synthetic dataframes have identical columns in order.
    """
    real_columns = list(real_df.columns)
    synthetic_columns = list(synthetic_df.columns)
    if synthetic_columns != real_columns:
        raise ValueError(
            "Synthetic data columns do not match real training data columns.\n"
            f"Expected: {real_columns}\n"
            f"Actual:   {synthetic_columns}"
        )


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


# ── Data loading ──────────────────────────────────────────────────────────────


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
    real_path = _real_train_path()
    synthetic_path = _synthetic_train_path(synthesizer_name, epsilon)

    real_df = _load_csv(real_path, "Real training split")
    synthetic_df = _load_csv(synthetic_path, "Synthetic training data")
    _validate_matching_schema(real_df, synthetic_df)

    return real_df, synthetic_df, real_path, synthetic_path


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


# ── Main evaluation ───────────────────────────────────────────────────────────


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
    data_source = _data_source_key(synthesizer_name, epsilon)
    run_name = f"eval_fidelity_{data_source.replace('/', '_')}"
    metadata_key = _metadata_key(synthesizer_name)

    parameters = {
        "synthesizer": synthesizer_name,
        "epsilon": epsilon,
        "data_source": data_source,
        "evaluation": "fidelity",
        "metadata_key": metadata_key,
        "use_wandb": use_wandb,
    }

    with RunLogger(
        run_name=run_name,
        script_name=SCRIPT_NAME,
        parameters=parameters,
        use_wandb=use_wandb,
    ) as logger:
        real_df, synthetic_df, real_path, synthetic_path = load_data(
            synthesizer_name=synthesizer_name,
            epsilon=epsilon,
        )
        metadata = load_metadata(
            MODELS_DIR,
            metadata_key,
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


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate fidelity of synthetic data using SDMetrics."
    )
    parser.add_argument(
        "--synthesizer",
        choices=SYNTHESIZERS + DP_SYNTHESIZERS,
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
