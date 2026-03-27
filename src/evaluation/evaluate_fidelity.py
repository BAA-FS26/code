"""
evaluate_fidelity.py

Fidelity evaluation for synthetic data using SDMetrics.

Runs two reports for each synthesizer:
  - QualityReport: measures statistical similarity between synthetic and
    real data across two dimensions:
      * Column Shapes: univariate distribution similarity per feature
      * Column Pair Trends: bivariate correlation similarity between features
  - DiagnosticReport: checks data validity of synthetic data, including
    out-of-range values and invalid categories

All results are logged to W&B. Per-column scores are logged as W&B tables
for detailed inspection.

Usage:
    python evaluate_fidelity.py --synthesizer gaussian_copula
    python evaluate_fidelity.py --synthesizer ctgan
    python evaluate_fidelity.py --synthesizer tvae
"""

import argparse
from pathlib import Path

import pandas as pd
import wandb
from sdv.evaluation.single_table import (
    evaluate_quality,
    run_diagnostic,
)
from sdv.metadata import Metadata

# ── Constants ────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models" / "synthesizers"

WANDB_PROJECT = "synthetic-data-eval"
WANDB_ENTITY = "baa_fs26_pm"  # TODO: replace with your W&B entity

SYNTHESIZERS = ["gaussian_copula", "ctgan", "tvae"]


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


def load_metadata(synthesizer_name: str) -> Metadata:
    """
    Load the saved SDV metadata for a given synthesizer.

    Args:
        synthesizer_name: One of 'gaussian_copula', 'ctgan', 'tvae'.

    Returns:
        Loaded Metadata object.
    """
    metadata_path = MODELS_DIR / f"{synthesizer_name}_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"No metadata found at {metadata_path}. "
            "Run synthesize.py first to generate and save the metadata."
        )
    return Metadata.load_from_json(str(metadata_path))


# ── Evaluation ────────────────────────────────────────────────────────────────


def evaluate_fidelity(synthesizer_name: str) -> None:
    """
    Run fidelity evaluation for a synthesizer and log results to W&B.

    Runs SDMetrics QualityReport and DiagnosticReport comparing synthetic
    data against real training data. Logs overall scores, per-property
    scores and per-column scores to W&B.

    Args:
        synthesizer_name: One of 'gaussian_copula', 'ctgan', 'tvae'.
    """
    run_name = f"eval_fidelity_{synthesizer_name}_default"

    with wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=run_name,
        config={"synthesizer": synthesizer_name, "evaluation": "fidelity"},
    ):
        real_df, synthetic_df = load_data(synthesizer_name)
        metadata = load_metadata(synthesizer_name)

        print(f"Running QualityReport for {synthesizer_name}...")
        quality_report = evaluate_quality(
            real_data=real_df,
            synthetic_data=synthetic_df,
            metadata=metadata,
        )

        print(f"Running DiagnosticReport for {synthesizer_name}...")
        diagnostic_report = run_diagnostic(
            real_data=real_df,
            synthetic_data=synthetic_df,
            metadata=metadata,
        )

        # ── Quality scores ────────────────────────────────────────────────

        overall_quality = quality_report.get_score()
        properties = quality_report.get_properties()

        column_shapes_score = properties.loc[
            properties["Property"] == "Column Shapes", "Score"
        ].values[0]

        column_pair_trends_score = properties.loc[
            properties["Property"] == "Column Pair Trends", "Score"
        ].values[0]

        wandb.log(
            {
                "quality_overall": overall_quality,
                "quality_column_shapes": column_shapes_score,
                "quality_column_pair_trends": column_pair_trends_score,
            }
        )

        print(f"Overall Quality Score:       {overall_quality:.4f}")
        print(f"Column Shapes Score:         {column_shapes_score:.4f}")
        print(f"Column Pair Trends Score:    {column_pair_trends_score:.4f}")

        # ── Per-column scores (Column Shapes) ─────────────────────────────

        column_shapes_details = quality_report.get_details("Column Shapes")
        wandb.log(
            {"column_shapes_details": wandb.Table(dataframe=column_shapes_details)}
        )

        # ── Per-column pair scores (Column Pair Trends) ───────────────────

        column_pair_details = quality_report.get_details("Column Pair Trends")
        wandb.log(
            {"column_pair_trends_details": wandb.Table(dataframe=column_pair_details)}
        )

        # ── Diagnostic scores ─────────────────────────────────────────────

        diagnostic_properties = diagnostic_report.get_properties()

        for _, row in diagnostic_properties.iterrows():
            property_name = row["Property"].lower().replace(" ", "_")
            wandb.log({f"diagnostic_{property_name}": row["Score"]})

        print("\nDiagnostic Results:")
        print(diagnostic_properties.to_string(index=False))

        diagnostic_details = diagnostic_report.get_details("Data Validity")
        wandb.log({"diagnostic_details": wandb.Table(dataframe=diagnostic_details)})


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Run fidelity evaluation for a synthesizer."
    )
    parser.add_argument(
        "--synthesizer",
        choices=SYNTHESIZERS,
        required=True,
        help="Synthesizer to evaluate.",
    )

    args = parser.parse_args()
    evaluate_fidelity(synthesizer_name=args.synthesizer)


if __name__ == "__main__":
    main()
