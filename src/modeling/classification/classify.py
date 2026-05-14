"""
classify.py

CLI entry point for classifier training and hyperparameter workflows in the
synthetic data utility evaluation pipeline.

This script supports:
  - classifier training on real or synthetic training data
  - validation-based utility evaluation
  - optional W&B hyperparameter sweeps
  - exporting best sweep parameters
  - saving trained model payloads

Supported classifiers:
  - logistic_regression
  - random_forest
  - gradient_boosting

Usage:

    # Real data
    python -m src.modeling.classification.classify --mode default --classifier logistic_regression --data_source real

    # Train with best parameters
    python -m src.modeling.classification.classify --mode best --classifier logistic_regression --data_source real --params best_logistic_regression_real.yaml

    # W&B sweep
    python -m src.modeling.classification.classify --mode sweep --classifier logistic_regression --data_source real --wandb

    # Fetch best sweep parameters
    python -m src.modeling.classification.classify --mode fetch_best --classifier logistic_regression --data_source real --wandb

    # TSTR with non-DP synthetic data
    python -m src.modeling.classification.classify --mode default --classifier logistic_regression --synthesizer ctgan

    # TSTR with DP synthetic data
    python -m src.modeling.classification.classify --mode default --classifier logistic_regression --synthesizer dpctgan --epsilon 1.0
"""

import argparse
from pathlib import Path
from typing import Any


import yaml

try:
    import wandb as _wandb
except ImportError:
    _wandb = None


from src.modeling.classification.classifier_sweeps import fetch_best_params, run_sweep
from src.modeling.classification.classifier_training import train_and_validate
from src.utility.constants import (
    CLASSIFIERS,
    CONFIG_DIR,
    DP_EPSILONS,
    DP_SYNTHESIZERS,
    RANDOM_STATE,
    SYNTHESIZERS,
)


from src.core.data_source import (
    resolve_training_data_source,
)

SCRIPT_NAME = "classify.py"


def _validate_params_dict(params: Any, source_description: str) -> dict[str, Any]:
    """
    Validate that a loaded params object is a dictionary.
    """
    if not isinstance(params, dict):
        raise ValueError(
            f"Expected parameter dictionary from {source_description}, "
            f"got {type(params).__name__}."
        )
    return params


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate classifiers for synthetic data utility evaluation."
    )
    parser.add_argument(
        "--mode",
        choices=["default", "sweep", "fetch_best", "best"],
        required=True,
        help=(
            "default: train with default params, evaluate on train and val. "
            "sweep: run W&B hyperparameter sweep (requires --wandb). "
            "fetch_best: fetch best params from W&B sweep (requires --wandb). "
            "best: train with best params, evaluate on train and val, save model."
        ),
    )
    parser.add_argument(
        "--classifier",
        choices=CLASSIFIERS,
        required=True,
        help="Classifier to train.",
    )
    parser.add_argument(
        "--data_source",
        choices=["real"],
        default="real",
        help="Data source to use. Default: real.",
    )
    parser.add_argument(
        "--synthesizer",
        choices=SYNTHESIZERS | DP_SYNTHESIZERS,
        default=None,
        help="Synthetic data source to train on instead of real data.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        choices=DP_EPSILONS,
        default=None,
        help="Privacy budget for DP synthesizers. Required for dpctgan/patectgan.",
    )
    parser.add_argument(
        "--params",
        type=str,
        default=None,
        help=(
            "Filename of YAML file in config/ with best hyperparameters. "
            "Required for --mode best."
        ),
    )

    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help=(
            "Log results to Weights & Biases. Local JSON logging remains primary. "
            "Required for sweep and fetch_best modes."
        ),
    )

    args = parser.parse_args()
    try:
        resolved_data_source = resolve_training_data_source(
            data_source=args.data_source,
            synthesizer=args.synthesizer,
            epsilon=args.epsilon,
        )
    except ValueError as exc:
        parser.error(str(exc))

    if args.mode in ("sweep", "fetch_best") and not args.wandb:
        parser.error(f"--mode {args.mode} requires --wandb.")

    if args.mode in ("sweep", "fetch_best") and _wandb is None:
        parser.error(
            "--mode sweep and --mode fetch_best require the 'wandb' package to be installed."
        )

    if args.mode == "default":
        train_and_validate(
            classifier_name=args.classifier,
            data_source=resolved_data_source,
            params={"seed": RANDOM_STATE},
            run_name=f"{resolved_data_source.replace('/', '_')}_{args.classifier}_default",
            mode="default",
            model_type="default",
            save_model=True,
            use_wandb=args.wandb,
        )

    elif args.mode == "sweep":
        run_sweep(classifier_name=args.classifier, data_source=resolved_data_source)

    elif args.mode == "fetch_best":
        fetch_best_params(
            classifier_name=args.classifier,
            data_source=resolved_data_source,
        )

    elif args.mode == "best":
        if args.params is None:
            parser.error("--params is required for --mode best.")

        params_path = CONFIG_DIR / Path(args.params).name
        if not params_path.exists():
            parser.error(f"Parameter file not found at {params_path}.")

        with open(params_path) as f:
            loaded_params = yaml.safe_load(f)

        best_params = _validate_params_dict(loaded_params, str(params_path))
        train_and_validate(
            classifier_name=args.classifier,
            data_source=resolved_data_source,
            params=best_params,
            run_name=f"{resolved_data_source.replace('/', '_')}_{args.classifier}_best",
            mode="best",
            model_type="best",
            save_model=True,
            use_wandb=args.wandb,
        )


if __name__ == "__main__":
    main()
