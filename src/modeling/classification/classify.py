"""
CLI entry point for classifier training and hyperparameter workflows.

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

from src.core.data_source import resolve_training_data_source
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

SCRIPT_NAME = "classify.py"


def _validate_params_dict(params: Any, source_description: str) -> dict[str, Any]:
    """Validate that a loaded params object is a dictionary."""
    if not isinstance(params, dict):
        raise ValueError(
            f"Expected parameter dictionary from {source_description}, "
            f"got {type(params).__name__}."
        )

    return params


def _load_params(params_path: Path | None) -> dict[str, Any]:
    """Load model parameters from YAML, or return an empty dict."""
    if params_path is None:
        return {}

    config_path = CONFIG_DIR / params_path

    if not config_path.exists():
        raise FileNotFoundError(f"Parameter file not found at {config_path}.")

    with open(config_path) as file:
        params = yaml.safe_load(file)

    return _validate_params_dict(params, str(config_path))


def _build_run_name(
    mode: str,
    classifier_name: str,
    data_source: str,
) -> str:
    """Build the stable classifier run name."""
    safe_data_source = data_source.replace("/", "_")
    return f"{safe_data_source}_{classifier_name}_{mode}"


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate classifiers for synthetic data utility evaluation."
    )

    parser.add_argument(
        "--mode",
        choices=["default", "sweep", "fetch_best", "best"],
        required=True,
        help=(
            "default: train with default params, evaluate on train and val. "
            "sweep: run W&B hyperparameter sweep. "
            "fetch_best: fetch best sweep params. "
            "best: train with params loaded from YAML."
        ),
    )
    parser.add_argument("--classifier", choices=CLASSIFIERS, required=True)
    parser.add_argument("--data_source", default="real")
    parser.add_argument("--synthesizer", choices=sorted(SYNTHESIZERS | DP_SYNTHESIZERS))
    parser.add_argument("--epsilon", type=float, choices=DP_EPSILONS)
    parser.add_argument("--params", type=Path)
    parser.add_argument("--wandb", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    data_source = resolve_training_data_source(
        data_source=args.data_source,
        synthesizer=args.synthesizer,
        epsilon=args.epsilon,
    )

    if args.mode == "sweep":
        run_sweep(
            classifier_name=args.classifier,
            data_source=data_source,
        )
        return

    if args.mode == "fetch_best":
        fetch_best_params(
            classifier_name=args.classifier,
            data_source=data_source,
        )
        return

    params = _load_params(args.params)
    params.setdefault("seed", RANDOM_STATE)

    model_type = "best" if args.mode == "best" else "default"
    run_name = _build_run_name(
        mode=args.mode,
        classifier_name=args.classifier,
        data_source=data_source,
    )

    train_and_validate(
        classifier_name=args.classifier,
        data_source=data_source,
        params=params,
        run_name=run_name,
        mode=args.mode,
        model_type=model_type,
        save_model=True,
        use_wandb=args.wandb,
    )


if __name__ == "__main__":
    main()
