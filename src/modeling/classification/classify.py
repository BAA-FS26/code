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

"""
classify.py

CLI entry point for classifier training and hyperparameter workflows in the
synthetic data utility evaluation pipeline.
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
from src.modeling.classification.classifier_sweeps import (
    fetch_best_params,
    run_sweep,
)
from src.modeling.classification.classifier_training import (
    train_and_validate,
)
from src.utility.constants import (
    CLASSIFIERS,
    CONFIG_DIR,
    DP_EPSILONS,
    DP_SYNTHESIZERS,
    SYNTHESIZERS,
)

SCRIPT_NAME = "classify.py"


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate classifiers for synthetic " "data utility evaluation."
        )
    )

    parser.add_argument(
        "--mode",
        choices=["default", "sweep", "fetch_best", "best"],
        required=True,
        help=(
            "default: train with default params, evaluate on train and val. "
            "sweep: run W&B hyperparameter sweep (requires --wandb). "
            "fetch_best: fetch best params from W&B sweep "
            "(requires --wandb). "
            "best: train with best params, evaluate on train and val, "
            "save model."
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
        help=("Privacy budget for DP synthesizers. " "Required for dpctgan/patectgan."),
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
        help="Log results to Weights & Biases.",
    )

    return parser


def _validate_params_dict(
    params: Any,
    source_description: str,
) -> dict[str, Any]:
    """
    Validate that a loaded params object is a dictionary.
    """
    if not isinstance(params, dict):
        raise ValueError(
            f"Expected parameter dictionary from "
            f"{source_description}, "
            f"got {type(params).__name__}."
        )

    return params


def _validate_mode_requirements(
    args: argparse.Namespace,
) -> None:
    """Validate mode-specific CLI requirements."""
    if args.mode in {"sweep", "fetch_best"} and not args.wandb:
        raise ValueError(f"--mode {args.mode} requires --wandb.")

    if args.mode == "best" and args.params is None:
        raise ValueError("--mode best requires --params.")


def _validate_wandb_support(
    use_wandb: bool,
) -> None:
    """Validate optional W&B availability."""
    if use_wandb and _wandb is None:
        raise ImportError("wandb is not installed but --wandb was requested.")


def _load_yaml_params(
    filename: str,
) -> dict[str, Any]:
    """
    Load classifier parameters from a YAML config file.
    """
    config_path = CONFIG_DIR / filename

    if not config_path.exists():
        raise FileNotFoundError(f"Parameter config not found at {config_path}.")

    with open(config_path) as file:
        params = yaml.safe_load(file)

    return _validate_params_dict(
        params=params,
        source_description=str(config_path),
    )


def _default_run_name(
    classifier_name: str,
    data_source: str,
) -> str:
    """Build the canonical default-mode run name."""
    return f"{data_source}_{classifier_name}_default"


def _best_run_name(
    classifier_name: str,
    data_source: str,
) -> str:
    """Build the canonical best-mode run name."""
    return f"{data_source}_{classifier_name}_best"


def _run_default_mode(
    classifier_name: str,
    data_source: str,
    use_wandb: bool,
) -> None:
    """Run default-parameter classifier training."""
    train_and_validate(
        classifier_name=classifier_name,
        data_source=data_source,
        params={},
        run_name=_default_run_name(
            classifier_name=classifier_name,
            data_source=data_source,
        ),
        mode="default",
        model_type="default",
        save_model=True,
        use_wandb=use_wandb,
    )


def _run_best_mode(
    classifier_name: str,
    data_source: str,
    params_filename: str,
    use_wandb: bool,
) -> None:
    """Run classifier training using best saved hyperparameters."""
    params = _load_yaml_params(params_filename)

    train_and_validate(
        classifier_name=classifier_name,
        data_source=data_source,
        params=params,
        run_name=_best_run_name(
            classifier_name=classifier_name,
            data_source=data_source,
        ),
        mode="best",
        model_type="best",
        save_model=True,
        use_wandb=use_wandb,
    )


def _run_sweep_mode(
    classifier_name: str,
    data_source: str,
) -> None:
    """Run W&B hyperparameter sweep."""
    run_sweep(
        classifier_name=classifier_name,
        data_source=data_source,
    )


def _run_fetch_best_mode(
    classifier_name: str,
    data_source: str,
) -> None:
    """Fetch best hyperparameters from W&B."""
    fetch_best_params(
        classifier_name=classifier_name,
        data_source=data_source,
    )


def _dispatch_mode(
    args: argparse.Namespace,
    data_source: str,
) -> None:
    """Dispatch execution to the selected CLI mode."""
    if args.mode == "default":
        _run_default_mode(
            classifier_name=args.classifier,
            data_source=data_source,
            use_wandb=args.wandb,
        )
        return

    if args.mode == "best":
        _run_best_mode(
            classifier_name=args.classifier,
            data_source=data_source,
            params_filename=args.params,
            use_wandb=args.wandb,
        )
        return

    if args.mode == "sweep":
        _run_sweep_mode(
            classifier_name=args.classifier,
            data_source=data_source,
        )
        return

    if args.mode == "fetch_best":
        _run_fetch_best_mode(
            classifier_name=args.classifier,
            data_source=data_source,
        )
        return

    raise ValueError(f"Unsupported mode: {args.mode}")


def main() -> None:
    """CLI entry point."""
    parser = _build_parser()

    args = parser.parse_args()

    _validate_mode_requirements(args)

    _validate_wandb_support(
        use_wandb=args.wandb,
    )

    data_source = resolve_training_data_source(
        data_source=args.data_source,
        synthesizer=args.synthesizer,
        epsilon=args.epsilon,
    )

    print(f"[classify] Resolved training data source: " f"{data_source}")

    _dispatch_mode(
        args=args,
        data_source=data_source,
    )


if __name__ == "__main__":
    main()
