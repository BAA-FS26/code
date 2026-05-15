"""
Held-out utility evaluation for trained classifiers.

This module loads saved classifier payloads, evaluates them on the real test
split, and logs test-set utility metrics.

Usage:

    # Real data baseline
    python -m src.evaluation.evaluate_utility --classifier logistic_regression --data_source real --model_type best

    # TSTR, non-DP synthetic source
    python -m src.evaluation.evaluate_utility --classifier logistic_regression --synthesizer ctgan --model_type best

    # TSTR, DP synthetic source
    python -m src.evaluation.evaluate_utility --classifier logistic_regression --synthesizer dpctgan --epsilon 1.0 --model_type best
"""

import argparse
import pickle
from pathlib import Path
from typing import Any

from src.core.data_source import (
    epsilon_from_data_source,
    resolve_training_data_source,
    synthesizer_from_data_source,
)
from src.core.paths import classifier_model_path
from src.evaluation.evaluation_data import load_utility_test_dataset
from src.evaluation.metrics import compute_classification_metrics
from src.modeling.classification.classifier_data import prepare_single
from src.utility.constants import (
    CLASSIFIERS,
    DP_EPSILONS,
    DP_SYNTHESIZERS,
    RANDOM_STATE,
    SYNTHESIZERS,
)
from src.utility.logger import RunLogger

SCRIPT_NAME = "evaluate_utility.py"


def _read_model_payload(model_path: Path) -> dict[str, Any]:
    """Read a saved classifier payload from disk."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"No saved model found at {model_path}. "
            "Run src.modeling.classification.classify with --mode default "
            "or --mode best first."
        )

    with open(model_path, "rb") as file:
        saved = pickle.load(file)

    if not isinstance(saved, dict):
        raise ValueError(
            f"Saved model payload at {model_path} is invalid: " "expected dictionary."
        )

    return saved


def _validate_payload_keys(
    saved: dict[str, Any],
    model_path: Path,
) -> None:
    """Validate that the saved payload contains the required top-level keys."""
    missing_keys = {"model", "preprocessor", "metadata"} - set(saved.keys())

    if missing_keys:
        raise ValueError(
            f"Saved model payload at {model_path} is missing required keys: "
            f"{sorted(missing_keys)}"
        )


def _validate_payload_metadata(
    metadata: Any,
    model_path: Path,
    classifier_name: str,
    data_source: str,
) -> dict[str, Any]:
    """Validate saved model metadata against the requested evaluation setup."""
    if not isinstance(metadata, dict):
        raise ValueError(
            f"Saved model metadata at {model_path} is invalid: " "expected dictionary."
        )

    if metadata.get("classifier") != classifier_name:
        raise ValueError(
            f"Saved model classifier mismatch at {model_path}: "
            f"expected '{classifier_name}', got '{metadata.get('classifier')}'."
        )

    if metadata.get("data_source") != data_source:
        raise ValueError(
            f"Saved model data_source mismatch at {model_path}: "
            f"expected '{data_source}', got '{metadata.get('data_source')}'."
        )

    return metadata


def load_saved_model(
    classifier_name: str,
    data_source: str,
    model_type: str,
) -> tuple[Any, Any, dict[str, Any], Path]:
    """
    Load and validate a saved classifier payload.

    The payload format is intentionally kept compatible with models saved by
    src.modeling.classification.classifier_training.

    Args:
        classifier_name: Classifier identifier.
        data_source: Canonical data-source key, e.g. real or dpctgan/eps_1.0.
        model_type: Saved model variant, usually default or best.

    Returns:
        Tuple of (model, preprocessor, metadata, model_path).
    """
    model_path = classifier_model_path(
        classifier_name,
        data_source,
        model_type,
    )

    saved = _read_model_payload(model_path)
    _validate_payload_keys(saved, model_path)

    metadata = _validate_payload_metadata(
        metadata=saved["metadata"],
        model_path=model_path,
        classifier_name=classifier_name,
        data_source=data_source,
    )

    return saved["model"], saved["preprocessor"], metadata, model_path


def _build_run_parameters(
    classifier_name: str,
    data_source: str,
    model_type: str,
    params: Any,
    random_state: int,
    use_wandb: bool,
    model_path: Path,
) -> dict[str, Any]:
    """
    Build logger metadata for held-out utility evaluation.

    Field names are intentionally stable for result compatibility.
    """
    return {
        "pipeline_stage": "evaluation",
        "evaluation": "utility",
        "mode": "test",
        "data_source": data_source,
        "synthesizer": synthesizer_from_data_source(data_source),
        "epsilon": epsilon_from_data_source(data_source),
        "classifier": classifier_name,
        "model_type": model_type,
        "params": params,
        "random_state": random_state,
        "use_wandb": use_wandb,
        "model_path": model_path,
    }


def evaluate_utility(
    classifier_name: str,
    data_source: str,
    model_type: str,
    use_wandb: bool = False,
) -> None:
    """
    Evaluate a trained classifier on the real held-out test split.

    Utility is evaluated in a TSTR setup for synthetic sources:
    Train on Synthetic, Test on Real.
    """
    model, preprocessor, metadata, model_path = load_saved_model(
        classifier_name=classifier_name,
        data_source=data_source,
        model_type=model_type,
    )

    params = metadata.get("params", {})
    random_state = (
        params.get("seed", RANDOM_STATE) if isinstance(params, dict) else RANDOM_STATE
    )

    test_df, test_path = load_utility_test_dataset()

    X_test, y_test = prepare_single(
        classifier_name=classifier_name,
        df=test_df,
        preprocessor=preprocessor,
    )

    run_name = (
        f"eval_utility_{classifier_name}_{data_source.replace('/', '_')}_{model_type}"
    )

    parameters = _build_run_parameters(
        classifier_name=classifier_name,
        data_source=data_source,
        model_type=model_type,
        params=params,
        random_state=random_state,
        use_wandb=use_wandb,
        model_path=model_path,
    )

    print(
        f"[evaluate_utility] Evaluating saved {classifier_name} model on real "
        f"test data for source '{data_source}' with '{model_type}' hyperparameters"
    )

    with RunLogger(
        run_name=run_name,
        script_name=SCRIPT_NAME,
        parameters=parameters,
        use_wandb=use_wandb,
        category="utility",
    ) as logger:
        test_predictions = model.predict(X_test)

        test_metrics = compute_classification_metrics(
            y_test,
            test_predictions,
            prefix="test",
        )

        logger.log(test_metrics)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate saved classifiers on the real held-out test split."
    )
    parser.add_argument(
        "--classifier",
        choices=CLASSIFIERS,
        required=True,
        help="Classifier to evaluate.",
    )
    parser.add_argument(
        "--data_source",
        choices=["real"],
        default="real",
        help="Data source used to train the saved model. Default: real.",
    )
    parser.add_argument(
        "--synthesizer",
        choices=SYNTHESIZERS | DP_SYNTHESIZERS,
        default=None,
        help="Synthetic data source used to train the saved model.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        choices=DP_EPSILONS,
        default=None,
        help="Privacy budget for DP synthesizers. Required for dpctgan/patectgan.",
    )
    parser.add_argument(
        "--model_type",
        choices=["default", "best"],
        default="best",
        help="Which saved model variant to evaluate.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="Log results to Weights & Biases. Local JSON logging remains primary.",
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

    evaluate_utility(
        classifier_name=args.classifier,
        data_source=resolved_data_source,
        model_type=args.model_type,
        use_wandb=args.wandb,
    )


if __name__ == "__main__":
    main()
