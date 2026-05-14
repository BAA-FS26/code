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
from typing import Any

import pandas as pd

from src.core.data_source import (
    epsilon_from_data_source,
    resolve_training_data_source,
    synthesizer_from_data_source,
)
from src.core.io import load_csv
from src.core.paths import classifier_model_path, processed_split_path
from src.dataset.feature_engineering import prepare_data, prepare_data_gradient_boosting
from src.evaluation.metrics import compute_classification_metrics
from src.utility.constants import (
    CLASSIFIERS,
    DP_EPSILONS,
    DP_SYNTHESIZERS,
    RANDOM_STATE,
    SYNTHESIZERS,
    TEST_FILENAME,
)
from src.utility.logger import RunLogger

SCRIPT_NAME = "evaluate_utility.py"


def load_test() -> pd.DataFrame:
    """
    Load the real held-out test split.

    Always loads real test data regardless of the training data source.
    This implements the TSTR setup: train on synthetic, test on real.
    """
    test_path = processed_split_path(TEST_FILENAME)
    test_df = load_csv(test_path, "Test split")
    print(f"[evaluate_utility] Loaded test data ({len(test_df)} rows)")
    return test_df


def prepare_test_features(
    classifier_name: str,
    test_df: pd.DataFrame,
    preprocessor: Any,
) -> tuple[Any, Any]:
    """
    Apply saved-model preprocessing to the held-out test split.

    This mirrors the preprocessing used during training without importing from
    the training CLI module. Logistic Regression and Random Forest reuse the
    fitted preprocessor stored with the model. Gradient Boosting uses the
    Adult-specific categorical dtype preparation because it handles categorical
    features natively.
    """
    if classifier_name == "gradient_boosting":
        return prepare_data_gradient_boosting(test_df)

    if preprocessor is None:
        raise ValueError(
            f"Saved model for classifier '{classifier_name}' is missing a preprocessor."
        )

    return prepare_data(preprocessor, test_df)


def load_saved_model(
    classifier_name: str,
    data_source: str,
    model_type: str,
) -> tuple[Any, Any, dict[str, Any], object]:
    """
    Load and validate a saved classifier payload.

    Args:
        classifier_name: Classifier identifier.
        data_source: Canonical data-source key, e.g. real or dpctgan/eps_1.0.
        model_type: Saved model variant, usually default or best.

    Returns:
        Tuple of (model, preprocessor, metadata, model_path).
    """
    model_path = classifier_model_path(classifier_name, data_source, model_type)
    if not model_path.exists():
        raise FileNotFoundError(
            f"No saved model found at {model_path}. "
            "Run src.modeling.classify with --mode default or --mode best first."
        )

    with open(model_path, "rb") as f:
        saved = pickle.load(f)

    if not isinstance(saved, dict):
        raise ValueError(
            f"Saved model payload at {model_path} is invalid: expected dictionary."
        )

    missing_keys = {"model", "preprocessor", "metadata"} - set(saved.keys())
    if missing_keys:
        raise ValueError(
            f"Saved model payload at {model_path} is missing required keys: "
            f"{sorted(missing_keys)}"
        )

    metadata = saved["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError(
            f"Saved model metadata at {model_path} is invalid: expected dictionary."
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

    return saved["model"], saved["preprocessor"], metadata, model_path


def evaluate_utility(
    classifier_name: str,
    data_source: str,
    model_type: str,
    use_wandb: bool = False,
) -> None:
    """
    Evaluate a trained model on the real held-out test set.
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

    test_df = load_test()
    X_test, y_test = prepare_test_features(classifier_name, test_df, preprocessor)

    run_name = (
        f"eval_utility_{classifier_name}_{data_source.replace('/', '_')}_{model_type}"
    )
    parameters = {
        "pipeline_stage": "utility",
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
        logger.log(
            compute_classification_metrics(y_test, model.predict(X_test), prefix="test")
        )


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
