"""
Classifier training and validation workflow.

This module trains classifiers on real or synthetic training data, evaluates
them on validation data, logs utility metrics, and saves trained model payloads
when requested.
"""

import pickle
from typing import Any

from src.core.data_source import (
    epsilon_from_data_source,
    synthesizer_from_data_source,
)
from src.core.paths import classifier_model_path
from src.dataset.adult_census import TARGET_COL
from src.evaluation.metrics import compute_classification_metrics
from src.modeling.classification.classifier_data import (
    load_splits,
    prepare_splits,
)
from src.modeling.classification.classifier_models import build_model
from src.utility.constants import MODELS_DIR, RANDOM_STATE
from src.utility.logger import RunLogger
from src.utility.utils import set_random_seeds

SCRIPT_NAME = "classifier_training.py"


def _build_run_parameters(
    classifier_name: str,
    data_source: str,
    params: dict[str, Any],
    mode: str,
    model_type: str | None,
    save_model: bool,
    use_wandb: bool,
    current_seed: int,
) -> dict[str, Any]:
    """
    Build logger metadata for a classifier training run.

    IMPORTANT:
    The returned structure must remain stable because downstream dashboard
    loading and result aggregation depend on these fields.
    """
    return {
        "pipeline_stage": "classification",
        "evaluation": "validation_utility",
        "mode": mode,
        "data_source": data_source,
        "synthesizer": synthesizer_from_data_source(data_source),
        "epsilon": epsilon_from_data_source(data_source),
        "classifier": classifier_name,
        "model_type": model_type,
        "params": params,
        "random_state": current_seed,
        "use_wandb": use_wandb,
        "save_model": save_model,
    }


def _save_model_payload(
    model: Any,
    preprocessor: Any,
    classifier_name: str,
    data_source: str,
    params: dict[str, Any],
    current_seed: int,
    model_type: str,
) -> None:
    """
    Save a trained classifier payload to disk.

    The payload intentionally stores both:
    - the fitted classifier
    - the fitted preprocessing pipeline

    This guarantees consistent preprocessing during later held-out test
    evaluation in src.evaluation.evaluate_utility.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = classifier_model_path(
        classifier_name,
        data_source,
        model_type,
    )

    payload = {
        "model": model,
        "preprocessor": preprocessor,
        "metadata": {
            "classifier": classifier_name,
            "data_source": data_source,
            "params": params,
            "seed": current_seed,
            "target_col": TARGET_COL,
        },
    }

    with open(model_path, "wb") as file:
        pickle.dump(payload, file)

    print(f"[classify] Model saved to {model_path.resolve()}")


def train_and_validate(
    classifier_name: str,
    data_source: str,
    params: dict[str, Any],
    run_name: str,
    mode: str,
    model_type: str | None = None,
    save_model: bool = False,
    use_wandb: bool = False,
) -> None:
    """
    Train a classifier and evaluate on train and validation sets.

    Results are always saved locally. W&B logging is optional.
    The held-out test set is never loaded in this function.

    For TSTR evaluation, the classifier trains on synthetic data but
    evaluates on real validation data.

    Args:
        classifier_name:
            One of:
                - logistic_regression
                - random_forest
                - gradient_boosting

        data_source:
            Real or synthetic training source.

        params:
            Hyperparameter dictionary passed to build_model().

        run_name:
            Identifier used for local result logging and optional W&B runs.

        mode:
            Pipeline mode (default, sweep, best, etc.).

        model_type:
            Optional model label used for saved model naming.

        save_model:
            If True, persist the trained model payload to disk.

        use_wandb:
            Whether W&B logging is enabled.
    """
    current_seed = params.get("seed", RANDOM_STATE)

    parameters = _build_run_parameters(
        classifier_name=classifier_name,
        data_source=data_source,
        params=params,
        mode=mode,
        model_type=model_type,
        save_model=save_model,
        use_wandb=use_wandb,
        current_seed=current_seed,
    )

    with RunLogger(
        run_name=run_name,
        script_name=SCRIPT_NAME,
        parameters=parameters,
        use_wandb=use_wandb,
        category="utility",
    ) as logger:

        set_random_seeds(current_seed)

        train_df, val_df = load_splits(data_source)

        X_train, y_train, X_val, y_val, preprocessor = prepare_splits(
            classifier_name,
            train_df,
            val_df,
        )

        print(
            f"[classify] Training {classifier_name} "
            f"on source '{data_source}' "
            f"in mode '{mode}'"
        )

        model = build_model(classifier_name, params)

        model.fit(X_train, y_train)

        train_predictions = model.predict(X_train)
        val_predictions = model.predict(X_val)

        train_metrics = compute_classification_metrics(
            y_train,
            train_predictions,
            prefix="train",
        )

        val_metrics = compute_classification_metrics(
            y_val,
            val_predictions,
            prefix="val",
        )

        logger.log(train_metrics)
        logger.log(val_metrics)

        if save_model:
            if model_type is None:
                raise ValueError("model_type must be provided when save_model=True.")

            _save_model_payload(
                model=model,
                preprocessor=preprocessor,
                classifier_name=classifier_name,
                data_source=data_source,
                params=params,
                current_seed=current_seed,
                model_type=model_type,
            )

            model_path = classifier_model_path(
                classifier_name,
                data_source,
                model_type,
            )

            logger.log({"model_path": model_path})
