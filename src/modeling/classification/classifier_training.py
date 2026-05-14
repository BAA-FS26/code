"""
Classifier training and validation workflow.

This module trains classifiers on real or synthetic training data, evaluates
them on validation data, logs utility metrics, and saves trained model payloads
when requested.
"""

import pickle
from typing import Any

from src.core.data_source import epsilon_from_data_source, synthesizer_from_data_source
from src.core.paths import classifier_model_path
from src.dataset.adult_census import TARGET_COL
from src.evaluation.metrics import compute_classification_metrics
from src.modeling.classification.classifier_data import load_splits, prepare_splits
from src.modeling.classification.classifier_models import build_model
from src.utility.constants import MODELS_DIR, RANDOM_STATE
from src.utility.logger import RunLogger
from src.utility.utils import set_random_seeds

SCRIPT_NAME = "classifier_training.py"


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
    The test set is never loaded or evaluated in this function.

    For TSTR evaluation, the classifier trains on synthetic data but
    evaluates on real validation data.

    Args:
        classifier_name: One of 'logistic_regression', 'random_forest',
                         'gradient_boosting'.
        data_source: One of 'real', 'gaussian_copula', 'ctgan', 'tvae', or a DP source.
        params: Hyperparameter dictionary passed to build_model().
        run_name: Identifier for this run (used for local save and W&B).
        mode: Pipeline mode for result metadata.
        save_model: If True, save the fitted model and preprocessor to
                    models/ for later use in src.evaluation.evaluate_utility.
        use_wandb: Whether to log results to W&B. Defaults to False.
    """
    current_seed = params.get("seed", RANDOM_STATE)
    parameters = {
        "pipeline_stage": "utility",
        "evaluation": None,
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
            classifier_name, train_df, val_df
        )

        print(
            f"[classify] Training {classifier_name} on source '{data_source}' "
            f"in mode '{mode}'"
        )
        model = build_model(classifier_name, params)
        model.fit(X_train, y_train)

        train_metrics = compute_classification_metrics(
            y_train, model.predict(X_train), prefix="train"
        )
        val_metrics = compute_classification_metrics(
            y_val, model.predict(X_val), prefix="val"
        )
        logger.log(train_metrics)
        logger.log(val_metrics)

        if save_model:
            if model_type is None:
                raise ValueError("model_type must be provided when save_model=True.")
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            model_path = classifier_model_path(classifier_name, data_source, model_type)
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
            with open(model_path, "wb") as f:
                pickle.dump(payload, f)

            print(f"[classify] Model saved to {model_path.resolve()}")
            logger.log({"model_path": model_path})
