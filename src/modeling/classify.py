"""
classify.py

Binary classification pipeline for the Adult Census Income dataset.
Supports five modes:
  - default:    train with default hyperparameters and evaluate on val set
  - sweep:      run a W&B hyperparameter sweep using val set only
  - fetch_best: fetch best hyperparameters from latest sweep and save to YAML
  - best:       train with best hyperparameters, evaluate on val, save model
  - test:       evaluate saved model on test set (call once after tuning)

The test set is structurally separated from all tuning modes.
It is only loaded and evaluated in 'test' mode to prevent data leakage.

Usage:
    # Run with default parameters
    python classify.py --mode default --classifier logistic_regression --data_source real

    # Run hyperparameter sweep
    python classify.py --mode sweep --classifier logistic_regression --data_source real

    # Fetch best parameters from sweep and save to config/
    python classify.py --mode fetch_best --classifier logistic_regression --data_source real

    # Run with best parameters (saves model to disk)
    python classify.py --mode best --classifier logistic_regression --data_source real --params config/best_logistic_regression.yaml

    # Final test evaluation (once per classifier, after tuning is complete)
    python classify.py --mode test --classifier logistic_regression --data_source real
"""

import argparse
import pickle
import yaml
from pathlib import Path

import pandas as pd
import wandb
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.dataset.feature_engineering import (
    build_preprocessor_logistic_regression,
    build_preprocessor_random_forest,
    prepare_data_logistic_regression,
    prepare_data_random_forest,
    prepare_data_gradient_boosting,
    TARGET_COL,
)

# ── Constants ────────────────────────────────────────────────────────────────

WANDB_PROJECT = "synthetic-data-eval"
WANDB_ENTITY = "baa_fs26_pm"  # TODO: replace with your W&B entity
BASE_DIR   = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
CONFIG_DIR = BASE_DIR / "config"
MODELS_DIR = BASE_DIR / "models"
RANDOM_STATE = 42
N_ESTIMATORS_RF = 300  # Fixed, not tuned (more trees is always better for RF)

CLASSIFIERS = ["logistic_regression", "random_forest", "gradient_boosting"]
DATA_SOURCES = ["real"]  # extend with synthesizer names later

PARAM_ABBREVIATIONS = {
    "max_features": "mf",
    "min_samples_leaf": "msl",
    "max_depth": "md",
    "learning_rate": "lr",
    "max_leaf_nodes": "mln",
    "C": "C",
}


# ── Data loading ─────────────────────────────────────────────────────────────


def load_splits(data_source: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train and validation splits from disk.

    The test split is intentionally excluded. Use load_test() only after
    hyperparameter tuning is complete.

    Args:
        data_source: Data source identifier. Use 'real' for real data
                     baseline. Extend with synthesizer names for TSTR.

    Returns:
        Tuple of (train_df, val_df) DataFrames.
    """
    base = DATA_DIR / data_source if data_source != "real" else DATA_DIR
    train_df = pd.read_csv(base / "train.csv")
    val_df = pd.read_csv(base / "validation.csv")
    return train_df, val_df


def load_test(data_source: str) -> pd.DataFrame:
    """
    Load the test split from disk.

    This function must only be called once per classifier after
    hyperparameter tuning is complete. Calling it during tuning
    constitutes data leakage.

    Args:
        data_source: Data source identifier. Use 'real' for real data
                     baseline.

    Returns:
        Test split DataFrame.
    """
    base = DATA_DIR / data_source if data_source != "real" else DATA_DIR
    return pd.read_csv(base / "test.csv")


# ── Preprocessing ─────────────────────────────────────────────────────────────


def prepare_splits(
    classifier_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> tuple:
    """
    Apply classifier-specific preprocessing to train and val splits.

    The preprocessor is fitted on the training set only and applied
    consistently to the val split. Returns the fitted preprocessor
    so it can be reused on test or synthetic data.

    Args:
        classifier_name: One of 'logistic_regression', 'random_forest',
                         'gradient_boosting'.
        train_df: Training split DataFrame including target column.
        val_df: Validation split DataFrame including target column.

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, preprocessor).
        preprocessor is None for gradient_boosting.
    """
    if classifier_name == "logistic_regression":
        preprocessor = build_preprocessor_logistic_regression(
            train_df.drop(columns=[TARGET_COL])
        )
        X_train, y_train = prepare_data_logistic_regression(preprocessor, train_df)
        X_val, y_val = prepare_data_logistic_regression(preprocessor, val_df)

    elif classifier_name == "random_forest":
        preprocessor = build_preprocessor_random_forest(
            train_df.drop(columns=[TARGET_COL])
        )
        X_train, y_train = prepare_data_random_forest(preprocessor, train_df)
        X_val, y_val = prepare_data_random_forest(preprocessor, val_df)

    elif classifier_name == "gradient_boosting":
        preprocessor = None
        X_train, y_train = prepare_data_gradient_boosting(train_df)
        X_val, y_val = prepare_data_gradient_boosting(val_df)

    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")

    return X_train, y_train, X_val, y_val, preprocessor


def prepare_single(
    classifier_name: str,
    df: pd.DataFrame,
    preprocessor,
) -> tuple:
    """
    Apply classifier-specific preprocessing to a single DataFrame.

    Uses a previously fitted preprocessor to transform the data
    consistently. Used for test set evaluation and synthetic data.

    Args:
        classifier_name: One of 'logistic_regression', 'random_forest',
                         'gradient_boosting'.
        df: DataFrame including target column to transform.
        preprocessor: Fitted preprocessor from prepare_splits().
                      Pass None for gradient_boosting.

    Returns:
        Tuple of (X, y).
    """
    if classifier_name == "logistic_regression":
        return prepare_data_logistic_regression(preprocessor, df)
    elif classifier_name == "random_forest":
        return prepare_data_random_forest(preprocessor, df)
    elif classifier_name == "gradient_boosting":
        return prepare_data_gradient_boosting(df)
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")


# ── Model building ────────────────────────────────────────────────────────────


def build_model(classifier_name: str, params: dict):
    """
    Build a classifier instance from a parameter dictionary.

    Handles the 'None' string to Python None conversion for max_depth
    in RandomForestClassifier when loading from YAML.

    Args:
        classifier_name: One of 'logistic_regression', 'random_forest',
                         'gradient_boosting'.
        params: Dictionary of hyperparameters. When called from a W&B
                sweep, pass dict(wandb.config).

    Returns:
        Unfitted scikit-learn classifier instance.
    """
    if classifier_name == "logistic_regression":
        return LogisticRegression(
            C=params.get("C", 1.0),
            max_iter=1000,
            random_state=RANDOM_STATE,
        )

    elif classifier_name == "random_forest":
        max_depth = params.get("max_depth", None)
        if max_depth == "None":
            max_depth = None
        return RandomForestClassifier(
            n_estimators=N_ESTIMATORS_RF,
            max_features=params.get("max_features", "sqrt"),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            max_depth=max_depth,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    elif classifier_name == "gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=params.get("learning_rate", 0.1),
            max_leaf_nodes=params.get("max_leaf_nodes", 31),
            min_samples_leaf=params.get("min_samples_leaf", 20),
            early_stopping=True,
            random_state=RANDOM_STATE,
            categorical_features="from_dtype",
        )

    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")


# ── Metrics ───────────────────────────────────────────────────────────────────


def compute_metrics(y_true, y_pred, y_proba, prefix: str) -> dict:
    """
    Compute classification metrics for the positive class (>50K, label=1).

    Args:
        y_true: True binary labels.
        y_pred: Predicted binary labels.
        y_proba: Predicted probabilities for the positive class.
        prefix: Metric name prefix, e.g. 'train', 'val' or 'test'.

    Returns:
        Dictionary of metrics ready for wandb.log().
    """
    return {
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}_precision": precision_score(y_true, y_pred, zero_division=0),
        f"{prefix}_recall": recall_score(y_true, y_pred, zero_division=0),
        f"{prefix}_f1": f1_score(y_true, y_pred, zero_division=0),
        f"{prefix}_auc_roc": roc_auc_score(y_true, y_proba),
    }


# ── Train and evaluate ────────────────────────────────────────────────────────


def train_and_evaluate(
    classifier_name: str,
    data_source: str,
    params: dict,
    run_name: str,
    save_model: bool = False,
) -> None:
    """
    Train a classifier and evaluate on train and validation sets.

    Logs all metrics and hyperparameters to W&B. The test set is never
    loaded or evaluated in this function.

    Optionally saves the fitted model and preprocessor to disk so they
    can be reused in evaluate_on_test().

    Args:
        classifier_name: One of 'logistic_regression', 'random_forest',
                         'gradient_boosting'.
        data_source: Data source identifier, e.g. 'real'.
        params: Hyperparameter dictionary passed to build_model().
        run_name: W&B run name.
        save_model: If True, save the fitted model and preprocessor to
                    models/ for later use in evaluate_on_test().
    """
    with wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=run_name,
        config=params,
    ):
        train_df, val_df = load_splits(data_source)
        X_train, y_train, X_val, y_val, preprocessor = prepare_splits(
            classifier_name, train_df, val_df
        )

        model = build_model(classifier_name, params)
        model.fit(X_train, y_train)

        # Train metrics
        y_train_pred = model.predict(X_train)
        y_train_proba = model.predict_proba(X_train)[:, 1]
        wandb.log(compute_metrics(y_train, y_train_pred, y_train_proba, prefix="train"))

        # Val metrics
        y_val_pred = model.predict(X_val)
        y_val_proba = model.predict_proba(X_val)[:, 1]
        wandb.log(compute_metrics(y_val, y_val_pred, y_val_proba, prefix="val"))

        if save_model:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            model_path = MODELS_DIR / f"{data_source}_{classifier_name}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump({"model": model, "preprocessor": preprocessor}, f)
            print(f"Model saved to {model_path.resolve()}")


# ── Test evaluation ───────────────────────────────────────────────────────────


def evaluate_on_test(
    classifier_name: str,
    data_source: str,
) -> None:
    """
    Evaluate a trained model on the test set.

    Loads the fitted model and preprocessor saved by train_and_evaluate()
    with save_model=True. This function must only be called once per
    classifier after hyperparameter tuning is complete.

    Logs test metrics to W&B under a dedicated run.

    Args:
        classifier_name: One of 'logistic_regression', 'random_forest',
                         'gradient_boosting'.
        data_source: Data source identifier, e.g. 'real'.
    """
    model_path = MODELS_DIR / f"{data_source}_{classifier_name}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No saved model found at {model_path}. "
            "Run --mode best first to train and save the model."
        )

    with open(model_path, "rb") as f:
        saved = pickle.load(f)

    model = saved["model"]
    preprocessor = saved["preprocessor"]

    test_df = load_test(data_source)
    X_test, y_test = prepare_single(classifier_name, test_df, preprocessor)

    run_name = f"{data_source}_{classifier_name}_test"
    with wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=run_name,
    ):
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1]
        wandb.log(compute_metrics(y_test, y_test_pred, y_test_proba, prefix="test"))


# ── Fetch best params ─────────────────────────────────────────────────────────


def fetch_best_params(classifier_name: str, data_source: str) -> None:
    """
    Fetch the best hyperparameters from the latest W&B sweep and save
    them to config/best_{classifier_name}.yaml.

    Queries W&B for the run with the highest val_f1 among all sweep
    runs for the given classifier and data source, then saves the
    parameters to disk for use with --mode best.

    Args:
        classifier_name: One of 'logistic_regression', 'random_forest',
                         'gradient_boosting'.
        data_source: Data source identifier, e.g. 'real'.
    """
    api = wandb.Api()
    runs = api.runs(
        f"{WANDB_ENTITY}/{WANDB_PROJECT}",
        filters={
            "display_name": {"$regex": f"^{data_source}_{classifier_name}_sweep"},
        },
    )

    best_run = max(runs, key=lambda r: r.summary.get("val_f1", 0))
    best_params = {
        k: v
        for k, v in best_run.config.items()
        if not k.startswith("_") and k != "classifier"
    }

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    output_path = CONFIG_DIR / f"best_{classifier_name}.yaml"
    with open(output_path, "w") as f:
        yaml.dump(best_params, f)

    print(f"Best params saved to {output_path.resolve()}")
    print(f"Best val_f1: {best_run.summary.get('val_f1'):.4f}")
    print(f"Params: {best_params}")


# ── Sweep ─────────────────────────────────────────────────────────────────────


def run_sweep(classifier_name: str, data_source: str) -> None:
    """
    Initialise and run a W&B hyperparameter sweep for a classifier.

    Loads sweep configuration from config/sweep_{classifier_name}.yaml.
    The sweep optimises val_f1 and never touches the test set.
    Each run is named with abbreviated hyperparameter values for
    readability in the W&B UI.

    Args:
        classifier_name: One of 'logistic_regression', 'random_forest',
                         'gradient_boosting'.
        data_source: Data source identifier, e.g. 'real'.
    """
    config_path = CONFIG_DIR / f"sweep_{classifier_name}.yaml"
    with open(config_path) as f:
        sweep_config = yaml.safe_load(f)

    sweep_id = wandb.sweep(sweep_config, project=WANDB_PROJECT, entity=WANDB_ENTITY)

    def sweep_run():
        with wandb.init() as run:
            if run is not None:
                params = dict(wandb.config)
                param_str = "_".join(
                    f"{PARAM_ABBREVIATIONS.get(k, k)}={v}"
                    for k, v in params.items()
                    if k != "classifier"
                )
                run_name = f"{data_source}_{classifier_name}_sweep_{param_str}"

                run.name = run_name

                train_and_evaluate(
                    classifier_name=classifier_name,
                    data_source=data_source,
                    params=params,
                    run_name=run_name,
                    save_model=False,
                )

    wandb.agent(sweep_id, function=sweep_run)


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate classifiers for synthetic data utility evaluation."
    )
    parser.add_argument(
        "--mode",
        choices=["default", "sweep", "fetch_best", "best", "test"],
        required=True,
        help=(
            "default: train with default params, evaluate on train and val. "
            "sweep: run W&B hyperparameter sweep using val set only. "
            "fetch_best: fetch best params from sweep and save to config/. "
            "best: train with best params, evaluate on train and val, save model. "
            "test: evaluate saved model on test set (once after tuning)."
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
        choices=DATA_SOURCES,
        default="real",
        help="Data source to use. Default: real.",
    )
    parser.add_argument(
        "--params",
        type=str,
        default=None,
        help="Path to YAML file containing best hyperparameters. Required for --mode best.",
    )

    args = parser.parse_args()

    if args.mode == "default":
        train_and_evaluate(
            classifier_name=args.classifier,
            data_source=args.data_source,
            params={},
            run_name=f"{args.data_source}_{args.classifier}_default",
            save_model=False,
        )

    elif args.mode == "sweep":
        run_sweep(
            classifier_name=args.classifier,
            data_source=args.data_source,
        )

    elif args.mode == "fetch_best":
        fetch_best_params(
            classifier_name=args.classifier,
            data_source=args.data_source,
        )

    elif args.mode == "best":
        if args.params is None:
            raise ValueError("--params is required for --mode best.")
        params_path = CONFIG_DIR / Path(args.params).name
        with open(params_path) as f:
            best_params = yaml.safe_load(f)
        train_and_evaluate(
            classifier_name=args.classifier,
            data_source=args.data_source,
            params=best_params,
            run_name=f"{args.data_source}_{args.classifier}_best",
            save_model=True,
        )

    elif args.mode == "test":
        evaluate_on_test(
            classifier_name=args.classifier,
            data_source=args.data_source,
        )


if __name__ == "__main__":
    main()
