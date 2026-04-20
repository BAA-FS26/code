"""
classify.py

Binary classification pipeline for the Adult Census Income dataset.
Supports five modes:
  - default:    train with default hyperparameters and evaluate on val set
  - sweep:      run a W&B hyperparameter sweep using val set only (requires --wandb)
  - fetch_best: fetch best hyperparameters from latest sweep and save to config/
  - best:       train with best hyperparameters, evaluate on val, save model
  - test:       evaluate saved model on test set (call once after tuning)

For TSTR evaluation, pass a synthesizer name as --data_source. The classifier
will train on synthetic data and evaluate on real val and real test data.

The test set is structurally separated from all tuning modes.
It is only loaded and evaluated in 'test' mode to prevent data leakage.

Usage:
    # Real data baseline — no W&B required
    python classify.py --mode default --classifier logistic_regression --data_source real
    python classify.py --mode best --classifier logistic_regression --data_source real --params best_logistic_regression.yaml
    python classify.py --mode test --classifier logistic_regression --data_source real

    # With W&B logging
    python classify.py --mode default --classifier logistic_regression --data_source real --wandb

    # W&B sweep (requires --wandb)
    python classify.py --mode sweep --classifier logistic_regression --data_source real --wandb
    python classify.py --mode fetch_best --classifier logistic_regression --data_source real --wandb

    # TSTR with synthetic data
    python classify.py --mode default --classifier logistic_regression --data_source gaussian_copula
    python classify.py --mode test --classifier logistic_regression --data_source gaussian_copula
"""

import argparse
import pickle
import yaml
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.dataset.feature_engineering import (
    TARGET_COL,
    build_preprocessor_logistic_regression,
    build_preprocessor_random_forest,
    prepare_data,
    prepare_data_gradient_boosting,
)
from src.utility.constants import (
    BASE_DIR,
    CLASSIFIERS,
    DATA_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    SYNTHESIZERS,
)
from src.utility.logger import RunLogger
from src.utility.utils import set_random_seeds

# ── Constants ────────────────────────────────────────────────────────────────

CONFIG_DIR = BASE_DIR / "config"
DATA_SOURCES = ["real"] + SYNTHESIZERS
N_ESTIMATORS_RF = 300

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

    For real data, both train and val are loaded from data/processed/.
    For synthetic data (TSTR), the synthetic train split is loaded as
    training data while the real val split is used for evaluation.
    This ensures that hyperparameter tuning is always evaluated against
    real data distributions.

    The test split is intentionally excluded here. Use load_test() only
    after hyperparameter tuning is complete.

    Args:
        data_source: One of 'real', 'gaussian_copula', 'ctgan', 'tvae'.

    Returns:
        Tuple of (train_df, val_df) DataFrames.
    """
    if data_source == "real":
        train_df = pd.read_csv(DATA_DIR / "processed" / "train.csv")
    else:
        train_df = pd.read_csv(
            DATA_DIR / "synthetic" / data_source / "default" / "synthetic_train.csv"
        )
    val_df = pd.read_csv(DATA_DIR / "processed" / "validation.csv")
    return train_df, val_df


def load_test() -> pd.DataFrame:
    """
    Load the real test split from disk.

    Always loads real test data regardless of data source — TSTR
    evaluates on real test data by design. This function must only
    be called once per classifier after hyperparameter tuning is
    complete. Calling it during tuning constitutes data leakage.

    Returns:
        Real test split DataFrame.
    """
    return pd.read_csv(DATA_DIR / "processed" / "test.csv")


# ── Preprocessing ─────────────────────────────────────────────────────────────


def _get_preprocessor(classifier_name: str, train_df: pd.DataFrame):
    """
    Build and fit the appropriate preprocessor for a given classifier.

    Returns None for gradient_boosting, which handles preprocessing natively.

    Args:
        classifier_name: One of 'logistic_regression', 'random_forest',
                         'gradient_boosting'.
        train_df: Training DataFrame including the target column.

    Returns:
        Fitted ColumnTransformer, or None for gradient_boosting.
    """
    builders = {
        "logistic_regression": build_preprocessor_logistic_regression,
        "random_forest": build_preprocessor_random_forest,
    }
    builder = builders.get(classifier_name)
    if builder is None:
        return None
    return builder(train_df.drop(columns=[TARGET_COL]))


def _apply_preprocessor(classifier_name: str, df: pd.DataFrame, preprocessor) -> tuple:
    """
    Apply classifier-specific preprocessing to a DataFrame.

    Args:
        classifier_name: One of 'logistic_regression', 'random_forest',
                         'gradient_boosting'.
        df: DataFrame including the target column to transform.
        preprocessor: Fitted preprocessor, or None for gradient_boosting.

    Returns:
        Tuple of (X, y).
    """
    if classifier_name == "gradient_boosting":
        return prepare_data_gradient_boosting(df)
    return prepare_data(preprocessor, df)


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
    preprocessor = _get_preprocessor(classifier_name, train_df)
    X_train, y_train = _apply_preprocessor(classifier_name, train_df, preprocessor)
    X_val, y_val = _apply_preprocessor(classifier_name, val_df, preprocessor)
    return X_train, y_train, X_val, y_val, preprocessor


def prepare_single(classifier_name: str, df: pd.DataFrame, preprocessor) -> tuple:
    """
    Apply classifier-specific preprocessing to a single DataFrame.

    Uses a previously fitted preprocessor to transform the data
    consistently. Used for test set evaluation.

    Args:
        classifier_name: One of 'logistic_regression', 'random_forest',
                         'gradient_boosting'.
        df: DataFrame including target column to transform.
        preprocessor: Fitted preprocessor from prepare_splits().
                      Pass None for gradient_boosting.

    Returns:
        Tuple of (X, y).
    """
    return _apply_preprocessor(classifier_name, df, preprocessor)


# ── Model building ────────────────────────────────────────────────────────────


def build_model(classifier_name: str, params: dict):
    """
    Build a classifier instance from a parameter dictionary.

    Args:
        classifier_name: One of 'logistic_regression', 'random_forest',
                         'gradient_boosting'.
        params: Dictionary of hyperparameters.

    Returns:
        Unfitted scikit-learn classifier instance.

    Raises:
        ValueError: If classifier_name is not recognised.
    """
    current_seed = params.get("seed", RANDOM_STATE)

    if classifier_name == "logistic_regression":
        return LogisticRegression(
            C=params.get("C", 1.0),
            max_iter=1000,
            random_state=current_seed,
        )

    if classifier_name == "random_forest":
        # max_depth may arrive as the string "None" when loaded from YAML
        raw_depth = params.get("max_depth", None)
        max_depth = None if raw_depth in (None, "None") else int(raw_depth)
        return RandomForestClassifier(
            n_estimators=N_ESTIMATORS_RF,
            max_features=params.get("max_features", "sqrt"),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            max_depth=max_depth,
            random_state=current_seed,
            n_jobs=-1,
        )

    if classifier_name == "gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=params.get("learning_rate", 0.1),
            max_leaf_nodes=params.get("max_leaf_nodes", 31),
            min_samples_leaf=params.get("min_samples_leaf", 20),
            early_stopping=True,
            random_state=current_seed,
            categorical_features="from_dtype",
        )

    raise ValueError(f"Unknown classifier: {classifier_name}")


# ── Metrics ───────────────────────────────────────────────────────────────────


def compute_metrics(y_true, y_pred, prefix: str) -> dict:
    """
    Compute macro-averaged classification metrics.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted labels.
        prefix: Prefix for metric keys (e.g. 'train', 'val', 'test').

    Returns:
        Dictionary of prefixed metric names to scalar values.
    """
    return {
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}_precision_macro": precision_score(
            y_true, y_pred, zero_division=0, average="macro"
        ),
        f"{prefix}_recall_macro": recall_score(
            y_true, y_pred, zero_division=0, average="macro"
        ),
        f"{prefix}_f1_macro": f1_score(
            y_true, y_pred, zero_division=0, average="macro"
        ),
    }


# ── Train and evaluate ────────────────────────────────────────────────────────


def train_and_evaluate(
    classifier_name: str,
    data_source: str,
    params: dict,
    run_name: str,
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
        data_source: One of 'real', 'gaussian_copula', 'ctgan', 'tvae'.
        params: Hyperparameter dictionary passed to build_model().
        run_name: Identifier for this run (used for local save and W&B).
        save_model: If True, save the fitted model and preprocessor to
                    models/ for later use in evaluate_on_test().
        use_wandb: Whether to log results to W&B. Defaults to False.
    """
    config = {**params, "classifier": classifier_name, "data_source": data_source}

    with RunLogger(run_name=run_name, config=config, use_wandb=use_wandb) as logger:
        current_seed = params.get("seed", RANDOM_STATE)
        set_random_seeds(current_seed)

        train_df, val_df = load_splits(data_source)
        X_train, y_train, X_val, y_val, preprocessor = prepare_splits(
            classifier_name, train_df, val_df
        )

        model = build_model(classifier_name, params)
        model.fit(X_train, y_train)

        logger.log(compute_metrics(y_train, model.predict(X_train), prefix="train"))
        logger.log(compute_metrics(y_val, model.predict(X_val), prefix="val"))

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
    use_wandb: bool = False,
) -> None:
    """
    Evaluate a trained model on the real test set.

    Loads the fitted model and preprocessor saved by train_and_evaluate()
    with save_model=True. Always evaluates on real test data regardless
    of data source — TSTR evaluates on real test data by design.

    This function must only be called once per classifier after
    hyperparameter tuning is complete.

    Args:
        classifier_name: One of 'logistic_regression', 'random_forest',
                         'gradient_boosting'.
        data_source: One of 'real', 'gaussian_copula', 'ctgan', 'tvae'.
        use_wandb: Whether to log results to W&B. Defaults to False.
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
    test_df = load_test()
    X_test, y_test = prepare_single(classifier_name, test_df, preprocessor)

    run_name = f"{data_source}_{classifier_name}_test"
    config = {"classifier": classifier_name, "data_source": data_source}

    with RunLogger(run_name=run_name, config=config, use_wandb=use_wandb) as logger:
        logger.log(compute_metrics(y_test, model.predict(X_test), prefix="test"))


# ── Fetch best params (W&B only) ──────────────────────────────────────────────


def fetch_best_params(classifier_name: str, data_source: str) -> None:
    """
    Fetch the best hyperparameters from the latest W&B sweep and save
    them to config/best_{classifier_name}.yaml.

    Queries W&B for the run with the highest val_f1_macro among all sweep
    runs for the given classifier and data source, then saves the
    parameters to disk for use with --mode best.

    Requires W&B to be configured (WANDB_ENTITY must be set).

    Args:
        classifier_name: One of 'logistic_regression', 'random_forest',
                         'gradient_boosting'.
        data_source: One of 'real', 'gaussian_copula', 'ctgan', 'tvae'.
    """
    import wandb
    from src.utility.wandb_config import get_wandb_entity, get_wandb_project

    api = wandb.Api()
    runs = api.runs(
        f"{get_wandb_entity()}/{get_wandb_project()}",
        filters={"display_name": {"$regex": f"^{data_source}_{classifier_name}_sweep"}},
    )

    best_run = max(runs, key=lambda r: r.summary.get("val_f1_macro", 0))
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
    print(f"Best val_f1_macro: {best_run.summary.get('val_f1_macro'):.4f}")
    print(f"Params: {best_params}")


# ── Sweep (W&B only) ──────────────────────────────────────────────────────────


def run_sweep(classifier_name: str, data_source: str) -> None:
    """
    Initialise and run a W&B hyperparameter sweep for a classifier.

    Loads sweep configuration from config/sweep_{classifier_name}.yaml.
    The sweep optimises val_f1_macro and never touches the test set.
    Each run is named with abbreviated hyperparameter values for
    readability in the W&B UI.

    Requires W&B to be configured (WANDB_ENTITY must be set).

    Args:
        classifier_name: One of 'logistic_regression', 'random_forest',
                         'gradient_boosting'.
        data_source: One of 'real', 'gaussian_copula', 'ctgan', 'tvae'.
    """
    import wandb
    from src.utility.wandb_config import get_wandb_entity, get_wandb_project

    config_path = CONFIG_DIR / f"sweep_{classifier_name}.yaml"
    with open(config_path) as f:
        sweep_config = yaml.safe_load(f)

    sweep_id = wandb.sweep(
        sweep_config,
        project=get_wandb_project(),
        entity=get_wandb_entity(),
    )

    def sweep_run():
        with wandb.init() as run:
            if run is not None:
                params = dict(wandb.config)
                param_str = "_".join(
                    f"{PARAM_ABBREVIATIONS.get(k, k)}={v}"
                    for k, v in params.items()
                    if k not in ["classifier", "seed"]
                )
                run_name = f"{data_source}_{classifier_name}_sweep_{param_str}"
                run.name = run_name
                train_and_evaluate(
                    classifier_name=classifier_name,
                    data_source=data_source,
                    params=params,
                    run_name=run_name,
                    save_model=False,
                    use_wandb=True,
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
            "sweep: run W&B hyperparameter sweep (requires --wandb). "
            "fetch_best: fetch best params from W&B sweep (requires --wandb). "
            "best: train with best params, evaluate on train and val, save model. "
            "test: evaluate saved model on real test set (once after tuning)."
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
            "Log results to Weights & Biases. Required for sweep and fetch_best modes. "
            "Requires WANDB_ENTITY to be set in the environment."
        ),
    )

    args = parser.parse_args()

    if args.mode in ("sweep", "fetch_best") and not args.wandb:
        parser.error(f"--mode {args.mode} requires --wandb.")

    if args.mode == "default":
        train_and_evaluate(
            classifier_name=args.classifier,
            data_source=args.data_source,
            params={"seed": RANDOM_STATE},
            run_name=f"{args.data_source}_{args.classifier}_default",
            use_wandb=args.wandb,
        )

    elif args.mode == "sweep":
        run_sweep(classifier_name=args.classifier, data_source=args.data_source)

    elif args.mode == "fetch_best":
        fetch_best_params(classifier_name=args.classifier, data_source=args.data_source)

    elif args.mode == "best":
        if args.params is None:
            parser.error("--params is required for --mode best.")
        params_path = CONFIG_DIR / Path(args.params).name
        with open(params_path) as f:
            best_params = yaml.safe_load(f)
        train_and_evaluate(
            classifier_name=args.classifier,
            data_source=args.data_source,
            params=best_params,
            run_name=f"{args.data_source}_{args.classifier}_best",
            save_model=True,
            use_wandb=args.wandb,
        )

    elif args.mode == "test":
        evaluate_on_test(
            classifier_name=args.classifier,
            data_source=args.data_source,
            use_wandb=args.wandb,
        )


if __name__ == "__main__":
    main()
