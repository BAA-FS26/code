"""
classify.py

Binary classification pipeline for the Adult Census Income dataset.

Supports five modes:
  - default:    train with default hyperparameters and evaluate on val set
  - sweep:      run a W&B hyperparameter sweep using val set only (requires --wandb)
  - fetch_best: fetch best hyperparameters from latest sweep and save to config/
  - best:       train with best hyperparameters, evaluate on val, save model to disk
  - test:       load model saved by --mode best and evaluate on real test set
                (call once after tuning)

For TSTR evaluation, pass a synthesizer name as --data_source. The classifier
will train on synthetic data and evaluate on real val and real test data.

The test set is structurally separated from all tuning modes.
It is only loaded and evaluated in 'test' mode to prevent data leakage.

Usage:
    # Real data baseline — no W&B required
    python -m src.modeling.classify --mode default --classifier logistic_regression --data_source real
    python -m src.modeling.classify --mode best --classifier logistic_regression --data_source real --params best_logistic_regression_real.yaml
    python -m src.modeling.classify --mode test --classifier logistic_regression --data_source real

    # With W&B logging
    python -m src.modeling.classify --mode default --classifier logistic_regression --data_source real --wandb

    # W&B sweep (requires --wandb)
    python -m src.modeling.classify --mode sweep --classifier logistic_regression --data_source real --wandb
    python -m src.modeling.classify --mode fetch_best --classifier logistic_regression --data_source real --wandb

    # TSTR with non-DP synthetic data
    python -m src.modeling.classify --mode default --classifier logistic_regression --synthesizer gaussian_copula
    python -m src.modeling.classify --mode best --classifier logistic_regression --synthesizer gaussian_copula --params best_logistic_regression_gaussian_copula.yaml
    python -m src.modeling.classify --mode test --classifier logistic_regression --synthesizer gaussian_copula

    # TSTR with DP synthetic data — --mode best must be run before --mode test
    python -m src.modeling.classify --mode default --classifier logistic_regression --synthesizer dpctgan --epsilon 1.0
    python -m src.modeling.classify --mode best --classifier logistic_regression --synthesizer dpctgan --epsilon 1.0 --params best_logistic_regression_dpctgan_eps_1.0.yaml
    python -m src.modeling.classify --mode test --classifier logistic_regression --synthesizer dpctgan --epsilon 1.0
"""

import argparse
import pickle
from pathlib import Path
from typing import Any, cast

import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

try:
    import wandb as _wandb
except ImportError:
    _wandb = None

from src.dataset.adult_census import TARGET_COL
from src.dataset.feature_engineering import (
    build_preprocessor_logistic_regression,
    build_preprocessor_random_forest,
    prepare_data,
    prepare_data_gradient_boosting,
)
from src.utility.constants import (
    CLASSIFIERS,
    CONFIG_DIR,
    DP_EPSILONS,
    DP_SYNTHESIZERS,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    SYNTHESIZERS,
    SYNTHETIC_DATA_DIR,
    SYNTHETIC_TRAIN_FILENAME,
    TEST_FILENAME,
    TRAIN_FILENAME,
    VALIDATION_FILENAME,
)
from src.utility.logger import RunLogger
from src.utility.utils import set_random_seeds
from src.utility.wandb_config import (
    get_wandb_entity,
    get_wandb_project,
    require_wandb_config,
)

# ── Constants ────────────────────────────────────────────────────────────────

SCRIPT_NAME = "classify.py"

N_ESTIMATORS_RF = 300

PARAM_ABBREVIATIONS = {
    "max_features": "mf",
    "min_samples_leaf": "msl",
    "max_depth": "md",
    "learning_rate": "lr",
    "max_leaf_nodes": "mln",
    "C": "C",
}


# ── Path helpers ──────────────────────────────────────────────────────────────


def _processed_split_path(filename: str) -> Path:
    """Return the canonical path to a processed split file."""
    return PROCESSED_DATA_DIR / filename


def _synthetic_train_path(data_source: str) -> Path:
    """
    Return the canonical synthetic training data path for a data source.
    """
    if "/" in data_source:
        return SYNTHETIC_DATA_DIR / data_source / SYNTHETIC_TRAIN_FILENAME
    return SYNTHETIC_DATA_DIR / data_source / "default" / SYNTHETIC_TRAIN_FILENAME


def _saved_model_path(classifier_name: str, data_source: str) -> Path:
    """Return the canonical saved model path."""
    return MODELS_DIR / f"{data_source.replace('/', '_')}_{classifier_name}.pkl"


def _best_params_path(classifier_name: str, data_source: str) -> Path:
    """Return the canonical best-params config path."""
    safe_source = data_source.replace("/", "_")
    return CONFIG_DIR / f"best_{classifier_name}_{safe_source}.yaml"


def _resolve_data_source(
    data_source: str,
    synthesizer: str | None,
    epsilon: float | None,
) -> str:
    """
    Resolve the canonical internal data-source key used by paths and model names.

    Returns:
        One of:
        - "real"
        - "<non-dp-synthesizer>"
        - "<dp-synthesizer>/eps_<epsilon>"
    """
    if synthesizer is None:
        if epsilon is not None:
            raise ValueError("--epsilon should only be used with DP synthesizers.")
        return data_source

    if data_source != "real":
        raise ValueError(
            "--synthesizer cannot be combined with a non-real --data_source."
        )

    if synthesizer in DP_SYNTHESIZERS:
        if epsilon is None:
            raise ValueError("--epsilon is required for DP synthesizers.")
        return f"{synthesizer}/eps_{epsilon}"

    if epsilon is not None:
        raise ValueError("--epsilon should only be used with DP synthesizers.")

    return synthesizer


# ── Validation helpers ────────────────────────────────────────────────────────


def _require_wandb_support() -> Any:
    """
    Validate that W&B package support and environment configuration exist.

    Returns:
        The imported wandb module.

    Raises:
        RuntimeError: If wandb is unavailable or required configuration is missing.
    """
    if _wandb is None:
        raise RuntimeError(
            "The 'wandb' package is not installed, but a W&B mode was requested. "
            "Install project dependencies including wandb to use sweep or fetch_best."
        )
    require_wandb_config()
    return cast(Any, _wandb)


def _load_csv(path: Path, description: str) -> pd.DataFrame:
    """
    Load a CSV from disk with a clear, pipeline-friendly error message.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found at {path}. "
            "Run the required earlier pipeline step first."
        )
    return pd.read_csv(path)


def _validate_dataframe_schema(
    df: pd.DataFrame,
    expected_columns: list[str],
    dataframe_name: str,
) -> None:
    """
    Validate that a DataFrame has exactly the expected columns in order.

    Raises:
        ValueError: If columns do not match.
    """
    actual_columns = list(df.columns)
    if actual_columns != expected_columns:
        raise ValueError(
            f"{dataframe_name} columns do not match expected schema.\n"
            f"Expected: {expected_columns}\n"
            f"Actual:   {actual_columns}"
        )


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


# ── Data loading ─────────────────────────────────────────────────────────────


def load_splits(data_source: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train and validation splits from disk.

    For real data, both train and val are loaded from processed data.
    For synthetic data (TSTR), the synthetic train split is loaded as
    training data while the real val split is used for evaluation.
    This ensures that hyperparameter tuning is always evaluated against
    real data distributions.

    The test split is intentionally excluded here. Use load_test() only
    after hyperparameter tuning is complete.

    Args:
        data_source: One of 'real', 'gaussian_copula', 'ctgan', 'tvae', or a
                     DP source in the form 'dpctgan/eps_1.0'.

    Returns:
        Tuple of (train_df, val_df) DataFrames.

    Raises:
        FileNotFoundError: If required input files are missing.
        ValueError: If loaded training data schema does not match validation schema.
    """
    if data_source == "real":
        train_path = _processed_split_path(TRAIN_FILENAME)
    else:
        train_path = _synthetic_train_path(data_source)

    val_path = _processed_split_path(VALIDATION_FILENAME)

    train_df = _load_csv(train_path, "Training split")
    val_df = _load_csv(val_path, "Validation split")

    _validate_dataframe_schema(
        train_df,
        expected_columns=list(val_df.columns),
        dataframe_name=f"Training data for source '{data_source}'",
    )

    print(
        f"[classify] Loaded training data for source '{data_source}' "
        f"({len(train_df)} rows)"
    )
    print(f"[classify] Loaded validation data ({len(val_df)} rows)")

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

    Raises:
        FileNotFoundError: If the canonical test split is missing.
    """
    test_path = _processed_split_path(TEST_FILENAME)
    test_df = _load_csv(test_path, "Test split")
    print(f"[classify] Loaded test data ({len(test_df)} rows)")
    return test_df


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


def _apply_preprocessor(
    classifier_name: str,
    df: pd.DataFrame,
    preprocessor,
) -> tuple:
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


def build_model(classifier_name: str, params: dict[str, Any]):
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


def compute_metrics(y_true, y_pred, prefix: str) -> dict[str, float]:
    """
    Compute macro-averaged classification metrics.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted labels.
        prefix: Prefix for metric keys (e.g. 'train', 'val', 'test').

    Returns:
        Dictionary of prefixed metric names to scalar float values.
    """
    return {
        f"{prefix}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}_precision_macro": float(
            precision_score(y_true, y_pred, zero_division=0, average="macro")
        ),
        f"{prefix}_recall_macro": float(
            recall_score(y_true, y_pred, zero_division=0, average="macro")
        ),
        f"{prefix}_f1_macro": float(
            f1_score(y_true, y_pred, zero_division=0, average="macro")
        ),
    }


# ── Train and evaluate ────────────────────────────────────────────────────────


def train_and_evaluate(
    classifier_name: str,
    data_source: str,
    params: dict[str, Any],
    run_name: str,
    mode: str,
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
                    models/ for later use in evaluate_on_test().
        use_wandb: Whether to log results to W&B. Defaults to False.
    """
    current_seed = params.get("seed", RANDOM_STATE)
    parameters = {
        "mode": mode,
        "classifier": classifier_name,
        "data_source": data_source,
        "params": params,
        "seed": current_seed,
        "save_model": save_model,
        "use_wandb": use_wandb,
    }

    with RunLogger(
        run_name=run_name,
        script_name=SCRIPT_NAME,
        parameters=parameters,
        use_wandb=use_wandb,
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

        train_metrics = compute_metrics(y_train, model.predict(X_train), prefix="train")
        val_metrics = compute_metrics(y_val, model.predict(X_val), prefix="val")
        logger.log(train_metrics)
        logger.log(val_metrics)

        if save_model:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            model_path = _saved_model_path(classifier_name, data_source)
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
        data_source: One of 'real', 'gaussian_copula', 'ctgan', 'tvae', or a DP source.
        use_wandb: Whether to log results to W&B. Defaults to False.
    """
    model_path = _saved_model_path(classifier_name, data_source)
    if not model_path.exists():
        raise FileNotFoundError(
            f"No saved model found at {model_path}. "
            "Run --mode best first to train and save the model."
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

    model = saved["model"]
    preprocessor = saved["preprocessor"]
    test_df = load_test()
    X_test, y_test = prepare_single(classifier_name, test_df, preprocessor)

    run_name = f"eval_utility_{classifier_name}_{data_source.replace('/', '_')}"
    parameters = {
        "mode": "test",
        "classifier": classifier_name,
        "data_source": data_source,
        "model_path": model_path,
        "use_wandb": use_wandb,
    }

    print(
        f"[classify] Evaluating saved {classifier_name} model on real test data "
        f"for source '{data_source}'"
    )

    with RunLogger(
        run_name=run_name,
        script_name=SCRIPT_NAME,
        parameters=parameters,
        use_wandb=use_wandb,
    ) as logger:
        logger.log(compute_metrics(y_test, model.predict(X_test), prefix="test"))


# ── Fetch best params (W&B only) ──────────────────────────────────────────────


def fetch_best_params(classifier_name: str, data_source: str) -> None:
    """
    Fetch the best hyperparameters from the latest W&B sweep and save
    them to config/best_{classifier}_{data_source}.yaml.

    Queries W&B for the run with the highest val_f1_macro among all sweep
    runs for the given classifier and data source, then saves the
    parameters to disk for use with --mode best.

    Requires W&B to be configured.
    """
    wandb_module = _require_wandb_support()
    api = wandb_module.Api()
    runs = list(
        api.runs(
            f"{get_wandb_entity()}/{get_wandb_project()}",
            filters={
                "display_name": {"$regex": f"^{data_source}_{classifier_name}_sweep"}
            },
        )
    )

    if not runs:
        raise RuntimeError(
            f"No W&B sweep runs found for classifier '{classifier_name}' "
            f"and data source '{data_source}'. Run --mode sweep first."
        )

    best_run = max(runs, key=lambda r: r.summary.get("val_f1_macro", 0))
    best_params = {
        k: v
        for k, v in best_run.config.items()
        if not k.startswith("_") and k != "classifier"
    }

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    output_path = _best_params_path(classifier_name, data_source)
    with open(output_path, "w") as f:
        yaml.safe_dump(best_params, f, sort_keys=True)

    print(f"[classify] Best params saved to {output_path.resolve()}")
    print(f"[classify] Best val_f1_macro: {best_run.summary.get('val_f1_macro'):.4f}")
    print(f"[classify] Params: {best_params}")


# ── Sweep (W&B only) ──────────────────────────────────────────────────────────


def run_sweep(classifier_name: str, data_source: str) -> None:
    """
    Initialise and run a W&B hyperparameter sweep for a classifier.

    Loads sweep configuration from config/sweep_{classifier_name}.yaml.
    The sweep optimises val_f1_macro and never touches the test set.
    Each run is named with abbreviated hyperparameter values for
    readability in the W&B UI.

    Requires W&B to be configured.
    """
    wandb_module = _require_wandb_support()

    config_path = CONFIG_DIR / f"sweep_{classifier_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Sweep configuration not found at {config_path}. "
            "Ensure the corresponding sweep YAML file exists."
        )

    with open(config_path) as f:
        sweep_config = yaml.safe_load(f)

    if not isinstance(sweep_config, dict):
        raise ValueError(
            f"Sweep configuration at {config_path} must load as a dictionary."
        )

    print(
        f"[classify] Starting W&B sweep for classifier '{classifier_name}' "
        f"on source '{data_source}'"
    )

    sweep_id = wandb_module.sweep(
        sweep_config,
        project=get_wandb_project(),
        entity=get_wandb_entity(),
    )

    def sweep_run() -> None:
        with wandb_module.init() as run:
            if run is None:
                raise RuntimeError("wandb.init() returned None during sweep run.")

            params = dict(wandb_module.config)
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
                mode="sweep",
                save_model=False,
                use_wandb=True,
            )

    wandb_module.agent(sweep_id, function=sweep_run)


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
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
            "test: load model saved by --mode best and evaluate on real test set."
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
        choices=SYNTHESIZERS + DP_SYNTHESIZERS,
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
        resolved_data_source = _resolve_data_source(
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
        train_and_evaluate(
            classifier_name=args.classifier,
            data_source=resolved_data_source,
            params={"seed": RANDOM_STATE},
            run_name=f"{resolved_data_source.replace('/', '_')}_{args.classifier}_default",
            mode="default",
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
        train_and_evaluate(
            classifier_name=args.classifier,
            data_source=resolved_data_source,
            params=best_params,
            run_name=f"{resolved_data_source.replace('/', '_')}_{args.classifier}_best",
            mode="best",
            save_model=True,
            use_wandb=args.wandb,
        )

    elif args.mode == "test":
        evaluate_on_test(
            classifier_name=args.classifier,
            data_source=resolved_data_source,
            use_wandb=args.wandb,
        )


if __name__ == "__main__":
    main()
