"""
Classifier model factory for the synthetic data utility pipeline.

This module builds the supported scikit-learn classifiers from parameter
dictionaries used by default, sweep, and best-parameter training runs.

Supported classifiers:
  - logistic_regression
  - random_forest
  - gradient_boosting
"""

from typing import Any


from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.utility.constants import RANDOM_STATE

N_ESTIMATORS_RF = 300


def _normalize_max_depth(value: Any) -> int | None:
    """
    Normalize Random Forest max_depth values loaded from YAML or W&B configs.

    Sweep configurations may serialize None values as the string "None".
    This helper converts both None and "None" to Python None while ensuring
    numeric values are returned as integers.
    """
    if value in (None, "None"):
        return None

    return int(value)


def _build_logistic_regression(
    params: dict[str, Any],
    seed: int,
) -> LogisticRegression:
    """Build a Logistic Regression classifier."""
    regularization_strength = params.get("C", 1.0)

    return LogisticRegression(
        C=regularization_strength,
        max_iter=1000,
        random_state=seed,
    )


def _build_random_forest(
    params: dict[str, Any],
    seed: int,
) -> RandomForestClassifier:
    """Build a Random Forest classifier."""
    max_features = params.get("max_features", "sqrt")
    min_samples_leaf = params.get("min_samples_leaf", 1)
    max_depth = _normalize_max_depth(params.get("max_depth"))

    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS_RF,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=-1,
    )


def _build_gradient_boosting(
    params: dict[str, Any],
    seed: int,
) -> HistGradientBoostingClassifier:
    """
    Build a HistGradientBoostingClassifier.

    HistGradientBoostingClassifier natively handles missing values and
    categorical features when categorical columns are provided with the
    pandas categorical dtype.
    """
    learning_rate = params.get("learning_rate", 0.1)
    max_leaf_nodes = params.get("max_leaf_nodes", 31)
    min_samples_leaf = params.get("min_samples_leaf", 20)

    return HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        early_stopping=True,
        random_state=seed,
        categorical_features="from_dtype",
    )


MODEL_BUILDERS = {
    "logistic_regression": _build_logistic_regression,
    "random_forest": _build_random_forest,
    "gradient_boosting": _build_gradient_boosting,
}


def build_model(
    classifier_name: str,
    params: dict[str, Any],
) -> Any:
    """
    Build a classifier instance from a parameter dictionary.

    Args:
        classifier_name: One of:
            - logistic_regression
            - random_forest
            - gradient_boosting
        params: Dictionary of hyperparameters.

    Returns:
        Unfitted scikit-learn classifier instance.

    Raises:
        ValueError:
            If classifier_name is not recognised.
    """
    current_seed = params.get("seed", RANDOM_STATE)

    builder = MODEL_BUILDERS.get(classifier_name)

    if builder is None:
        available = sorted(MODEL_BUILDERS.keys())

        raise ValueError(
            f"Unknown classifier '{classifier_name}'. "
            f"Available classifiers: {available}"
        )

    return builder(params, current_seed)
