"""
Classifier model factory for the synthetic data utility pipeline.
"""

from collections.abc import Callable
from typing import Any, TypeAlias


from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression

from src.utility.constants import RANDOM_STATE

N_ESTIMATORS_RF = 300

ClassifierModel: TypeAlias = (
    LogisticRegression | RandomForestClassifier | HistGradientBoostingClassifier
)


def _normalize_max_depth(value: Any) -> int | None:
    """
    Normalize Random Forest max_depth values loaded from YAML or W&B configs.
    """
    if value in (None, "None"):
        return None

    return int(value)


def _build_logistic_regression(
    params: dict[str, Any],
    seed: int,
) -> LogisticRegression:
    """Build a Logistic Regression classifier."""
    return LogisticRegression(
        C=params.get("C", 1.0),
        max_iter=1000,
        random_state=seed,
    )


def _build_random_forest(
    params: dict[str, Any],
    seed: int,
) -> RandomForestClassifier:
    """Build a Random Forest classifier."""
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS_RF,
        max_features=params.get("max_features", "sqrt"),
        min_samples_leaf=params.get("min_samples_leaf", 1),
        max_depth=_normalize_max_depth(params.get("max_depth")),
        random_state=seed,
        n_jobs=-1,
    )


def _build_gradient_boosting(
    params: dict[str, Any],
    seed: int,
) -> HistGradientBoostingClassifier:
    """Build a HistGradientBoostingClassifier."""
    return HistGradientBoostingClassifier(
        learning_rate=params.get("learning_rate", 0.1),
        max_leaf_nodes=params.get("max_leaf_nodes", 31),
        random_state=seed,
    )


def build_model(
    classifier_name: str,
    params: dict[str, Any] | None = None,
    seed: int = RANDOM_STATE,
) -> ClassifierModel:
    """
    Build a supported classifier from a parameter dictionary.
    """
    params = params or {}

    builders: dict[str, Callable] = {
        "logistic_regression": _build_logistic_regression,
        "random_forest": _build_random_forest,
        "gradient_boosting": _build_gradient_boosting,
    }

    builder = builders.get(classifier_name)

    if builder is None:
        raise ValueError(
            f"Unsupported classifier '{classifier_name}'. "
            f"Available classifiers: {sorted(builders)}"
        )

    return builder(params, seed)
