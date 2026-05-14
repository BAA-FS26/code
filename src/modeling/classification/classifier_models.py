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
