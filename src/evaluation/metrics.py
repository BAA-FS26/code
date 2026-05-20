"""
Shared classification metric helpers.

Computes classification metrics used by both validation-time training and
held-out Utility evaluation.
"""

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

LabelArray: TypeAlias = Sequence[int] | npt.NDArray[np.integer]


def compute_classification_metrics(
    y_true: LabelArray,
    y_pred: LabelArray,
    prefix: str,
) -> dict[str, float]:
    """
    Compute standard classification metrics with a shared metric-name prefix.

    Returned keys:
        {prefix}_accuracy
        {prefix}_precision_macro
        {prefix}_recall_macro
        {prefix}_f1_macro
    """
    _validate_metric_inputs(y_true, y_pred, prefix)

    return {
        f"{prefix}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}_precision_macro": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
                average="macro",
            )
        ),
        f"{prefix}_recall_macro": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
                average="macro",
            )
        ),
        f"{prefix}_f1_macro": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
                average="macro",
            )
        ),
    }


def _validate_metric_inputs(
    y_true: LabelArray,
    y_pred: LabelArray,
    prefix: str,
) -> None:
    """Validate classification metric inputs."""
    if not prefix.strip():
        raise ValueError("Metric prefix must not be empty.")

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must contain the same number of samples. "
            f"Got {len(y_true)} and {len(y_pred)}."
        )
