"""
Shared classification metric helpers.

This module computes classification metrics used by validation and held-out
utility evaluation workflows.
"""

from collections.abc import Sequence

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    prefix: str,
) -> dict[str, float]:
    """
    Compute standard classification metrics with a shared metric-name prefix.

    Metrics are returned using the naming convention:
        {prefix}_accuracy
        {prefix}_precision_macro
        {prefix}_recall_macro
        {prefix}_f1_macro

    Macro averaging is used for precision, recall, and F1 to weight both
    target classes equally regardless of class imbalance.

    Args:
        y_true: Ground-truth class labels.
        y_pred: Predicted class labels.
        prefix: Metric prefix such as 'train', 'val', or 'test'.

    Returns:
        Dictionary containing accuracy, macro precision, macro recall,
        and macro F1 scores.

    Raises:
        ValueError:
            If prefix is empty or prediction lengths do not match.
    """
    if not prefix.strip():
        raise ValueError("Metric prefix must not be empty.")

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must contain the same number of samples. "
            f"Got {len(y_true)} and {len(y_pred)}."
        )

    accuracy = float(accuracy_score(y_true, y_pred))

    precision_macro = float(
        precision_score(
            y_true,
            y_pred,
            zero_division=0,
            average="macro",
        )
    )

    recall_macro = float(
        recall_score(
            y_true,
            y_pred,
            zero_division=0,
            average="macro",
        )
    )

    f1_macro = float(
        f1_score(
            y_true,
            y_pred,
            zero_division=0,
            average="macro",
        )
    )

    return {
        f"{prefix}_accuracy": accuracy,
        f"{prefix}_precision_macro": precision_macro,
        f"{prefix}_recall_macro": recall_macro,
        f"{prefix}_f1_macro": f1_macro,
    }
