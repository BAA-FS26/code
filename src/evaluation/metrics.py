"""
Shared classification metric helpers.

This module computes classification metrics used by validation and held-out
utility evaluation workflows.
"""

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_classification_metrics(y_true, y_pred, prefix: str) -> dict[str, float]:
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
