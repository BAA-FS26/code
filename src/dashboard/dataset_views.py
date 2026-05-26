"""Dashboard dataset-specific display configuration."""

from typing import Final

from src.dataset.dataset_config import get_dataset_config

ADULT_PRIVACY_LABELS: Final[dict[str, str]] = {
    "income": "Inference (income)",
    "occupation": "Inference (occupation)",
    "sex": "Inference (sex)",
    "relationship": "Inference (relationship)",
}


def build_privacy_metric_config(
    dataset_name: str = "adult_census",
) -> tuple[list[str], dict[str, str]]:
    """Build dashboard privacy metric labels and result-key mappings."""
    config = get_dataset_config(dataset_name)

    metric_labels = ["Singling Out (multivariate)"]
    metric_keys = {
        "Singling Out (multivariate)": "singling_out_risk_multivariate",
    }

    for sensitive_col in config.sensitive_cols:
        display = ADULT_PRIVACY_LABELS.get(
            sensitive_col,
            f"Inference ({sensitive_col})",
        )

        metric_labels.append(display)
        metric_keys[display] = f"inference_risk_{sensitive_col}"

    return metric_labels, metric_keys
