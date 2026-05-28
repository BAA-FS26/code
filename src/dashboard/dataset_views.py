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


def build_privacy_raw_metric_config(
    dataset_name: str = "adult_census",
) -> dict[str, str]:
    """Build full raw-table privacy metric label to result-key mapping."""
    config = get_dataset_config(dataset_name)

    metric_keys = {
        "Singling Out (univariate)": "singling_out_risk_univariate",
        "Singling Out (univariate) CI lower": "singling_out_risk_univariate_ci_lower",
        "Singling Out (univariate) CI upper": "singling_out_risk_univariate_ci_upper",
        "Singling Out (multivariate)": "singling_out_risk_multivariate",
        "Singling Out (multivariate) CI lower": "singling_out_risk_multivariate_ci_lower",
        "Singling Out (multivariate) CI upper": "singling_out_risk_multivariate_ci_upper",
        "Linkability": "linkability_risk",
        "Linkability CI lower": "linkability_risk_ci_lower",
        "Linkability CI upper": "linkability_risk_ci_upper",
    }

    for sensitive_col in config.sensitive_cols:
        display = ADULT_PRIVACY_LABELS.get(
            sensitive_col,
            f"Inference ({sensitive_col})",
        )

        metric_keys[display] = f"inference_risk_{sensitive_col}"
        metric_keys[f"{display} CI lower"] = f"inference_risk_{sensitive_col}_ci_lower"
        metric_keys[f"{display} CI upper"] = f"inference_risk_{sensitive_col}_ci_upper"

    return metric_keys
