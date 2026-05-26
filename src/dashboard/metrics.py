"""Shared dashboard metric definitions and table column constants."""

FIDELITY_METRICS = ["Column shapes", "Column-pair trends", "Quality overall"]
FIDELITY_KEYS = {
    "Quality overall": "quality_overall",
    "Column shapes": "quality_column_shapes",
    "Column-pair trends": "quality_column_pair_trends",
    "Diagnostic overall": "diagnostic_overall",
    "Data validity": "diagnostic_data_validity",
    "Data structure": "diagnostic_data_structure",
}

PRIVACY_METRICS = [
    "Singling Out (multivariate)",
    "Inference (income)",
    "Inference (occupation)",
    "Inference (sex)",
    "Inference (relationship)",
]
PRIVACY_KEYS = {
    "Singling Out (univariate)": "singling_out_risk_univariate",
    "Singling Out (multivariate)": "singling_out_risk_multivariate",
    "Linkability": "linkability_risk",
    "Inference (income)": "inference_risk_income",
    "Inference (occupation)": "inference_risk_occupation",
    "Inference (sex)": "inference_risk_sex",
    "Inference (relationship)": "inference_risk_relationship",
}
DCR_KEYS = {
    "DCR-Baseline-Protection": "dcr_baseline_protection",
    "DCR-Overfitting-Protection": "dcr_overfitting_protection",
}

METRIC_COLUMNS = ["Accuracy", "Precision", "Recall", "F1 (macro)"]
RAW_TABLE_COLUMNS = ["Source", "Classifier", "Run date", *METRIC_COLUMNS]
