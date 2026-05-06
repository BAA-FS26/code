"""

config.py

"""

COLORS = {
    "real": "#2563EB",
    "gaussian_copula": "#10B981",
    "ctgan": "#F59E0B",
    "tvae": "#8B5CF6",
    "dpctgan": "#EF4444",
    "patectgan": "#EC4899",
}
EPSILON_SHADE = {0.1: 0.3, 1.0: 0.55, 5.0: 0.75, 10.0: 1.0}

METRIC_LABELS = {
    "test_accuracy": "Accuracy",
    "test_precision_macro": "Precision (macro)",
    "test_recall_macro": "Recall (macro)",
    "test_f1_macro": "F1 (macro)",
    "quality_overall": "Quality overall",
    "quality_column_shapes": "Column shapes",
    "quality_column_pair_trends": "Column-pair trends",
    "diagnostic_overall": "Diagnostic overall",
}

CLASSIFIER_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
}

SYNTHESIZER_LABELS = {
    "real": "Real data (baseline)",
    "gaussian_copula": "Gaussian Copula",
    "ctgan": "CTGAN",
    "tvae": "TVAE",
    "dpctgan": "DP-CTGAN",
    "patectgan": "PATE-CTGAN",
}
