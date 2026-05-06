"""Dashboard display configuration."""

from __future__ import annotations

COLORS: dict[str, str] = {
    "real": "#111111",
    "dpctgan": "#1F77B4",
    "patectgan": "#FF7F0E",
    "gaussian_copula": "#9467BD",
    "ctgan": "#2CA02C",
    "tvae": "#D62728",
}

DEFAULT_COLOR = "#94A3B8"
EPSILON_SHADE: dict[float, float] = {0.1: 0.30, 1.0: 0.55, 5.0: 0.75, 10.0: 1.0}
GRID_COLOR = "#E5E7EB"
TRANSPARENT = "rgba(0,0,0,0)"

CLASSIFIER_LABELS: dict[str, str] = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
}

SYNTHESIZER_LABELS: dict[str, str] = {
    "real": "Real data (baseline)",
    "gaussian_copula": "Gaussian Copula",
    "ctgan": "CTGAN",
    "tvae": "TVAE",
    "dpctgan": "DP-CTGAN",
    "patectgan": "PATE-CTGAN",
}
