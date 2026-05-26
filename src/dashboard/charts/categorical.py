""" """

import pandas as pd
import plotly.graph_objects as go
from src.dashboard.charts.base import apply_common_layout, format_value


def grouped_metric_bars(
    df: pd.DataFrame,
    *,
    metrics: list[str],
    metric_labels: list[str],
    title: str,
    y_title: str,
    y_range: list[float] | None = None,
    reference_line: float | None = None,
) -> go.Figure:
    """Create thesis-style grouped bars for non-DP synthesizer comparisons."""
    fig = go.Figure()

    for metric, label in zip(metrics, metric_labels):
        fig.add_trace(
            go.Bar(
                name=label,
                x=df["Source"],
                y=df[metric],
                text=[format_value(value, ".1f") for value in df[metric]],
                textposition="outside",
            )
        )

    if reference_line is not None:
        fig.add_hline(
            y=reference_line,
            line_dash="dash",
            line_color="#111827",
            line_width=2,
            annotation_text=f"reference: {reference_line:g} %",
            annotation_position="top left",
        )

    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text=y_title, range=y_range)

    return apply_common_layout(fig, title=title, height=500, bottom_margin=80)
