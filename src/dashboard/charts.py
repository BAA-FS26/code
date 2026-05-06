"""

charts.py

"""

import plotly.graph_objects as go
import pandas as pd

from typing import Optional
from src.dashboard.config import METRIC_LABELS


def bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_cols: list[str],
    title: str,
    y_range=(0, 1),
    color_map: Optional[dict] = None,
) -> go.Figure:
    fig = go.Figure()
    for col in y_cols:
        label = METRIC_LABELS.get(col, col)
        fig.add_trace(
            go.Bar(
                name=label,
                x=df[x_col],
                y=df[col],
                text=df[col].map(lambda v: f"{v:.3f}"),
                textposition="outside",
            )
        )
    fig.update_layout(
        title=title,
        barmode="group",
        yaxis=dict(range=list(y_range), tickformat=".2f"),
        legend=dict(orientation="h", y=-0.2),
        height=420,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=50, b=80),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#E5E7EB", gridwidth=1)
    return fig


def risk_bar_with_ci(
    labels: list[str],
    values: list[float],
    ci_lower: list[float],
    ci_upper: list[float],
    colors: list[str],
    title: str,
) -> go.Figure:
    fig = go.Figure()
    for i, (lbl, val, lo, hi, col) in enumerate(
        zip(labels, values, ci_lower, ci_upper, colors)
    ):
        fig.add_trace(
            go.Bar(
                name=lbl,
                x=[lbl],
                y=[val],
                marker_color=col,
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=[hi - val],
                    arrayminus=[val - lo],
                    color="#374151",
                    thickness=1.5,
                    width=6,
                ),
                text=[f"{val:.4f}"],
                textposition="outside",
                showlegend=False,
            )
        )
    fig.update_layout(
        title=title,
        yaxis=dict(range=[0, 1], tickformat=".2f", title="Risk score"),
        height=400,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=50, b=40),
        barmode="group",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#E5E7EB")
    return fig
