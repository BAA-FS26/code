"""Reusable dashboard heatmap helpers."""

import pandas as pd
import plotly.graph_objects as go

from src.dashboard.charts.base import apply_common_layout, format_value


def heatmap(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    z: str,
    title: str,
    colorbar_title: str,
    zmin: float | None = None,
    zmax: float | None = None,
) -> go.Figure:
    """Create an annotated heatmap."""
    pivot = df.pivot_table(index=y, columns=x, values=z, aggfunc="first")

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale="Blues",
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title=colorbar_title),
            text=[
                [format_value(value, ".2f") for value in row] for row in pivot.values
            ],
            texttemplate="%{text}",
            hovertemplate="%{y}<br>%{x}: %{z:.2f}<extra></extra>",
        )
    )

    return apply_common_layout(
        fig,
        title=title,
        height=460,
        bottom_margin=70,
        hovermode=None,
    )
