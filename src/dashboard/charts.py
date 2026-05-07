"""Reusable Plotly chart helpers with thesis-style visual grammar."""

from __future__ import annotations

from numbers import Real
from typing import TypeAlias

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.dashboard.config import COLORS, GRID_COLOR, SYNTHESIZER_LABELS, TRANSPARENT

FormattableValue: TypeAlias = Real | str | None


def apply_common_layout(
    fig: go.Figure,
    *,
    height: int = 420,
    bottom_margin: int = 80,
    title: str | None = None,
) -> go.Figure:
    """Apply a shared scientific/dashboard chart style."""
    fig.update_layout(
        title=title,
        height=height,
        plot_bgcolor=TRANSPARENT,
        paper_bgcolor=TRANSPARENT,
        margin=dict(t=70 if title else 50, b=bottom_margin, l=60, r=30),
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center", yanchor="bottom"),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False)
    return fig


def format_value(value: FormattableValue, spec: str = ".3f") -> str:
    """Format numeric values for chart labels."""
    if value is None:
        return "—"
    if isinstance(value, Real):
        return f"{float(value):{spec}}"
    return value


def to_percent(value: object) -> float | None:
    """Convert result metrics stored as 0..1 fractions into percentage points."""
    if value is None:
        return None
    try:
        return float(value) * 100 # type: ignore
    except (TypeError, ValueError):
        return None


def synth_label(synth: str) -> str:
    """Return display label for a synthesizer without epsilon."""
    return SYNTHESIZER_LABELS.get(synth, synth)


def synth_color(synth: str) -> str:
    """Return thesis-style color for a synthesizer."""
    return COLORS.get(synth, "#94A3B8")


def add_baseline_lines(
    fig: go.Figure,
    baselines: pd.DataFrame,
    *,
    metric: str,
    row: int = 1,
    col: int = 1,
    showlegend: bool = True,
) -> None:
    """Add dashed horizontal non-DP/real baseline traces to a subplot."""
    for _, baseline in baselines.dropna(subset=[metric]).iterrows():
        synth = str(baseline["Synthesizer"])
        value = float(baseline[metric])
        fig.add_hline(
            y=value,
            line_dash="dash",
            line_color=synth_color(synth),
            line_width=2,
            row=row, # type: ignore
            col=col, # type: ignore
        )
        # Invisible scatter keeps the dashed baseline in the legend.
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name=synth_label(synth),
                line=dict(color=synth_color(synth), dash="dash", width=2),
                showlegend=showlegend,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )


def dp_metric_grid(
    df: pd.DataFrame,
    *,
    metrics: list[str],
    titles: list[str],
    title: str,
    dp_synths: set[str],
    baseline_synths: set[str],
    y_title: str,
    y_range: list[float] | None = None,
    cols: int = 3,
    height: int = 540,
) -> go.Figure:
    """Create thesis-style epsilon panels: DP lines + dashed baseline references."""
    rows = (len(metrics) + cols - 1) // cols
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles, shared_yaxes=True)

    dp_df = df[df["Synthesizer"].isin(dp_synths) & df["Epsilon"].notna()]
    baseline_df = df[df["Synthesizer"].isin(baseline_synths)]
    legend_seen: set[str] = set()

    for idx, metric in enumerate(metrics):
        row = idx // cols + 1
        col = idx % cols + 1
        for synth, group in dp_df.dropna(subset=[metric]).groupby("Synthesizer"):
            group = group.sort_values("Epsilon")
            label = synth_label(str(synth))
            fig.add_trace(
                go.Scatter(
                    x=group["Epsilon"],
                    y=group[metric],
                    mode="lines+markers",
                    name=label,
                    line=dict(color=synth_color(str(synth)), width=3),
                    marker=dict(size=9),
                    showlegend=label not in legend_seen,
                ),
                row=row,
                col=col,
            )
            legend_seen.add(label)

        add_baseline_lines(
            fig,
            baseline_df,
            metric=metric,
            row=row,
            col=col,
            showlegend=idx == 0,
        )
        fig.update_xaxes(title_text="Privacy budget ε", type="log", row=row, col=col)
        fig.update_yaxes(
            title_text=y_title if col == 1 else None, range=y_range, row=row, col=col
        )

    for empty_idx in range(len(metrics), rows * cols):
        fig.update_xaxes(
            visible=False, row=empty_idx // cols + 1, col=empty_idx % cols + 1
        )
        fig.update_yaxes(
            visible=False, row=empty_idx // cols + 1, col=empty_idx % cols + 1
        )

    return apply_common_layout(fig, title=title, height=height, bottom_margin=70)


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
                text=[format_value(v, ".1f") for v in df[metric]],
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
    fig.update_layout(barmode="group", yaxis=dict(title=y_title, range=y_range))
    return apply_common_layout(fig, title=title, height=500, bottom_margin=80)


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
            text=[[format_value(v, ".2f") for v in row] for row in pivot.values],
            texttemplate="%{text}",
            hovertemplate="%{y}<br>%{x}: %{z:.2f}<extra></extra>",
        )
    )
    return apply_common_layout(fig, title=title, height=460, bottom_margin=70)
