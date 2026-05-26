"""Reusable Plotly chart helpers"""

from __future__ import annotations

from numbers import Real
from typing import TypeAlias, cast

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
    hovermode: str | None = "x unified",
) -> go.Figure:
    """Apply a shared scientific/dashboard chart style."""
    fig.update_layout(
        title=title,
        height=height,
        plot_bgcolor=TRANSPARENT,
        paper_bgcolor=TRANSPARENT,
        margin=dict(t=70 if title else 50, b=bottom_margin, l=60, r=30),
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center", yanchor="bottom"),
        hovermode=hovermode,
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
        return float(cast(float, value)) * 100
    except (TypeError, ValueError):
        return None


def synth_label(synth: str) -> str:
    """Return display label for a synthesizer without epsilon."""
    return SYNTHESIZER_LABELS.get(synth, synth)


def synth_color(synth: str) -> str:
    """Return thesis-style color for a synthesizer."""
    return COLORS.get(synth, "#94A3B8")


def add_baseline_line(
    fig: go.Figure,
    *,
    value: float,
    label: str,
    color: str,
    row: int | None = None,
    col: int | None = None,
    showlegend: bool = True,
    annotation_text: str | None = None,
    annotation_position: str = "right",
) -> None:
    """Add one dashed horizontal baseline line and optional legend entry."""
    if row is not None and col is not None:
        fig.add_hline(
            y=value,
            line_dash="dash",
            line_color=color,
            line_width=2,
            annotation_text=annotation_text,
            annotation_position=annotation_position,
            row=row,   # type: ignore
            col=col,   # type: ignore
        )
    else:
        fig.add_hline(
            y=value,
            line_dash="dash",
            line_color=color,
            line_width=2,
            annotation_text=annotation_text,
            annotation_position=annotation_position,
        )

    baseline_trace = go.Scatter(
        x=[None],
        y=[None],
        mode="lines",
        name=label,
        line=dict(color=color, dash="dash", width=2),
        showlegend=showlegend,
        hoverinfo="skip",
    )

    if row is not None and col is not None:
        fig.add_trace(baseline_trace, row=row, col=col)
    else:
        fig.add_trace(baseline_trace)


def add_baseline_lines(
    fig: go.Figure,
    baselines: pd.DataFrame,
    *,
    metric: str,
    row: int | None = None,
    col: int | None = None,
    showlegend: bool = True,
    annotate: bool = False,
) -> None:
    """Add dashed horizontal baseline lines from a baseline dataframe."""
    for _, baseline in baselines.dropna(subset=[metric]).iterrows():
        synth = str(baseline["Synthesizer"])
        label = synth_label(synth)
        value = float(baseline[metric])

        add_baseline_line(
            fig,
            value=value,
            label=label,
            color=synth_color(synth),
            row=row,
            col=col,
            showlegend=showlegend,
            annotation_text=label if annotate else None,
        )


def add_dp_epsilon_traces(
    fig: go.Figure,
    df: pd.DataFrame,
    *,
    metric: str,
    legend_seen: set[str] | None = None,
    row: int | None = None,
    col: int | None = None,
) -> set[str]:
    """Add DP ε line traces grouped by synthesizer."""
    if legend_seen is None:
        legend_seen = set()

    for synth, group in df.dropna(subset=[metric]).groupby("Synthesizer"):
        group = group.sort_values("Epsilon")
        synth_name = str(synth)
        label = synth_label(synth_name)

        fig.add_trace(
            go.Scatter(
                x=group["Epsilon"],
                y=group[metric],
                mode="lines+markers",
                name=label,
                line=dict(color=synth_color(synth_name), width=3),
                marker=dict(size=9),
                showlegend=label not in legend_seen,
            ),
            row=row,
            col=col,
        )
        legend_seen.add(label)

    return legend_seen


def dp_epsilon_chart(
    df: pd.DataFrame,
    *,
    metric: str,
    title: str,
    dp_synths: set[str],
    baseline_synths: set[str],
    y_title: str,
    y_range: list[float] | None = None,
    height: int = 500,
    bottom_margin: int = 70,
    annotate_baselines: bool = True,
) -> go.Figure:
    """Create a single DP ε line chart with optional baseline references."""
    fig = go.Figure()

    dp_df = df[df["Synthesizer"].isin(dp_synths) & df["Epsilon"].notna()]
    baseline_df = df[df["Synthesizer"].isin(baseline_synths)]

    add_dp_epsilon_traces(fig, dp_df, metric=metric)
    add_baseline_lines(
        fig,
        baseline_df,
        metric=metric,
        annotate=annotate_baselines,
    )

    fig.update_xaxes(title_text="Privacy budget ε", type="log")
    fig.update_yaxes(title_text=y_title, range=y_range)

    return apply_common_layout(
        fig,
        title=title,
        height=height,
        bottom_margin=bottom_margin,
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

        legend_seen = add_dp_epsilon_traces(
            fig,
            dp_df,
            metric=metric,
            legend_seen=legend_seen,
            row=row,
            col=col,
        )

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
            title_text=y_title if col == 1 else None,
            range=y_range,
            row=row,
            col=col,
        )

    hide_empty_subplots(fig, used=len(metrics), total=rows * cols, cols=cols)

    return apply_common_layout(fig, title=title, height=height, bottom_margin=70)


def hide_empty_subplots(
    fig: go.Figure,
    *,
    used: int,
    total: int,
    cols: int,
) -> None:
    """Hide unused subplot cells."""
    for idx in range(used, total):
        row = idx // cols + 1
        col = idx % cols + 1
        fig.update_xaxes(visible=False, row=row, col=col)
        fig.update_yaxes(visible=False, row=row, col=col)


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
