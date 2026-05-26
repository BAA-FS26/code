""" """

from numbers import Real
from typing import TypeAlias

import plotly.graph_objects as go

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


def synth_label(synth: str) -> str:
    """Return display label for a synthesizer without epsilon."""
    return SYNTHESIZER_LABELS.get(synth, synth)


def synth_color(synth: str) -> str:
    """Return thesis-style color for a synthesizer."""
    return COLORS.get(synth, "#94A3B8")
