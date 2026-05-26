""" """

from numbers import Real
from typing import TypeAlias

import plotly.graph_objects as go

from src.dashboard.config import (
    COLORS,
    DEFAULT_COLOR,
    EPSILON_SHADE,
    GRID_COLOR,
    SYNTHESIZER_LABELS,
    TRANSPARENT,
)
from src.utility.constants import DP_SYNTHESIZERS

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


def get_color(synth: str, epsilon: float | None = None) -> str:
    """Return a stable color for a synthesizer/epsilon pair."""
    base_hex = COLORS.get(synth, DEFAULT_COLOR)
    if epsilon is None or synth not in DP_SYNTHESIZERS:
        return base_hex

    shade = EPSILON_SHADE.get(epsilon, 1.0)
    red, green, blue = (int(base_hex[i : i + 2], 16) for i in (1, 3, 5))
    shaded = [
        round(channel * shade + 255 * (1 - shade)) for channel in (red, green, blue)
    ]
    return "#{:02X}{:02X}{:02X}".format(*shaded)
