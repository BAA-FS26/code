"""Dashboard display helpers."""

from __future__ import annotations
from typing import Any

from src.dashboard.config import (
    COLORS,
    DEFAULT_COLOR,
    EPSILON_SHADE,
    SYNTHESIZER_LABELS,
)
from src.dashboard.loader import Result, epsilon_of, run_date, run_timestamp, synthesizer_key
from src.utility.constants import DP_SYNTHESIZERS


def source_label(synth: str, epsilon: float | None) -> str:
    """Return a readable label for a synthesizer/epsilon pair."""
    label = SYNTHESIZER_LABELS.get(synth, synth)
    return f"{label} ε={epsilon:g}" if epsilon is not None else label


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

def build_base_row(record: Result) -> dict[str, Any]:
    """Build shared dataframe metadata columns for dashboard tables/charts."""
    synth = synthesizer_key(record)
    epsilon = epsilon_of(record)

    return {
        "Source": source_label(synth, epsilon),
        "Synthesizer": synth,
        "Epsilon": epsilon,
        "Run date": run_date(record),
        "Timestamp": run_timestamp(record),
    }