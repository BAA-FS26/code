"""Sidebar filters for the dashboard."""

from __future__ import annotations

import streamlit as st

from src.dashboard.config import SYNTHESIZER_LABELS
from src.dashboard.loader import (
    RUN_MODES,
    ResultMap,
    RunMode,
    available_run_dates,
    epsilon_of,
    run_date,
    synthesizer_key,
)
from src.utility.constants import DP_SYNTHESIZERS, RESULTS_DIR


def render_sidebar(
    all_results: ResultMap,
) -> tuple[set[str], set[float], RunMode, str | None]:
    """Render sidebar controls and return selected filters and run mode."""
    st.sidebar.image("assets/HSLU_2022_log.png")

    st.sidebar.markdown("## Synthetic Data Evaluation")
    st.sidebar.markdown("BAA · HSLU · FS26")
    st.sidebar.markdown("---")

    all_synths, all_epsilons = collect_filter_values(all_results)
    selected_synths = render_synthesizer_filters(all_synths)
    selected_epsilons = render_epsilon_filters(all_epsilons)
    run_mode, selected_date = render_run_filters(all_results)

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Results path: `{RESULTS_DIR}`")
    return selected_synths, selected_epsilons, run_mode, selected_date


def collect_filter_values(all_results: ResultMap) -> tuple[set[str], set[float]]:
    """Collect available synthesizer and epsilon values from result records."""
    synths: set[str] = set()
    epsilons: set[float] = set()
    for records in all_results.values():
        for record in records:
            synths.add(synthesizer_key(record))
            epsilon = epsilon_of(record)
            if epsilon is not None:
                epsilons.add(epsilon)
    return synths, epsilons


def render_synthesizer_filters(all_synths: set[str]) -> set[str]:
    """Render synthesizer checkboxes."""
    selected: set[str] = {"real"} if "real" in all_synths else set()

    visible_synths = all_synths - {"real"}

    groups = [
        ("Synthesizers", sorted(visible_synths - DP_SYNTHESIZERS)),
        ("DP Synthesizers", sorted(visible_synths & DP_SYNTHESIZERS)),
    ]

    for heading, synths in groups:
        if not synths:
            continue
        st.sidebar.markdown(f"**{heading}**")
        for synth in synths:
            if st.sidebar.checkbox(
                SYNTHESIZER_LABELS.get(synth, synth), value=True, key=f"cb_{synth}"
            ):
                selected.add(synth)

    return selected


def render_epsilon_filters(all_epsilons: set[float]) -> set[float]:
    """Render epsilon checkboxes."""
    if not all_epsilons:
        return set()

    selected: set[float] = set()
    st.sidebar.markdown("**Epsilon (ε)**")
    for epsilon in sorted(all_epsilons):
        if st.sidebar.checkbox(f"ε = {epsilon:g}", value=True, key=f"eps_{epsilon:g}"):
            selected.add(epsilon)
    return selected


def render_run_filters(all_results: ResultMap) -> tuple[RunMode, str | None]:
    """Render run selection controls."""
    st.sidebar.markdown("**Runs**")
    run_mode = st.sidebar.radio(
        "Display mode",
        RUN_MODES,
        index=0,
        help="Latest only is the recommended default for clean comparison charts.",
    )

    selected_date: str | None = None
    dates = available_run_dates(all_results)
    if run_mode == "Specific date":
        if dates:
            selected_date = st.sidebar.selectbox("Run date", dates, index=0)
            render_date_availability(all_results, selected_date)
        else:
            st.sidebar.info("No dated result folders found.")
    return run_mode, selected_date


def render_date_availability(all_results: ResultMap, selected_date: str) -> None:
    """Show which result categories have runs for the selected date."""
    counts = {
        category: sum(1 for record in records if run_date(record) == selected_date)
        for category, records in all_results.items()
    }
    missing = [category for category, count in counts.items() if count == 0]
    if not missing:
        return

    available = (
        ", ".join(
            f"{category}: {count}" for category, count in counts.items() if count > 0
        )
        or "none"
    )
    st.sidebar.caption(f"Available on {selected_date}: {available}")
    st.sidebar.caption(f"No runs on this date: {', '.join(missing)}")
