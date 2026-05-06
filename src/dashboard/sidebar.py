"""Sidebar filters for the dashboard."""

from __future__ import annotations

import streamlit as st

from src.dashboard.config import SYNTHESIZER_LABELS
from src.dashboard.loader import ResultMap, epsilon_of, synthesizer_key
from src.utility.constants import DP_SYNTHESIZERS, RESULTS_DIR


def render_sidebar(all_results: ResultMap) -> tuple[set[str], set[float]]:
    """Render sidebar controls and return selected synthesizers and epsilons."""
    st.sidebar.image("assets/HSLU_2022_log.png")

    st.sidebar.markdown("## 🔬 Synthetic Data Evaluation")
    st.sidebar.markdown("BAA · HSLU · FS26")
    st.sidebar.markdown("---")

    all_synths, all_epsilons = collect_filter_values(all_results)
    selected_synths = render_synthesizer_filters(all_synths)
    selected_epsilons = render_epsilon_filters(all_epsilons)

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Results path: `{RESULTS_DIR}`")
    return selected_synths, selected_epsilons


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
    selected: set[str] = set()
    groups = [
        ("Synthesizers", sorted(all_synths - DP_SYNTHESIZERS)),
        ("DP Synthesizers", sorted(all_synths & DP_SYNTHESIZERS)),
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
