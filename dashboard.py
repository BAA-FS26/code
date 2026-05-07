"""Streamlit entrypoint for the Synthetic Data as PET evaluation dashboard.

Run with:
    streamlit run dashboard.py
"""

from __future__ import annotations

import streamlit as st

from src.dashboard.loader import load_all_results
from src.dashboard.sidebar import render_sidebar
from src.dashboard.tabs.fidelity import render_fidelity_tab
from src.dashboard.tabs.privacy import render_privacy_tab
from src.dashboard.tabs.tradeoff import render_tradeoff_tab
from src.dashboard.tabs.utility import render_utility_tab
from src.utility.constants import RESULTS_DIR

APP_TITLE = "Evaluating Synthetic Data as PET"
APP_CAPTION = (
    "Bachelor Thesis · HSLU · FS26 · " "Adult Census Income dataset"
)
TAB_TITLES = ["Utility", "Privacy", "Trade-off", "Fidelity"]


def configure_page() -> None:
    """Configure global Streamlit page settings."""
    st.set_page_config(page_title="Synthetic Data as PET", layout="wide")


def render_header() -> None:
    """Render the dashboard title and subtitle."""
    st.title(APP_TITLE)
    st.caption(APP_CAPTION)
    st.markdown("---")


def main() -> None:
    """Load result files and render the dashboard."""
    configure_page()

    all_results = load_all_results(RESULTS_DIR)
    selected_synths, selected_epsilons, run_mode, selected_date = render_sidebar(
        all_results
    )

    render_header()
    utility_tab, privacy_tab, tradeoff_tab, fidelity_tab = st.tabs(TAB_TITLES)

    with utility_tab:
        render_utility_tab(
            all_results["utility"],
            selected_synths,
            selected_epsilons,
            run_mode,
            selected_date,
        )
    with privacy_tab:
        render_privacy_tab(
            all_results["privacy"],
            selected_synths,
            selected_epsilons,
            run_mode,
            selected_date,
        )
    with fidelity_tab:
        render_fidelity_tab(
            all_results["fidelity"],
            selected_synths,
            selected_epsilons,
            run_mode,
            selected_date,
        )

    with tradeoff_tab:
        render_tradeoff_tab(
            all_results["utility"],
            all_results["privacy"],
            selected_synths,
            selected_epsilons,
            run_mode,
            selected_date,
        )


if __name__ == "__main__":
    main()
