"""
BAA Dashboard — Synthetic Data as PET
Evaluation of Fidelity · Privacy · Utility across synthesizers
Run: streamlit run dashboard.py
"""

import streamlit as st

from src.dashboard.loader import load_all_results
from src.dashboard.overview import kpi_banner
from src.dashboard.sidebar import sidebar
from src.dashboard.tabs.fidelity import tab_fidelity
from src.dashboard.tabs.privacy import tab_privacy
from src.dashboard.tabs.tradeoff import tab_tradeoff
from src.dashboard.tabs.utility import tab_utility
from src.utility.constants import RESULTS_DIR


def main():
    all_results = load_all_results(RESULTS_DIR)

    sel_synths, sel_eps = sidebar(all_results)

    st.title("🔬 Synthetic Data as PET — Evaluation Dashboard")
    st.caption(
        "Bachelor's Thesis · HSLU · FS26 · "
        "Adult Census Income dataset · RANDOM_STATE=42"
    )
    st.markdown("---")

    kpi_banner(all_results)
    st.markdown("---")

    tab_names = ["⚖️ Trade-off", "🎯 Utility", "🔒 Privacy", "📊 Fidelity"]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        tab_tradeoff(
            all_results["utility"],
            all_results["privacy"],
            sel_synths,
            sel_eps,
        )
    with tabs[1]:
        tab_utility(all_results["utility"], sel_synths, sel_eps)
    with tabs[2]:
        tab_privacy(all_results["privacy"], sel_synths, sel_eps)
    with tabs[3]:
        tab_fidelity(all_results["fidelity"], sel_synths, sel_eps)


if __name__ == "__main__":
    main()
