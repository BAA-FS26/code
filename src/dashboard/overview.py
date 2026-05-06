"""$

overview.py

"""

import streamlit as st
from src.dashboard.loader import epsilon_of, synthesizer_key


def kpi_banner(all_results: dict):
    n_synths = len(
        {
            (synthesizer_key(r), epsilon_of(r))
            for cat in all_results.values()
            for r in cat
        }
    )
    n_util = len(all_results["utility"])
    n_priv = len(all_results["privacy"])
    n_fidel = len(all_results["fidelity"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Synthesizer configs", n_synths)
    c2.metric("Utility runs", n_util)
    c3.metric("Privacy runs", n_priv)
    c4.metric("Fidelity runs", n_fidel)
