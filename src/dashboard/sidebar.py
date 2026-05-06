"""

sidebar.py


"""

import streamlit as st

from src.dashboard.config import SYNTHESIZER_LABELS
from src.dashboard.loader import epsilon_of, synthesizer_key
from src.utility.constants import DP_SYNTHESIZERS, RESULTS_DIR


def sidebar(all_results: dict) -> tuple[set, set]:
    """Render sidebar filters; return (selected_synthesizers, selected_epsilons)."""
    st.sidebar.image(
        "assets/HSLU_2022_log.png",
    )
    st.sidebar.markdown("## 🔬 Synthetic Data Evaluation")
    st.sidebar.markdown("BAA · HSLU · FS26")
    st.sidebar.markdown("---")

    all_synths: set[str] = set()
    all_eps: set[float] = set()
    for cat_records in all_results.values():
        for r in cat_records:
            s = synthesizer_key(r)
            e = epsilon_of(r)
            all_synths.add(s)
            if e is not None:
                all_eps.add(e)

    non_dp = sorted(all_synths - DP_SYNTHESIZERS)
    dp = sorted(all_synths & DP_SYNTHESIZERS)

    st.sidebar.markdown("**Synthesizers**")
    sel_non_dp = set()
    for s in non_dp:
        if st.sidebar.checkbox(SYNTHESIZER_LABELS.get(s, s), value=True, key=f"cb_{s}"):
            sel_non_dp.add(s)

    sel_dp = set()
    if dp:
        st.sidebar.markdown("**DP Synthesizers**")
        for s in dp:
            if st.sidebar.checkbox(
                SYNTHESIZER_LABELS.get(s, s), value=True, key=f"cb_{s}"
            ):
                sel_dp.add(s)

    sel_eps = set()
    if all_eps:
        st.sidebar.markdown("**Epsilon (ε)**")
        for e in sorted(all_eps):
            if st.sidebar.checkbox(f"ε = {e}", value=True, key=f"eps_{e}"):
                sel_eps.add(e)

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Results path: `{RESULTS_DIR}`")

    return sel_non_dp | sel_dp, sel_eps
