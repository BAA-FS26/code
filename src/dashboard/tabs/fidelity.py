"""

fidelity.py

"""

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from src.dashboard.loader import (
    dedup_latest,
    epsilon_of,
    get_color,
    source_label,
    synthesizer_key,
)


def tab_fidelity(records: list[dict], sel_synths: set, sel_eps: set):
    st.markdown(
        "**Fidelity:** How faithfully does the synthetic data reproduce the "
        "statistical properties of the real data? Measured via SDV quality and diagnostic scores."
    )

    filtered = [
        r
        for r in records
        if synthesizer_key(r) in sel_synths
        and (epsilon_of(r) is None or epsilon_of(r) in sel_eps)
    ]
    filtered = dedup_latest(filtered, lambda r: (synthesizer_key(r), epsilon_of(r)))

    if not filtered:
        st.info("No fidelity results match the current filter.")
        return

    rows = []
    for r in sorted(filtered, key=lambda r: (synthesizer_key(r), epsilon_of(r) or 0)):
        s = synthesizer_key(r)
        e = epsilon_of(r)
        summ = r.get("results", {}).get("summary", {})
        rows.append(
            {
                "Source": source_label(s, e),
                "Quality overall": summ.get("quality_overall"),
                "Column shapes": summ.get("quality_column_shapes"),
                "Column-pair trends": summ.get("quality_column_pair_trends"),
                "Diagnostic overall": summ.get("diagnostic_overall"),
                "Data validity": summ.get("diagnostic_data_validity"),
                "Data structure": summ.get("diagnostic_data_structure"),
                "_color": get_color(s, e),
            }
        )
    df = pd.DataFrame(rows)

    quality_cols = ["Quality overall", "Column shapes", "Column-pair trends"]
    diag_cols = ["Diagnostic overall", "Data validity", "Data structure"]

    col1, col2 = st.columns(2)

    with col1:
        fig_q = go.Figure()
        for _, row in df.iterrows():
            fig_q.add_trace(
                go.Bar(
                    name=row["Source"],
                    x=quality_cols,
                    y=[row[c] for c in quality_cols],
                    marker_color=row["_color"],
                    text=[
                        f"{row[c]:.3f}" if row[c] is not None else "—"
                        for c in quality_cols
                    ],
                    textposition="outside",
                )
            )
        fig_q.update_layout(
            title="SDV quality scores (↑ better)",
            barmode="group",
            yaxis=dict(range=[0, 1.1], tickformat=".2f"),
            legend=dict(orientation="h", y=-0.3),
            height=430,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=50, b=100),
        )
        fig_q.update_yaxes(gridcolor="#E5E7EB")
        st.plotly_chart(fig_q, use_container_width=True)

    with col2:
        fig_d = go.Figure()
        for _, row in df.iterrows():
            fig_d.add_trace(
                go.Bar(
                    name=row["Source"],
                    x=diag_cols,
                    y=[row[c] for c in diag_cols],
                    marker_color=row["_color"],
                    text=[
                        f"{row[c]:.3f}" if row[c] is not None else "—"
                        for c in diag_cols
                    ],
                    textposition="outside",
                )
            )
        fig_d.update_layout(
            title="SDV diagnostic scores (↑ better)",
            barmode="group",
            yaxis=dict(range=[0, 1.1], tickformat=".2f"),
            legend=dict(orientation="h", y=-0.3),
            height=430,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=50, b=100),
        )
        fig_d.update_yaxes(gridcolor="#E5E7EB")
        st.plotly_chart(fig_d, use_container_width=True)

    # ── Radar chart — quality profile per synthesizer
    st.markdown("#### Quality profile (radar)")
    all_dims = quality_cols + diag_cols
    fig_r = go.Figure()
    for _, row in df.iterrows():
        vals = [row[d] for d in all_dims]
        if any(v is not None for v in vals):
            cleaned = [v if v is not None else 0 for v in vals]
            fig_r.add_trace(
                go.Scatterpolar(
                    r=cleaned + [cleaned[0]],
                    theta=all_dims + [all_dims[0]],
                    fill="toself",
                    name=row["Source"],
                    marker_color=row["_color"],
                    opacity=0.7,
                )
            )
    fig_r.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=450,
        title="Fidelity radar — all dimensions",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_r, use_container_width=True)

    with st.expander("📄 Raw numbers"):
        display = [c for c in df.columns if not c.startswith("_")]
        st.dataframe(df[display], use_container_width=True, hide_index=True)
