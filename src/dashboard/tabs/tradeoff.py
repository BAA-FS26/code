"""
tradeoff.py

"""

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.dashboard.config import SYNTHESIZER_LABELS
from src.dashboard.loader import (
    dedup_latest,
    epsilon_of,
    get_color,
    source_label,
    synthesizer_key,
)
from src.utility.constants import DP_SYNTHESIZERS


def tab_tradeoff(
    util_records: list[dict],
    priv_records: list[dict],
    sel_synths: set,
    sel_eps: set,
):
    st.markdown(
        "**FF3 — Trade-off:** Privacy vs. utility across synthesizers and ε levels. "
        "A good synthesizer sits in the **top-left** quadrant: high utility, low privacy risk."
    )

    # ── Utility: best F1 per (synth, eps) across classifiers
    util_filtered = [
        r
        for r in util_records
        if synthesizer_key(r) in sel_synths
        and (epsilon_of(r) is None or epsilon_of(r) in sel_eps)
    ]
    util_dedup = dedup_latest(
        util_filtered,
        lambda r: (
            synthesizer_key(r),
            epsilon_of(r),
            r.get("parameters", {}).get("classifier"),
        ),
    )

    best_f1: dict[tuple, float] = {}
    for r in util_dedup:
        k = (synthesizer_key(r), epsilon_of(r))
        f1 = r.get("results", {}).get("summary", {}).get("test_f1_macro")
        if f1 is not None:
            best_f1[k] = max(best_f1.get(k, 0), f1)

    # ── Privacy: singling-out multivariate as main risk metric
    priv_filtered = [
        r
        for r in priv_records
        if synthesizer_key(r) in sel_synths
        and (epsilon_of(r) is None or epsilon_of(r) in sel_eps)
    ]
    priv_dedup = dedup_latest(
        priv_filtered, lambda r: (synthesizer_key(r), epsilon_of(r))
    )

    risk_map: dict[tuple, float] = {}
    for r in priv_dedup:
        k = (synthesizer_key(r), epsilon_of(r))
        risk = (
            r.get("results", {})
            .get("summary", {})
            .get("singling_out_risk_multivariate")
        )
        if risk is not None:
            risk_map[k] = risk

    # Build combined table
    all_keys = set(best_f1) | set(risk_map)
    rows = []
    for k in sorted(all_keys, key=lambda x: (x[0], x[1] or 0)):
        s, e = k
        rows.append(
            {
                "Source": source_label(s, e),
                "Synth": s,
                "Epsilon": e,
                "F1 (macro)": best_f1.get(k),
                "Privacy risk (singling-out multivariate)": risk_map.get(k),
                "_color": get_color(s, e),
                "_size": 14,
            }
        )
    df = pd.DataFrame(rows)
    df_both = df.dropna(
        subset=["F1 (macro)", "Privacy risk (singling-out multivariate)"]
    )

    # ── Scatter: privacy risk vs utility
    if df_both.empty:
        st.warning(
            "Not enough data for the trade-off plot. "
            "Ensure both privacy and utility results exist for the same synthesizers."
        )
    else:
        fig_scatter = go.Figure()
        for _, row in df_both.iterrows():
            fig_scatter.add_trace(
                go.Scatter(
                    x=[row["Privacy risk (singling-out multivariate)"]],
                    y=[row["F1 (macro)"]],
                    mode="markers+text",
                    name=row["Source"],
                    text=[row["Source"]],
                    textposition="top center",
                    marker=dict(
                        color=row["_color"],
                        size=16,
                        line=dict(color="white", width=2),
                    ),
                    showlegend=True,
                )
            )
        # Reference quadrant shading
        fig_scatter.add_shape(
            type="rect",
            x0=0,
            x1=0.05,
            y0=0.75,
            y1=1.0,
            fillcolor="rgba(16,185,129,0.08)",
            line_width=0,
            layer="below",
        )
        fig_scatter.add_annotation(
            x=0.025,
            y=0.99,
            text="✅ Ideal zone",
            showarrow=False,
            font=dict(color="#10B981", size=11),
        )
        fig_scatter.update_layout(
            title="Privacy-utility trade-off (best F1 vs singling-out multivariate risk)",
            xaxis=dict(
                title="Privacy risk — singling-out multivariate (↓ safer)",
                tickformat=".3f",
                range=[
                    -0.01,
                    max(
                        df_both["Privacy risk (singling-out multivariate)"].max() * 1.3,
                        0.1,
                    ),
                ],
            ),
            yaxis=dict(
                title="Utility — F1 macro (↑ better)",
                tickformat=".3f",
                range=[max(df_both["F1 (macro)"].min() - 0.05, 0), 1.0],
            ),
            height=500,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.2),
            margin=dict(t=60, b=100),
        )
        fig_scatter.update_xaxes(gridcolor="#E5E7EB")
        fig_scatter.update_yaxes(gridcolor="#E5E7EB")
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Epsilon effect on DP synthesizers (line chart)
    dp_rows = df[df["Synth"].isin(DP_SYNTHESIZERS) & df["Epsilon"].notna()]
    if not dp_rows.empty and not dp_rows["F1 (macro)"].isna().all():
        st.markdown("#### Effect of ε on DP synthesizers")
        col1, col2 = st.columns(2)
        with col1:
            fig_eps_u = go.Figure()
            for synth, grp in dp_rows.groupby("Synth"):
                synth = str(synth)
                grp = grp.sort_values("Epsilon")
                fig_eps_u.add_trace(
                    go.Scatter(
                        x=grp["Epsilon"],
                        y=grp["F1 (macro)"],
                        mode="lines+markers",
                        name=SYNTHESIZER_LABELS.get(synth, synth),
                        marker=dict(size=10, color=get_color(synth)),
                        line=dict(color=get_color(synth)),
                    )
                )
            fig_eps_u.update_layout(
                title="Utility (F1) vs ε",
                xaxis_title="ε",
                yaxis_title="F1 (macro)",
                xaxis=dict(type="log"),
                height=350,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            fig_eps_u.update_yaxes(gridcolor="#E5E7EB", tickformat=".3f")
            st.plotly_chart(fig_eps_u, use_container_width=True)

        dp_priv = df_both[
            df_both["Synth"].isin(DP_SYNTHESIZERS) & df_both["Epsilon"].notna()
        ]
        with col2:
            fig_eps_p = go.Figure()
            for synth, grp in dp_priv.groupby("Synth"):
                synth = str(synth)
                grp = grp.sort_values("Epsilon")
                fig_eps_p.add_trace(
                    go.Scatter(
                        x=grp["Epsilon"],
                        y=grp["Privacy risk (singling-out multivariate)"],
                        mode="lines+markers",
                        name=SYNTHESIZER_LABELS.get(synth, synth),
                        marker=dict(size=10, color=get_color(synth)),
                        line=dict(color=get_color(synth)),
                    )
                )
            fig_eps_p.update_layout(
                title="Privacy risk vs ε",
                xaxis_title="ε",
                yaxis_title="Singling-out risk (multivariate)",
                xaxis=dict(type="log"),
                height=350,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            fig_eps_p.update_yaxes(gridcolor="#E5E7EB", tickformat=".3f")
            st.plotly_chart(fig_eps_p, use_container_width=True)

    # ── Summary table
    with st.expander("📄 Trade-off summary table"):
        display_cols = [c for c in df.columns if not c.startswith("_")]
        st.dataframe(
            df[display_cols].sort_values("F1 (macro)", ascending=False),
            use_container_width=True,
            hide_index=True,
        )
