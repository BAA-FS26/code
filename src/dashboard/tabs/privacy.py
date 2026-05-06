"""

privacy.py


"""

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from src.dashboard.charts import risk_bar_with_ci
from src.dashboard.loader import (
    dedup_latest,
    epsilon_of,
    get_color,
    source_label,
    synthesizer_key,
)


def tab_privacy(records: list[dict], sel_synths: set, sel_eps: set):
    st.markdown(
        "**FF2 — Privacy:** Re-identification and inference risks measured via "
        "Anonymeter (singling-out, linkability, inference) and DCR analysis."
    )

    filtered = [
        r
        for r in records
        if synthesizer_key(r) in sel_synths
        and (epsilon_of(r) is None or epsilon_of(r) in sel_eps)
    ]
    filtered = dedup_latest(filtered, lambda r: (synthesizer_key(r), epsilon_of(r)))

    if not filtered:
        st.info("No privacy results match the current filter.")
        return

    # ── Singling-out risk
    st.markdown("#### Singling-out risk")
    labels, uni_vals, uni_lo, uni_hi = [], [], [], []
    multi_vals, multi_lo, multi_hi = [], [], []
    bar_colors = []

    for r in sorted(filtered, key=lambda r: (synthesizer_key(r), epsilon_of(r) or 0)):
        s = synthesizer_key(r)
        e = epsilon_of(r)
        summ = r.get("results", {}).get("summary", {})
        labels.append(source_label(s, e))
        uni_vals.append(summ.get("singling_out_risk_univariate", 0))
        uni_lo.append(summ.get("singling_out_risk_univariate_ci_lower", 0))
        uni_hi.append(summ.get("singling_out_risk_univariate_ci_upper", 0))
        multi_vals.append(summ.get("singling_out_risk_multivariate", 0))
        multi_lo.append(summ.get("singling_out_risk_multivariate_ci_lower", 0))
        multi_hi.append(summ.get("singling_out_risk_multivariate_ci_upper", 0))
        bar_colors.append(get_color(s, e))

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            risk_bar_with_ci(
                labels,
                uni_vals,
                uni_lo,
                uni_hi,
                bar_colors,
                "Univariate singling-out risk (↓ better)",
            ),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            risk_bar_with_ci(
                labels,
                multi_vals,
                multi_lo,
                multi_hi,
                bar_colors,
                "Multivariate singling-out risk (↓ better)",
            ),
            use_container_width=True,
        )

    # ── Linkability + DCR
    st.markdown("#### Linkability risk & DCR")
    link_vals, link_lo, link_hi = [], [], []
    dcr_base, dcr_overfit = [], []

    for r in sorted(filtered, key=lambda r: (synthesizer_key(r), epsilon_of(r) or 0)):
        summ = r.get("results", {}).get("summary", {})
        link_vals.append(summ.get("linkability_risk", 0))
        link_lo.append(summ.get("linkability_risk_ci_lower", 0))
        link_hi.append(summ.get("linkability_risk_ci_upper", 0))
        dcr_base.append(summ.get("dcr_baseline_protection", 0))
        dcr_overfit.append(summ.get("dcr_overfitting_protection", 0))

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(
            risk_bar_with_ci(
                labels,
                link_vals,
                link_lo,
                link_hi,
                bar_colors,
                "Linkability risk (↓ better)",
            ),
            use_container_width=True,
        )
    with col4:
        fig_dcr = go.Figure()
        fig_dcr.add_trace(
            go.Bar(
                name="DCR baseline protection",
                x=labels,
                y=dcr_base,
                marker_color=bar_colors,
                text=[f"{v:.3f}" for v in dcr_base],
                textposition="outside",
            )
        )
        fig_dcr.add_trace(
            go.Bar(
                name="DCR overfitting protection",
                x=labels,
                y=dcr_overfit,
                marker_color=bar_colors,
                opacity=0.5,
                text=[f"{v:.3f}" for v in dcr_overfit],
                textposition="outside",
            )
        )
        fig_dcr.update_layout(
            title="DCR protection scores (↑ better)",
            barmode="group",
            yaxis=dict(range=[0, 1.1], tickformat=".2f"),
            legend=dict(orientation="h", y=-0.3),
            height=400,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=50, b=80),
        )
        fig_dcr.update_yaxes(gridcolor="#E5E7EB")
        st.plotly_chart(fig_dcr, use_container_width=True)

    # ── Inference risk per sensitive column
    st.markdown("#### Inference risk — sensitive columns")
    sensitive = ["income", "occupation", "sex", "relationship"]
    inf_data = []
    for r in sorted(filtered, key=lambda r: (synthesizer_key(r), epsilon_of(r) or 0)):
        s = synthesizer_key(r)
        e = epsilon_of(r)
        summ = r.get("results", {}).get("summary", {})
        for col in sensitive:
            val = summ.get(f"inference_risk_{col}", None)
            lo = summ.get(f"inference_risk_{col}_ci_lower", None)
            hi = summ.get(f"inference_risk_{col}_ci_upper", None)
            if val is not None:
                inf_data.append(
                    {
                        "Source": source_label(s, e),
                        "Column": col,
                        "Risk": val,
                        "CI lower": lo,
                        "CI upper": hi,
                        "_color": get_color(s, e),
                    }
                )
    if inf_data:
        df_inf = pd.DataFrame(inf_data)
        fig_inf = px.bar(
            df_inf,
            x="Column",
            y="Risk",
            color="Source",
            barmode="group",
            error_y=df_inf["CI upper"] - df_inf["Risk"],
            error_y_minus=df_inf["Risk"] - df_inf["CI lower"],
            title="Inference risk per sensitive column (↓ better)",
            range_y=[0, max(df_inf["CI upper"].max() * 1.2, 0.05)],
        )
        fig_inf.update_layout(
            height=420,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.3),
            margin=dict(t=50, b=100),
        )
        fig_inf.update_yaxes(gridcolor="#E5E7EB", tickformat=".3f")
        st.plotly_chart(fig_inf, use_container_width=True)
