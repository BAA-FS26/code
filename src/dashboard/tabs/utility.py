"""

utility.py

"""

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from src.dashboard.config import CLASSIFIER_LABELS
from src.dashboard.loader import (
    dedup_latest,
    epsilon_of,
    get_color,
    source_label,
    synthesizer_key,
)


def tab_utility(records: list[dict], sel_synths: set, sel_eps: set):
    st.markdown(
        "**FF1 — Utility:** How well do classifiers trained on synthetic data "
        "perform on real held-out test data? *(TSTR — Train on Synthetic, Test on Real)*"
    )

    # Filter
    filtered = [
        r
        for r in records
        if synthesizer_key(r) in sel_synths
        and (epsilon_of(r) is None or epsilon_of(r) in sel_eps)
    ]

    # Dedup by (synthesizer, epsilon, classifier)
    def util_key(r):
        return (
            synthesizer_key(r),
            epsilon_of(r),
            r.get("parameters", {}).get("classifier", ""),
        )

    filtered = dedup_latest(filtered, util_key)

    if not filtered:
        st.info(
            "No utility results match the current filter. Check your `results/utility/` folder."
        )
        return

    rows = []
    for r in filtered:
        s = synthesizer_key(r)
        e = epsilon_of(r)
        clf = r.get("parameters", {}).get("classifier", "?")
        summ = r.get("results", {}).get("summary", {})
        rows.append(
            {
                "Source": source_label(s, e),
                "Synthesizer": s,
                "Epsilon": e,
                "Classifier": CLASSIFIER_LABELS.get(clf, clf),
                "Accuracy": summ.get("test_accuracy", None),
                "Precision": summ.get("test_precision_macro", None),
                "Recall": summ.get("test_recall_macro", None),
                "F1 (macro)": summ.get("test_f1_macro", None),
                "_color": get_color(s, e),
            }
        )
    df = pd.DataFrame(rows).dropna(subset=["F1 (macro)"])

    # ── Classifier selector
    classifiers = sorted(df["Classifier"].unique())
    selected_clf = st.selectbox("Classifier", options=classifiers, key="util_clf")
    df_clf = df[df["Classifier"] == selected_clf]

    # ── Metric bar chart
    metric_cols = ["Accuracy", "Precision", "Recall", "F1 (macro)"]
    fig = go.Figure()
    for _, row in df_clf.iterrows():
        fig.add_trace(
            go.Bar(
                name=row["Source"],
                x=metric_cols,
                y=[row[m] for m in metric_cols],
                marker_color=row["_color"],
                text=[f"{row[m]:.3f}" for m in metric_cols],
                textposition="outside",
            )
        )
    fig.update_layout(
        title=f"Classification metrics — {selected_clf}",
        barmode="group",
        yaxis=dict(range=[0, 1.05], tickformat=".2f"),
        legend=dict(orientation="h", y=-0.25),
        height=430,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=50, b=100),
    )
    fig.update_yaxes(gridcolor="#E5E7EB")
    st.plotly_chart(fig, use_container_width=True)

    # ── Cross-classifier F1 heatmap (all classifiers × all sources)
    st.markdown("#### F1 (macro) across all classifiers")
    pivot = df.pivot_table(
        index="Source", columns="Classifier", values="F1 (macro)", aggfunc="first"
    )
    fig2 = px.imshow(
        pivot,
        color_continuous_scale="Blues",
        zmin=0.5,
        zmax=1.0,
        text_auto=".3f",  # type: ignore
        aspect="auto",
        title="F1 heatmap — source × classifier",
    )
    fig2.update_layout(height=300, margin=dict(t=50, b=30))
    st.plotly_chart(fig2, use_container_width=True)

    # ── Raw table
    with st.expander("📄 Raw numbers"):
        display_cols = [
            "Source",
            "Classifier",
            "Accuracy",
            "Precision",
            "Recall",
            "F1 (macro)",
        ]
        st.dataframe(
            df[display_cols].sort_values(
                ["Classifier", "F1 (macro)"], ascending=[True, False]
            ),
            use_container_width=True,
            hide_index=True,
        )
