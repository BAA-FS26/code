"""Utility tab rendering."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.dashboard.charts import (
    apply_common_layout,
    synth_color,
    synth_label,
    to_percent,
)
from src.dashboard.config import CLASSIFIER_LABELS
from src.dashboard.loader import (
    Result,
    classifier_key,
    epsilon_of,
    filter_results,
    run_date,
    run_timestamp,
    select_runs,
    get_color,
    source_label,
    summary,
    synthesizer_key,
    utility_key,
)
from src.utility.constants import DP_SYNTHESIZERS

CLASSIFIER_ORDER = ["logistic_regression", "random_forest", "gradient_boosting"]


def render_utility_tab(
    records: list[Result],
    selected_synths: set[str],
    selected_epsilons: set[float],
    run_mode: str,
    selected_date: str | None,
) -> None:
    """Render thesis-style utility charts and tables."""
    st.markdown(
        "**FF1 — Utility:** How well do classifiers trained on synthetic data perform "
        "on real held-out test data? *(TSTR — Train on Synthetic, Test on Real)*"
    )

    filtered = select_runs(
        filter_results(records, selected_synths, selected_epsilons),
        utility_key,
        run_mode,  # type: ignore[arg-type]
        selected_date,
    )
    if not filtered:
        st.info(
            "No utility results match the current filter. Check your `results/utility/` folder."
        )
        return

    classifier_df = pd.DataFrame(build_utility_rows(filtered)).dropna(
        subset=["F1 (macro)"]
    )
    df = aggregate_utility_by_source(classifier_df)
    if df.empty:
        st.info("Utility result files were found, but none contain F1 scores.")
        return

    st.markdown("#### Average utility vs. real baseline")
    render_average_utility_lollipop(df)

    st.markdown("#### Average DP utility across ε")
    render_dp_utility(df)

    render_raw_table(classifier_df)


def build_utility_rows(records: list[Result]) -> list[dict]:
    """Transform utility result records into dataframe rows in percent units."""
    rows: list[dict] = []
    for record in records:
        synth = synthesizer_key(record)
        epsilon = epsilon_of(record)
        clf = classifier_key(record)
        metrics = summary(record)
        rows.append(
            {
                "Source": source_label(synth, epsilon),
                "Synthesizer": synth,
                "Epsilon": epsilon,
                "ClassifierKey": clf,
                "Classifier": CLASSIFIER_LABELS.get(clf, clf),
                "Run date": run_date(record),
                "Timestamp": run_timestamp(record),
                "Accuracy": to_percent(metrics.get("test_accuracy")),
                "Precision": to_percent(metrics.get("test_precision_macro")),
                "Recall": to_percent(metrics.get("test_recall_macro")),
                "F1 (macro)": to_percent(metrics.get("test_f1_macro")),
            }
        )
    return rows


def render_dp_utility(df: pd.DataFrame) -> None:
    """Render average macro-F1 across classifiers for DP synthesizers over epsilon."""
    dp_df = df[df["Synthesizer"].isin(DP_SYNTHESIZERS) & df["Epsilon"].notna()]
    baseline_df = df[(~df["Synthesizer"].isin(DP_SYNTHESIZERS)) & df["Epsilon"].isna()]

    if dp_df.empty:
        return

    fig = go.Figure()

    for synth, group in dp_df.groupby("Synthesizer"):
        group = group.sort_values("Epsilon")
        fig.add_trace(
            go.Scatter(
                x=group["Epsilon"],
                y=group["F1 (macro)"],
                mode="lines+markers",
                name=synth_label(str(synth)),
                line=dict(color=synth_color(str(synth)), width=3),
                marker=dict(size=9),
            )
        )

    for _, baseline in baseline_df.dropna(subset=["F1 (macro)"]).iterrows():
        synth = str(baseline["Synthesizer"])
        fig.add_hline(
            y=float(baseline["F1 (macro)"]),
            line_dash="dash",
            line_color=synth_color(synth),
            line_width=2,
            annotation_text=synth_label(synth),
            annotation_position="right",
        )

    fig.update_xaxes(title_text="Privacy Budget ε", type="log")
    fig.update_yaxes(title_text="Average Macro-F1 Score (%)", range=[0, 100])

    st.plotly_chart(
        apply_common_layout(
            fig,
            title="Average Utility of DP Synthesizers across ε",
            height=500,
            bottom_margin=70,
        ),
        use_container_width=True,
    )


def render_average_utility_lollipop(df: pd.DataFrame) -> None:
    """Render averaged utility as a lollipop chart against the real-data baseline."""
    real = df[(df["Synthesizer"] == "real") & df["Epsilon"].isna()]
    if real.empty:
        st.info("Utility chart requires real-data baseline results.")
        return

    baseline = float(real["F1 (macro)"].mean())

    non_dp = df[
        (~df["Synthesizer"].isin(DP_SYNTHESIZERS))
        & (df["Synthesizer"] != "real")
        & df["Epsilon"].isna()
    ]

    dp_df = df[df["Synthesizer"].isin(DP_SYNTHESIZERS) & df["Epsilon"].notna()]
    selected_epsilon = select_epsilon(dp_df)

    plot_df = non_dp.copy()
    if selected_epsilon is not None:
        plot_df = pd.concat(
            [plot_df, dp_df[dp_df["Epsilon"] == selected_epsilon]],
            ignore_index=True,
        )

    if plot_df.empty:
        st.info("No synthetic utility results match the current filters.")
        return

    plot_df = plot_df.sort_values("F1 (macro)", ascending=False).reset_index(drop=True)

    fig = go.Figure()
    x_positions = {source: i for i, source in enumerate(plot_df["Source"])}

    fig.add_hline(
        y=baseline,
        line_dash="dash",
        line_color="black",
        line_width=3,
        annotation_text=f"Real data baseline: {baseline:.1f}%",
        annotation_position="top left",
    )

    for _, row in plot_df.iterrows():
        x = x_positions[row["Source"]]
        y = float(row["F1 (macro)"])

        color = get_color(
            str(row["Synthesizer"]),
            row["Epsilon"] if pd.notna(row["Epsilon"]) else None,
        )

        fig.add_shape(
            type="line",
            x0=x,
            x1=x,
            y0=y,
            y1=baseline,
            line=dict(color=color, width=2),
        )

        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers+text",
                marker=dict(size=16, color=color, line=dict(color="black", width=1.5)),
                text=[f"{y:.1f}%"],
                textposition="bottom center",
                name=row["Source"],
                hovertemplate=(
                    f"{row['Source']}<br>"
                    f"Average F1: {y:.1f}%<br>"
                    f"Loss vs real: {baseline - y:.1f} pp"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )

    fig.update_xaxes(
        tickmode="array",
        tickvals=list(x_positions.values()),
        ticktext=list(x_positions.keys()),
    )

    fig.update_yaxes(
        title_text="Average Macro-F1 Score (%)",
        range=[0, 100],
    )

    st.plotly_chart(
        apply_common_layout(
            fig,
            title="Average Utility Compared to Real Data",
            height=520,
            bottom_margin=110,
        ),
        use_container_width=True,
    )


def select_epsilon(dp_df: pd.DataFrame) -> float | None:
    """Render a compact ε selector for the DP overlay."""
    epsilons = sorted(float(eps) for eps in dp_df["Epsilon"].dropna().unique())
    if not epsilons:
        return None

    show_dp = st.checkbox(
        "Show DP variants in utility-loss chart",
        value=True,
        help="Adds DP synthesizer results for one selected ε.",
    )
    if not show_dp:
        return None

    slider_col, _ = st.columns([1, 4])

    with slider_col:
        return float(
            st.select_slider(
                "DP ε shown",
                options=epsilons,
                value=epsilons[0],
                format_func=lambda value: f"ε = {value:g}",
            )
        )


def synthesizer_offsets(synths: list[str]) -> dict[str, float]:
    """Return centered x offsets for all synthesizers shown in a classifier group."""
    if not synths:
        return {}
    if len(synths) == 1:
        return {synths[0]: 0.0}
    span = 0.52
    step = span / (len(synths) - 1)
    start = -span / 2
    return {synth: start + idx * step for idx, synth in enumerate(synths)}


def render_raw_table(df: pd.DataFrame) -> None:
    """Render classifier-level utility metrics table."""
    with st.expander("Classifier-specific raw numbers"):
        display_cols = [
            "Source",
            "Classifier",
            "Run date",
            "Accuracy",
            "Precision",
            "Recall",
            "F1 (macro)",
        ]

        st.dataframe(
            df[display_cols].sort_values(
                ["Source", "Classifier"], ascending=[True, True]
            ),
            use_container_width=True,
            hide_index=True,
        )


def aggregate_utility_by_source(df: pd.DataFrame) -> pd.DataFrame:
    """Average utility metrics across classifiers per synthesizer/epsilon/run."""
    group_cols = ["Source", "Synthesizer", "Epsilon", "Run date"]

    return df.groupby(group_cols, dropna=False, as_index=False).agg(
        {
            "Accuracy": "mean",
            "Precision": "mean",
            "Recall": "mean",
            "F1 (macro)": "mean",
            "Timestamp": "max",
        }
    )
