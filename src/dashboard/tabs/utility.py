"""Utility tab rendering."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.charts.base import apply_common_layout
from src.dashboard.charts.dp import dp_epsilon_chart
from src.dashboard.config import CLASSIFIER_LABELS
from src.dashboard.display import build_base_row, get_color
from src.dashboard.loader import (
    Result,
    RunMode,
    classifier_key,
    prepare_records,
    summary,
    to_percent,
    utility_key,
)
from src.dashboard.metrics import RAW_TABLE_COLUMNS
from src.utility.constants import DP_SYNTHESIZERS


def render_utility_tab(
    records: list[Result],
    selected_synths: set[str],
    selected_epsilons: set[float],
    run_mode: RunMode,
    selected_date: str | None,
) -> None:
    """Render utility charts and classifier-level result table."""
    st.markdown(
        "**FF1 — Utility:** How well do classifiers trained on synthetic data perform "
        "on real held-out test data? *(TSTR — Train on Synthetic, Test on Real)*"
    )

    selected_records = prepare_records(
        records,
        selected_synths,
        selected_epsilons,
        utility_key,
        run_mode,
        selected_date,
    )
    if not selected_records:
        st.info("No utility results match the current filter.")
        return

    classifier_df = pd.DataFrame(build_utility_rows(selected_records)).dropna(
        subset=["F1 (macro)"]
    )
    if classifier_df.empty:
        st.info("Utility result files were found, but none contain F1 scores.")
        return

    average_df = aggregate_utility_by_source(classifier_df)

    st.markdown("#### Average utility vs. real baseline")
    render_average_utility_lollipop(average_df)

    st.markdown("#### Average DP utility across ε")
    render_dp_utility(average_df)

    render_raw_table(classifier_df)


def build_utility_rows(records: list[Result]) -> list[dict]:
    """Transform utility records into classifier-level rows in percent units."""
    rows: list[dict] = []

    for record in records:
        classifier = classifier_key(record)
        metrics = summary(record)

        row = build_base_row(record)

        row.update(
            {
                "ClassifierKey": classifier,
                "Classifier": CLASSIFIER_LABELS.get(classifier, classifier),
                "Accuracy": to_percent(metrics.get("test_accuracy")),
                "Precision": to_percent(metrics.get("test_precision_macro")),
                "Recall": to_percent(metrics.get("test_recall_macro")),
                "F1 (macro)": to_percent(metrics.get("test_f1_macro")),
            }
        )

    return rows


def aggregate_utility_by_source(df: pd.DataFrame) -> pd.DataFrame:
    """Average utility metrics across classifiers per synthesizer, epsilon, and run."""
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


def render_dp_utility(df: pd.DataFrame) -> None:
    """Render average macro-F1 across ε for DP synthesizers."""
    if df.empty:
        return

    fig = dp_epsilon_chart(
        df=df,
        metric="F1 (macro)",
        title="Average Utility of DP Synthesizers across ε",
        dp_synths=set(DP_SYNTHESIZERS),
        baseline_synths={
            synth
            for synth in df["Synthesizer"].unique()
            if synth not in DP_SYNTHESIZERS
        },
        y_title="Average Macro-F1 Score (%)",
        y_range=[0, 100],
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)


def render_average_utility_lollipop(df: pd.DataFrame) -> None:
    """Render average utility against the real-data baseline."""
    real_df = df[(df["Synthesizer"] == "real") & df["Epsilon"].isna()]
    if real_df.empty:
        st.info("Utility chart requires real-data baseline results.")
        return

    baseline = float(real_df["F1 (macro)"].mean())

    plot_df = build_lollipop_dataframe(df)
    if plot_df.empty:
        st.info("No synthetic utility results match the current filters.")
        return

    plot_df = plot_df.sort_values(
        "F1 (macro)",
        ascending=False,
    ).reset_index(drop=True)

    fig = build_lollipop_chart(
        plot_df,
        baseline=baseline,
        title="Average Utility Compared to Real Data",
    )

    st.plotly_chart(fig, use_container_width=True)


def build_lollipop_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return non-DP rows plus optionally one selected DP epsilon."""
    non_dp_df = df[
        (~df["Synthesizer"].isin(DP_SYNTHESIZERS))
        & (df["Synthesizer"] != "real")
        & df["Epsilon"].isna()
    ]

    dp_df = df[df["Synthesizer"].isin(DP_SYNTHESIZERS) & df["Epsilon"].notna()]
    selected_epsilon = select_epsilon(dp_df)

    if selected_epsilon is None:
        return non_dp_df.copy()

    return pd.concat(
        [non_dp_df, dp_df[dp_df["Epsilon"] == selected_epsilon]],
        ignore_index=True,
    )


def select_epsilon(dp_df: pd.DataFrame) -> float | None:
    """Render ε selector for the lollipop DP overlay."""
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


def build_lollipop_chart(
    df: pd.DataFrame,
    *,
    baseline: float,
    title: str,
) -> go.Figure:
    """Build lollipop chart comparing utility against real baseline."""
    fig = go.Figure()

    fig.add_hline(
        y=baseline,
        line_dash="dash",
        line_color="black",
        line_width=3,
        annotation_text=f"Real data baseline: {baseline:.1f}%",
        annotation_position="top left",
    )

    for x_position, (_, row) in enumerate(df.iterrows()):
        y = float(row["F1 (macro)"])
        synth = str(row["Synthesizer"])
        epsilon = row["Epsilon"] if pd.notna(row["Epsilon"]) else None
        color = get_color(synth, epsilon)

        fig.add_shape(
            type="line",
            x0=x_position,
            x1=x_position,
            y0=y,
            y1=baseline,
            line=dict(color=color, width=2),
        )

        fig.add_trace(
            go.Scatter(
                x=[x_position],
                y=[y],
                mode="markers+text",
                marker=dict(
                    size=16,
                    color=color,
                    line=dict(color="black", width=1.5),
                ),
                text=[f"{y:.1f}%"],
                textposition="bottom center",
                name=str(row["Source"]),
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
        tickvals=list(range(len(df))),
        ticktext=list(df["Source"]),
    )

    fig.update_yaxes(
        title_text="Average Macro-F1 Score (%)",
        range=[0, 100],
    )

    return apply_common_layout(
        fig,
        title=title,
        height=520,
        bottom_margin=110,
    )


def render_raw_table(df: pd.DataFrame) -> None:
    """Render classifier-level utility metrics."""
    with st.expander("Classifier-specific raw numbers"):
        available_cols = [col for col in RAW_TABLE_COLUMNS if col in df.columns]

        st.dataframe(
            df[available_cols].sort_values(
                ["Source", "Classifier"], ascending=[True, True]
            ),
            use_container_width=True,
            hide_index=True,
        )
