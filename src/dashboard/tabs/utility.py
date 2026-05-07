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

    df = pd.DataFrame(build_utility_rows(filtered)).dropna(subset=["F1 (macro)"])
    if df.empty:
        st.info("Utility result files were found, but none contain F1 scores.")
        return

    st.markdown("#### Utility loss vs. real baseline")
    render_utility_loss_lollipop(df)

    st.markdown("#### DP utility across ε")
    render_dp_utility(df)

    render_raw_table(df)


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
    """Render classifier panels with DP epsilon curves and dashed baselines."""
    dp_df = df[df["Synthesizer"].isin(DP_SYNTHESIZERS) & df["Epsilon"].notna()]
    baseline_df = df[(~df["Synthesizer"].isin(DP_SYNTHESIZERS)) & df["Epsilon"].isna()]
    classifiers = [
        c for c in CLASSIFIER_ORDER if c in set(df["ClassifierKey"])
    ] or sorted(df["ClassifierKey"].unique())
    if dp_df.empty:
        return

    fig = make_subplots(
        rows=1,
        cols=len(classifiers),
        subplot_titles=[CLASSIFIER_LABELS.get(c, c) for c in classifiers],
        shared_yaxes=True,
    )
    legend_seen: set[str] = set()
    for col, clf in enumerate(classifiers, start=1):
        clf_dp = dp_df[dp_df["ClassifierKey"] == clf]
        clf_base = baseline_df[baseline_df["ClassifierKey"] == clf]

        for synth, group in clf_dp.groupby("Synthesizer"):
            group = group.sort_values("Epsilon")
            label = synth_label(str(synth))
            fig.add_trace(
                go.Scatter(
                    x=group["Epsilon"],
                    y=group["F1 (macro)"],
                    mode="lines+markers",
                    name=label,
                    line=dict(color=synth_color(str(synth)), width=3),
                    marker=dict(size=9),
                    showlegend=label not in legend_seen,
                ),
                row=1,
                col=col,
            )
            legend_seen.add(label)

        for _, baseline in clf_base.dropna(subset=["F1 (macro)"]).iterrows():
            synth = str(baseline["Synthesizer"])
            value = float(baseline["F1 (macro)"])
            fig.add_hline(
                y=value,
                line_dash="dash",
                line_color=synth_color(synth),
                line_width=2,
                row=1,  # type: ignore
                col=col,  # type: ignore
            )
            label = "Real Data Baseline" if synth == "real" else synth_label(synth)
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    name=label,
                    line=dict(color=synth_color(synth), dash="dash", width=2),
                    showlegend=label not in legend_seen,
                ),
                row=1,
                col=col,
            )
            legend_seen.add(label)

        fig.update_xaxes(title_text="Privacy Budget ε", type="log", row=1, col=col)
        fig.update_yaxes(
            title_text="Macro F1 Score (%)" if col == 1 else None,
            range=[0, 100],
            row=1,
            col=col,
        )

    st.plotly_chart(
        apply_common_layout(
            fig,
            title="Utility of DP Synthesizers compared to CTGAN and Real Data",
            height=500,
            bottom_margin=70,
        ),
        use_container_width=True,
    )


def render_utility_loss_lollipop(df: pd.DataFrame) -> None:
    """Render utility loss against the real baseline, with optional one-ε DP overlay."""
    real = df[(df["Synthesizer"] == "real") & df["Epsilon"].isna()]
    if real.empty:
        st.info("Utility-loss chart requires real-data baseline results.")
        return

    non_dp = df[
        (~df["Synthesizer"].isin(DP_SYNTHESIZERS))
        & (df["Synthesizer"] != "real")
        & df["Epsilon"].isna()
    ]
    dp_df = df[df["Synthesizer"].isin(DP_SYNTHESIZERS) & df["Epsilon"].notna()]

    selected_epsilon = select_lollipop_epsilon(dp_df)
    plot_df = non_dp.copy()
    if selected_epsilon is not None:
        plot_df = pd.concat(
            [plot_df, dp_df[dp_df["Epsilon"] == selected_epsilon]],
            ignore_index=True,
        )

    if plot_df.empty:
        st.info("No synthetic utility results match the current filters.")
        return

    classifiers = [
        c for c in CLASSIFIER_ORDER if c in set(df["ClassifierKey"])
    ] or sorted(df["ClassifierKey"].unique())
    x_positions = {clf: i for i, clf in enumerate(classifiers)}
    offsets = synthesizer_offsets(sorted(plot_df["Synthesizer"].dropna().unique()))

    fig = go.Figure()
    legend_seen: set[str] = set()
    for clf in classifiers:
        x_center = x_positions[clf]
        real_score = real.loc[real["ClassifierKey"] == clf, "F1 (macro)"]
        if real_score.empty:
            continue

        baseline = float(real_score.iloc[0])
        fig.add_shape(
            type="line",
            x0=x_center - 0.36,
            x1=x_center + 0.36,
            y0=baseline,
            y1=baseline,
            line=dict(color="black", width=3, dash="dash"),
        )

        clf_rows = plot_df[plot_df["ClassifierKey"] == clf]
        for _, row in clf_rows.sort_values(["Synthesizer", "Epsilon"]).iterrows():
            y = row["F1 (macro)"]
            if pd.isna(y):
                continue
            synth = str(row["Synthesizer"])
            epsilon = row["Epsilon"] if pd.notna(row["Epsilon"]) else None
            x = x_center + offsets.get(synth, 0)
            label = source_label(synth, float(epsilon) if epsilon is not None else None)
            color = synth_color(synth)

            fig.add_shape(
                type="line",
                x0=x,
                x1=x,
                y0=float(y),
                y1=baseline,
                line=dict(color=color, width=2),
            )
            fig.add_trace(
                go.Scatter(
                    x=[x],
                    y=[y],
                    mode="markers",
                    name=label,
                    marker=dict(
                        size=17,
                        color=color,
                        line=dict(color="black", width=2),
                    ),
                    hovertemplate=(
                        f"{label}<br>"
                        f"{CLASSIFIER_LABELS.get(clf, clf)}<br>"
                        "F1: %{y:.1f}%<br>"
                        f"Loss vs real: {baseline - float(y):.1f} pp"
                        "<extra></extra>"
                    ),
                    showlegend=label not in legend_seen,
                )
            )
            legend_seen.add(label)

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            name="Real Data Baseline",
            line=dict(color="black", width=3, dash="dash"),
        )
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(x_positions.values()),
        ticktext=[CLASSIFIER_LABELS.get(c, c) for c in classifiers],
    )
    fig.update_yaxes(title_text="Macro F1 Score (%)", range=[0, 100])
    title = "Utility-Loss: Distance from the real-data baseline"    
    st.plotly_chart(
        apply_common_layout(fig, title=title, height=560, bottom_margin=95),
        use_container_width=True,
    )


def select_lollipop_epsilon(dp_df: pd.DataFrame) -> float | None:
    """Render a compact ε selector for the lollipop DP overlay."""
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
    """Render utility raw metrics table."""
    with st.expander("Raw numbers"):
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
                ["Classifier", "F1 (macro)"], ascending=[True, False]
            ),
            use_container_width=True,
            hide_index=True,
        )
