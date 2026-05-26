"""Privacy tab rendering."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard.charts.categorical import grouped_metric_bars
from src.dashboard.charts.dp import dp_epsilon_chart, dp_metric_grid
from src.dashboard.charts.heatmaps import heatmap
from src.dashboard.display import build_base_row
from src.dashboard.loader import (
    Result,
    RunMode,
    add_percent_metrics,
    epsilon_of,
    prepare_records,
    result_key,
    summary,
    synthesizer_key,
)
from src.dashboard.metrics import DCR_KEYS, PRIVACY_KEYS, PRIVACY_METRICS
from src.utility.constants import DP_SYNTHESIZERS


def render_privacy_tab(
    records: list[Result],
    selected_synths: set[str],
    selected_epsilons: set[float],
    run_mode: RunMode,
    selected_date: str | None,
) -> None:
    """Render thesis-style privacy charts."""
    st.markdown(
        "**FF2 — Privacy:** Re-identification and inference risks measured via "
        "Anonymeter (singling-out, linkability, inference) and SDMetrics DCR analysis."
    )

    filtered = prepare_records(
        records,
        selected_synths,
        selected_epsilons,
        result_key,
        run_mode,
        selected_date,
    )
    if not filtered:
        st.info("No privacy results match the current filter.")
        return

    df = pd.DataFrame(build_privacy_rows(filtered))
    dp_df = df[df["Synthesizer"].isin(DP_SYNTHESIZERS) & df["Epsilon"].notna()]
    non_dp_df = df[~df["Synthesizer"].isin(DP_SYNTHESIZERS) & df["Epsilon"].isna()]

    if not non_dp_df.empty:
        render_non_dp_heatmap(non_dp_df)
        render_non_dp_dcr(non_dp_df)

    if not dp_df.empty:
        st.plotly_chart(
            dp_metric_grid(
                df,
                metrics=PRIVACY_METRICS,
                titles=PRIVACY_METRICS,
                title="Anonymeter Risk Scores of DP Synthesizer across ε",
                dp_synths=set(DP_SYNTHESIZERS),
                baseline_synths=set(non_dp_df["Synthesizer"]),
                y_title="Risk (%)",
                y_range=[
                    -0.5,
                    10,
                ],
                cols=3,
                height=720,
            ),
            use_container_width=True,
        )

        render_dp_dcr(df, non_dp_df)

    render_raw_table(df)


def build_privacy_rows(records: list[Result]) -> list[dict]:
    """Transform privacy result records into dataframe rows in percent units."""
    rows: list[dict] = []
    for record in sorted(
        records, key=lambda item: (synthesizer_key(item), epsilon_of(item) or 0)
    ):
        metrics = summary(record)
        row = build_base_row(record)
        metrics = summary(record)
        add_percent_metrics(row, metrics, PRIVACY_KEYS)
        add_percent_metrics(row, metrics, DCR_KEYS)
        rows.append(row)
    return rows


def render_dp_dcr(df: pd.DataFrame, non_dp_df: pd.DataFrame) -> None:
    """Render DCR protection over epsilon with dashed baseline references."""
    dp_df = df[df["Synthesizer"].isin(DP_SYNTHESIZERS) & df["Epsilon"].notna()].dropna(
        subset=["DCR-Baseline-Protection"]
    )

    if dp_df.empty:
        return

    fig = dp_epsilon_chart(
        df=df,
        metric="DCR-Baseline-Protection",
        title="DCR Baseline Protection of DP Synthesizer across ε",
        dp_synths=set(DP_SYNTHESIZERS),
        baseline_synths=set(non_dp_df["Synthesizer"]),
        y_title="DCR-Baseline-Protection (%)",
        y_range=[0, 100],
        height=520,
    )

    st.plotly_chart(fig, use_container_width=True)


def render_non_dp_heatmap(non_dp_df: pd.DataFrame) -> None:
    """Render non-DP Anonymeter risks as a compact heatmap."""
    rows: list[dict] = []
    for _, row in non_dp_df.iterrows():
        for metric in PRIVACY_METRICS:
            if pd.notna(row.get(metric)):
                rows.append(
                    {
                        "Synthesizer": row["Source"],
                        "Metric": metric,
                        "Risk": row[metric],
                    }
                )
    if not rows:
        return
    st.plotly_chart(
        heatmap(
            pd.DataFrame(rows),
            x="Synthesizer",
            y="Metric",
            z="Risk",
            title="Anonymeter Risk Scores of SDV Synthesizer",
            colorbar_title="Risiko (%)",
            zmin=0,
            zmax=max(10, max(item["Risk"] for item in rows)),
        ),
        use_container_width=True,
    )


def render_non_dp_dcr(non_dp_df: pd.DataFrame) -> None:
    """Render non-DP DCR protection as a thesis-style bar chart."""
    dcr_df = non_dp_df.dropna(subset=["DCR-Baseline-Protection"])
    if dcr_df.empty:
        return
    st.plotly_chart(
        grouped_metric_bars(
            dcr_df,
            metrics=["DCR-Baseline-Protection"],
            metric_labels=["DCR-Baseline-Protection"],
            title="DCR Baseline Protection of SDV Synthesizer",
            y_title="DCR-Baseline-Protection (%)",
            y_range=[0, 100],
        ),
        use_container_width=True,
    )


def render_raw_table(df: pd.DataFrame) -> None:
    """Render privacy raw metrics table."""
    with st.expander("Raw numbers"):
        st.dataframe(df, use_container_width=True, hide_index=True)
