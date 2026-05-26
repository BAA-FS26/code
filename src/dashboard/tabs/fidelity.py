"""Fidelity tab rendering."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard.charts.categorical import grouped_metric_bars
from src.dashboard.charts.dp import dp_metric_grid
from src.dashboard.loader import (
    Result,
    RunMode,
    add_percent_metrics,
    build_base_row,
    epsilon_of,
    prepare_records,
    result_key,
    summary,
    synthesizer_key,
)
from src.dashboard.metrics import FIDELITY_KEYS, FIDELITY_METRICS
from src.utility.constants import DP_SYNTHESIZERS


def render_fidelity_tab(
    records: list[Result],
    selected_synths: set[str],
    selected_epsilons: set[float],
    run_mode: RunMode,
    selected_date: str | None,
) -> None:
    """Render thesis-style fidelity charts and table."""
    st.markdown(
        "**Fidelity:** How faithfully does the synthetic data reproduce the statistical "
        "properties of the real data? Measured via SDMetrics quality analysis"
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
        st.info("No fidelity results match the current filter.")
        return

    df = pd.DataFrame(build_fidelity_rows(filtered))
    dp_df = df[df["Synthesizer"].isin(DP_SYNTHESIZERS) & df["Epsilon"].notna()]
    non_dp_df = df[~df["Synthesizer"].isin(DP_SYNTHESIZERS) & df["Epsilon"].isna()]

    if not non_dp_df.empty:
        st.plotly_chart(
            grouped_metric_bars(
                non_dp_df,
                metrics=FIDELITY_METRICS,
                metric_labels=[
                    "Column shapes",
                    "Column-pair trends",
                    "Quality overall",
                ],
                title="Fidelity von SDV Synthesizern",
                y_title="Score (%)",
                y_range=[0, 120],
                reference_line=100,
            ),
            use_container_width=True,
        )

    if not dp_df.empty:
        st.plotly_chart(
            dp_metric_grid(
                df,
                metrics=FIDELITY_METRICS,
                titles=[
                    "Column shapes",
                    "Column-pair trends",
                    "Quality overall",
                ],
                title="Fidelity of the DP synthesizers across ε",
                dp_synths=set(DP_SYNTHESIZERS),
                baseline_synths=set(non_dp_df["Synthesizer"]),
                y_title="Score (%)",
                y_range=[0, 100],
                cols=3,
                height=500,
            ),
            use_container_width=True,
        )

    render_raw_table(df)


def build_fidelity_rows(records: list[Result]) -> list[dict]:
    """Transform fidelity result records into dataframe rows in percent units."""
    rows: list[dict] = []
    for record in sorted(
        records, key=lambda item: (synthesizer_key(item), epsilon_of(item) or 0)
    ):
        row = build_base_row(record)
        add_percent_metrics(row, summary(record), FIDELITY_KEYS)
        rows.append(row)
    return rows


def render_raw_table(df: pd.DataFrame) -> None:
    """Render fidelity raw metrics table."""
    with st.expander("Raw numbers"):
        st.dataframe(df, use_container_width=True, hide_index=True)
