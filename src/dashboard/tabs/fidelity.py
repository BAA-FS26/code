"""Fidelity tab rendering."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard.charts import dp_metric_grid, grouped_metric_bars, to_percent
from src.dashboard.loader import (
    Result,
    epsilon_of,
    filter_results,
    result_key,
    run_date,
    run_timestamp,
    select_runs,
    source_label,
    summary,
    synthesizer_key,
)
from src.utility.constants import DP_SYNTHESIZERS

FIDELITY_METRICS = ["Column shapes", "Column-pair trends", "Quality overall"]
FIDELITY_KEYS = {
    "Quality overall": "quality_overall",
    "Column shapes": "quality_column_shapes",
    "Column-pair trends": "quality_column_pair_trends",
    "Diagnostic overall": "diagnostic_overall",
    "Data validity": "diagnostic_data_validity",
    "Data structure": "diagnostic_data_structure",
}


def render_fidelity_tab(
    records: list[Result],
    selected_synths: set[str],
    selected_epsilons: set[float],
    run_mode: str,
    selected_date: str | None,
) -> None:
    """Render thesis-style fidelity charts and table."""
    st.markdown(
        "**Fidelity:** How faithfully does the synthetic data reproduce the statistical "
        "properties of the real data? Measured via SDV quality and diagnostic scores."
    )

    filtered = select_runs(
        filter_results(records, selected_synths, selected_epsilons),
        result_key,
        run_mode,  # type: ignore[arg-type]
        selected_date,
    )
    if not filtered:
        st.info("No fidelity results match the current filter.")
        return

    df = pd.DataFrame(build_fidelity_rows(filtered))
    dp_df = df[df["Synthesizer"].isin(DP_SYNTHESIZERS) & df["Epsilon"].notna()]
    non_dp_df = df[~df["Synthesizer"].isin(DP_SYNTHESIZERS) & df["Epsilon"].isna()]

    if not dp_df.empty:
        st.plotly_chart(
            dp_metric_grid(
                df,
                metrics=FIDELITY_METRICS,
                titles=[
                    "Spaltenverteilungen",
                    "Zusammenhänge zwischen Spalten",
                    "Gesamtqualität",
                ],
                title="Fidelity der DP-Synthesizer in Abhängigkeit von ε",
                dp_synths=set(DP_SYNTHESIZERS),
                baseline_synths=set(non_dp_df["Synthesizer"]),
                y_title="Score (%)",
                y_range=[0, 100],
                cols=3,
                height=500,
            ),
            use_container_width=True,
        )

    if not non_dp_df.empty:
        st.plotly_chart(
            grouped_metric_bars(
                non_dp_df,
                metrics=FIDELITY_METRICS,
                metric_labels=[
                    "Spaltenverteilungen",
                    "Zusammenhänge zwischen Spalten",
                    "Gesamtqualität",
                ],
                title="Fidelity von SDV Synthesizern",
                y_title="Score (%)",
                y_range=[0, 120],
                reference_line=100,
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
        synth = synthesizer_key(record)
        epsilon = epsilon_of(record)
        metrics = summary(record)
        row = {
            "Source": source_label(synth, epsilon),
            "Synthesizer": synth,
            "Epsilon": epsilon,
            "Run date": run_date(record),
            "Timestamp": run_timestamp(record),
        }
        for label, key in FIDELITY_KEYS.items():
            row[label] = to_percent(metrics.get(key))
        rows.append(row)
    return rows


def render_raw_table(df: pd.DataFrame) -> None:
    """Render fidelity raw metrics table."""
    with st.expander("📄 Raw numbers"):
        st.dataframe(df, use_container_width=True, hide_index=True)
