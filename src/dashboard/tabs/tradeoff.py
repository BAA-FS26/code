"""Privacy/utility trade-off tab rendering."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.config import GRID_COLOR, TRANSPARENT
from src.dashboard.loader import (
    Result,
    epsilon_of,
    filter_results,
    get_color,
    latest_by,
    result_key,
    source_label,
    summary,
    synthesizer_key,
    utility_key,
)
from src.utility.constants import DP_SYNTHESIZERS

RISK_COLUMN = "Privacy risk (singling-out multivariate)"
F1_COLUMN = "F1 (macro)"


def render_tradeoff_tab(
    utility_records: list[Result],
    privacy_records: list[Result],
    selected_synths: set[str],
    selected_epsilons: set[float],
) -> None:
    """Render privacy/utility trade-off charts and table."""
    st.markdown(
        "**FF3 — Trade-off:** Privacy vs. utility across synthesizers and ε levels. "
        "A good synthesizer sits in the **top-left** quadrant: high utility, low privacy risk."
    )

    df = build_tradeoff_dataframe(
        utility_records, privacy_records, selected_synths, selected_epsilons
    )
    df_both = df.dropna(subset=[F1_COLUMN, RISK_COLUMN])

    if df_both.empty:
        st.warning(
            "Not enough data for the trade-off plot. Ensure both privacy and utility results exist for the same synthesizers."
        )
    else:
        render_tradeoff_scatter(df_both)

    render_summary_table(df)


def build_tradeoff_dataframe(
    utility_records: list[Result],
    privacy_records: list[Result],
    selected_synths: set[str],
    selected_epsilons: set[float],
) -> pd.DataFrame:
    """Combine best utility F1 and privacy risk by synthesizer/epsilon."""
    utility_latest = latest_by(
        filter_results(utility_records, selected_synths, selected_epsilons), utility_key
    )
    privacy_latest = latest_by(
        filter_results(privacy_records, selected_synths, selected_epsilons), result_key
    )

    best_f1: dict[tuple[str, float | None], float] = {}
    for record in utility_latest:
        key = (synthesizer_key(record), epsilon_of(record))
        f1 = summary(record).get("test_f1_macro")
        if f1 is not None:
            best_f1[key] = max(best_f1.get(key, 0), float(f1))

    risk_map: dict[tuple[str, float | None], float] = {}
    for record in privacy_latest:
        key = (synthesizer_key(record), epsilon_of(record))
        risk = summary(record).get("singling_out_risk_multivariate")
        if risk is not None:
            risk_map[key] = float(risk)

    rows = []
    for synth, epsilon in sorted(
        set(best_f1) | set(risk_map), key=lambda item: (item[0], item[1] or 0)
    ):
        rows.append(
            {
                "Source": source_label(synth, epsilon),
                "Synth": synth,
                "Epsilon": epsilon,
                F1_COLUMN: best_f1.get((synth, epsilon)),
                RISK_COLUMN: risk_map.get((synth, epsilon)),
                "_color": get_color(synth, epsilon),
            }
        )
    return pd.DataFrame(rows)


def render_tradeoff_scatter(df: pd.DataFrame) -> None:
    """Render privacy risk vs. utility scatter chart."""
    fig = go.Figure()
    for _, row in df.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row[RISK_COLUMN]],
                y=[row[F1_COLUMN]],
                mode="markers+text",
                name=row["Source"],
                text=[row["Source"]],
                textposition="top center",
                marker=dict(
                    color=row["_color"], size=16, line=dict(color="white", width=2)
                ),
            )
        )
            
    fig.update_layout(
        title="Privacy-utility trade-off (best F1 vs singling-out multivariate risk)",
        xaxis=dict(
            title="Privacy risk — singling-out multivariate (↓ safer)",
            tickformat=".3f",
            range=[-0.01, max(df[RISK_COLUMN].max() * 1.3, 0.1)],
        ),
        yaxis=dict(
            title="Utility — F1 macro (↑ better)",
            tickformat=".3f",
            range=[max(df[F1_COLUMN].min() - 0.05, 0), 1.0],
        ),
        height=500,
        plot_bgcolor=TRANSPARENT,
        paper_bgcolor=TRANSPARENT,
        legend=dict(orientation="h", y=-0.2),
        margin=dict(t=60, b=100),
    )
    fig.update_xaxes(gridcolor=GRID_COLOR)
    fig.update_yaxes(gridcolor=GRID_COLOR)
    st.plotly_chart(fig, use_container_width=True)


def render_summary_table(df: pd.DataFrame) -> None:
    """Render trade-off summary table."""
    with st.expander("📄 Trade-off summary table"):
        if df.empty:
            st.info("No trade-off rows are available.")
            return
        display_cols = [column for column in df.columns if not column.startswith("_")]
        st.dataframe(
            df[display_cols].sort_values(F1_COLUMN, ascending=False),
            use_container_width=True,
            hide_index=True,
        )
