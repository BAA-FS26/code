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
    result_key,
    run_date,
    select_runs,
    source_label,
    summary,
    synthesizer_key,
    utility_key,
)

RISK_COLUMN = "Privacy risk (singling-out multivariate)"
F1_COLUMN = "F1 (macro)"


def render_tradeoff_tab(
    utility_records: list[Result],
    privacy_records: list[Result],
    selected_synths: set[str],
    selected_epsilons: set[float],
    run_mode: str,
    selected_date: str | None,
) -> None:
    """Render privacy/utility trade-off charts and table."""
    st.markdown(
        "**FF3 — Trade-off:** Privacy vs. utility across synthesizers and ε levels. "
    )

    df = build_tradeoff_dataframe(
        utility_records,
        privacy_records,
        selected_synths,
        selected_epsilons,
        run_mode,
        selected_date,
    )
    if df.empty or F1_COLUMN not in df.columns or RISK_COLUMN not in df.columns:
        st.info(
            "No trade-off rows match the current run selection. "
            "This can happen when utility and privacy results were not run on the selected date."
        )
        render_summary_table(df)
        return

    df_both = df.dropna(subset=[F1_COLUMN, RISK_COLUMN])

    if df_both.empty:
        st.warning(
            "Not enough paired data for the trade-off plot. "
            "Utility and privacy results must exist for the same synthesizer/ε in the current run selection."
        )
    else:
        render_tradeoff_scatter(df_both)

    render_summary_table(df)


def build_tradeoff_dataframe(
    utility_records: list[Result],
    privacy_records: list[Result],
    selected_synths: set[str],
    selected_epsilons: set[float],
    run_mode: str,
    selected_date: str | None,
) -> pd.DataFrame:
    """Combine average utility F1 and privacy risk by synthesizer/epsilon."""
    utility_latest = select_runs(
        filter_results(utility_records, selected_synths, selected_epsilons),
        utility_key,
        run_mode,  # type: ignore[arg-type]
        selected_date,
    )
    privacy_latest = select_runs(
        filter_results(privacy_records, selected_synths, selected_epsilons),
        result_key,
        run_mode,  # type: ignore[arg-type]
        selected_date,
    )

    def join_key(record: Result) -> tuple[str, float | None, str | None]:
        date = run_date(record) if run_mode == "All runs" else None
        return (synthesizer_key(record), epsilon_of(record), date)

    f1_values: dict[tuple[str, float | None, str | None], list[float]] = {}

    for record in utility_latest:
        key = join_key(record)
        f1 = summary(record).get("test_f1_macro")
        if f1 is not None:
            f1_values.setdefault(key, []).append(float(f1))

    avg_f1 = {
        key: sum(values) / len(values) for key, values in f1_values.items() if values
    }

    risk_map: dict[tuple[str, float | None, str | None], float] = {}
    for record in privacy_latest:
        key = join_key(record)
        risk = summary(record).get("singling_out_risk_multivariate")
        if risk is not None:
            risk_map[key] = float(risk)

    rows = []
    all_keys = set(avg_f1) | set(risk_map)
    if not all_keys:
        return pd.DataFrame(
            columns=[
                "Source",
                "Synth",
                "Epsilon",
                "Run date",
                F1_COLUMN,
                RISK_COLUMN,
                "_color",
            ]
        )

    for synth, epsilon, date in sorted(
        all_keys, key=lambda item: (item[0], item[1] or 0, item[2] or "")
    ):
        label = source_label(synth, epsilon)
        if date is not None:
            label = f"{label} ({date})"

        test_f1_raw = avg_f1.get((synth, epsilon, date))
        risk_raw = risk_map.get((synth, epsilon, date))
        rows.append(
            {
                "Source": label,
                "Synth": synth,
                "Epsilon": epsilon,
                "Run date": date,
                F1_COLUMN: test_f1_raw * 100 if test_f1_raw is not None else None,
                RISK_COLUMN: risk_raw * 100 if risk_raw is not None else None,
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
        title="Privacy-utility trade-off (average F1 vs singling-out multivariate risk)",
        xaxis=dict(
            title="Privacy risk — singling-out multivariate (↓ safer)",
            tickformat=".3f",
            range=[0, 10],
        ),
        yaxis=dict(
            title="Utility — F1 macro (↑ better)",
            tickformat=".3f",
            range=[0, 105.0],
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
    with st.expander("Trade-off summary table"):
        if df.empty:
            st.info("No trade-off rows are available for the current run selection.")
            return
        display_cols = [column for column in df.columns if not column.startswith("_")]
        sort_cols = [F1_COLUMN] if F1_COLUMN in df.columns else display_cols[:1]
        ascending = [False] if F1_COLUMN in df.columns else [True]
        st.dataframe(
            df[display_cols].sort_values(sort_cols, ascending=ascending),
            use_container_width=True,
            hide_index=True,
        )
