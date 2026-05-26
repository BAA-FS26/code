"""Privacy tab rendering."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard.charts import (
    dp_epsilon_chart,
    dp_metric_grid,
    grouped_metric_bars,
    heatmap,
    to_percent,
)
from src.dashboard.loader import (
    Result,
    RunMode,
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

PRIVACY_METRICS = [
    "Singling Out (multivariate)",
    "Inference (income)",
    "Inference (occupation)",
    "Inference (sex)",
    "Inference (relationship)",
]
PRIVACY_KEYS = {
    "Singling Out (univariate)": "singling_out_risk_univariate",
    "Singling Out (multivariate)": "singling_out_risk_multivariate",
    "Linkability": "linkability_risk",
    "Inference (income)": "inference_risk_income",
    "Inference (occupation)": "inference_risk_occupation",
    "Inference (sex)": "inference_risk_sex",
    "Inference (relationship)": "inference_risk_relationship",
}
DCR_KEYS = {
    "DCR-Baseline-Protection": "dcr_baseline_protection",
    "DCR-Overfitting-Protection": "dcr_overfitting_protection",
}


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

    filtered = select_runs(
        filter_results(records, selected_synths, selected_epsilons),
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
        for label, key in PRIVACY_KEYS.items():
            row[label] = to_percent(metrics.get(key))
        for label, key in DCR_KEYS.items():
            row[label] = to_percent(metrics.get(key))
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
