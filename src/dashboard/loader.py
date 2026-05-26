"""Result loading and normalization helpers."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Literal

import streamlit as st

from src.dashboard.config import (
    COLORS,
    DEFAULT_COLOR,
    EPSILON_SHADE,
    SYNTHESIZER_LABELS,
)
from src.utility.constants import DP_SYNTHESIZERS

LOGGER = logging.getLogger(__name__)
Result = dict[str, Any]
ResultMap = dict[str, list[Result]]
RecordKey = tuple[Any, ...]
RunMode = Literal["Latest only", "Specific date", "All runs"]

RESULT_CATEGORIES = ("fidelity", "privacy", "utility")
RUN_MODES: tuple[RunMode, ...] = (
    "Latest only",
    "Specific date",
    "All runs",
)


@st.cache_data(show_spinner=False)
def load_all_results(results_dir: Path) -> ResultMap:
    """Load JSON result files grouped by category.

    Expected layout::

        results/{category}/{YYYY-MM-DD}/*.json

    The loader still accepts deeper category subfolders via ``rglob``. Category,
    synthesizer, epsilon, classifier, and timestamp are read from the JSON record
    whenever possible, so dashboard behavior does not depend on filename parsing.
    """
    results: ResultMap = {category: [] for category in RESULT_CATEGORIES}
    if not results_dir.exists():
        LOGGER.warning("Results directory does not exist: %s", results_dir)
        return results

    for category in RESULT_CATEGORIES:
        category_dir = results_dir / category
        if not category_dir.exists():
            continue

        for json_file in category_dir.rglob("*.json"):
            try:
                record = json.loads(json_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                LOGGER.warning("Skipping invalid JSON file %s: %s", json_file, exc)
                continue
            except OSError as exc:
                LOGGER.warning("Could not read result file %s: %s", json_file, exc)
                continue

            if not isinstance(record, dict):
                LOGGER.warning("Skipping non-object JSON result: %s", json_file)
                continue

            record_category = str(record.get("category") or category)
            if record_category not in RESULT_CATEGORIES:
                LOGGER.warning(
                    "Skipping unknown result category %r in %s",
                    record_category,
                    json_file,
                )
                continue

            record["_file"] = str(json_file)
            record["_date_dir"] = json_file.parent.name
            results[record_category].append(record)

    for category in RESULT_CATEGORIES:
        results[category].sort(
            key=lambda record: str(record.get("timestamp", "")), reverse=True
        )

    return results


def parameters(record: Result) -> dict[str, Any]:
    """Return the parameters section for a result record."""
    return record.get("parameters", {}) or {}


def summary(record: Result) -> dict[str, Any]:
    """Return the nested results.summary section for a result record."""
    return record.get("results", {}).get("summary", {}) or {}


def run_timestamp(record: Result) -> str:
    """Return the result timestamp as stored in the JSON record."""
    return str(record.get("timestamp", ""))


def run_date(record: Result) -> str:
    """Return the run date, preferring the date folder and falling back to timestamp."""
    date_dir = str(record.get("_date_dir", ""))
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_dir):
        return date_dir
    timestamp = run_timestamp(record)
    return timestamp[:10] if timestamp else "unknown"


def available_run_dates(all_results: ResultMap) -> list[str]:
    """Return available run dates newest-first."""
    dates = {
        run_date(record)
        for records in all_results.values()
        for record in records
        if run_date(record) != "unknown"
    }
    return sorted(dates, reverse=True)


def filter_by_run_date(
    records: Iterable[Result], selected_date: str | None
) -> list[Result]:
    """Keep records from a selected run date."""
    if not selected_date:
        return list(records)
    return [record for record in records if run_date(record) == selected_date]


def select_runs(
    records: Iterable[Result],
    key_fn: Callable[[Result], RecordKey],
    run_mode: RunMode,
    selected_date: str | None,
) -> list[Result]:
    """Apply the user-selected run display mode.

    ``Latest only`` keeps one newest record per logical configuration.
    ``Specific date`` shows all records from the selected date, still deduplicated by
    configuration in case the same experiment was rerun on that date.
    ``All runs`` returns the records unchanged, newest-first.
    """
    records_list = sorted(
        list(records), key=lambda record: run_timestamp(record), reverse=True
    )
    if run_mode == "All runs":
        return records_list
    if run_mode == "Specific date":
        return latest_by(filter_by_run_date(records_list, selected_date), key_fn)
    return latest_by(records_list, key_fn)


def synthesizer_key(record: Result) -> str:
    """Return a canonical synthesizer identifier from a result record."""
    params = parameters(record)
    synth = params.get("synthesizer") or params.get("data_source") or "unknown"
    synth_key = str(synth).lower().strip().replace("\\", "/")
    return synth_key.split("/", 1)[0]


def classifier_key(record: Result) -> str:
    """Return the classifier identifier for a utility result."""
    return str(parameters(record).get("classifier", ""))


def epsilon_of(record: Result) -> float | None:
    """Return epsilon as a float, or ``None`` for non-DP records."""
    params = parameters(record)
    epsilon = params.get("epsilon")
    if epsilon not in (None, ""):
        try:
            return float(epsilon)
        except (TypeError, ValueError):
            LOGGER.warning(
                "Invalid epsilon value %r in %s", epsilon, record.get("_file", "record")
            )
            return None

    data_source = str(params.get("data_source", "")).lower().replace("\\", "/")
    match = re.search(r"(?:^|/)eps(?:ilon)?[_=-]([0-9]+(?:\.[0-9]+)?)", data_source)
    if match:
        return float(match.group(1))
    return None


def source_label(synth: str, epsilon: float | None) -> str:
    """Return a readable label for a synthesizer/epsilon pair."""
    label = SYNTHESIZER_LABELS.get(synth, synth)
    return f"{label} ε={epsilon:g}" if epsilon is not None else label


def get_color(synth: str, epsilon: float | None = None) -> str:
    """Return a stable color for a synthesizer/epsilon pair."""
    base_hex = COLORS.get(synth, DEFAULT_COLOR)
    if epsilon is None or synth not in DP_SYNTHESIZERS:
        return base_hex

    shade = EPSILON_SHADE.get(epsilon, 1.0)
    red, green, blue = (int(base_hex[i : i + 2], 16) for i in (1, 3, 5))
    shaded = [
        round(channel * shade + 255 * (1 - shade)) for channel in (red, green, blue)
    ]
    return "#{:02X}{:02X}{:02X}".format(*shaded)


def latest_by(
    records: Iterable[Result],
    key_fn: Callable[[Result], RecordKey],
) -> list[Result]:
    """Keep only the newest result for each key, based on the timestamp field."""
    latest: dict[RecordKey, Result] = {}
    latest_timestamps: dict[RecordKey, str] = {}

    for record in records:
        key = key_fn(record)
        timestamp = run_timestamp(record)

        current = latest_timestamps.get(key)
        if current is None or timestamp > current:
            latest[key] = record
            latest_timestamps[key] = timestamp

    return list(latest.values())


def result_key(record: Result) -> RecordKey:
    """Key for one result per synthesizer/epsilon pair."""
    return (synthesizer_key(record), epsilon_of(record))


def utility_key(record: Result) -> RecordKey:
    """Key for one utility result per synthesizer/epsilon/classifier."""
    return (synthesizer_key(record), epsilon_of(record), classifier_key(record))


def filter_results(
    records: Iterable[Result],
    selected_synths: set[str],
    selected_epsilons: set[float],
) -> list[Result]:
    """Apply dashboard synthesizer and epsilon filters."""
    filtered: list[Result] = []

    for record in records:
        synth = synthesizer_key(record)
        epsilon = epsilon_of(record)

        if synth not in selected_synths:
            continue

        if epsilon is not None and epsilon not in selected_epsilons:
            continue

        filtered.append(record)

    return filtered
