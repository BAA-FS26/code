"""Result loading and normalization helpers."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import streamlit as st

from src.dashboard.config import (
    COLORS,
    DEFAULT_COLOR,
    EPSILON_SHADE,
    SYNTHESIZER_LABELS,
)
from src.utility.constants import DP_SYNTHESIZERS

LOGGER = logging.getLogger(__name__)
RESULT_CATEGORIES = ("fidelity", "privacy", "utility")
Result = dict[str, Any]
ResultMap = dict[str, list[Result]]
RecordKey = tuple[Any, ...]


@st.cache_data(show_spinner=False)
def load_all_results(results_dir: Path) -> ResultMap:
    """Load all JSON result files grouped by category.

    Files are read from ``results/{category}/**/*.json`` and returned newest-first by
    path order. Invalid JSON files are skipped and logged instead of silently ignored.
    """
    results: ResultMap = {category: [] for category in RESULT_CATEGORIES}
    if not results_dir.exists():
        LOGGER.warning("Results directory does not exist: %s", results_dir)
        return results

    for category in RESULT_CATEGORIES:
        category_dir = results_dir / category
        if not category_dir.exists():
            continue

        for json_file in sorted(category_dir.rglob("*.json"), reverse=True):
            try:
                record = json.loads(json_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                LOGGER.warning("Skipping invalid JSON file %s: %s", json_file, exc)
                continue
            except OSError as exc:
                LOGGER.warning("Could not read result file %s: %s", json_file, exc)
                continue

            if isinstance(record, dict):
                record["_file"] = str(json_file)
                results[category].append(record)
            else:
                LOGGER.warning("Skipping non-object JSON result: %s", json_file)

    return results


def parameters(record: Result) -> dict[str, Any]:
    """Return the parameters section for a result record."""
    return record.get("parameters", {}) or {}


def summary(record: Result) -> dict[str, Any]:
    """Return the nested results.summary section for a result record."""
    return record.get("results", {}).get("summary", {}) or {}


def synthesizer_key(record: Result) -> str:
    """Return a canonical synthesizer identifier from a result record.

    Some result files store DP runs in ``data_source`` values such as
    ``dpctgan/eps_0.1`` instead of separate ``synthesizer`` and ``epsilon``
    parameters.  For dashboard grouping we only want the model name here.
    """
    params = parameters(record)
    synth = params.get("synthesizer") or params.get("data_source") or "unknown"
    synth_key = str(synth).lower().strip().replace("\\", "/")
    return synth_key.split("/", 1)[0]


def classifier_key(record: Result) -> str:
    """Return the classifier identifier for a utility result."""
    return str(parameters(record).get("classifier", ""))


def epsilon_of(record: Result) -> float | None:
    """Return epsilon as a float, or ``None`` for non-DP records.

    Supports both explicit ``parameters.epsilon`` and encoded data-source paths
    such as ``dpctgan/eps_0.1``.
    """
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
    records: Iterable[Result], key_fn: Callable[[Result], RecordKey]
) -> list[Result]:
    """Keep only the newest result for each key, based on the timestamp field."""
    latest: dict[RecordKey, Result] = {}
    for record in records:
        key = key_fn(record)
        timestamp = str(record.get("timestamp", ""))
        if key not in latest or timestamp > str(latest[key].get("timestamp", "")):
            latest[key] = record
    return list(latest.values())


def result_key(record: Result) -> RecordKey:
    """Key for one result per synthesizer/epsilon pair."""
    return (synthesizer_key(record), epsilon_of(record))


def utility_key(record: Result) -> RecordKey:
    """Key for one utility result per synthesizer/epsilon/classifier."""
    return (synthesizer_key(record), epsilon_of(record), classifier_key(record))


def filter_results(
    records: Iterable[Result], selected_synths: set[str], selected_epsilons: set[float]
) -> list[Result]:
    """Apply dashboard synthesizer and epsilon filters."""
    return [
        record
        for record in records
        if synthesizer_key(record) in selected_synths
        and (epsilon_of(record) is None or epsilon_of(record) in selected_epsilons)
    ]
