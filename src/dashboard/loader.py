"""

loader.py

"""

import json
import streamlit as st
from pathlib import Path
from typing import Optional

from src.dashboard.config import COLORS, EPSILON_SHADE, SYNTHESIZER_LABELS
from src.utility.constants import DP_SYNTHESIZERS


@st.cache_data
def load_all_results(results_dir: Path) -> dict[str, list[dict]]:
    """
    Scan results/{category}/YYYY-MM-DD/*.json and return
    {category: [parsed_result, ...]} sorted newest-first.
    """
    out: dict[str, list[dict]] = {"fidelity": [], "privacy": [], "utility": []}
    if not results_dir.exists():
        return out
    for category in out:
        cat_dir = results_dir / category
        if not cat_dir.exists():
            continue
        for json_file in sorted(cat_dir.rglob("*.json"), reverse=True):
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
                data["_file"] = str(json_file)
                out[category].append(data)
            except Exception:
                pass
    return out


def synthesizer_key(result: dict) -> str:
    """Return a canonical synthesizer identifier from a result record."""
    params = result.get("parameters", {})
    synth = params.get("synthesizer") or params.get("data_source") or "unknown"
    return synth.lower()


def epsilon_of(result: dict) -> Optional[float]:
    eps = result.get("parameters", {}).get("epsilon")
    return float(eps) if eps is not None else None


def source_label(synth: str, eps: Optional[float]) -> str:
    base = SYNTHESIZER_LABELS.get(synth, synth)
    if eps is not None:
        return f"{base} ε={eps}"
    return base


def get_color(synth: str, eps: Optional[float] = None) -> str:
    base_hex = COLORS.get(synth, "#94A3B8")
    if eps is None or synth not in DP_SYNTHESIZERS:
        return base_hex
    # Darken/lighten by blending with white using shade factor
    shade = EPSILON_SHADE.get(eps, 1.0)
    r = int(base_hex[1:3], 16)
    g = int(base_hex[3:5], 16)
    b = int(base_hex[5:7], 16)
    r2 = int(r * shade + 255 * (1 - shade))
    g2 = int(g * shade + 255 * (1 - shade))
    b2 = int(b * shade + 255 * (1 - shade))
    return f"#{r2:02X}{g2:02X}{b2:02X}"


def dedup_latest(records: list[dict], key_fn) -> list[dict]:
    """Keep only the most-recent result per key (by timestamp)."""
    seen: dict = {}
    for r in records:
        k = key_fn(r)
        ts = r.get("timestamp", "")
        if k not in seen or ts > seen[k]["timestamp"]:
            seen[k] = r
    return list(seen.values())
