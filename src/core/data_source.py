# src/core/data_source.py

"""
Canonical data-source key helpers.

The data_source key is used consistently across:
- synthetic data paths
- classifier model paths
- result logging
- dashboard loading
- privacy/utility trade-off joins
"""

from src.utility.constants import DP_SYNTHESIZERS

REAL_DATA_SOURCE = "real"
EPSILON_PREFIX = "eps_"


def build_data_source_key(
    synthesizer_name: str,
    epsilon: float | None = None,
) -> str:
    """Build the canonical data-source key."""
    if synthesizer_name in DP_SYNTHESIZERS:
        if epsilon is None:
            raise ValueError(
                f"Epsilon is required for DP synthesizer '{synthesizer_name}'."
            )
        return f"{synthesizer_name}/{EPSILON_PREFIX}{epsilon}"

    if epsilon is not None:
        raise ValueError("--epsilon should only be used with DP synthesizers.")

    return synthesizer_name


def resolve_training_data_source(
    data_source: str,
    synthesizer: str | None,
    epsilon: float | None,
) -> str:
    """Resolve the canonical training data source used by classification."""
    if synthesizer is None:
        if epsilon is not None:
            raise ValueError("--epsilon should only be used with DP synthesizers.")
        return data_source

    if data_source != REAL_DATA_SOURCE:
        raise ValueError(
            "--synthesizer cannot be combined with a non-real --data_source."
        )

    return build_data_source_key(synthesizer_name=synthesizer, epsilon=epsilon)


def synthesizer_from_data_source(data_source: str) -> str | None:
    """Extract synthesizer name from a canonical data-source key."""
    if data_source == REAL_DATA_SOURCE:
        return None

    return data_source.split("/")[0]


def epsilon_from_data_source(data_source: str) -> float | None:
    """Extract epsilon from a canonical DP data-source key."""
    marker = f"/{EPSILON_PREFIX}"
    if marker not in data_source:
        return None

    return float(data_source.split(EPSILON_PREFIX, 1)[1])


def is_dp_data_source(data_source: str) -> bool:
    """Return True if data_source refers to a DP synthetic source."""
    synthesizer = synthesizer_from_data_source(data_source)
    return synthesizer in DP_SYNTHESIZERS
