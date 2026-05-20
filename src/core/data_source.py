"""
Canonical data-source key helpers.

The canonical data_source key is used consistently across:
- synthetic-data paths
- saved classifier model paths
- result logging
- dashboard loading
- privacy–utility joins

Examples:
    real
    ctgan
    tvae
    dpctgan/eps_1.0
    patectgan/eps_5.0
"""

from src.utility.constants import DP_SYNTHESIZERS

REAL_DATA_SOURCE = "real"
EPSILON_PREFIX = "eps_"


def build_data_source_key(
    synthesizer_name: str,
    epsilon: float | None = None,
) -> str:
    """Build the canonical data-source key for a synthesizer run."""
    is_dp_synthesizer = synthesizer_name in DP_SYNTHESIZERS

    if is_dp_synthesizer:
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
    """Resolve the canonical training data source used for classification."""
    if synthesizer is None:
        if epsilon is not None:
            raise ValueError("--epsilon should only be used with DP synthesizers.")

        return data_source

    if data_source != REAL_DATA_SOURCE:
        raise ValueError(
            "--synthesizer cannot be combined with a non-real --data_source."
        )

    return build_data_source_key(
        synthesizer_name=synthesizer,
        epsilon=epsilon,
    )


def synthesizer_from_data_source(data_source: str) -> str | None:
    """Extract the synthesizer name from a canonical data-source key."""
    if data_source == REAL_DATA_SOURCE:
        return None

    return data_source.split("/", maxsplit=1)[0]


def epsilon_from_data_source(data_source: str) -> float | None:
    """Extract epsilon from a canonical DP data-source key."""
    marker = f"/{EPSILON_PREFIX}"

    if marker not in data_source:
        return None

    epsilon_str = data_source.split(EPSILON_PREFIX, maxsplit=1)[1]

    try:
        return float(epsilon_str)
    except ValueError as exc:
        raise ValueError(
            f"Could not parse epsilon from data_source '{data_source}'."
        ) from exc


def is_dp_data_source(data_source: str) -> bool:
    """Return True if a data-source key refers to a DP synthesizer run."""
    synthesizer = synthesizer_from_data_source(data_source)
    return synthesizer in DP_SYNTHESIZERS
