# src/core/data_source.py

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
    """
    Build the canonical data-source key for a synthesizer run.

    Non-DP synthesizers use:
        {synthesizer_name}

    DP synthesizers use:
        {synthesizer_name}/eps_{epsilon}

    Examples:
        ctgan -> ctgan
        dpctgan + 1.0 -> dpctgan/eps_1.0

    Args:
        synthesizer_name:
            Synthesizer identifier.

        epsilon:
            Privacy budget for DP synthesizers.

    Returns:
        Canonical data-source key.

    Raises:
        ValueError:
            If epsilon usage is inconsistent with the synthesizer type.
    """
    is_dp_synthesizer = synthesizer_name in DP_SYNTHESIZERS

    if is_dp_synthesizer:
        if epsilon is None:
            raise ValueError(
                f"Epsilon is required for DP synthesizer " f"'{synthesizer_name}'."
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
    """
    Resolve the canonical training data source used for classification.

    Real-data training:
        data_source='real'
        synthesizer=None

    Synthetic-data training:
        data_source='real'
        synthesizer='ctgan'
        -> returns 'ctgan'

    DP synthetic-data training:
        data_source='real'
        synthesizer='dpctgan'
        epsilon=1.0
        -> returns 'dpctgan/eps_1.0'

    Args:
        data_source:
            Base training source. Currently expected to be 'real'.

        synthesizer:
            Optional synthetic-data generator.

        epsilon:
            Optional DP privacy budget.

    Returns:
        Canonical training data-source key.

    Raises:
        ValueError:
            If incompatible argument combinations are provided.
    """
    if synthesizer is None:
        if epsilon is not None:
            raise ValueError("--epsilon should only be used with DP synthesizers.")

        return data_source

    if data_source != REAL_DATA_SOURCE:
        raise ValueError(
            "--synthesizer cannot be combined with a non-real " "--data_source."
        )

    return build_data_source_key(
        synthesizer_name=synthesizer,
        epsilon=epsilon,
    )


def synthesizer_from_data_source(
    data_source: str,
) -> str | None:
    """
    Extract the synthesizer name from a canonical data-source key.

    Examples:
        real -> None
        ctgan -> ctgan
        dpctgan/eps_1.0 -> dpctgan
    """
    if data_source == REAL_DATA_SOURCE:
        return None

    return data_source.split("/")[0]


def epsilon_from_data_source(
    data_source: str,
) -> float | None:
    """
    Extract epsilon from a canonical DP data-source key.

    Examples:
        real -> None
        ctgan -> None
        dpctgan/eps_1.0 -> 1.0

    Raises:
        ValueError:
            If the epsilon component cannot be parsed as a float.
    """
    marker = f"/{EPSILON_PREFIX}"

    if marker not in data_source:
        return None

    epsilon_str = data_source.split(EPSILON_PREFIX, 1)[1]

    try:
        return float(epsilon_str)

    except ValueError as exc:
        raise ValueError(
            f"Could not parse epsilon from data_source " f"'{data_source}'."
        ) from exc


def is_dp_data_source(data_source: str) -> bool:
    """
    Return True if a data-source key refers to a DP synthesizer run.

    Examples:
        real -> False
        ctgan -> False
        dpctgan/eps_1.0 -> True
    """
    synthesizer = synthesizer_from_data_source(data_source)

    return synthesizer in DP_SYNTHESIZERS
