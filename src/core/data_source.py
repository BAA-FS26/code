# src/core/data_source.py

"""
Helpers for resolving canonical data-source keys.

Keeps naming consistent across synthesis, evaluation, classification,
result logging, and dashboard loading.
"""

from src.utility.constants import DP_SYNTHESIZERS


def build_data_source_key(
    synthesizer_name: str,
    epsilon: float | None = None,
) -> str:
    """
    Build the canonical data-source key.

    Examples:
        ctgan -> "ctgan"
        dpctgan + 1.0 -> "dpctgan/eps_1.0"
    """
    if synthesizer_name in DP_SYNTHESIZERS:
        if epsilon is None:
            raise ValueError(
                f"Epsilon is required for DP synthesizer '{synthesizer_name}'."
            )
        return f"{synthesizer_name}/eps_{epsilon}"

    if epsilon is not None:
        raise ValueError("--epsilon should only be used with DP synthesizers.")

    return synthesizer_name


def resolve_training_data_source(
    data_source: str,
    synthesizer: str | None,
    epsilon: float | None,
) -> str:
    """
    Resolve the canonical training data source used by classification.

    Returns:
        - "real"
        - "ctgan"
        - "dpctgan/eps_1.0"
    """
    if synthesizer is None:
        if epsilon is not None:
            raise ValueError("--epsilon should only be used with DP synthesizers.")
        return data_source

    if data_source != "real":
        raise ValueError(
            "--synthesizer cannot be combined with a non-real --data_source."
        )

    return build_data_source_key(synthesizer_name=synthesizer, epsilon=epsilon)


def synthesizer_from_data_source(data_source: str) -> str | None:
    """
    Extract the synthesizer name from a canonical data-source key.

    Examples:
        "real" -> None
        "ctgan" -> "ctgan"
        "dpctgan/eps_1.0" -> "dpctgan"
    """
    if data_source == "real":
        return None

    return data_source.split("/")[0]


def epsilon_from_data_source(data_source: str) -> float | None:
    """
    Extract epsilon from a canonical DP data-source key.

    Examples:
        "ctgan" -> None
        "dpctgan/eps_1.0" -> 1.0
    """
    if "/eps_" not in data_source:
        return None

    return float(data_source.split("eps_", 1)[1])
