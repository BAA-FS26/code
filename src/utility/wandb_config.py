"""
wandb_config.py

Loads W&B configuration from environment variables.
Import this module only in code paths where W&B is actually used.

Required environment variables when W&B is enabled:
    WANDB_ENTITY   — your W&B username or team name

Optional environment variables:
    WANDB_PROJECT  — overrides the default project name defined in constants.py

Set these in a .env file (see .env.example) or export them in your shell.
They are never hardcoded in source so the repository is safe to share publicly.

Usage:
    from src.utility.wandb_config import get_wandb_entity, get_wandb_project
"""

import os

from src.utility.constants import WANDB_PROJECT as _DEFAULT_PROJECT


def get_wandb_entity() -> str:
    """
    Return the W&B entity from the environment.

    Raises:
        EnvironmentError: If WANDB_ENTITY is not set.
    """
    entity = os.environ.get("WANDB_ENTITY")
    if not entity:
        raise EnvironmentError(
            "WANDB_ENTITY environment variable is not set. "
            "Add it to your .env file or export it in your shell:\n"
            "    export WANDB_ENTITY=your_wandb_username\n"
            "See .env.example for all required variables."
        )
    return entity


def get_wandb_project() -> str:
    """
    Return the W&B project name, falling back to the default in constants.py.

    Override by setting WANDB_PROJECT in your environment or .env file.
    """
    return os.environ.get("WANDB_PROJECT", _DEFAULT_PROJECT)
