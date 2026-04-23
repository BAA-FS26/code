"""
wandb_config.py

Loads W&B configuration from environment variables.

Import this module only in code paths where W&B is actually requested.
W&B is optional in this project, and local execution without W&B is the
primary happy path. This module should therefore never be imported at
top level unless W&B usage is guaranteed for that code path.

Required environment variables when W&B logging is enabled:
    WANDB_ENTITY   — your W&B username or team name

Optional environment variables:
    WANDB_PROJECT  — overrides the default project name defined in constants.py

Set these in a .env file (see .env.example) or export them in your shell.
They are never hardcoded in source so the repository is safe to share publicly.

Usage:
    from src.utility.wandb_config import (
        get_wandb_entity,
        get_wandb_project,
        is_wandb_configured,
        require_wandb_config,
    )
"""

import os

from src.utility.constants import WANDB_PROJECT as _DEFAULT_PROJECT


def is_wandb_configured() -> bool:
    """
    Return True if the minimum required W&B environment configuration exists.

    W&B logging in this project requires WANDB_ENTITY. The project name is
    optional because it falls back to the default defined in constants.py.
    """
    return bool(os.environ.get("WANDB_ENTITY"))


def require_wandb_config() -> None:
    """
    Raise a RuntimeError if W&B logging was requested but not configured.
    """
    if not is_wandb_configured():
        raise RuntimeError(
            "WANDB_ENTITY is not set but W&B logging was requested. "
            "Set it in your environment or .env file, for example:\n"
            "    export WANDB_ENTITY=your_wandb_username\n"
            "See .env.example for the expected configuration."
        )


def get_wandb_entity() -> str:
    """
    Return the W&B entity from the environment.

    Raises:
        RuntimeError: If WANDB_ENTITY is not set.
    """
    require_wandb_config()
    return os.environ["WANDB_ENTITY"]


def get_wandb_project() -> str:
    """
    Return the W&B project name, falling back to the default in constants.py.

    Override by setting WANDB_PROJECT in your environment or .env file.
    """
    return os.environ.get("WANDB_PROJECT", _DEFAULT_PROJECT)
