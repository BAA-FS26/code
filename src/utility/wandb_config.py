"""
Weights & Biases configuration helpers.

W&B is optional in this project. Local JSON logging remains the primary source
of truth; W&B logging is additive and only used when explicitly requested.
"""

import os

from src.utility.constants import WANDB_PROJECT as _DEFAULT_PROJECT

WANDB_ENTITY_ENV = "WANDB_ENTITY"
WANDB_PROJECT_ENV = "WANDB_PROJECT"


def is_wandb_configured() -> bool:
    """Return True if the minimum required W&B configuration exists."""
    return bool(os.environ.get(WANDB_ENTITY_ENV))


def require_wandb_config() -> None:
    """Raise an error if W&B logging was requested but is not configured."""
    if not is_wandb_configured():
        raise RuntimeError(
            f"{WANDB_ENTITY_ENV} is not set but W&B logging was requested. "
            "Set it in your environment or .env file, for example:\n"
            f"    export {WANDB_ENTITY_ENV}=your_wandb_username\n"
            "See .env.example for the expected configuration."
        )


def get_wandb_entity() -> str:
    """Return the configured W&B entity."""
    require_wandb_config()
    return os.environ[WANDB_ENTITY_ENV]


def get_wandb_project() -> str:
    """Return the configured W&B project name."""
    return os.environ.get(WANDB_PROJECT_ENV, _DEFAULT_PROJECT)
