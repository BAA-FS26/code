"""
Weights & Biases sweep workflows for classifier hyperparameter tuning.

Runs validation-based hyperparameter sweeps and exports the best parameters
to YAML files for later best-model training.
"""

from typing import Any, cast

import yaml

try:
    import wandb as _wandb
except ImportError:
    _wandb = None

from src.core.paths import best_params_path
from src.modeling.classification.classifier_training import train_and_validate
from src.utility.constants import CONFIG_DIR
from src.utility.wandb_config import (
    get_wandb_entity,
    get_wandb_project,
    require_wandb_config,
)

PARAM_ABBREVIATIONS = {
    "max_features": "mf",
    "min_samples_leaf": "msl",
    "max_depth": "md",
    "learning_rate": "lr",
    "max_leaf_nodes": "mln",
    "C": "C",
}

EXCLUDED_RUN_NAME_PARAMS = {"classifier", "seed"}
SWEEP_SCORE_KEY = "val_f1_macro"


def _require_wandb_support() -> Any:
    """Validate W&B package availability and required environment config."""
    if _wandb is None:
        raise RuntimeError(
            "The 'wandb' package is not installed, but a W&B mode was requested. "
            "Install project dependencies including wandb to use sweep or fetch_best."
        )

    require_wandb_config()
    return cast(Any, _wandb)


def _load_sweep_config(classifier_name: str) -> dict[str, Any]:
    """Load a classifier sweep config from config/sweep_{classifier}.yaml."""
    config_path = CONFIG_DIR / f"sweep_{classifier_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Sweep configuration not found at {config_path}. "
            "Ensure the corresponding sweep YAML file exists."
        )

    with open(config_path) as file:
        sweep_config = yaml.safe_load(file)

    if not isinstance(sweep_config, dict):
        raise ValueError(
            f"Sweep configuration at {config_path} must load as a dictionary."
        )

    return sweep_config


def _build_sweep_run_name(
    classifier_name: str,
    data_source: str,
    params: dict[str, Any],
) -> str:
    """Build a readable W&B sweep run name."""
    param_str = "_".join(
        f"{PARAM_ABBREVIATIONS.get(key, key)}={value}"
        for key, value in params.items()
        if key not in EXCLUDED_RUN_NAME_PARAMS
    )

    return f"{data_source}_{classifier_name}_sweep_{param_str}"


def fetch_best_params(classifier_name: str, data_source: str) -> None:
    """Fetch best W&B sweep params and save them as YAML."""
    wandb_module = _require_wandb_support()
    api = wandb_module.Api()

    runs = list(
        api.runs(
            f"{get_wandb_entity()}/{get_wandb_project()}",
            filters={
                "display_name": {"$regex": f"^{data_source}_{classifier_name}_sweep"}
            },
        )
    )

    if not runs:
        raise RuntimeError(
            f"No W&B sweep runs found for classifier '{classifier_name}' "
            f"and data source '{data_source}'. Run --mode sweep first."
        )

    best_run = max(runs, key=lambda run: run.summary.get(SWEEP_SCORE_KEY, 0))
    best_params = {
        key: value
        for key, value in best_run.config.items()
        if not key.startswith("_") and key != "classifier"
    }

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    output_path = best_params_path(classifier_name, data_source)
    with open(output_path, "w") as file:
        yaml.safe_dump(best_params, file, sort_keys=True)

    print(f"[classify] Best params saved to {output_path.resolve()}")
    print(
        f"[classify] Best {SWEEP_SCORE_KEY}: {best_run.summary.get(SWEEP_SCORE_KEY):.4f}"
    )
    print(f"[classify] Params: {best_params}")


def run_sweep(classifier_name: str, data_source: str) -> None:
    """Initialize and run a W&B hyperparameter sweep."""
    wandb_module = _require_wandb_support()
    sweep_config = _load_sweep_config(classifier_name)

    print(
        f"[classify] Starting W&B sweep for classifier '{classifier_name}' "
        f"on source '{data_source}'"
    )

    sweep_id = wandb_module.sweep(
        sweep_config,
        project=get_wandb_project(),
        entity=get_wandb_entity(),
    )

    def sweep_run() -> None:
        with wandb_module.init() as run:
            if run is None:
                raise RuntimeError("wandb.init() returned None during sweep run.")

            params = dict(wandb_module.config)
            run_name = _build_sweep_run_name(
                classifier_name=classifier_name,
                data_source=data_source,
                params=params,
            )

            run.name = run_name

            train_and_validate(
                classifier_name=classifier_name,
                data_source=data_source,
                params=params,
                run_name=run_name,
                mode="sweep",
                save_model=False,
                use_wandb=True,
            )

    wandb_module.agent(sweep_id, function=sweep_run)
