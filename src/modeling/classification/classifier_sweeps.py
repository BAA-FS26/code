"""
Weights & Biases sweep workflows for classifier hyperparameter tuning.

This module runs validation-based hyperparameter sweeps and exports the best
parameters to YAML configuration files for later training runs.
"""

try:
    import wandb as _wandb
except ImportError:
    _wandb = None

from typing import Any, cast

import yaml

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


def fetch_best_params(classifier_name: str, data_source: str) -> None:
    """
    Fetch the best hyperparameters from the latest W&B sweep and save
    them to config/best_{classifier}_{data_source}.yaml.

    Queries W&B for the run with the highest val_f1_macro among all sweep
    runs for the given classifier and data source, then saves the
    parameters to disk for use with --mode best.

    Requires W&B to be configured.
    """
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

    best_run = max(runs, key=lambda r: r.summary.get("val_f1_macro", 0))
    best_params = {
        k: v
        for k, v in best_run.config.items()
        if not k.startswith("_") and k != "classifier"
    }

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    output_path = best_params_path(classifier_name, data_source)
    with open(output_path, "w") as f:
        yaml.safe_dump(best_params, f, sort_keys=True)

    print(f"[classify] Best params saved to {output_path.resolve()}")
    print(f"[classify] Best val_f1_macro: {best_run.summary.get('val_f1_macro'):.4f}")
    print(f"[classify] Params: {best_params}")


def run_sweep(classifier_name: str, data_source: str) -> None:
    """
    Initialise and run a W&B hyperparameter sweep for a classifier.

    Loads sweep configuration from config/sweep_{classifier_name}.yaml.
    The sweep optimises val_f1_macro and never touches the test set.
    Each run is named with abbreviated hyperparameter values for
    readability in the W&B UI.

    Requires W&B to be configured.
    """
    wandb_module = _require_wandb_support()

    config_path = CONFIG_DIR / f"sweep_{classifier_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Sweep configuration not found at {config_path}. "
            "Ensure the corresponding sweep YAML file exists."
        )

    with open(config_path) as f:
        sweep_config = yaml.safe_load(f)

    if not isinstance(sweep_config, dict):
        raise ValueError(
            f"Sweep configuration at {config_path} must load as a dictionary."
        )

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
            param_str = "_".join(
                f"{PARAM_ABBREVIATIONS.get(k, k)}={v}"
                for k, v in params.items()
                if k not in ["classifier", "seed"]
            )
            run_name = f"{data_source}_{classifier_name}_sweep_{param_str}"
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


def _require_wandb_support() -> Any:
    """
    Validate that W&B package support and environment configuration exist.

    Returns:
        The imported wandb module.

    Raises:
        RuntimeError: If wandb is unavailable or required configuration is missing.
    """
    if _wandb is None:
        raise RuntimeError(
            "The 'wandb' package is not installed, but a W&B mode was requested. "
            "Install project dependencies including wandb to use sweep or fetch_best."
        )
    require_wandb_config()
    return cast(Any, _wandb)
