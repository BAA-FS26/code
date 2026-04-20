"""
logger.py

Run logger abstraction for the synthetic data evaluation pipeline.

Provides a unified interface for logging metrics and config that works
with or without W&B. When W&B is disabled, all results are persisted
locally as JSON in the results/ directory.

This abstraction keeps all pipeline scripts free of direct wandb calls,
making W&B purely opt-in without scattering conditionals everywhere.

Usage:
    from src.utility.logger import RunLogger

    with RunLogger(run_name="eval_fidelity_ctgan", config={"synthesizer": "ctgan"}, use_wandb=True) as logger:
        logger.log({"quality_overall": 0.85})
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utility.constants import RESULTS_DIR


class RunLogger:
    """
    Context manager that logs metrics locally and optionally to W&B.

    All metrics are always written to a local JSON file under results/.
    W&B logging is additive — enabling it does not replace local logging.

    Args:
        run_name: Unique identifier for this run, used as the local
                  filename and the W&B run name.
        config:   Hyperparameter / metadata dictionary describing the run.
        use_wandb: If True, also log to W&B. Requires WANDB_ENTITY to
                   be set in the environment. Defaults to False.
        results_dir: Directory for local JSON results. Defaults to
                     RESULTS_DIR from constants.
    """

    def __init__(
        self,
        run_name: str,
        config: dict[str, Any],
        use_wandb: bool = False,
        results_dir: Path = RESULTS_DIR,
    ) -> None:
        self.run_name = run_name
        self.config = config
        self.use_wandb = use_wandb
        self.results_dir = Path(results_dir)
        self._metrics: dict[str, Any] = {}
        self._wandb_run = None

    def __enter__(self) -> "RunLogger":
        if self.use_wandb:
            self._init_wandb()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is None:
            self._save_locally()
        if self._wandb_run is not None:
            self._wandb_run.finish()

    def log(self, metrics: dict[str, Any]) -> None:
        """
        Log a dictionary of metrics.

        Merges into the accumulated metrics for local saving. Also logs
        to W&B immediately if enabled (preserving step-based logging for
        loss curves).

        Args:
            metrics: Dictionary of metric names to scalar values.
        """
        self._metrics.update(metrics)
        if self._wandb_run is not None:
            import wandb

            wandb.log(metrics)

    def log_table(self, key: str, dataframe) -> None:
        """
        Log a pandas DataFrame as a W&B Table (no-op locally).

        Args:
            key: W&B table key.
            dataframe: pandas DataFrame to log.
        """
        if self._wandb_run is not None:
            import wandb

            self._wandb_run.log({key: wandb.Table(dataframe=dataframe)})

    def _init_wandb(self) -> None:
        from src.utility.wandb_config import get_wandb_entity, get_wandb_project
        import wandb

        self._wandb_run = wandb.init(
            project=get_wandb_project(),
            entity=get_wandb_entity(),
            name=self.run_name,
            config=self.config,
        )

    def _save_locally(self) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        output = {
            "run_name": self.run_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": self.config,
            "metrics": self._metrics,
        }
        out_path = self.results_dir / f"{self.run_name}.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {out_path.resolve()}")
