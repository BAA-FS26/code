"""
logger.py

Run logger abstraction for the synthetic data evaluation pipeline.

Provides a unified interface for logging results that works with or without
W&B. Local JSON output is the primary source of truth. W&B logging is purely
additive and must never replace local persistence.

All result files written under results/ follow the same envelope so they are
easy to consume later from an orchestration layer or dashboard.

Usage:
    from src.utility.logger import RunLogger

    with RunLogger(
        run_name="eval_fidelity_ctgan",
        script_name="evaluate_fidelity.py",
        parameters={"synthesizer": "ctgan"},
        use_wandb=True,
    ) as logger:
        logger.log({"quality_overall": 0.85})
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import wandb

from src.utility.constants import (
    DEFAULT_ENCODING,
    JSON_INDENT,
    RESULTS_DIR,
    RESULTS_KEY_PARAMETERS,
    RESULTS_KEY_RESULTS,
    RESULTS_KEY_RUN_NAME,
    RESULTS_KEY_SCHEMA_VERSION,
    RESULTS_KEY_SCRIPT,
    RESULTS_KEY_TIMESTAMP,
    RESULTS_SCHEMA_VERSION,
)
from src.utility.wandb_config import (
    get_wandb_entity,
    get_wandb_project,
    require_wandb_config,
)


class RunLogger:
    """
    Context manager that logs results locally and optionally to W&B.

    Local JSON output is always written under results/. W&B logging is
    additive and must not replace local persistence.

    Args:
        run_name: Unique identifier for this run, used as the local
                  filename and the W&B run name.
        script_name: Canonical script identifier for the result envelope.
        parameters: Parameter / metadata dictionary describing the run.
        use_wandb: If True, also log to W&B. Requires WANDB_ENTITY to
                   be set in the environment. Defaults to False.
        results_dir: Directory for local JSON results. Defaults to
                     RESULTS_DIR from constants.
        category: Optional result category subdirectory below results_dir,
            for example "fidelity", "privacy", or "utility".
    """

    def __init__(
        self,
        run_name: str,
        script_name: str,
        parameters: dict[str, Any],
        use_wandb: bool = False,
        results_dir: Path = RESULTS_DIR,
        category: str | None = None,  # NEW
    ) -> None:
        self.run_name = run_name
        self.script_name = script_name
        self.parameters = parameters
        self.use_wandb = use_wandb
        base_results_dir = Path(results_dir)
        self.results_dir = (
            base_results_dir / category if category is not None else base_results_dir
        )

        self._results: dict[str, Any] = {}
        self._history: list[dict[str, Any]] = []
        self._artifacts: list[dict[str, Any]] = []
        self._status: str = "success"
        self._error: dict[str, str] | None = None

        self._wandb_run: Any = None

    def __enter__(self) -> "RunLogger":
        if self.use_wandb:
            self._init_wandb()
        return self

    def __exit__(self, exc_type, exc_val, _exc_tb) -> None:
        if exc_type is not None:
            self._status = "failed"
            self._error = {
                "type": exc_type.__name__,
                "message": str(exc_val),
            }

        self._save_locally()

        if self._wandb_run is not None:
            self._wandb_run.finish()

    def log(self, results: dict[str, Any]) -> None:
        """
        Log a dictionary of results.

        The latest values are merged into the run-level results summary.
        Each call is also appended to local history so repeated keys
        (for example epoch-wise losses) are preserved in local output.

        Args:
            results: Dictionary of result names to scalar or JSON-serializable
                     values.
        """
        normalized = self._normalize_value(results)
        if not isinstance(normalized, dict):
            raise TypeError("RunLogger.log() expects a dictionary of results.")

        self._results.update(normalized)
        self._history.append(normalized)

        if self._wandb_run is not None:
            self._wandb_run.log(normalized)

    def log_table(self, key: str, dataframe: Any) -> None:
        """
        Log a tabular artifact.

        For local JSON output, a lightweight artifact manifest entry is
        recorded. For W&B, the full table is logged as a W&B Table.

        Args:
            key: Artifact key.
            dataframe: Tabular object expected to behave like a pandas DataFrame.
        """
        artifact_info = {
            "key": key,
            "type": "table",
            "rows": int(len(dataframe)) if hasattr(dataframe, "__len__") else None,
            "columns": (
                list(map(str, dataframe.columns))
                if hasattr(dataframe, "columns")
                else None
            ),
            "n_columns": (
                int(len(dataframe.columns)) if hasattr(dataframe, "columns") else None
            ),
        }
        self._artifacts.append(artifact_info)

        if self._wandb_run is not None:
            self._wandb_run.log({key: wandb.Table(dataframe=dataframe)})

    def _init_wandb(self) -> None:
        require_wandb_config()
        self._wandb_run = wandb.init(
            project=get_wandb_project(),
            entity=get_wandb_entity(),
            name=self.run_name,
            config=self._normalize_value(self.parameters),
        )

    def _save_locally(self) -> None:
        timestamp_dt = datetime.now(timezone.utc)
        timestamp = timestamp_dt.isoformat()

        date_str = timestamp_dt.strftime("%Y-%m-%d")
        time_str = timestamp_dt.strftime("%H%M%S")

        category = (
            self.results_dir.name if self.results_dir != Path(RESULTS_DIR) else None
        )

        output_dir = self.results_dir / date_str
        output_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            RESULTS_KEY_SCHEMA_VERSION: RESULTS_SCHEMA_VERSION,
            RESULTS_KEY_SCRIPT: self.script_name,
            "category": category,
            RESULTS_KEY_RUN_NAME: self.run_name,
            RESULTS_KEY_TIMESTAMP: timestamp,
            RESULTS_KEY_PARAMETERS: self._normalize_value(self.parameters),
            RESULTS_KEY_RESULTS: {
                "status": self._status,
                "summary": self._normalize_value(self._results),
                "history": self._normalize_value(self._history),
                "artifacts": self._normalize_value(self._artifacts),
            },
        }

        if self._error is not None:
            payload[RESULTS_KEY_RESULTS]["error"] = self._normalize_value(self._error)

        out_path = output_dir / f"{self.run_name}_{time_str}.json"

        with open(out_path, "w", encoding=DEFAULT_ENCODING) as f:
            json.dump(payload, f, indent=JSON_INDENT)

        print(f"Results saved to {out_path.resolve()}")

    def _normalize_value(self, value: Any) -> Any:
        """
        Convert common non-JSON-native values into JSON-safe equivalents.
        """
        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, Path):
            return str(value)

        if isinstance(value, dict):
            return {str(k): self._normalize_value(v) for k, v in value.items()}

        if isinstance(value, (list, tuple, set)):
            return [self._normalize_value(v) for v in value]

        if hasattr(value, "item") and callable(value.item):
            try:
                return self._normalize_value(value.item())
            except (ValueError, TypeError):
                pass

        if hasattr(value, "tolist") and callable(value.tolist):
            try:
                return self._normalize_value(value.tolist())
            except (ValueError, TypeError):
                pass

        return str(value)
