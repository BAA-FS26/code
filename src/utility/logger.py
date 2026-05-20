"""
Run logger abstraction for the synthetic data evaluation pipeline.

Local JSON output is the primary source of truth. W&B logging is optional and
additive.
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
    """Context manager that logs results locally and optionally to W&B."""

    def __init__(
        self,
        run_name: str,
        script_name: str,
        parameters: dict[str, Any],
        use_wandb: bool = False,
        results_dir: Path = RESULTS_DIR,
        category: str | None = None,
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
        self._status = "success"
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
        """Log a dictionary of results."""
        normalized = self._normalize_value(results)

        if not isinstance(normalized, dict):
            raise TypeError("RunLogger.log() expects a dictionary of results.")

        self._results.update(normalized)
        self._history.append(normalized)

        if self._wandb_run is not None:
            self._wandb_run.log(normalized)

    def log_table(self, key: str, dataframe: Any) -> None:
        """Log a tabular artifact."""
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
        """Initialize a W&B run."""
        require_wandb_config()

        self._wandb_run = wandb.init(
            project=get_wandb_project(),
            entity=get_wandb_entity(),
            name=self.run_name,
            config=self._normalize_value(self.parameters),
        )

    def _save_locally(self) -> None:
        """Write the local JSON result file."""
        timestamp_dt = datetime.now(timezone.utc)
        output_path = self._build_output_path(timestamp_dt)

        payload = self._build_payload(timestamp_dt)

        with open(output_path, "w", encoding=DEFAULT_ENCODING) as file:
            json.dump(payload, file, indent=JSON_INDENT)

        print(f"Results saved to {output_path.resolve()}")

    def _build_output_path(self, timestamp_dt: datetime) -> Path:
        """Build the dated local JSON output path."""
        date_str = timestamp_dt.strftime("%Y-%m-%d")
        time_str = timestamp_dt.strftime("%H%M%S")

        output_dir = self.results_dir / date_str
        output_dir.mkdir(parents=True, exist_ok=True)

        return output_dir / f"{self.run_name}_{time_str}.json"

    def _build_payload(self, timestamp_dt: datetime) -> dict[str, Any]:
        """Build the stable result JSON envelope."""
        category = (
            self.results_dir.name if self.results_dir != Path(RESULTS_DIR) else None
        )

        payload = {
            RESULTS_KEY_SCHEMA_VERSION: RESULTS_SCHEMA_VERSION,
            RESULTS_KEY_SCRIPT: self.script_name,
            "category": category,
            RESULTS_KEY_RUN_NAME: self.run_name,
            RESULTS_KEY_TIMESTAMP: timestamp_dt.isoformat(),
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

        return payload

    def _normalize_value(self, value: Any) -> Any:
        """Convert common non-JSON-native values into JSON-safe equivalents."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, Path):
            return str(value)

        if isinstance(value, dict):
            return {str(key): self._normalize_value(val) for key, val in value.items()}

        if isinstance(value, (list, tuple, set)):
            return [self._normalize_value(item) for item in value]

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
