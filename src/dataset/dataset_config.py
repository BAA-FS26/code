# src/dataset/dataset_config.py

from dataclasses import dataclass

from src.dataset.adult_census import (
    CATEGORICAL_COLS,
    NUMERICAL_COLS,
    TARGET_COL,
    TARGET_MAP,
)


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    target_col: str
    target_map: dict[str, int]
    categorical_cols: list[str]
    numerical_cols: list[str]
    sensitive_cols: list[str]
    ordinal_cols: list[str]


ADULT_CENSUS_CONFIG = DatasetConfig(
    name="adult_census",
    target_col=TARGET_COL,
    target_map=TARGET_MAP,
    categorical_cols=CATEGORICAL_COLS,
    numerical_cols=NUMERICAL_COLS,
    sensitive_cols=["income", "occupation", "sex", "relationship"],
    ordinal_cols=["education-num"],
)


DATASET_CONFIGS = {
    ADULT_CENSUS_CONFIG.name: ADULT_CENSUS_CONFIG,
}


DEFAULT_DATASET = ADULT_CENSUS_CONFIG.name


def get_dataset_config(dataset_name: str = DEFAULT_DATASET) -> DatasetConfig:
    try:
        return DATASET_CONFIGS[dataset_name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. "
            f"Available datasets: {sorted(DATASET_CONFIGS)}"
        ) from exc
