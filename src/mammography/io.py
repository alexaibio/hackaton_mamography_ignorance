from pathlib import Path

import pandas as pd

from mammography.config import get_settings

STAGES = ("raw_fhir", "raw_csv", "interim", "processed")


def table_path(stage: str, name: str) -> Path:
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}, expected one of {STAGES}")
    return get_settings().data_folder / stage / f"{name}.csv"


def read_table(stage: str, name: str, parse_dates: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(table_path(stage, name), parse_dates=parse_dates)


def write_table(df: pd.DataFrame, stage: str, name: str) -> Path:
    path = table_path(stage, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def model_path(name: str) -> Path:
    return get_settings().models_folder / name
