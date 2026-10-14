from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        extra="ignore",
    )

    log_level: str = "INFO"
    log_dir: Path
    data_folder: Path
    models_folder: Path
    fhir_base_url: str = ""
    fhir_token: str = ""

    @field_validator("log_dir", "data_folder", "models_folder")
    @classmethod
    def _anchor_to_project_root(cls, value: Path) -> Path:
        return value if value.is_absolute() else PROJECT_ROOT / value

    @property
    def raw_dir(self) -> Path:
        return self.data_folder / "raw"

    @property
    def interim_dir(self) -> Path:
        return self.data_folder / "interim"

    @property
    def processed_dir(self) -> Path:
        return self.data_folder / "processed"


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)
