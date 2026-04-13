from __future__ import annotations

import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any

import yaml

FIGURES_DIR = Path("data/figures")
PROCESSED_DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")


class ConfigLoader:

    _ENV_PREFIX = "MLOPS__"

    def __init__(self, path: str = "config/config.yaml") -> None:
        self._path = Path(path)
        self._data: dict[str, Any] = self._load()
        self._apply_env_overrides()

    def get(self, section: str, key: str | None = None, default: Any = None) -> Any:
        section_data = self._data.get(section, {})
        if key is None:
            return section_data
        return section_data.get(key, default)

    def __getitem__(self, item: str) -> Any:
        return self._data[item]

    def __repr__(self) -> str:
        return f"ConfigLoader(path={self._path})"

    def _load(self) -> dict[str, Any]:
        if not self._path.exists():
            raise FileNotFoundError(
                f"Config file not found: {self._path}\n"
                f"Expected at: {self._path.resolve()}"
            )
        with self._path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    def _apply_env_overrides(self) -> None:
        for env_key, env_val in os.environ.items():
            if not env_key.startswith(self._ENV_PREFIX):
                continue
            parts = env_key[len(self._ENV_PREFIX) :].lower().split("__")
            if len(parts) == 2:
                section, key = parts
                if section in self._data and isinstance(self._data[section], dict):
                    if key in self._data[section]:
                        self._data[section][key] = _cast(env_val)

    @classmethod
    def from_env(cls) -> "ConfigLoader":
        path = os.getenv("MLOPS_CONFIG", "config/config.yaml")
        return cls(path)


def get_logger(
    name: str,
    log_dir: str = "logs",
    level: str = "INFO",
    log_to_file: bool = True,
) -> logging.Logger:

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_to_file:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = TimedRotatingFileHandler(
            filename=Path(log_dir) / "pipeline.log",
            when="midnight",
            backupCount=30,
            encoding="utf-8",
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger


_DEFAULT_CONFIG_PATH = "config/config.yaml"

SUPPORTED_SOURCE_TYPES = frozenset({"csv", "parquet", "json", "xlsx"})

SUPPORTED_ALGORITHMS = frozenset({"random_forest", "xgboost", "lightgbm"})
SUPPORTED_SCALERS = frozenset({"standard", "minmax", "robust"})


def _cast(value: str) -> Any:
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    return value
