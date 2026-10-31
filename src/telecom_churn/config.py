"""Load and validate project configuration."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    raw_path: Path
    target_column: str


@dataclass(frozen=True)
class TrainingConfig:
    test_size: float
    model_path: Path


@dataclass(frozen=True)
class AppConfig:
    project_name: str
    random_seed: int
    data: DataConfig
    training: TrainingConfig
    log_level: str


def find_project_root() -> Path:
    """Find the repository root by locating pyproject.toml."""
    current = Path(__file__).resolve()

    for candidate in current.parents:
        if (candidate / "pyproject.toml").exists():
            return candidate

    raise FileNotFoundError("Could not find project root containing pyproject.toml")


def load_config(config_path: str | Path = "configs/dev.yaml") -> AppConfig:
    """Load YAML configuration and resolve project paths."""
    root = find_project_root()
    path = Path(config_path)

    if not path.is_absolute():
        path = root / path

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        raw: dict[str, Any] = yaml.safe_load(file) or {}

    project = raw["project"]
    data = raw["data"]
    training = raw["training"]
    logging = raw.get("logging", {})

    test_size = float(training["test_size"])
    if not 0 < test_size < 1:
        raise ValueError("training.test_size must be between 0 and 1")

    return AppConfig(
        project_name=str(project["name"]),
        random_seed=int(project["random_seed"]),
        data=DataConfig(
            raw_path=root / data["raw_path"],
            target_column=str(data["target_column"]),
        ),
        training=TrainingConfig(
            test_size=test_size,
            model_path=root / training["model_path"],
        ),
        log_level=str(logging.get("level", "INFO")).upper(),
    )