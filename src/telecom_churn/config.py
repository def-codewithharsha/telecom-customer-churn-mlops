"""Load and validate project configuration."""

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from telecom_churn.exception import ConfigurationError


@dataclass(frozen=True)
class DataConfig:
    raw_path: Path
    target_column: str


@dataclass(frozen=True)
class TrainingConfig:
    test_size: float
    threshold_beta: float
    calibration_method: str
    model_path: Path


@dataclass(frozen=True)
class QualityGateConfig:
    min_cv_average_precision: float
    min_holdout_average_precision: float
    min_improvement_over_baseline: float


@dataclass(frozen=True)
class AppConfig:
    project_name: str
    random_seed: int
    data: DataConfig
    training: TrainingConfig
    quality_gates: QualityGateConfig
    log_level: str


def find_project_root() -> Path:
    """Find the repository root by locating pyproject.toml."""
    current = Path(__file__).resolve()

    for candidate in current.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate

    raise ConfigurationError("Could not find project root containing pyproject.toml")


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Load the explicitly selected config or TELECOM_CHURN_CONFIG profile.

    There is intentionally no implicit development fallback. Each process must
    select its configuration explicitly so a production service cannot silently
    start with development settings.
    """
    if config_path is None:
        config_path = os.getenv("TELECOM_CHURN_CONFIG")
    if config_path is None or not str(config_path).strip():
        raise ConfigurationError(
            "Configuration profile is not selected. Pass config_path or set "
            "TELECOM_CHURN_CONFIG (for example, configs/dev.yaml or configs/prod.yaml)."
        )

    root = find_project_root()
    path = Path(config_path)
    if not path.is_absolute():
        path = root / path

    if not path.is_file():
        raise ConfigurationError(f"Configuration file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            raw: Any = yaml.safe_load(file)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Could not read configuration at {path}: {exc}") from exc

    if not isinstance(raw, Mapping):
        raise ConfigurationError("Configuration document must be a YAML mapping.")

    _check_keys(
        raw,
        {"project", "data", "training", "quality_gates", "logging"},
        "root",
    )
    project = _require_mapping(raw, "project")
    data = _require_mapping(raw, "data")
    training = _require_mapping(raw, "training")
    quality_gates = _require_mapping(raw, "quality_gates")
    logging_config = raw.get("logging", {})
    if not isinstance(logging_config, Mapping):
        raise ConfigurationError("Configuration section 'logging' must be a mapping.")

    _check_keys(project, {"name", "random_seed"}, "project")
    _check_keys(data, {"raw_path", "target_column"}, "data")
    _check_keys(
        training,
        {"test_size", "threshold_beta", "calibration_method", "model_path"},
        "training",
    )
    _check_keys(
        quality_gates,
        {
            "min_cv_average_precision",
            "min_holdout_average_precision",
            "min_improvement_over_baseline",
        },
        "quality_gates",
    )
    _check_keys(logging_config, {"level"}, "logging")

    project_name = _require_nonempty_string(project, "name", "project")
    random_seed = _require_integer(project, "random_seed", "project")
    raw_path = _resolve_path(root, _require_nonempty_string(data, "raw_path", "data"))
    target_column = _require_nonempty_string(data, "target_column", "data")
    model_path = _resolve_path(
        root,
        _require_nonempty_string(training, "model_path", "training"),
    )
    test_size = training.get("test_size")
    if (
        isinstance(test_size, bool)
        or not isinstance(test_size, (int, float))
        or not math.isfinite(test_size)
        or not 0 < test_size < 1
    ):
        raise ConfigurationError("training.test_size must be a finite number between 0 and 1.")
    threshold_beta = training.get("threshold_beta")
    if (
        isinstance(threshold_beta, bool)
        or not isinstance(threshold_beta, (int, float))
        or not math.isfinite(threshold_beta)
        or threshold_beta <= 0
    ):
        raise ConfigurationError("training.threshold_beta must be a finite number greater than 0.")
    calibration_method = training.get("calibration_method")
    if not isinstance(calibration_method, str) or calibration_method not in {
        "sigmoid",
        "isotonic",
    }:
        raise ConfigurationError(
            "training.calibration_method must be either 'sigmoid' or 'isotonic'."
        )

    min_cv_average_precision = _require_metric(
        quality_gates,
        "min_cv_average_precision",
    )
    min_holdout_average_precision = _require_metric(
        quality_gates,
        "min_holdout_average_precision",
    )
    min_improvement_over_baseline = _require_metric(
        quality_gates,
        "min_improvement_over_baseline",
    )

    log_level = logging_config.get("level", "INFO")
    if not isinstance(log_level, str) or log_level.upper() not in {
        "CRITICAL",
        "ERROR",
        "WARNING",
        "INFO",
        "DEBUG",
        "NOTSET",
    }:
        raise ConfigurationError(
            "logging.level must be one of CRITICAL, ERROR, WARNING, INFO, DEBUG, or NOTSET."
        )

    return AppConfig(
        project_name=project_name,
        random_seed=random_seed,
        data=DataConfig(raw_path=raw_path, target_column=target_column),
        training=TrainingConfig(
            test_size=float(test_size),
            threshold_beta=float(threshold_beta),
            calibration_method=calibration_method,
            model_path=model_path,
        ),
        quality_gates=QualityGateConfig(
            min_cv_average_precision=min_cv_average_precision,
            min_holdout_average_precision=min_holdout_average_precision,
            min_improvement_over_baseline=min_improvement_over_baseline,
        ),
        log_level=log_level.upper(),
    )


def _require_mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Return a required section as a mapping or raise a useful config error."""
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"Configuration section {key!r} must be a mapping.")
    return value


def _check_keys(section: Mapping[str, Any], allowed: set[str], name: str) -> None:
    """Reject misspelled or unsupported configuration keys."""
    if any(not isinstance(key, str) for key in section):
        raise ConfigurationError(f"Configuration section {name!r} keys must be strings.")
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ConfigurationError(f"Configuration section {name!r} has unknown keys: {unknown}.")


def _require_nonempty_string(
    section: Mapping[str, Any],
    key: str,
    section_name: str,
) -> str:
    """Read a required non-empty string without coercing YAML values."""
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(
            f"Configuration value {section_name}.{key} must be a non-empty string."
        )
    return value.strip()


def _require_integer(section: Mapping[str, Any], key: str, section_name: str) -> int:
    """Read a required integer and reject booleans, which YAML also treats as ints."""
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"Configuration value {section_name}.{key} must be an integer.")
    if value < 0:
        raise ConfigurationError(f"Configuration value {section_name}.{key} must be non-negative.")
    return value


def _require_metric(section: Mapping[str, Any], key: str) -> float:
    """Read a required finite probability or metric floor in the range [0, 1]."""
    value = section.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ConfigurationError(
            f"Configuration value quality_gates.{key} must be a finite number from 0 to 1."
        )
    return float(value)


def _resolve_path(root: Path, value: str) -> Path:
    """Resolve relative paths from the repository root; preserve absolute paths."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path
