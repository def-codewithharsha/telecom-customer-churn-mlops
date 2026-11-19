"""Tests for configuration loading and validation."""

from pathlib import Path

import pytest
import yaml

from telecom_churn import config
from telecom_churn.exception import ConfigurationError


@pytest.fixture
def config_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Provide an isolated project root for config loading tests."""
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    (tmp_path / "configs").mkdir()
    monkeypatch.setattr(config, "find_project_root", lambda: tmp_path)
    return tmp_path


def _write_config(root: Path, contents: str) -> Path:
    path = root / "configs" / "dev.yaml"
    path.write_text(contents, encoding="utf-8")
    return path


def _valid_config() -> dict[str, object]:
    return {
        "project": {"name": "churn-test", "random_seed": 17},
        "data": {"raw_path": "data/churn.csv", "target_column": "Churn"},
        "training": {"test_size": 0.25, "model_path": "models/model.skops"},
        "quality_gates": {
            "min_cv_average_precision": 0.6,
            "min_holdout_average_precision": 0.55,
            "min_improvement_over_baseline": 0.05,
        },
        "logging": {"level": "debug"},
    }


def test_load_config_validates_values_and_resolves_relative_paths(
    config_root: Path,
) -> None:
    _write_config(config_root, yaml.safe_dump(_valid_config()))

    result = config.load_config()

    assert result.project_name == "churn-test"
    assert result.random_seed == 17
    assert result.training.test_size == 0.25
    assert result.quality_gates.min_cv_average_precision == 0.6
    assert result.log_level == "DEBUG"
    assert result.data.raw_path == config_root / "data" / "churn.csv"
    assert result.training.model_path == config_root / "models" / "model.skops"


def test_load_config_reports_missing_file(config_root: Path) -> None:
    with pytest.raises(ConfigurationError, match="Configuration file not found"):
        config.load_config("configs/missing.yaml")


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("", "must be a YAML mapping"),
        ("project: [unterminated", "Could not read configuration"),
        ("project: {}\ndata: {}\n", "section 'training'"),
    ],
)
def test_load_config_rejects_malformed_or_incomplete_documents(
    config_root: Path,
    contents: str,
    message: str,
) -> None:
    _write_config(config_root, contents)

    with pytest.raises(ConfigurationError, match=message):
        config.load_config()


@pytest.mark.parametrize("test_size", [0, 1, -0.1, True, float("nan"), "0.2"])
def test_load_config_rejects_invalid_test_size(
    config_root: Path,
    test_size: object,
) -> None:
    values = _valid_config()
    values["training"]["test_size"] = test_size  # type: ignore[index]
    _write_config(config_root, yaml.safe_dump(values))

    with pytest.raises(ConfigurationError, match="training.test_size"):
        config.load_config()


def test_load_config_rejects_unknown_keys(config_root: Path) -> None:
    values = _valid_config()
    values["training"]["test_szie"] = 0.2  # type: ignore[index]
    _write_config(config_root, yaml.safe_dump(values))

    with pytest.raises(ConfigurationError, match="unknown keys.*test_szie"):
        config.load_config()


def test_load_config_rejects_invalid_random_seed_and_log_level(
    config_root: Path,
) -> None:
    values = _valid_config()
    values["project"]["random_seed"] = True  # type: ignore[index]
    _write_config(config_root, yaml.safe_dump(values))

    with pytest.raises(ConfigurationError, match="random_seed must be an integer"):
        config.load_config()

    values = _valid_config()
    values["logging"]["level"] = "TRACE"  # type: ignore[index]
    _write_config(config_root, yaml.safe_dump(values))

    with pytest.raises(ConfigurationError, match="logging.level"):
        config.load_config()


@pytest.mark.parametrize(
    "metric_value",
    [-0.1, 1.1, True, float("nan"), "0.6"],
)
def test_load_config_rejects_invalid_quality_gate_metrics(
    config_root: Path,
    metric_value: object,
) -> None:
    values = _valid_config()
    values["quality_gates"]["min_cv_average_precision"] = metric_value  # type: ignore[index]
    _write_config(config_root, yaml.safe_dump(values))

    with pytest.raises(ConfigurationError, match="quality_gates.min_cv_average_precision"):
        config.load_config()
