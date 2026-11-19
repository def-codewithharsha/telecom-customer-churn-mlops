"""Tests for loading a model artifact and producing predictions."""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from telecom_churn.exception import TelecomChurnError
from telecom_churn.model_artifact import save_model_artifact
from telecom_churn.predict import ModelService


def _save_test_model(model_path) -> None:
    features = pd.DataFrame(
        {
            "tenure": [0, 1, 2, 3, 4, 5, 6, 7],
            "MonthlyCharges": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        }
    )
    target = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    model = LogisticRegression(random_state=42).fit(features, target)
    save_model_artifact(
        model,
        model_path,
        {
            "model_name": "test-model",
            "model_version": "test-version",
            "target_column": "Churn",
            "threshold": 0.5,
        },
    )


def test_model_service_returns_prediction_and_metadata(tmp_path) -> None:
    model_path = tmp_path / "model.skops"
    _save_test_model(model_path)
    service = ModelService(model_path, threshold=0.75)

    result = service.predict({"tenure": 7, "MonthlyCharges": 80.0})

    assert result["churn_prediction"] == "Yes"
    assert result["churn_probability"] > 0.75
    assert result["threshold"] == 0.75
    assert result["model_name"] == "test-model"
    assert result["model_version"] == "test-version"


def test_model_service_rejects_missing_or_unexpected_features(tmp_path) -> None:
    model_path = tmp_path / "model.skops"
    _save_test_model(model_path)
    service = ModelService(model_path)

    with pytest.raises(ValueError, match="Feature mismatch"):
        service.predict({"tenure": 12, "unknown": 1})


def test_model_service_rejects_model_incompatible_with_api_schema(tmp_path) -> None:
    model_path = tmp_path / "model.skops"
    _save_test_model(model_path)

    with pytest.raises(TelecomChurnError, match="do not match the API schema"):
        ModelService(
            model_path,
            expected_features=("tenure", "MonthlyCharges", "Partner"),
        )


def test_model_service_rejects_invalid_threshold(tmp_path) -> None:
    model_path = tmp_path / "model.skops"
    _save_test_model(model_path)

    with pytest.raises(ValueError, match="between 0 and 1"):
        ModelService(model_path, threshold=1.5)


def test_model_service_requires_metadata_sidecar(tmp_path) -> None:
    model_path = tmp_path / "model.skops"
    joblib.dump({}, model_path)

    with pytest.raises(TelecomChurnError, match="Model metadata not found"):
        ModelService(model_path)

def test_model_service_rejects_corrupted_model_artifact(tmp_path) -> None:
    model_path = tmp_path / "model.skops"
    _save_test_model(model_path)
    with model_path.open("ab") as file:
        file.write(b"corruption")

    with pytest.raises(TelecomChurnError, match="checksum does not match"):
        ModelService(model_path)

def test_model_service_rejects_artifact_not_matching_external_digest(
    tmp_path,
    monkeypatch,
) -> None:
    model_path = tmp_path / "model.skops"
    _save_test_model(model_path)
    monkeypatch.setenv("MODEL_ARTIFACT_SHA256", "0" * 64)

    with pytest.raises(TelecomChurnError, match="MODEL_ARTIFACT_SHA256"):
        ModelService(model_path)