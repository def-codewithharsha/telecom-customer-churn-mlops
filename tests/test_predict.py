"""Tests for loading a model bundle and producing predictions."""

import joblib
import numpy as np
import pytest

from telecom_churn.predict import ModelService


class StubClassifier:
    """Small deterministic estimator for prediction-service tests."""

    feature_names_in_ = np.array(["tenure", "MonthlyCharges"])

    def predict_proba(self, dataframe):
        assert list(dataframe.columns) == ["tenure", "MonthlyCharges"]
        return np.array([[0.2, 0.8]])


def test_model_service_returns_prediction_and_metadata(tmp_path) -> None:
    model_path = tmp_path / "model.joblib"
    joblib.dump(
        {
            "model": StubClassifier(),
            "model_name": "test-model",
            "model_version": "test-version",
        },
        model_path,
    )
    service = ModelService(model_path, threshold=0.75)

    result = service.predict({"tenure": 12, "MonthlyCharges": 65.5})

    assert result == {
        "churn_prediction": "Yes",
        "churn_probability": 0.8,
        "threshold": 0.75,
        "model_name": "test-model",
        "model_version": "test-version",
    }


def test_model_service_rejects_missing_or_unexpected_features(tmp_path) -> None:
    model_path = tmp_path / "model.joblib"
    joblib.dump(
        {
            "model": StubClassifier(),
            "model_name": "test-model",
            "model_version": "test-version",
        },
        model_path,
    )
    service = ModelService(model_path)

    with pytest.raises(ValueError, match="Feature mismatch"):
        service.predict({"tenure": 12, "unknown": 1})