"""HTTP route tests for the prediction API."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from telecom_churn import api


class StubModelService:
    """Small stand-in for the loaded prediction service."""

    def __init__(self, model_path: Path, expected_features: object) -> None:
        self.model_path = model_path
        self.expected_features = expected_features

    def predict(self, features: dict[str, object]) -> dict[str, object]:
        if features.get("tenure") == 71:
            raise ValueError("Invalid feature values")
        return {
            "churn_prediction": "Yes",
            "churn_probability": 0.8,
            "threshold": 0.5,
            "model_name": "test-model",
            "model_version": "test-version",
        }


@pytest.fixture
def api_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Run the app lifespan with a predictable config and model service."""
    config = SimpleNamespace(
        log_level="INFO",
        training=SimpleNamespace(model_path=Path("models/test.skops")),
    )
    monkeypatch.setattr(api, "load_config", lambda: config)
    monkeypatch.setattr(api, "configure_logging", lambda _level: None)
    monkeypatch.setattr(api, "ModelService", StubModelService)
    with TestClient(api.app) as client:
        yield client


def _valid_features() -> dict[str, object]:
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "Yes",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 65.5,
        "TotalCharges": 786.0,
    }


def test_health_and_ready_routes(api_client: TestClient) -> None:
    assert api_client.get("/health").json() == {"status": "ok"}
    assert api_client.get("/ready").json() == {"status": "ready"}


def test_predict_route_returns_prediction(api_client: TestClient) -> None:
    response = api_client.post("/predict", json={"features": _valid_features()})

    assert response.status_code == 200
    assert response.json() == {
        "churn_prediction": "Yes",
        "churn_probability": 0.8,
        "threshold": 0.5,
        "model_name": "test-model",
        "model_version": "test-version",
    }


def test_predict_route_rejects_invalid_request_schema(api_client: TestClient) -> None:
    features = _valid_features()
    features["Contract"] = "weekly"

    response = api_client.post("/predict", json={"features": features})

    assert response.status_code == 422


def test_predict_route_maps_prediction_value_error_to_422(
    api_client: TestClient,
) -> None:
    features = _valid_features()
    features["tenure"] = 71

    response = api_client.post("/predict", json={"features": features})

    assert response.status_code == 422
    assert response.json()["detail"] == "Invalid feature values"


def test_model_load_failure_fails_application_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(
        log_level="INFO",
        training=SimpleNamespace(model_path=Path("models/missing.skops")),
    )
    monkeypatch.setattr(api, "load_config", lambda: config)
    monkeypatch.setattr(api, "configure_logging", lambda _level: None)

    def fail_to_load(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("Model artifact unavailable")

    monkeypatch.setattr(api, "ModelService", fail_to_load)

    with pytest.raises(RuntimeError, match="Model artifact unavailable"):
        with TestClient(api.app):
            pass
