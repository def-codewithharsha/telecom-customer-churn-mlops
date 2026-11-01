"""HTTP API for churn predictions."""

from contextlib import asynccontextmanager
from pathlib import Path


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from telecom_churn.config import load_config
from telecom_churn.predict import ModelService
from telecom_churn.schema import CustomerFeatures


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: CustomerFeatures


class PredictResponse(BaseModel):
    churn_prediction: str
    churn_probability: float
    threshold: float
    model_name: str
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()
    model_path: Path = config.training.model_path
    app.state.model_service = ModelService(model_path)
    yield


app = FastAPI(
    title="Telecom Churn Prediction API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check for the API process."""
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str]:
    """Readiness check confirming a model is loaded."""
    if not hasattr(app.state, "model_service"):
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return {"status": "ready"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Return a churn prediction for one customer."""
    service: ModelService = app.state.model_service

    try:
        result = service.predict(request.features.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return PredictResponse(**result)