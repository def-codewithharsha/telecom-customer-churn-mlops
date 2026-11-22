# Telecom Customer Churn MLOps

A Python project for training and serving a telecom customer churn classifier. It includes a scikit-learn preprocessing/model pipeline, MLflow experiment logging, a FastAPI prediction API, Docker packaging, and a GitHub Actions training smoke check.

> **Status:** Development/learning project. The repository does not yet include a production deployment workflow, cloud infrastructure, or a committed trained model. See [Current scope and production gaps](#current-scope-and-production-gaps).

## Repository layout

```text
configs/dev.yaml             Local development settings
configs/prod.yaml            Container/inference settings
data/raw/                    Put the input CSV here (not committed)
models/                      Locally generated model bundle (not committed)
src/telecom_churn/           Data checks, features, training, prediction, and API
scripts/create_ci_dataset.py Synthetic dataset used by CI
.github/workflows/ci.yml     Lint, syntax, and training smoke check
Dockerfile                   API container definition
```

## Requirements

- Python 3.11 or 3.12
- Git
- Optional: Docker, to build and run the API container

## Setup

From the repository root, create and activate a virtual environment, then install the project and development dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The `dev` extra installs the runtime dependencies, MLflow, pytest, and Ruff. To install only the API/runtime dependencies, use `python -m pip install -r requirements.txt`. Training and MLflow dependencies are in `requirements-training.txt`.

## Prepare data and train

Place a CSV dataset at `data/raw/churn.csv`, or update `data.raw_path` in `configs/dev.yaml`. The target column defaults to `Churn`. Feature columns must match the fields in `src/telecom_churn/schema.py`; an optional customer ID column is removed before training. The input data is checked for required columns, duplicate column names, and missing targets.

Run training from the project root:

```powershell
$env:TELECOM_CHURN_CONFIG = "configs/dev.yaml"
python -m telecom_churn.train
```

The profile is selected explicitly with `TELECOM_CHURN_CONFIG`; there is no implicit fallback. Set the variable in each PowerShell session where you run training or the local API. In ECS, set it in the task definition to `configs/prod.yaml` (the Docker image also defaults to that profile).

Training splits the data into stratified train and holdout sets, compares logistic regression, random forest, and extra trees using five-fold stratified cross-validation, and selects by mean average precision. Before saving a deployable model artifact, it checks configured minimum CV and holdout PR-AUC and requires a minimum CV PR-AUC improvement over a `DummyClassifier` prior baseline. The initial floors in `configs/dev.yaml` are project learning policies based on observed full-dataset metrics; they are not business-approved production criteria. Failed candidates are recorded in MLflow without logging a model artifact, and training exits with an error. Passing candidates are saved as a safe sklearn model at `models/churn_model.skops` with JSON metadata at `models/churn_model.json`. The metadata includes SHA-256 digests for corruption detection. For deployment, set `MODEL_ARTIFACT_SHA256` to the independently recorded artifact digest so the API also checks the expected artifact identity.

By default, MLflow uses a local SQLite database (`mlflow.db`) and local artifact storage. To use a tracking server, set `MLFLOW_TRACKING_URI` before training. For example:

```powershell
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
python -m telecom_churn.train
```

The dataset, model bundle, MLflow database, and artifacts are ignored by Git. Back them up or use durable shared storage if you need to retain them.

## Run the prediction API

Train a model first. Then start the API from the repository root:

```powershell
uvicorn telecom_churn.api:app --app-dir src --host 127.0.0.1 --port 8080
```

Set `$env:TELECOM_CHURN_CONFIG = "configs/dev.yaml"` in that PowerShell session before starting Uvicorn.

The API provides:

- `GET /health` — process liveness
- `GET /ready` — readiness check for a loaded model
- `POST /predict` — one customer prediction
- `/docs` — interactive OpenAPI documentation

Example PowerShell request (replace values as needed):

```powershell
$body = @{
  features = @{
    gender = "Female"
    SeniorCitizen = 0
    Partner = "Yes"
    Dependents = "No"
    tenure = 12
    PhoneService = "Yes"
    MultipleLines = "No"
    InternetService = "DSL"
    OnlineSecurity = "Yes"
    OnlineBackup = "No"
    DeviceProtection = "No"
    TechSupport = "No"
    StreamingTV = "No"
    StreamingMovies = "No"
    Contract = "One year"
    PaperlessBilling = "Yes"
    PaymentMethod = "Electronic check"
    MonthlyCharges = 65.5
    TotalCharges = 786.0
  }
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Uri http://127.0.0.1:8080/predict -Method Post -ContentType "application/json" -Body $body
```

The request schema rejects unknown fields and validates categorical values and numeric ranges. The response contains the predicted label, churn probability, threshold, model name, and model version.

## Docker

The Dockerfile packages the API and a trained model artifact, runs the service as UID 10001, and defines a container health check. Since model artifacts are not committed, train first so `models/churn_model.skops` and `models/churn_model.json` exist, then build and run:

```powershell
docker build -t telecom-churn-api:local .
$modelMetadata = Get-Content models/churn_model.json | ConvertFrom-Json
docker run --rm -p 8080:8080 -e "MODEL_ARTIFACT_SHA256=$($modelMetadata.artifact_sha256)" telecom-churn-api:local
```

Check `http://localhost:8080/health` or open `http://localhost:8080/docs`. The digest is included in the metadata and training logs. In production, inject `MODEL_ARTIFACT_SHA256` from a trusted release record or deployment configuration; do not calculate the expected value from the same artifact at container startup. This local example is for development only.

## CI

The current GitHub Actions workflow runs on pull requests and pushes to `main`. It installs dependencies, runs Ruff and syntax checks, executes unit and API route tests, creates a synthetic CI dataset, runs a training smoke check, then builds and starts a container to check its user and health endpoints. The smoke image is not published or deployed. Broader integration and production deployment tests are still pending.

## Current scope and production gaps

Implemented today:

- YAML configuration for local development
- Basic training-data structure validation and typed prediction input validation
- Cross-validation model comparison and holdout metrics
- Local MLflow experiment tracking and local model bundle output
- FastAPI single-record inference endpoint and basic health/readiness routes
- Dockerfile and CI training smoke check

Not implemented yet:

- Automated deployment, cloud infrastructure, and environment promotion
- Shared, durable MLflow tracking/artifact storage and governed model registry/approval
- A release flow that pins and promotes a specific model artifact
- Business-approved model-quality thresholds and a business-selected classification threshold
- Production monitoring for latency, errors, traffic, data quality, and model performance
- Authentication/authorization, rate limits, secrets handling, and deployment security controls
- Rollback, incident runbooks, and scheduled/event-driven retraining
- Comprehensive unit, integration, API contract, and container tests

These are next steps for a production service; the right choices depend on its availability, latency, privacy, and business requirements.

## Configuration

Configuration selection is explicit: pass a path to `load_config()` or set `TELECOM_CHURN_CONFIG` to a YAML profile. `configs/dev.yaml` is for local training and API development; `configs/prod.yaml` is the container/inference profile. Paths are resolved relative to the repository root. Review the quality-gate thresholds against a frozen, representative evaluation set and business costs before production promotion. Do not put credentials or secrets in YAML files.

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE).
