"""Compare candidate churn classifiers and save the selected model."""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from telecom_churn.config import AppConfig
from telecom_churn.features import build_preprocessor, split_features_target
from telecom_churn.logger import get_logger

import os

import mlflow
import mlflow.data
import mlflow.sklearn
from mlflow.models import infer_signature

logger = get_logger(__name__)

SELECTION_METRIC = "average_precision"


def _build_candidates(config: AppConfig) -> dict[str, Pipeline]:
    """Build candidate pipelines with shared preprocessing."""
    classifiers = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=config.random_seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=config.random_seed,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=config.random_seed,
            n_jobs=-1,
        ),
    }

    return {
        name: Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", classifier),
            ]
        )
        for name, classifier in classifiers.items()
    }


def train_model(dataframe: pd.DataFrame, config: AppConfig) -> dict[str, Any]:
    """Select a candidate with CV, evaluate it on holdout data, and save it."""
    features, target = split_features_target(dataframe, config)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=config.training.test_size,
        random_state=config.random_seed,
        stratify=target,
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=config.random_seed,
    )
    scoring = {
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }

    candidate_results: dict[str, dict[str, float]] = {}
    candidates = _build_candidates(config)

    for name, candidate in candidates.items():
        scores = cross_validate(
            candidate,
            x_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            error_score="raise",
        )
        candidate_results[name] = {
            metric: float(scores[f"test_{metric}"].mean())
            for metric in scoring
        }
        logger.info(
            "CV %s: PR-AUC=%.4f ROC-AUC=%.4f",
            name,
            candidate_results[name]["average_precision"],
            candidate_results[name]["roc_auc"],
        )

    selected_name = max(
        candidate_results,
        key=lambda name: candidate_results[name][SELECTION_METRIC],
    )
    selected_model = candidates[selected_name]
    selected_model.fit(x_train, y_train)

    probabilities = selected_model.predict_proba(x_test)[:, 1]
    predictions = selected_model.predict(x_test)

    holdout_metrics: dict[str, Any] = {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            target_names=["no_churn", "churn"],
            output_dict=True,
            zero_division=0,
        ),
    }

    results: dict[str, Any] = {
        "selection_metric": SELECTION_METRIC,
        "selected_model": selected_name,
        "cross_validation": candidate_results,
        "holdout": holdout_metrics,
    }

    run_id = _log_mlflow_run(
        dataframe=dataframe,
        train_features=x_train,
        selected_model=selected_model,
        candidates=candidates,
        results=results,
        config=config,
    )

    results["mlflow_run_id"] = run_id
    logger.info("MLflow tracking run: %s", run_id)

    _save_model_bundle(selected_model, config, results, selected_name)
    logger.info("Selected model: %s", selected_name)
    logger.info(
        "Holdout: PR-AUC=%.4f ROC-AUC=%.4f precision=%.4f recall=%.4f",
        holdout_metrics["average_precision"],
        holdout_metrics["roc_auc"],
        holdout_metrics["precision"],
        holdout_metrics["recall"],
    )

    return results


def _save_model_bundle(
    model: Pipeline,
    config: AppConfig,
    results: dict[str, Any],
    model_name: str,
) -> Path:
    """Persist the fitted pipeline and evaluation metadata."""
    model_path = config.training.model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "model_name": model_name,
            "model_version": "0.2.0",
            "target_column": config.data.target_column,
            "results": results,
        },
        model_path,
    )

    logger.info("Saved selected model bundle to %s", model_path)
    return model_path

def _log_mlflow_run(
    dataframe: pd.DataFrame,
    train_features: pd.DataFrame,
    selected_model: Pipeline,
    candidates: dict[str, Pipeline],
    results: dict[str, Any],
    config: AppConfig,
) -> str:
    """Record candidate comparisons, dataset metadata, and selected model in MLflow."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config.project_name)

    dataset = mlflow.data.from_pandas(
        dataframe,
        source=str(config.data.raw_path),
        name=config.project_name,
        targets=config.data.target_column,
    )

    with mlflow.start_run(run_name="churn-model-comparison") as parent_run:
        mlflow.log_params(
            {
                "selection_metric": results["selection_metric"],
                "selected_model": results["selected_model"],
                "random_seed": config.random_seed,
                "test_size": config.training.test_size,
                "cv_folds": 5,
                "target_column": config.data.target_column,
            }
        )
        mlflow.set_tag("project", config.project_name)
        mlflow.log_input(dataset, context="training")

        for name, metrics in results["cross_validation"].items():
            candidate = candidates[name]
            classifier_params = candidate.named_steps["classifier"].get_params(
                deep=False
            )

            with mlflow.start_run(run_name=name, nested=True):
                mlflow.log_param("algorithm", name)
                mlflow.log_params(
                    {
                        f"classifier_{key}": str(value)
                        for key, value in classifier_params.items()
                    }
                )
                mlflow.log_metrics(
                    {f"cv_{key}": float(value) for key, value in metrics.items()}
                )

        holdout_metrics = {
            f"holdout_{key}": float(value)
            for key, value in results["holdout"].items()
            if isinstance(value, (int, float))
        }
        mlflow.log_metrics(holdout_metrics)

        mlflow.log_dict(
            {
                "confusion_matrix": results["holdout"]["confusion_matrix"],
                "classification_report": results["holdout"]["classification_report"],
            },
            "evaluation/holdout_report.json",
        )

        model_example = train_features.head(3)
        signature = infer_signature(
            model_example,
            selected_model.predict(model_example),
        )
        mlflow.sklearn.log_model(
            selected_model,
            name="selected_model",
            signature=signature,
            input_example=model_example,
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.compose._column_transformer.make_column_selector",
            ],
        )

        run_id = parent_run.info.run_id

    return run_id