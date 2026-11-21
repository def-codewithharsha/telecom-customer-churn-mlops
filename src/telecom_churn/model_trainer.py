"""Coordinate candidate training, evaluation, tracking, and artifact saving."""

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from telecom_churn.config import AppConfig
from telecom_churn.exception import ModelQualityGateError
from telecom_churn.features import build_preprocessor, split_features_target
from telecom_churn.logger import get_logger
from telecom_churn.mlflow_tracking import log_mlflow_run
from telecom_churn.model_artifact import save_model_artifact
from telecom_churn.model_evaluation import evaluate_holdout, evaluate_release_gate

logger = get_logger(__name__)

SELECTION_METRIC = "average_precision"
CV_FOLDS = 5


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
    """Select a candidate, evaluate it, enforce release policy, and save it."""
    features, target = split_features_target(dataframe, config)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=config.training.test_size,
        random_state=config.random_seed,
        stratify=target,
    )

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
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
    candidates = _build_candidates(config)
    candidate_results = _compare_candidates(candidates, x_train, y_train, cv, scoring)
    baseline_average_precision = _evaluate_prior_baseline(x_train, y_train, cv)

    selected_name = max(
        candidate_results,
        key=lambda name: candidate_results[name][SELECTION_METRIC],
    )
    selected_model = candidates[selected_name]
    selected_model.fit(x_train, y_train)

    holdout_metrics = evaluate_holdout(selected_model, x_test, y_test)
    quality_gate = evaluate_release_gate(
        cv_average_precision=candidate_results[selected_name][SELECTION_METRIC],
        baseline_average_precision=baseline_average_precision,
        holdout_average_precision=holdout_metrics["average_precision"],
        config=config,
    )
    results: dict[str, Any] = {
        "selection_metric": SELECTION_METRIC,
        "selected_model": selected_name,
        "cross_validation": candidate_results,
        "baseline": {"average_precision": baseline_average_precision},
        "holdout": holdout_metrics,
        "quality_gate": quality_gate,
    }

    run_id = log_mlflow_run(
        dataframe=dataframe,
        train_features=x_train,
        selected_model=selected_model,
        candidates=candidates,
        results=results,
        config=config,
    )
    results["mlflow_run_id"] = run_id
    logger.info("MLflow tracking run: %s", run_id)

    if not quality_gate["passed"]:
        failures = "; ".join(quality_gate["failures"])
        logger.error("Model release gate failed; artifact was not saved: %s", failures)
        raise ModelQualityGateError(f"Model release gate failed: {failures}")

    logger.info("Model release gate passed: %s", quality_gate["checks"])
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


def _compare_candidates(
    candidates: dict[str, Pipeline],
    features: pd.DataFrame,
    target: pd.Series,
    cv: StratifiedKFold,
    scoring: dict[str, str],
) -> dict[str, dict[str, float]]:
    """Run cross-validation and return mean scores for each candidate."""
    results: dict[str, dict[str, float]] = {}
    for name, candidate in candidates.items():
        scores = cross_validate(
            candidate,
            features,
            target,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            error_score="raise",
        )
        results[name] = {
            metric: float(scores[f"test_{metric}"].mean()) for metric in scoring
        }
        logger.info(
            "CV %s: PR-AUC=%.4f ROC-AUC=%.4f",
            name,
            results[name]["average_precision"],
            results[name]["roc_auc"],
        )
    return results


def _evaluate_prior_baseline(
    features: pd.DataFrame,
    target: pd.Series,
    cv: StratifiedKFold,
) -> float:
    """Score an always-prior classifier using the same CV splits as candidates."""
    scores = cross_validate(
        DummyClassifier(strategy="prior"),
        features,
        target,
        cv=cv,
        scoring={"average_precision": "average_precision"},
        n_jobs=1,
        error_score="raise",
    )
    average_precision = float(scores["test_average_precision"].mean())
    logger.info("CV prior baseline: PR-AUC=%.4f", average_precision)
    return average_precision


def _save_model_bundle(
    model: Pipeline,
    config: AppConfig,
    results: dict[str, Any],
    model_name: str,
) -> Path:
    """Persist the fitted pipeline in skops with a JSON metadata sidecar."""
    model_path = config.training.model_path
    artifact_digest = save_model_artifact(
        model=model,
        model_path=model_path,
        metadata={
            "model_name": model_name,
            "model_version": "0.2.0",
            "target_column": config.data.target_column,
            "threshold": 0.5,
            "mlflow_run_id": results.get("mlflow_run_id"),
            "results": results,
        },
    )
    logger.info("Saved selected safe model artifact to %s", model_path)
    logger.info("Model artifact SHA-256: %s", artifact_digest)
    logger.info("Saved model metadata to %s", model_path.with_suffix(".json"))
    return model_path
