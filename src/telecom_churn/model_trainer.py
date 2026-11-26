"""Coordinate candidate training, evaluation, tracking, and artifact saving."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from telecom_churn.config import AppConfig
from telecom_churn.exception import ModelQualityGateError
from telecom_churn.features import (
    add_service_count_features,
    build_preprocessor,
    split_features_target,
)
from telecom_churn.logger import get_logger
from telecom_churn.mlflow_tracking import log_mlflow_run
from telecom_churn.model_artifact import save_model_artifact
from telecom_churn.model_evaluation import (
    evaluate_holdout,
    evaluate_release_gate,
    select_fbeta_threshold,
)

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

    candidates: dict[str, Pipeline] = {}
    for name, classifier in classifiers.items():
        candidates[name] = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", classifier),
            ]
        )
        candidates[f"{name}_with_service_counts"] = Pipeline(
            steps=[
                (
                    "feature_engineering",
                    FunctionTransformer(add_service_count_features, validate=False),
                ),
                ("preprocessor", build_preprocessor()),
                ("classifier", classifier),
            ]
        )
    return candidates


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

    selected_name, highest_cv_candidate = _select_candidate(candidate_results)
    selected_model = CalibratedClassifierCV(
        estimator=candidates[selected_name],
        method=config.training.calibration_method,
        cv=cv,
        ensemble=False,
        n_jobs=-1,
    )
    oof_probabilities = cross_val_predict(
        selected_model,
        x_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    threshold_selection = select_fbeta_threshold(
        y_train,
        oof_probabilities,
        beta=config.training.threshold_beta,
    )
    selected_threshold = float(threshold_selection["threshold"])
    selected_model.fit(x_train, y_train)

    holdout_metrics = evaluate_holdout(
        selected_model,
        x_test,
        y_test,
        threshold=selected_threshold,
        beta=config.training.threshold_beta,
    )
    quality_gate = evaluate_release_gate(
        cv_average_precision=candidate_results[selected_name][SELECTION_METRIC],
        baseline_average_precision=baseline_average_precision,
        holdout_average_precision=holdout_metrics["average_precision"],
        config=config,
    )
    results: dict[str, Any] = {
        "selection_metric": SELECTION_METRIC,
        "selection_policy": "one_standard_error_for_service_count_features",
        "highest_cv_candidate": highest_cv_candidate,
        "selected_model": selected_name,
        "calibration_method": config.training.calibration_method,
        "threshold_selection": threshold_selection,
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
    logger.info(
        "Decision threshold %.4f selected by maximizing F-beta (beta=%.2f); "
        "holdout F-beta=%.4f",
        selected_threshold,
        config.training.threshold_beta,
        holdout_metrics["f_beta"],
    )
    logger.info("Probability calibration method: %s", config.training.calibration_method)
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
        results[name] = {}
        for metric in scoring:
            fold_scores = scores[f"test_{metric}"]
            results[name][metric] = float(fold_scores.mean())
            results[name][f"{metric}_std"] = float(fold_scores.std(ddof=1))
            results[name][f"{metric}_se"] = float(
                fold_scores.std(ddof=1) / np.sqrt(len(fold_scores))
            )
        logger.info(
            "CV %s: PR-AUC=%.4f ± %.4f ROC-AUC=%.4f",
            name,
            results[name]["average_precision"],
            results[name]["average_precision_std"],
            results[name]["roc_auc"],
        )
    return results


def _select_candidate(
    candidate_results: dict[str, dict[str, float]],
) -> tuple[str, str]:
    """Prefer raw features if service-count gains are within one CV standard error."""
    highest_cv_candidate = max(
        candidate_results,
        key=lambda name: candidate_results[name][SELECTION_METRIC],
    )
    suffix = "_with_service_counts"
    if highest_cv_candidate.endswith(suffix):
        base_candidate = highest_cv_candidate[: -len(suffix)]
        base_result = candidate_results.get(base_candidate)
        engineered_result = candidate_results[highest_cv_candidate]
        if (
            base_result is not None
            and base_result[SELECTION_METRIC]
            >= engineered_result[SELECTION_METRIC]
            - engineered_result[f"{SELECTION_METRIC}_se"]
        ):
            logger.info(
                "Service-count feature gain is within one CV standard error; "
                "preferring simpler candidate %s over %s",
                base_candidate,
                highest_cv_candidate,
            )
            return base_candidate, highest_cv_candidate

    return highest_cv_candidate, highest_cv_candidate


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
    model: BaseEstimator,
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
            "model_version": "0.3.0",
            "target_column": config.data.target_column,
            "threshold": float(results["threshold_selection"]["threshold"]),
            "calibration_method": config.training.calibration_method,
            "mlflow_run_id": results.get("mlflow_run_id"),
            "results": results,
        },
    )
    logger.info("Saved selected safe model artifact to %s", model_path)
    logger.info("Model artifact SHA-256: %s", artifact_digest)
    logger.info("Saved model metadata to %s", model_path.with_suffix(".json"))
    return model_path
