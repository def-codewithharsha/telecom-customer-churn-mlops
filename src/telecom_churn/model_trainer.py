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