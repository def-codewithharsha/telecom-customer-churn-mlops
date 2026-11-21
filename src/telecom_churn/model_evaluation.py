"""Evaluate trained candidates and apply model release criteria."""

from typing import Any

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from telecom_churn.config import AppConfig


def evaluate_holdout(
    model: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
) -> dict[str, Any]:
    """Calculate threshold-based and ranking metrics on the holdout partition."""
    probabilities = model.predict_proba(features)[:, 1]
    predictions = model.predict(features)

    return {
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "average_precision": float(average_precision_score(target, probabilities)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(target, predictions).tolist(),
        "classification_report": classification_report(
            target,
            predictions,
            target_names=["no_churn", "churn"],
            output_dict=True,
            zero_division=0,
        ),
    }


def evaluate_release_gate(
    *,
    cv_average_precision: float,
    baseline_average_precision: float,
    holdout_average_precision: float,
    config: AppConfig,
) -> dict[str, Any]:
    """Compare the selected candidate with configured performance requirements."""
    gate_config = config.quality_gates
    improvement = cv_average_precision - baseline_average_precision
    checks = {
        "cv_average_precision": {
            "actual": cv_average_precision,
            "minimum": gate_config.min_cv_average_precision,
            "passed": cv_average_precision >= gate_config.min_cv_average_precision,
        },
        "holdout_average_precision": {
            "actual": holdout_average_precision,
            "minimum": gate_config.min_holdout_average_precision,
            "passed": holdout_average_precision
            >= gate_config.min_holdout_average_precision,
        },
        "improvement_over_prior_baseline": {
            "actual": improvement,
            "minimum": gate_config.min_improvement_over_baseline,
            "passed": improvement >= gate_config.min_improvement_over_baseline,
        },
    }
    failures = [
        f"{name}={check['actual']:.4f} is below minimum {check['minimum']:.4f}"
        for name, check in checks.items()
        if not check["passed"]
    ]
    return {"passed": not failures, "checks": checks, "failures": failures}
