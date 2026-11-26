"""Evaluate trained candidates and apply model release criteria."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from telecom_churn.config import AppConfig

CALIBRATION_BIN_COUNT = 10


def select_fbeta_threshold(
    target: pd.Series,
    probabilities: np.ndarray,
    beta: float,
) -> dict[str, float | str]:
    """Choose the decision threshold that maximizes F-beta on validation scores.

    The caller must supply out-of-fold predictions from training data, not the
    final holdout. F-beta greater than 1 gives recall more weight than precision.
    """
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be a finite number greater than 0.")
    if len(target) != len(probabilities) or len(target) == 0:
        raise ValueError("target and probabilities must have the same non-zero length.")
    invalid_probabilities = (
        not np.isfinite(probabilities).all()
        or ((probabilities < 0) | (probabilities > 1)).any()
    )
    if invalid_probabilities:
        raise ValueError("probabilities must be finite values between 0 and 1.")

    precision, recall, thresholds = precision_recall_curve(target, probabilities)
    beta_squared = beta**2
    denominator = beta_squared * precision[:-1] + recall[:-1]
    fbeta = np.divide(
        (1 + beta_squared) * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )
    best_index = int(np.argmax(fbeta))
    threshold = float(thresholds[best_index])

    return {
        "strategy": "maximize_f_beta",
        "beta": float(beta),
        "threshold": threshold,
        "precision": float(precision[best_index]),
        "recall": float(recall[best_index]),
        "f_beta": float(fbeta[best_index]),
    }


def _calibration_table(
    target: pd.Series,
    probabilities: np.ndarray,
    bin_count: int = CALIBRATION_BIN_COUNT,
) -> list[dict[str, float | int]]:
    """Summarize predicted and observed churn rates in quantile bins."""
    edges = np.unique(np.quantile(probabilities, np.linspace(0.0, 1.0, bin_count + 1)))
    if len(edges) == 1:
        bin_indexes = np.zeros(len(probabilities), dtype=int)
    else:
        bin_indexes = np.digitize(probabilities, edges[1:-1], right=True)

    table: list[dict[str, float | int]] = []
    target_values = target.to_numpy()
    for bin_index in np.unique(bin_indexes):
        mask = bin_indexes == bin_index
        table.append(
            {
                "bin": int(bin_index),
                "count": int(mask.sum()),
                "mean_predicted_probability": float(probabilities[mask].mean()),
                "observed_churn_rate": float(target_values[mask].mean()),
            }
        )
    return table


def evaluate_holdout(
    model: BaseEstimator,
    features: pd.DataFrame,
    target: pd.Series,
    threshold: float,
    beta: float,
) -> dict[str, Any]:
    """Calculate ranking and threshold-based metrics on untouched holdout data."""
    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    return {
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "average_precision": float(average_precision_score(target, probabilities)),
        "brier_score": float(brier_score_loss(target, probabilities)),
        "log_loss": float(log_loss(target, probabilities, labels=[0, 1])),
        "threshold": float(threshold),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
        "f_beta": float(fbeta_score(target, predictions, beta=beta, zero_division=0)),
        "confusion_matrix": confusion_matrix(target, predictions).tolist(),
        "classification_report": classification_report(
            target,
            predictions,
            target_names=["no_churn", "churn"],
            output_dict=True,
            zero_division=0,
        ),
        "calibration_by_bin": _calibration_table(target, probabilities),
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
