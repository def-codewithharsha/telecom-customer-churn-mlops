"""Tests for release criteria applied to trained model candidates."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from telecom_churn.model_evaluation import evaluate_release_gate, select_fbeta_threshold


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        quality_gates=SimpleNamespace(
            min_cv_average_precision=0.60,
            min_holdout_average_precision=0.55,
            min_improvement_over_baseline=0.05,
        )
    )


def test_release_gate_passes_when_each_criterion_is_met() -> None:
    result = evaluate_release_gate(
        cv_average_precision=0.66,
        baseline_average_precision=0.30,
        holdout_average_precision=0.63,
        config=_config(),
    )

    assert result["passed"] is True
    assert result["failures"] == []
    assert all(check["passed"] for check in result["checks"].values())


def test_release_gate_reports_all_failed_criteria() -> None:
    result = evaluate_release_gate(
        cv_average_precision=0.59,
        baseline_average_precision=0.56,
        holdout_average_precision=0.54,
        config=_config(),
    )

    assert result["passed"] is False
    assert set(result["checks"]) == {
        "cv_average_precision",
        "holdout_average_precision",
        "improvement_over_prior_baseline",
    }
    assert len(result["failures"]) == 3


def test_select_fbeta_threshold_uses_validation_scores() -> None:
    target = pd.Series([0, 0, 1, 1])
    probabilities = np.array([0.2, 0.4, 0.45, 0.8])

    result = select_fbeta_threshold(target, probabilities, beta=2.0)

    assert result["strategy"] == "maximize_f_beta"
    assert result["threshold"] == 0.45
    assert result["recall"] == 1.0
