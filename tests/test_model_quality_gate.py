"""Tests for release criteria applied to trained model candidates."""

from types import SimpleNamespace

from telecom_churn.model_trainer import _evaluate_release_gate


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        quality_gates=SimpleNamespace(
            min_cv_average_precision=0.60,
            min_holdout_average_precision=0.55,
            min_improvement_over_baseline=0.05,
        )
    )


def test_release_gate_passes_when_each_criterion_is_met() -> None:
    result = _evaluate_release_gate(
        cv_average_precision=0.66,
        baseline_average_precision=0.30,
        holdout_average_precision=0.63,
        config=_config(),
    )

    assert result["passed"] is True
    assert result["failures"] == []
    assert all(check["passed"] for check in result["checks"].values())


def test_release_gate_reports_all_failed_criteria() -> None:
    result = _evaluate_release_gate(
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
