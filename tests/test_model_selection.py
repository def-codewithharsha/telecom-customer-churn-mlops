"""Tests for parsimonious selection between raw and derived features."""

from telecom_churn.model_trainer import _select_candidate


def test_select_candidate_prefers_raw_features_for_gain_within_one_standard_error() -> None:
    scores = {
        "logistic_regression": {
            "average_precision": 0.660023,
            "average_precision_se": 0.004,
        },
        "logistic_regression_with_service_counts": {
            "average_precision": 0.660028,
            "average_precision_se": 0.004,
        },
    }

    selected, highest = _select_candidate(scores)

    assert selected == "logistic_regression"
    assert highest == "logistic_regression_with_service_counts"


def test_select_candidate_keeps_engineered_features_for_gain_over_one_standard_error() -> None:
    scores = {
        "random_forest": {
            "average_precision": 0.64,
            "average_precision_se": 0.004,
        },
        "random_forest_with_service_counts": {
            "average_precision": 0.66,
            "average_precision_se": 0.004,
        },
    }

    selected, highest = _select_candidate(scores)

    assert selected == highest == "random_forest_with_service_counts"
