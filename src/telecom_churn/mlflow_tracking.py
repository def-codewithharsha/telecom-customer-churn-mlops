"""Log training comparisons, metrics, and accepted model artifacts to MLflow."""

import os
from typing import Any

import mlflow
import mlflow.data
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from telecom_churn.config import AppConfig
from telecom_churn.model_artifact import TRUSTED_SKOPS_TYPES


def log_mlflow_run(
    dataframe: pd.DataFrame,
    train_features: pd.DataFrame,
    selected_model: BaseEstimator,
    candidates: dict[str, Pipeline],
    results: dict[str, Any],
    config: AppConfig,
) -> str:
    """Record candidate comparisons, dataset metadata, and a passing model."""
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
                "selection_policy": results["selection_policy"],
                "highest_cv_candidate": results["highest_cv_candidate"],
                "selected_model": results["selected_model"],
                "random_seed": config.random_seed,
                "test_size": config.training.test_size,
                "cv_folds": 5,
                "threshold_strategy": results["threshold_selection"]["strategy"],
                "threshold_beta": results["threshold_selection"]["beta"],
                "decision_threshold": results["threshold_selection"]["threshold"],
                "calibration_method": results["calibration_method"],
                "target_column": config.data.target_column,
                "minimum_cv_average_precision": (
                    config.quality_gates.min_cv_average_precision
                ),
                "minimum_holdout_average_precision": (
                    config.quality_gates.min_holdout_average_precision
                ),
                "minimum_improvement_over_baseline": (
                    config.quality_gates.min_improvement_over_baseline
                ),
            }
        )
        mlflow.set_tag("project", config.project_name)
        mlflow.set_tag(
            "quality_gate_status",
            "passed" if results["quality_gate"]["passed"] else "failed",
        )
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
        mlflow.log_metrics(
            {
                f"threshold_selection_{key}": float(value)
                for key, value in results["threshold_selection"].items()
                if isinstance(value, (int, float))
            }
        )
        mlflow.log_metric(
            "cv_prior_baseline_average_precision",
            results["baseline"]["average_precision"],
        )
        mlflow.log_metrics(
            {
                "quality_gate_passed": float(results["quality_gate"]["passed"]),
                "cv_average_precision_improvement_over_baseline": results[
                    "quality_gate"
                ]["checks"]["improvement_over_prior_baseline"]["actual"],
            }
        )
        mlflow.log_dict(results["quality_gate"], "evaluation/quality_gate.json")
        mlflow.log_dict(
            {
                "confusion_matrix": results["holdout"]["confusion_matrix"],
                "classification_report": results["holdout"]["classification_report"],
            },
            "evaluation/holdout_report.json",
        )
        mlflow.log_dict(
            results["holdout"]["calibration_by_bin"],
            "evaluation/holdout_calibration.json",
        )

        if results["quality_gate"]["passed"]:
            # Infer only the feature/output schema. Do not attach real customer
            # rows as MLflow's input_example artifact.
            model_example = train_features.head(3)
            signature = infer_signature(
                model_example,
                selected_model.predict(model_example),
            )
            mlflow.sklearn.log_model(
                selected_model,
                name="selected_model",
                signature=signature,
                skops_trusted_types=list(TRUSTED_SKOPS_TYPES),
            )

        return parent_run.info.run_id
