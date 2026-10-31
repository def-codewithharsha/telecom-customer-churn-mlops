"""Train and evaluate the churn classification pipeline."""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from telecom_churn.config import AppConfig
from telecom_churn.features import build_preprocessor, split_features_target
from telecom_churn.logger import get_logger

logger = get_logger(__name__)


def train_model(dataframe: pd.DataFrame, config: AppConfig) -> dict[str, Any]:
    """Train the model, report holdout metrics, and save the model bundle."""
    features, target = split_features_target(dataframe, config)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=config.training.test_size,
        random_state=config.random_seed,
        stratify=target,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=config.random_seed,
                ),
            ),
        ]
    )

    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = model.predict(x_test)

    metrics: dict[str, Any] = {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            target_names=["no_churn", "churn"],
            output_dict=True,
            zero_division=0,
        ),
    }

    _save_model_bundle(model, config, metrics)
    logger.info(
        "Training complete. Holdout ROC AUC: %.4f",
        metrics["roc_auc"],
    )

    return metrics


def _save_model_bundle(
    model: Pipeline,
    config: AppConfig,
    metrics: dict[str, Any],
) -> Path:
    """Persist the fitted pipeline and metadata as one artifact."""
    model_path = config.training.model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model": model,
        "model_version": "0.1.0",
        "target_column": config.data.target_column,
        "metrics": metrics,
    }
    joblib.dump(bundle, model_path)

    logger.info("Saved model bundle to %s", model_path)
    return model_path