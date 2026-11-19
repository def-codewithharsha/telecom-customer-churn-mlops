"""Load, validate, and use the selected model artifact."""

from collections.abc import Iterable
from math import isfinite
from pathlib import Path
from typing import Any

import pandas as pd

from telecom_churn.exception import TelecomChurnError
from telecom_churn.model_artifact import load_model_artifact


class ModelService:
    """Serve predictions using a safely serialized sklearn model pipeline."""

    def __init__(
        self,
        model_path: Path,
        threshold: float | None = None,
        expected_features: Iterable[str] | None = None,
    ) -> None:
        self.model_path = model_path
        self.model, self.bundle = load_model_artifact(model_path)

        metadata_threshold = float(self.bundle["threshold"])
        self.threshold = metadata_threshold if threshold is None else threshold
        if (
            isinstance(self.threshold, bool)
            or not isinstance(self.threshold, (int, float))
            or not isfinite(self.threshold)
            or not 0.0 <= self.threshold <= 1.0
        ):
            raise ValueError("Prediction threshold must be finite and between 0 and 1.")
        self.threshold = float(self.threshold)

        if not callable(getattr(self.model, "predict_proba", None)):
            raise TelecomChurnError("Loaded model does not support predict_proba().")

        raw_feature_names = getattr(self.model, "feature_names_in_", None)
        if raw_feature_names is None:
            raise TelecomChurnError("Loaded model does not define feature_names_in_.")

        self.feature_names = list(raw_feature_names)
        if not self.feature_names or not all(
            isinstance(name, str) and name for name in self.feature_names
        ):
            raise TelecomChurnError("Loaded model has invalid feature names.")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise TelecomChurnError("Loaded model has duplicate feature names.")

        if expected_features is not None:
            expected = set(expected_features)
            actual = set(self.feature_names)
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            if missing or unexpected:
                raise TelecomChurnError(
                    "Loaded model features do not match the API schema. "
                    f"Missing: {missing}; unexpected: {unexpected}"
                )

        classes = getattr(self.model, "classes_", None)
        if classes is None or len(classes) != 2:
            raise TelecomChurnError("Loaded model must expose two binary classes.")
        positive_indices = [index for index, label in enumerate(classes) if label == 1]
        if len(positive_indices) != 1:
            raise TelecomChurnError("Loaded model must contain positive class 1 (churn).")

        self.positive_class_index = positive_indices[0]

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        """Validate one request and return its churn prediction."""
        supplied = set(features)
        expected = set(self.feature_names)

        missing = sorted(expected - supplied)
        unexpected = sorted(supplied - expected)

        if missing or unexpected:
            raise ValueError(
                f"Feature mismatch. Missing: {missing}; unexpected: {unexpected}"
            )

        row = pd.DataFrame(
            [[features[name] for name in self.feature_names]],
            columns=self.feature_names,
        )
        probabilities = self.model.predict_proba(row)
        expected_shape = (1, len(self.model.classes_))
        if getattr(probabilities, "shape", None) != expected_shape:
            raise TelecomChurnError(
                "Model returned probabilities with an unexpected shape."
            )

        churn_probability = float(probabilities[0, self.positive_class_index])
        if not isfinite(churn_probability) or not 0.0 <= churn_probability <= 1.0:
            raise TelecomChurnError("Model returned an invalid churn probability.")

        prediction = churn_probability >= self.threshold

        return {
            "churn_prediction": "Yes" if prediction else "No",
            "churn_probability": churn_probability,
            "threshold": self.threshold,
            "model_name": self.bundle["model_name"],
            "model_version": self.bundle["model_version"],
        }