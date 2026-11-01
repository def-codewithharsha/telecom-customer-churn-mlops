"""Load the selected model and produce a churn prediction."""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


class ModelService:
    """Serve predictions using a saved preprocessing-and-model pipeline."""

    def __init__(self, model_path: Path, threshold: float = 0.5) -> None:
        self.model_path = model_path
        self.threshold = threshold
        self.bundle: dict[str, Any] = joblib.load(model_path)
        self.model = self.bundle["model"]
        self.feature_names = list(self.model.feature_names_in_)

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

        churn_probability = float(self.model.predict_proba(row)[0, 1])
        prediction = churn_probability >= self.threshold

        return {
            "churn_prediction": "Yes" if prediction else "No",
            "churn_probability": churn_probability,
            "threshold": self.threshold,
            "model_name": self.bundle["model_name"],
            "model_version": self.bundle["model_version"],
        }