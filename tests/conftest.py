"""Shared fixtures for the telecom churn test suite."""

import pandas as pd
import pytest


@pytest.fixture

def churn_dataframe() -> pd.DataFrame:
    """Return valid deterministic rows with the project feature columns."""
    rows = []
    for index in range(12):
        internet = ["DSL", "Fiber optic", "No"][index % 3]
        internet_feature = (
            "No internet service"
            if internet == "No"
            else ["Yes", "No"][index % 2]
        )
        rows.append(
            {
                "customerID": f"customer-{index:03}",
                "gender": ["Female", "Male"][index % 2],
                "SeniorCitizen": index % 2,
                "Partner": ["Yes", "No"][index % 2],
                "Dependents": ["No", "Yes"][index % 2],
                "tenure": index * 5,
                "PhoneService": "Yes",
                "MultipleLines": ["Yes", "No"][index % 2],
                "InternetService": internet,
                "OnlineSecurity": internet_feature,
                "OnlineBackup": internet_feature,
                "DeviceProtection": internet_feature,
                "TechSupport": internet_feature,
                "StreamingTV": internet_feature,
                "StreamingMovies": internet_feature,
                "Contract": ["Month-to-month", "One year", "Two year"][index % 3],
                "PaperlessBilling": ["Yes", "No"][index % 2],
                "PaymentMethod": [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ][index % 4],
                "MonthlyCharges": 25.0 + index,
                "TotalCharges": "" if index == 0 else str((25 + index) * index * 5),
                "Churn": "Yes" if index % 2 else "No",
            }
        )
    return pd.DataFrame(rows)