"""Create a small synthetic churn dataset for the CI training smoke check."""

from pathlib import Path

import pandas as pd

rows = []

for i in range(120):
    internet = ["DSL", "Fiber optic", "No"][i % 3]
    internet_feature = "No internet service" if internet == "No" else ["Yes", "No"][i % 2]

    rows.append(
        {
            "gender": ["Female", "Male"][i % 2],
            "SeniorCitizen": i % 2,
            "Partner": ["Yes", "No"][i % 2],
            "Dependents": ["No", "Yes"][i % 2],
            "tenure": (i * 7) % 73,
            "PhoneService": ["Yes", "No"][i % 2],
            "MultipleLines": ["Yes", "No", "No phone service"][i % 3],
            "InternetService": internet,
            "OnlineSecurity": internet_feature,
            "OnlineBackup": internet_feature,
            "DeviceProtection": internet_feature,
            "TechSupport": internet_feature,
            "StreamingTV": internet_feature,
            "StreamingMovies": internet_feature,
            "Contract": ["Month-to-month", "One year", "Two year"][i % 3],
            "PaperlessBilling": ["Yes", "No"][i % 2],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ][i % 4],
            "MonthlyCharges": 20.0 + (i % 80),
            "TotalCharges": "" if i % 17 == 0 else f"{(20.0 + (i % 80)) * (i % 73):.2f}",
            "Churn": "Yes" if i % 3 == 0 else "No",
        }
    )

output_path = Path("data/raw/churn.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(output_path, index=False)
print(f"Created {len(rows)} synthetic CI rows at {output_path}")