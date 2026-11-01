"""Prepare churn features and build model preprocessing."""

import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from telecom_churn.config import AppConfig
from telecom_churn.exception import TelecomChurnError


def split_features_target(
    dataframe: pd.DataFrame,
    config: AppConfig,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate input features and convert Yes/No churn labels to 1/0."""
    target_column = config.data.target_column

    if target_column not in dataframe.columns:
        raise TelecomChurnError(f"Target column {target_column!r} is missing.")

    features = dataframe.drop(columns=[target_column]).copy()

    numeric_columns = (
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    )

    for column in numeric_columns:
        if column not in features.columns:
            continue

        cleaned = features[column].astype("string").str.strip()
        cleaned = cleaned.replace("", pd.NA)
        converted = pd.to_numeric(cleaned, errors="coerce")

        invalid = cleaned.notna() & converted.isna()
        if invalid.any():
            examples = cleaned[invalid].drop_duplicates().head(5).tolist()
            raise TelecomChurnError(
                f"Column {column!r} has nonnumeric values: {examples}"
            )

        # Whitespace-only values become NaN; the numeric pipeline imputes them.
        features[column] = converted.astype("float64")

    # Identifiers are not predictive features and should not be used by the model.
    identifier_columns = [
        column
        for column in features.columns
        if column.strip().lower() in {"customerid", "customer_id", "id"}
    ]
    features = features.drop(columns=identifier_columns)

    target_values = dataframe[target_column].astype(str).str.strip().str.lower()
    target = target_values.map({"yes": 1, "no": 0})

    if target.isna().any():
        unexpected = sorted(target_values[target.isna()].unique().tolist())
        raise TelecomChurnError(f"Unexpected churn labels: {unexpected}")

    if target.nunique() != 2:
        raise TelecomChurnError("Expected exactly two churn target classes.")

    if features.shape[1] == 0:
        raise TelecomChurnError("No feature columns remain after removing target and IDs.")

    return features, target.astype("int64")


def build_preprocessor() -> ColumnTransformer:
    """Create preprocessing that learns imputers, scaling, and encoding on fit."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                make_column_selector(dtype_include="number"),
            ),
            (
                "categorical",
                categorical_pipeline,
                make_column_selector(dtype_exclude="number"),
            ),
        ],
        remainder="drop",
    )