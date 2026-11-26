"""Prepare churn features and build model preprocessing."""

import pandas as pd
from pydantic import TypeAdapter, ValidationError
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from telecom_churn.config import AppConfig
from telecom_churn.exception import DataValidationError
from telecom_churn.schema import CustomerFeatures


def split_features_target(
    dataframe: pd.DataFrame,
    config: AppConfig,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate input features and convert Yes/No churn labels to 1/0."""
    target_column = config.data.target_column

    if target_column not in dataframe.columns:
        raise DataValidationError(f"Target column {target_column!r} is missing.")

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
            raise DataValidationError(
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

    expected_columns = set(CustomerFeatures.model_fields)
    actual_columns = set(features.columns)
    missing = sorted(expected_columns - actual_columns)
    unexpected = sorted(actual_columns - expected_columns)
    if missing or unexpected:
        raise DataValidationError(
            "Training columns do not match the prediction API schema. "
            f"Missing: {missing}; unexpected: {unexpected}"
        )

    _validate_feature_values(features)

    target_values = dataframe[target_column].astype(str).str.strip().str.lower()
    target = target_values.map({"yes": 1, "no": 0})

    if target.isna().any():
        unexpected = sorted(target_values[target.isna()].unique().tolist())
        raise DataValidationError(f"Unexpected churn labels: {unexpected}")

    if target.nunique() != 2:
        raise DataValidationError("Expected exactly two churn target classes.")

    if features.shape[1] == 0:
        raise DataValidationError("No feature columns remain after removing target and IDs.")

    return features, target.astype("int64")


def add_service_count_features(features: pd.DataFrame) -> pd.DataFrame:
    """Add compact counts of subscribed services without changing raw inputs.

    These features are an experiment, not a guaranteed improvement. Missing
    source values produce missing counts so the numeric imputer handles them.
    """
    support_columns = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
    ]
    streaming_columns = ["StreamingTV", "StreamingMovies"]
    all_service_columns = [
        "PhoneService",
        "MultipleLines",
        "InternetService",
        *support_columns,
        *streaming_columns,
    ]
    missing = sorted(set(all_service_columns) - set(features.columns))
    if missing:
        raise DataValidationError(
            f"Cannot derive service counts; required columns are missing: {missing}"
        )

    transformed = features.copy()
    support_complete = transformed[support_columns].notna().all(axis=1)
    streaming_complete = transformed[streaming_columns].notna().all(axis=1)
    all_complete = transformed[all_service_columns].notna().all(axis=1)

    transformed["support_service_count"] = transformed[support_columns].eq("Yes").sum(
        axis=1
    ).astype("float64").where(support_complete)
    transformed["streaming_service_count"] = (
        transformed[streaming_columns]
        .eq("Yes")
        .sum(axis=1)
        .astype("float64")
        .where(streaming_complete)
    )

    active_service_flags = pd.DataFrame(
        {
            "phone": transformed["PhoneService"].eq("Yes"),
            "multiple_lines": transformed["MultipleLines"].eq("Yes"),
            "internet": transformed["InternetService"].isin(["DSL", "Fiber optic"]),
            **{
                column: transformed[column].eq("Yes")
                for column in (*support_columns, *streaming_columns)
            },
        },
        index=transformed.index,
    )
    transformed["active_service_count"] = (
        active_service_flags.sum(axis=1).astype("float64").where(all_complete)
    )
    return transformed


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

def _validate_feature_values(features: pd.DataFrame) -> None:
    """Validate each present training value against the inference field rules."""
    problems: list[str] = []

    for column in features.columns:
        adapter = TypeAdapter(CustomerFeatures.model_fields[column].rebuild_annotation())
        for row_number, value in enumerate(features[column].array, start=2):
            # Missing feature values remain valid for training because the
            # preprocessing pipeline imputes them. TotalCharges is also nullable
            # in the serving schema.
            if pd.isna(value):
                continue

            try:
                adapter.validate_python(value)
            except ValidationError as exc:
                error = exc.errors(include_input=False)[0]
                problems.append(
                    f"{column} at CSV row {row_number}: {error['msg']}"
                )
                if len(problems) >= 10:
                    break
        if len(problems) >= 10:
            break

    if problems:
        raise DataValidationError(
            "Training data contains values outside the prediction schema: "
            + "; ".join(problems)
        )
