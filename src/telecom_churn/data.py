"""Read the raw churn dataset and validate its basic structure."""

from pathlib import Path

import pandas as pd

from telecom_churn.config import AppConfig
from telecom_churn.exception import TelecomChurnError
from telecom_churn.logger import get_logger
from telecom_churn.schema import CustomerFeatures

logger = get_logger(__name__)


def load_raw_data(config: AppConfig) -> pd.DataFrame:
    """Load the configured CSV file and check its basic structure."""
    path: Path = config.data.raw_path

    if not path.exists():
        raise TelecomChurnError(
            f"Dataset not found at {path}. Place the CSV there or update configs/dev.yaml."
        )

    try:
        dataframe = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise TelecomChurnError(f"Could not read dataset at {path}: {exc}") from exc

    validate_raw_data(dataframe, config.data.target_column)
    logger.info("Loaded %d rows and %d columns from %s", *dataframe.shape, path)

    return dataframe


def validate_raw_data(dataframe: pd.DataFrame, target_column: str) -> None:
    """Reject empty, malformed, or unlabeled training data."""
    if dataframe.empty:
        raise TelecomChurnError("Dataset is empty.")

    if dataframe.columns.duplicated().any():
        duplicates = dataframe.columns[dataframe.columns.duplicated()].tolist()
        raise TelecomChurnError(f"Dataset has duplicate column names: {duplicates}")

    if target_column not in dataframe.columns:
        raise TelecomChurnError(
            f"Target column {target_column!r} is missing. "
            f"Available columns: {dataframe.columns.tolist()}"
        )
    expected_features = set(CustomerFeatures.model_fields)
    actual_features = set(dataframe.columns) - {target_column}

    # IDs may be present in source data but are removed before training.
    identifier_columns = {
        column
        for column in actual_features
        if column.strip().lower() in {"customerid", "customer_id", "id"}
    }
    actual_features -= identifier_columns

    missing = sorted(expected_features - actual_features)
    unexpected = sorted(actual_features - expected_features)

    if missing or unexpected:
        raise TelecomChurnError(
            "Training columns do not match the prediction API schema. "
            f"Missing: {missing}; unexpected: {unexpected}"
        )

    if dataframe[target_column].isna().any():
        missing_count = int(dataframe[target_column].isna().sum())
        raise TelecomChurnError(
            f"Target column {target_column!r} has {missing_count} missing values."
        )

    if dataframe.shape[1] < 2:
        raise TelecomChurnError("Dataset must contain at least one feature and the target.")