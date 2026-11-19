"""Tests for raw data checks and feature preparation."""

import pandas as pd
import pytest

from telecom_churn.data import validate_raw_data
from telecom_churn.exception import TelecomChurnError
from telecom_churn.features import split_features_target


def test_validate_raw_data_accepts_expected_columns(churn_dataframe: pd.DataFrame) -> None:
    validate_raw_data(churn_dataframe, "Churn")


def test_validate_raw_data_rejects_missing_feature(churn_dataframe: pd.DataFrame) -> None:
    malformed = churn_dataframe.drop(columns=["tenure"])

    with pytest.raises(TelecomChurnError, match="Missing: \\['tenure'\\]"):
        validate_raw_data(malformed, "Churn")


def test_validate_raw_data_rejects_duplicate_columns(churn_dataframe: pd.DataFrame) -> None:
    malformed = pd.concat(
        [churn_dataframe, churn_dataframe[["gender"]]],
        axis=1,
    )

    with pytest.raises(TelecomChurnError, match="duplicate column names"):
        validate_raw_data(malformed, "Churn")


def test_validate_raw_data_rejects_missing_targets(churn_dataframe: pd.DataFrame) -> None:
    malformed = churn_dataframe.copy()
    malformed.loc[0, "Churn"] = None

    with pytest.raises(TelecomChurnError, match="missing values"):
        validate_raw_data(malformed, "Churn")


def test_split_features_target_maps_labels_drops_id_and_coerces_blank_charge(
    churn_dataframe: pd.DataFrame,
) -> None:
    class Config:
        class Data:
            target_column = "Churn"

        data = Data()

    features, target = split_features_target(churn_dataframe, Config())

    assert "customerID" not in features.columns
    assert target.tolist() == [0, 1] * 6
    assert pd.isna(features.loc[0, "TotalCharges"])
    assert features["TotalCharges"].dtype == "float64"


def test_split_features_target_rejects_unknown_label(churn_dataframe: pd.DataFrame) -> None:
    malformed = churn_dataframe.copy()
    malformed.loc[0, "Churn"] = "maybe"

    class Config:
        class Data:
            target_column = "Churn"

        data = Data()

    with pytest.raises(TelecomChurnError, match="Unexpected churn labels"):
        split_features_target(malformed, Config())