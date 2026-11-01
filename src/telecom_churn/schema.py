"""Validate the prediction request against the churn feature contract."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

YesNo = Literal["Yes", "No"]
InternetService = Literal["DSL", "Fiber optic", "No"]
InternetFeature = Literal["Yes", "No", "No internet service"]
MultipleLines = Literal["Yes", "No", "No phone service"]
Contract = Literal["Month-to-month", "One year", "Two year"]
PaymentMethod = Literal[
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]


class CustomerFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gender: Literal["Female", "Male"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=72)
    PhoneService: YesNo
    MultipleLines: MultipleLines
    InternetService: InternetService
    OnlineSecurity: InternetFeature
    OnlineBackup: InternetFeature
    DeviceProtection: InternetFeature
    TechSupport: InternetFeature
    StreamingTV: InternetFeature
    StreamingMovies: InternetFeature
    Contract: Contract
    PaperlessBilling: YesNo
    PaymentMethod: PaymentMethod
    MonthlyCharges: float = Field(ge=0, allow_inf_nan=False)
    TotalCharges: float | None = Field(ge=0, allow_inf_nan=False)