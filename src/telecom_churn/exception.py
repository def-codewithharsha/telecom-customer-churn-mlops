"""Project-specific exceptions."""


class TelecomChurnError(Exception):
    """Base exception for expected project errors."""


class ConfigurationError(TelecomChurnError):
    """Raised when project configuration is missing or invalid."""


class DataValidationError(TelecomChurnError):
    """Raised when training input does not satisfy the data contract."""


class PredictionInputError(TelecomChurnError):
    """Raised when prediction input cannot be accepted by the model service."""
