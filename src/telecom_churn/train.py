"""Local training entry point."""

from telecom_churn.config import load_config
from telecom_churn.data import load_raw_data
from telecom_churn.logger import get_logger
from telecom_churn.model_trainer import train_model

logger = get_logger(__name__)


def main() -> None:
    config = load_config()
    dataframe = load_raw_data(config)
    metrics = train_model(dataframe, config)
    logger.info("Holdout ROC AUC: %.4f", metrics["roc_auc"])


if __name__ == "__main__":
    main()