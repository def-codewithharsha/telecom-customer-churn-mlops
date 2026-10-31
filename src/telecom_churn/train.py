"""Local training entry point."""

from telecom_churn.config import load_config
from telecom_churn.data import load_raw_data
from telecom_churn.logger import get_logger
from telecom_churn.model_trainer import train_model

logger = get_logger(__name__)


def main() -> None:
    config = load_config()
    dataframe = load_raw_data(config)
    results = train_model(dataframe, config)

    logger.info("Selected model: %s", results["selected_model"])
    logger.info("Holdout metrics: %s", results["holdout"])


if __name__ == "__main__":
    main()