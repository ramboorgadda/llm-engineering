import logging


logging.basicConfig(level=logging.INFO,
format="%(asctime)s - %(levelname)s - %(message)s",
handlers=[
    logging.FileHandler("applog"),
    logging.StreamHandler()
]
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("logging is woring as expected")