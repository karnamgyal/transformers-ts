# Logging Utility
import logging

def setup_logging(level=logging.INFO):
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=level
    )   

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
