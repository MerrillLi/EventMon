import logging
import sys
import warnings
from loguru import logger

def setup_logger(args, path):
    warnings.filterwarnings("ignore", module='pandas')
    logger.remove()
    format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | " \
             "<level>{level: <8}</level> | " \
             "<level>{message}</level>"

    logger.add(sys.stdout, format=format, colorize=True)

    logger.add(f"{path}.log", format=format)
