"""Logging setup. Use it in any file like this:

    from app.logger.custom_logger import get_logger

    logger = get_logger(__name__)
    logger.info("file_uploaded", filename="sales.csv", rows=120)

Every log line is JSON, e.g. {"filename": "sales.csv", "rows": 120, "event": "file_uploaded", ...}.
It is printed to the console and also saved in logs/<start time>.log.
"""

import logging
import os
from datetime import datetime

import structlog

from app.config.settings import settings

os.makedirs("logs", exist_ok=True)
LOG_FILE = os.path.join("logs", datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".log")

# Python's standard logging decides WHERE lines go: the console and the log file
logging.basicConfig(
    level=settings.log_level,
    format="%(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE, encoding="utf-8")],
)

# structlog decides WHAT a line looks like: a timestamp, the level, and everything as JSON
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
        structlog.processors.add_log_level,
        structlog.processors.format_exc_info,  # logger.exception(...) adds the traceback
        structlog.processors.JSONRenderer(),
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
)


def get_logger(name: str):
    return structlog.get_logger(name)
