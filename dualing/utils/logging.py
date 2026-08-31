"""Logging helpers."""

import logging
import sys
from logging import StreamHandler
from logging.handlers import TimedRotatingFileHandler

FORMATTER = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
LOG_FILE = "dualing.log"
LOG_LEVEL = logging.DEBUG


class Logger(logging.Logger):
    """Logger that can emit a message only to its file handler."""

    def to_file(self, msg: str, *args, **kwargs) -> None:
        console_level = self.handlers[0].level
        self.handlers[0].setLevel(logging.CRITICAL)

        self.info(msg, *args, **kwargs)

        self.handlers[0].setLevel(console_level)


def get_console_handler() -> StreamHandler:
    """Create the console handler."""

    handler = StreamHandler(sys.stdout)
    handler.setFormatter(FORMATTER)

    return handler


def get_timed_file_handler() -> TimedRotatingFileHandler:
    """Create the rotating file handler."""

    handler = TimedRotatingFileHandler(LOG_FILE, delay=True, when="midnight")
    handler.setFormatter(FORMATTER)

    return handler


def get_logger(logger_name: str) -> Logger:
    """Return a configured project logger."""

    logging.setLoggerClass(Logger)

    logger = logging.getLogger(logger_name)
    logger.setLevel(LOG_LEVEL)

    if not logger.handlers:
        logger.addHandler(get_console_handler())
        logger.addHandler(get_timed_file_handler())

    logger.propagate = False

    return logger
