from dualing.utils import logging


def test_logger_file():
    logger = logging.get_logger(__name__)

    assert logger.to_file("testing") is None


def test_get_console_handler():
    c = logging.get_console_handler()

    assert c is not None


def test_get_timed_file_handler():
    f = logging.get_timed_file_handler()

    assert f is not None


def test_get_logger():
    logger = logging.get_logger(__name__)

    assert logger.name == "test_logging"

    assert logger.hasHandlers() is True
