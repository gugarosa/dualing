"""Project exception classes."""

from dualing.utils import logging

logger = logging.get_logger(__name__)


class Error(Exception):
    """Base Dualing error."""

    def __init__(self, cls: str, msg: str) -> None:
        message = f"{cls}: {msg}"

        logger.error(message)

        super().__init__(message)


class ArgumentError(Error):
    """Wrong number of arguments."""

    def __init__(self, error: str) -> None:
        super().__init__("ArgumentError", error)


class BuildError(Error):
    """Object has not been built."""

    def __init__(self, error: str) -> None:
        super().__init__("BuildError", error)


class SizeError(Error):
    """Invalid variable size."""

    def __init__(self, error: str) -> None:
        super().__init__("SizeError", error)


class TypeError(Error):
    """Invalid variable type."""

    def __init__(self, error: str) -> None:
        super().__init__("TypeError", error)


class ValueError(Error):
    """Invalid variable value."""

    def __init__(self, error: str) -> None:
        super().__init__("ValueError", error)
