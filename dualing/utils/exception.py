# Copyright (c) 2020-2026 Gustavo Rosa.
# Licensed under the Apache License, Version 2.0.

"""Project exception classes."""

from dualing.utils.logging import get_logger

logger = get_logger(__name__)


class Error(Exception):
    """Base Dualing error."""

    def __init__(self, cls: str, msg: str) -> None:
        """Initialize category-prefixed exception text and emit an error diagnostic.

        Args:
            cls: Exception category name.
            msg: Caller-provided message retained in the exception text.

        """

        message = f"{cls}: {msg}"

        logger.error(f"`exception={cls}` was raised with message {msg!r}.")

        super().__init__(message)


class ArgumentError(Error):
    """Wrong number of arguments."""

    def __init__(self, error: str) -> None:
        """Initialize an argument error.

        Args:
            error: Error message.

        """

        super().__init__("ArgumentError", error)


class BuildError(Error):
    """Object has not been built."""

    def __init__(self, error: str) -> None:
        """Initialize a build error.

        Args:
            error: Error message.

        """

        super().__init__("BuildError", error)


class SizeError(Error):
    """Invalid variable size."""

    def __init__(self, error: str) -> None:
        """Initialize a size error.

        Args:
            error: Error message.

        """

        super().__init__("SizeError", error)


class TypeError(Error):
    """Invalid variable type."""

    def __init__(self, error: str) -> None:
        """Initialize a type error.

        Args:
            error: Error message.

        """

        super().__init__("TypeError", error)


class ValueError(Error):
    """Invalid variable value."""

    def __init__(self, error: str) -> None:
        """Initialize a value error.

        Args:
            error: Error message.

        """

        super().__init__("ValueError", error)
