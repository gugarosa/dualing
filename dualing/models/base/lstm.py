# Copyright (c) 2020-2026 Gustavo Rosa.
# Licensed under the Apache License, Version 2.0.

"""Long short-term memory embedding model."""

from dualing.core.model import Base
from dualing.embedders import LSTM

__all__ = ["Base", "LSTM"]
