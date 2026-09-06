# Copyright (c) 2020-2026 Gustavo Rosa.
# Licensed under the Apache License, Version 2.0.

"""Gated recurrent embedding model."""

from dualing.core.model import Base
from dualing.embedders import GRU

__all__ = ["Base", "GRU"]
