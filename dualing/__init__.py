# Copyright (c) 2020-2026 Gustavo Rosa.
# Licensed under the Apache License, Version 2.0.

"""Dual-based neural learning with TensorFlow."""

from importlib.metadata import version

from dualing.data import (
    balanced_pair_dataset,
    batch_dataset,
    preprocess,
    random_pair_dataset,
)
from dualing.embedders import CNN, GRU, LSTM, MLP, RNN
from dualing.models import ContrastiveSiamese, CrossEntropySiamese, TripletSiamese

__version__ = version("dualing")

__all__ = [
    "CNN",
    "GRU",
    "LSTM",
    "MLP",
    "RNN",
    "ContrastiveSiamese",
    "CrossEntropySiamese",
    "TripletSiamese",
    "__version__",
    "balanced_pair_dataset",
    "batch_dataset",
    "preprocess",
    "random_pair_dataset",
]
