# Copyright (c) 2020-2026 Gustavo Rosa.
# Licensed under the Apache License, Version 2.0.

"""Core dataset, loss, and model abstractions."""

from dualing.core.dataset import Dataset
from dualing.core.loss import (
    BinaryCrossEntropy,
    ContrastiveLoss,
    TripletHardLoss,
    TripletSemiHardLoss,
)
from dualing.core.model import Base, Siamese

__all__ = [
    "Base",
    "BinaryCrossEntropy",
    "ContrastiveLoss",
    "Dataset",
    "Siamese",
    "TripletHardLoss",
    "TripletSemiHardLoss",
]
