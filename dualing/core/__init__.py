"""A core package, containing all the basic class and functions that serves as
    the foundation of Dualing common modules.
"""

from dualing.core.dataset import Dataset
from dualing.core.loss import (
    BinaryCrossEntropy,
    ContrastiveLoss,
    TripletHardLoss,
    TripletSemiHardLoss,
)
from dualing.core.model import Base, Siamese
