"""Dataset.
"""

from typing import Optional, Tuple

import numpy as np
import tensorflow as tf

import dualing.utils.exception as e
from dualing.utils import logging

logger = logging.get_logger(__name__)


class Dataset:
    """A Dataset class is responsible for receiving raw data, pre-processing it and
    persisting batches that will be feed as inputs to the networks.

    """

    def __init__(
        self,
        batch_size: Optional[int] = 1,
        input_shape: Optional[Tuple[int, ...]] = None,
        normalize: Optional[Tuple[int, int]] = (0, 1),
        shuffle: Optional[bool] = True,
        seed: Optional[int] = 0,
    ):
        """Initialization method.

        Args:
            batch_size: Batch size.
            input_shape: Shape of the reshaped array.
            normalize: Normalization bounds.
            shuffle: Whether data should be shuffled or not.
            seed: Provides deterministic traits when using `random` module.

        """

        # Batch size
        self.batch_size = batch_size

        # Shape of the input tensors
        self.input_shape = input_shape

        # Normalization bounds
        self.normalize = normalize

        # Whether data should be shuffled or not
        self.shuffle = shuffle

        # Defines the tensorflow random seed
        tf.random.set_seed(seed)

        logger.debug(
            "Size: %s | Batch size: %s | Normalization: %s | Shuffle: %s.",
            input_shape,
            batch_size,
            normalize,
            shuffle,
        )

    @property
    def batch_size(self) -> int:
        """Batch size."""

        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int) -> None:
        if not isinstance(batch_size, int):
            raise e.TypeError("`batch_size` should be a integer")
        if batch_size <= 0:
            raise e.ValueError("`batch_size` should be greater than 0")

        self._batch_size = batch_size

    @property
    def input_shape(self: Tuple[int, ...]):
        """Shape of the input tensors."""

        return self._input_shape

    @input_shape.setter
    def input_shape(self, input_shape: Tuple[int, ...]) -> None:
        if not (isinstance(input_shape, tuple) or input_shape is None):
            raise e.TypeError("`input_shape` should be a tuple or None")

        self._input_shape = input_shape

    @property
    def normalize(self) -> Tuple[int, int]:
        """Normalization bounds."""

        return self._normalize

    @normalize.setter
    def normalize(self, normalize: Tuple[int, int]) -> None:
        if not (isinstance(normalize, tuple) or normalize is None):
            raise e.TypeError("`normalize` should be a tuple or None")

        self._normalize = normalize

    @property
    def shuffle(self):
        """Whether data should be shuffled or not."""

        return self._shuffle

    @shuffle.setter
    def shuffle(self, shuffle: bool) -> None:
        if not isinstance(shuffle, bool):
            raise e.TypeError("`shuffle` should be a boolean")

        self._shuffle = shuffle

    def preprocess(self, data: np.array) -> np.array:
        """Pre-process the data by reshaping and normalizing, if necessary.

        Args:
            data: Array of data.

        Returns:
            (np.array): Pre-processed data.

        """

        data = data.astype("float32")

        if self.input_shape:
            data = data.reshape(self.input_shape)

        if self.normalize:
            # Gathers the lower and upper bounds of normalization
            low, high = self.normalize[0], self.normalize[1]

            # Gathers the minimum and maximum values of the data
            _min, _max = tf.math.reduce_min(data), tf.math.reduce_max(data)

            # Normalizes the data between `low` and `high`
            data = (high - low) * ((data - _min) / (_max - _min)) + low

        return data

    def _build(self) -> None:
        """Method that builds the class.

        Note that you need to implement this method directly on its child. Essentially,
        each Dataset has its building procedure.

        Raises:
            NotImplementedError.

        """

        raise NotImplementedError
