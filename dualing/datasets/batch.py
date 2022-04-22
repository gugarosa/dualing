"""Batch-based dataset.
"""

from typing import Optional, Tuple

import numpy as np
import tensorflow as tf

import dualing.utils.constants as c
import dualing.utils.exception as e
from dualing.core import Dataset
from dualing.utils import logging

logger = logging.get_logger(__name__)


class BatchDataset(Dataset):
    """A BatchDataset class is responsible for implementing a standard dataset
    that uses input data and labels to provide batches.

    """

    def __init__(
        self,
        data: np.array,
        labels: np.array,
        batch_size: Optional[int] = 1,
        input_shape: Optional[Tuple[int, ...]] = None,
        normalize: Optional[Tuple[int, int]] = (0, 1),
        shuffle: Optional[bool] = True,
        seed: Optional[int] = 0,
    ) -> None:
        """Initialization method.

        Args:
            data: Array of samples.
            labels: Array of labels.
            batch_size: Batch size.
            input_shape: Shape of the reshaped array.
            normalize: Normalization bounds.
            shuffle: Whether data should be shuffled or not.
            seed: Provides deterministic traits when using `random` module.

        """

        logger.info("Overriding class: Dataset -> BatchDataset.")

        super(BatchDataset, self).__init__(
            batch_size, input_shape, normalize, shuffle, seed
        )

        data = self.preprocess(data)

        self._build(data, labels)

        logger.info("Class overrided.")

    @property
    def batches(self) -> tf.data.Dataset:
        """Batches of data (samples, labels)."""

        return self._batches

    @batches.setter
    def batches(self, batches: tf.data.Dataset) -> None:
        if not isinstance(batches, tf.data.Dataset):
            raise e.TypeError("`batches` should be a tf.data.Dataset")

        self._batches = batches

    def _build(self, data: np.array, labels: np.array) -> None:
        """Builds the class.

        Args:
            data: Array of samples.
            labels: Array of labels.

        """

        # Creates a dataset from tensor slices
        batches = tf.data.Dataset.from_tensor_slices((data, labels))

        if self.shuffle:
            batches = batches.shuffle(c.BUFFER_SIZE)

        # Creates the actual batches
        self.batches = batches.batch(self.batch_size)
