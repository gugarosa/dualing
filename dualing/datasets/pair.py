"""Balanced- and random-pair datasets.
"""

from typing import Optional, Tuple

import numpy as np
import tensorflow as tf

import dualing.utils.constants as c
import dualing.utils.exception as e
from dualing.core import Dataset
from dualing.utils import logging

logger = logging.get_logger(__name__)


class BalancedPairDataset(Dataset):
    """A BalancedPairDataset class is responsible for implementing a dataset
    that creates balanced pairs of data, as well as their similarity (1) or dissimilarity (0).

    """

    def __init__(
        self,
        data: np.array,
        labels: np.array,
        n_pairs: Optional[int] = 2,
        batch_size: Optional[int] = 1,
        input_shape: Optional[Tuple[int, ...]] = None,
        normalize: Optional[Tuple[int, int]] = (0, 1),
        shuffle: Optional[bool] = True,
        seed: Optional[int] = 0,
    ):
        """Initialization method.

        Args:
            data: Array of samples.
            labels: Array of labels.
            n_pairs: Number of pairs.
            batch_size: Batch size.
            input_shape: Shape of the reshaped array.
            normalize: Normalization bounds.
            shuffle: Whether data should be shuffled or not.
            seed: Provides deterministic traits when using `random` module.

        """

        logger.info("Overriding class: Dataset -> BalancedPairDataset.")

        super(BalancedPairDataset, self).__init__(
            batch_size, input_shape, normalize, shuffle, seed
        )

        try:
            # Checks if supplied labels are not equal
            assert not np.all(labels == labels[0])

        except:
            raise e.ValueError("`labels` should have distinct values")

        # Amount of pairs
        self.n_pairs = n_pairs

        data = self.preprocess(data)
        pairs = self.create_pairs(data, labels)

        self._build(pairs)

        logger.info("Class overrided.")

    @property
    def n_pairs(self) -> int:
        """Amount of pairs."""

        return self._n_pairs

    @n_pairs.setter
    def n_pairs(self, n_pairs: int) -> None:
        if not isinstance(n_pairs, int):
            raise e.TypeError("`n_pairs` should be a integer")

        self._n_pairs = n_pairs

    @property
    def batches(self) -> tf.data.Dataset:
        """Batches of data (samples, labels)."""

        return self._batches

    @batches.setter
    def batches(self, batches: tf.data.Dataset) -> None:
        if not isinstance(batches, tf.data.Dataset):
            raise e.TypeError("`batches` should be a tf.data.Dataset")

        self._batches = batches

    def create_pairs(
        self, data: np.array, labels: np.array
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Creates balanced pairs from data and labels.

        Args:
            data (np.array): Array of samples.
            labels (np.array): Array of labels.

        Returns:
            (Tuple[tf.Tensor, tf.Tensor, tf.Tensor]): Tuple containing pairs of samples along their labels.

        """

        logger.debug("Creating pairs ...")

        # Defines the number of samples and number of pairs
        n_samples = data.shape[0]
        n_pairs = self.n_pairs // 2

        # Defines the positive and negative lists
        x1_p, x2_p, y_p = [], [], []
        x1_n, x2_n, y_n = [], [], []

        # Iterates until both positive and negative pairs
        while len(y_p) < n_pairs or len(y_n) < n_pairs:
            # Samples two random indexes
            idx = tf.random.uniform([2], maxval=n_samples, dtype="int32")

            # If labels are equal on the particular indexes
            if tf.equal(tf.gather(labels, idx[0]), tf.gather(labels, idx[1])):
                # Appends positive data to `x1` and negative to `x2`
                x1_p.append(tf.gather(data, idx[0]))
                x2_p.append(tf.gather(data, idx[1]))

                # Appends positive label to `y`
                y_p.append(1.0)

            # If labels are not equal on the particular indexes
            else:
                # Appends negative data to `x1` and negative to `x2`
                x1_n.append(tf.gather(data, idx[0]))
                x2_n.append(tf.gather(data, idx[1]))

                # Appends negative label to `y`
                y_n.append(0.0)

        # Merges the positive and negative `x1`, `x2` and labels
        x1 = x1_p[:n_pairs] + x1_n[:n_pairs]
        x2 = x2_p[:n_pairs] + x2_n[:n_pairs]
        y = y_p[:n_pairs] + y_n[:n_pairs]

        logger.debug("Pairs: %s.", self.n_pairs)

        return x1, x2, y

    def _build(self, pairs: Tuple[tf.Tensor, tf.Tensor]) -> None:
        """Builds the class.

        Args:
            pairs: Pairs of samples along their labels.

        """

        if self.shuffle:
            self.batches = (
                tf.data.Dataset.from_tensor_slices(pairs)
                .shuffle(c.BUFFER_SIZE)
                .batch(self.batch_size)
            )

        else:
            self.batches = tf.data.Dataset.from_tensor_slices(pairs).batch(
                self.batch_size
            )


class RandomPairDataset(Dataset):
    """A RandomPairDataset class is responsible for implementing a dataset that
    randomly creates pairs of data, as well as their similarity (1) or not (0).

    """

    def __init__(
        self,
        data: np.array,
        labels: np.array,
        batch_size: Optional[int] = 1,
        input_shape: Optional[Tuple[int, ...]] = None,
        normalize: Optional[Tuple[int, int]] = (0, 1),
        seed: Optional[int] = 0,
    ):
        """Initialization method.

        Args:
            data: Array of samples.
            labels: Array of labels.
            batch_size: Batch size.
            input_shape: Shape of the reshaped array.
            normalize: Normalization bounds.
            seed: Provides deterministic traits when using `random` module.

        """

        logger.info("Overriding class: Dataset -> RandomPairDataset.")

        super(RandomPairDataset, self).__init__(
            batch_size, input_shape, normalize, False, seed
        )

        data = self.preprocess(data)
        pairs = self.create_pairs(data, labels)

        self._build(pairs)

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

    def _build(self, pairs: Tuple[tf.Tensor, tf.Tensor]) -> None:
        """Builds the class.

        Args:
            pairs: Pairs of samples along their labels.

        """

        self.batches = tf.data.Dataset.from_tensor_slices(pairs).batch(self.batch_size)

    def create_pairs(
        self, data: np.array, labels: np.array
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Creates random pairs from data and labels.

        Args:
            data (np.array): Array of samples.
            labels (np.array): Array of labels.

        Returns:
            (Tuple[tf.Tensor, tf.Tensor, tf.Tensor]): Tuple containing pairs of samples along their labels.

        """

        logger.debug("Creating pairs ...")

        # Defines the number of samples and pairs
        n_samples = data.shape[0]
        n_pairs = n_samples // 2

        # Randomly samples indexes
        indexes = tf.random.shuffle(tf.range(n_samples))

        # Gathers samples and their labels
        x1, x2 = tf.gather(data, indexes[:n_pairs]), tf.gather(data, indexes[n_pairs:])
        y1, y2 = tf.gather(labels, indexes[:n_pairs]), tf.gather(
            labels, indexes[n_pairs:]
        )

        # If labels are equal, it means that samples are similar
        y = tf.cast(tf.equal(y1, y2), "float32")

        # Calculates the number of positive and negative pairs
        n_pos_pairs = tf.math.count_nonzero(y)
        n_neg_pairs = y.shape[0] - n_pos_pairs

        logger.debug(
            "Positive pairs: %s | Negative pairs: %s.", n_pos_pairs, n_neg_pairs
        )

        return (x1, x2, y)
