"""Balanced and random pair dataset classes."""

import numpy as np
import tensorflow as tf

from dualing.core import Dataset
from dualing.data import random_pair_dataset
from dualing.utils import constants, exception


class _PairDataset(Dataset):
    @property
    def batches(self) -> tf.data.Dataset:
        """Batches of paired samples and labels."""

        return self._batches

    @batches.setter
    def batches(self, batches: tf.data.Dataset) -> None:
        if not isinstance(batches, tf.data.Dataset):
            raise exception.TypeError("`batches` should be a tf.data.Dataset")

        self._batches = batches

    def _build(self, pairs) -> None:
        batches = tf.data.Dataset.from_tensor_slices(pairs)

        if self.shuffle:
            batches = batches.shuffle(constants.BUFFER_SIZE)

        self.batches = batches.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)


class BalancedPairDataset(_PairDataset):
    """Expose balanced pairs through the original three-item batch layout.

    Args:
        data: Numeric samples, with the sample dimension first.
        labels: One class label per sample, with at least two distinct classes.
        n_pairs: Requested pair count. Prefer a positive even value; original
            generation rounds positive odd counts down to preserve balance.
        batch_size: Positive maximum number of pairs per batch.
        input_shape: Optional full shape passed to Dataset preprocessing.
        normalize: Global scaling bounds, or None to disable scaling.
        shuffle: Shuffle generated pairs before batching.
        seed: Sampling seed; construction also resets TensorFlow's global seed.

    ``batches`` is a prefetched dataset yielding ``(left, right, targets)``.
    Targets are 1 for similar and 0 for dissimilar pairs. Without shuffling,
    similar pairs precede dissimilar pairs.

    Sampling retains the original sample-based rejection scheme, including
    repeats and self-pairs; it is not the native helper's uniform-class sampler.
    """

    def __init__(
        self,
        data,
        labels,
        n_pairs: int = 2,
        batch_size: int = 1,
        input_shape: tuple[int, ...] | None = None,
        normalize: tuple[float, float] | None = (0.0, 1.0),
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__(batch_size, input_shape, normalize, shuffle, seed)

        if np.all(np.asarray(labels) == np.asarray(labels)[0]):
            raise exception.ValueError("`labels` should have distinct values")

        self.n_pairs = n_pairs

        pairs = self.create_pairs(self.preprocess(data), labels)

        self._build(pairs)

    @property
    def n_pairs(self) -> int:
        """Requested pair count; positive odd counts are rounded down."""

        return self._n_pairs

    @n_pairs.setter
    def n_pairs(self, n_pairs: int) -> None:
        if not isinstance(n_pairs, int):
            raise exception.TypeError("`n_pairs` should be an integer")

        self._n_pairs = n_pairs

    def create_pairs(self, data, labels):
        """Generate pairs from supplied samples without preprocessing them.

        Args:
            data: Samples to gather into pairs.
            labels: Corresponding class labels.

        Returns:
            A tuple of left-sample, right-sample, and target lists. Targets
            are 1 for similar and 0 for dissimilar pairs, in that order.
        """

        labels = np.asarray(labels)

        if np.all(labels == labels[0]):
            raise exception.ValueError("`labels` should have distinct values")

        rng = np.random.default_rng(self.seed)
        n_pairs = self.n_pairs // 2
        positive_left, positive_right, positive_targets = [], [], []
        negative_left, negative_right, negative_targets = [], [], []

        while len(positive_targets) < n_pairs or len(negative_targets) < n_pairs:
            left_index, right_index = rng.integers(0, len(data), size=2)

            if labels[left_index] == labels[right_index]:
                positive_left.append(tf.gather(data, left_index))
                positive_right.append(tf.gather(data, right_index))
                positive_targets.append(1.0)
            else:
                negative_left.append(tf.gather(data, left_index))
                negative_right.append(tf.gather(data, right_index))
                negative_targets.append(0.0)

        return (
            positive_left[:n_pairs] + negative_left[:n_pairs],
            positive_right[:n_pairs] + negative_right[:n_pairs],
            positive_targets[:n_pairs] + negative_targets[:n_pairs],
        )


class RandomPairDataset(_PairDataset):
    """Expose disjoint random pairs through the original batch interface.

    Args:
        data: At least two numeric samples, with the sample dimension first.
        labels: One class label per sample.
        batch_size: Positive maximum number of pairs per batch.
        input_shape: Optional full shape passed to Dataset preprocessing.
        normalize: Global scaling bounds, or None to disable scaling.
        seed: Sampling seed; construction also resets TensorFlow's global seed.

    ``batches`` is a prefetched dataset yielding ``(left, right, targets)``.
    Targets are 1 for equal labels and 0 otherwise. Pairing does not reuse
    samples; an odd leftover is unused. Traversal is not shuffled.
    """

    def __init__(
        self,
        data,
        labels,
        batch_size: int = 1,
        input_shape: tuple[int, ...] | None = None,
        normalize: tuple[float, float] | None = (0.0, 1.0),
        seed: int = 0,
    ) -> None:
        super().__init__(batch_size, input_shape, normalize, False, seed)

        pairs = self.create_pairs(self.preprocess(data), labels)

        self._build(pairs)

    def create_pairs(self, data, labels):
        """Return disjoint pairs without reshaping or normalizing the samples.

        Args:
            data: Numeric samples, converted to float32.
            labels: Corresponding class labels.

        Returns:
            Three tensors: left samples, right samples, and float32 targets.
        """

        n_pairs = len(data) // 2
        dataset = random_pair_dataset(
            data,
            labels,
            batch_size=n_pairs,
            normalize=None,
            shuffle=False,
            seed=self.seed,
        )

        (left, right), targets = next(iter(dataset))

        return left, right, targets
