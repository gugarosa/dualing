# Copyright (c) 2020-2026 Gustavo Rosa.
# Licensed under the Apache License, Version 2.0.

"""Batch-based dataset class."""

import tensorflow as tf

from dualing.core import Dataset
from dualing.utils import constants, exception


class BatchDataset(Dataset):
    """Preprocess samples and expose the original labeled-batch interface."""

    def __init__(
        self,
        data,
        labels,
        batch_size: int = 1,
        input_shape: tuple[int, ...] | None = None,
        normalize: tuple[float, float] | None = (0.0, 1.0),
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        """Initialize a prefetched dataset and reset TensorFlow's global random seed.

        Args:
            data: Numeric samples with the sample dimension first.
            labels: Targets whose first dimension matches the preprocessed samples.
            batch_size: Positive maximum number of samples per batch.
            input_shape: Full shape passed to preprocessing, or None to preserve the shape.
            normalize: Global scaling bounds, or None to disable scaling.
            shuffle: Whether to shuffle before batching on each traversal.
            seed: Seed for TensorFlow's global random state.

        """

        super().__init__(batch_size, input_shape, normalize, shuffle, seed)

        self._build(self.preprocess(data), labels)

    @property
    def batches(self) -> tf.data.Dataset:
        """Return prefetched sample and label batches.

        Samples are float32, labels retain their dtype, and the final partial batch is retained.

        Returns:
            Dataset yielding (samples, labels).

        """

        return self._batches

    @batches.setter
    def batches(self, batches: tf.data.Dataset) -> None:
        if not isinstance(batches, tf.data.Dataset):
            raise exception.TypeError("`batches` must be a tf.data.Dataset.")

        self._batches = batches

    def _build(self, data, labels) -> None:
        batches = tf.data.Dataset.from_tensor_slices((data, labels))

        if self.shuffle:
            batches = batches.shuffle(constants.BUFFER_SIZE)

        self.batches = batches.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
