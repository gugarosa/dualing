"""Batch-based dataset class."""

import tensorflow as tf

from dualing.core import Dataset
from dualing.utils import constants, exception


class BatchDataset(Dataset):
    """Preprocess samples and expose the original labeled-batch interface.

    Args:
        data: Numeric samples, with the sample dimension first.
        labels: Targets whose first dimension matches the preprocessed samples.
        batch_size: Positive maximum number of samples per batch.
        input_shape: Optional full shape passed to Dataset preprocessing.
        normalize: Global scaling bounds, or None to disable scaling.
        shuffle: Shuffle before batching, reshuffling on each traversal.
        seed: Dataset seed; construction also resets TensorFlow's global seed.

    Access the prefetched dataset through the ``batches`` property.
    """

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
        super().__init__(batch_size, input_shape, normalize, shuffle, seed)

        self._build(self.preprocess(data), labels)

    @property
    def batches(self) -> tf.data.Dataset:
        """Prefetched ``(samples, labels)`` batches.

        Samples are float32, label dtype is retained, and the final partial
        batch is not discarded.
        """

        return self._batches

    @batches.setter
    def batches(self, batches: tf.data.Dataset) -> None:
        if not isinstance(batches, tf.data.Dataset):
            raise exception.TypeError("`batches` should be a tf.data.Dataset")

        self._batches = batches

    def _build(self, data, labels) -> None:
        batches = tf.data.Dataset.from_tensor_slices((data, labels))

        if self.shuffle:
            batches = batches.shuffle(constants.BUFFER_SIZE)

        self.batches = batches.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
