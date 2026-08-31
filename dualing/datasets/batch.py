"""Batch-based dataset class."""

import tensorflow as tf

from dualing.core import Dataset
from dualing.utils import constants, exception


class BatchDataset(Dataset):
    """Preprocess samples and expose labeled batches."""

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
        """Batches of samples and labels."""

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
