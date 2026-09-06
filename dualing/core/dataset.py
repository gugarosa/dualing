"""Legacy-compatible dataset base class."""

import tensorflow as tf

from dualing.data import preprocess
from dualing.utils import exception


class Dataset:
    """Store the original dataset API's preprocessing and batching options.

    Args:
        batch_size: Positive maximum number of samples or pairs per batch.
        input_shape: Optional full tensor shape, including the sample dimension.
        normalize: Increasing global scaling bounds, or None to disable scaling.
        shuffle: Whether concrete datasets shuffle before batching.
        seed: Seed used by concrete datasets. Construction also resets
            TensorFlow's global random seed, as in the original interface.

    Native dataset helper functions do not reset the global seed. Subclasses
    build and expose their own ``batches`` dataset.
    """

    def __init__(
        self,
        batch_size: int = 1,
        input_shape: tuple[int, ...] | None = None,
        normalize: tuple[float, float] | None = (0.0, 1.0),
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.normalize = normalize
        self.shuffle = shuffle
        self.seed = seed

        tf.random.set_seed(seed)

    @property
    def batch_size(self) -> int:
        """Batch size."""

        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int) -> None:
        if not isinstance(batch_size, int):
            raise exception.TypeError("`batch_size` should be an integer")

        if batch_size <= 0:
            raise exception.ValueError("`batch_size` should be greater than 0")

        self._batch_size = batch_size

    @property
    def input_shape(self) -> tuple[int, ...] | None:
        """Shape of the input tensors."""

        return self._input_shape

    @input_shape.setter
    def input_shape(self, input_shape: tuple[int, ...] | None) -> None:
        if not isinstance(input_shape, tuple) and input_shape is not None:
            raise exception.TypeError("`input_shape` should be a tuple or None")

        self._input_shape = input_shape

    @property
    def normalize(self) -> tuple[float, float] | None:
        """Normalization bounds."""

        return self._normalize

    @normalize.setter
    def normalize(self, normalize: tuple[float, float] | None) -> None:
        if not isinstance(normalize, tuple) and normalize is not None:
            raise exception.TypeError("`normalize` should be a tuple or None")

        self._normalize = normalize

    @property
    def shuffle(self) -> bool:
        """Whether data should be shuffled."""

        return self._shuffle

    @shuffle.setter
    def shuffle(self, shuffle: bool) -> None:
        if not isinstance(shuffle, bool):
            raise exception.TypeError("`shuffle` should be a boolean")

        self._shuffle = shuffle

    def preprocess(self, data) -> tf.Tensor:
        """Apply the configured shape and global normalization.

        Args:
            data: Numeric array-like values or an eager tensor.

        Returns:
            A float32 tensor. Constant values map to the lower bound when
            normalization is enabled.
        """

        return preprocess(data, self.input_shape, self.normalize)

    def _build(self) -> None:
        """Build the concrete dataset."""

        raise NotImplementedError
