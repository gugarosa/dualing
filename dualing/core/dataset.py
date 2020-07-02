import tensorflow as tf

import dualing.utils.exception as e
import dualing.utils.logging as l

logger = l.get_logger(__name__)


class Dataset:
    """A Dataset class is responsible for receiving raw data, pre-processing it and
    persisting batches that will be feed as inputs to the networks.

    """

    def __init__(self, batch_size=1, input_shape=None, normalize=(-1, 1), shuffle=True, seed=0):
        """Initialization method.

        Args:
            batch_size (int): Batch size.
            input_shape (tuple): Shape of the reshaped array.
            normalize (tuple): Normalization bounds.
            shuffle (bool): Whether data should be shuffled or not.
            seed (int): Provides deterministic traits when using `random` module.

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

        # Debugs important information
        logger.debug(f'Size: {input_shape} | Batch size: {batch_size} | Normalization: {normalize} | Shuffle: {shuffle}.')

    @property
    def batch_size(self):
        """int: Batch size.

        """

        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size):
        if not isinstance(batch_size, int):
            raise e.TypeError('`batch_size` should be a integer')

        self._batch_size = batch_size

    @property
    def input_shape(self):
        """tuple: Shape of the input tensors.

        """

        return self._input_shape

    @input_shape.setter
    def input_shape(self, input_shape):
        if not (isinstance(input_shape, tuple) or input_shape is None):
            raise e.TypeError('`input_shape` should be a tuple or None')

        self._input_shape = input_shape

    @property
    def normalize(self):
        """tuple: Normalization bounds.

        """

        return self._normalize

    @normalize.setter
    def normalize(self, normalize):
        if not isinstance(normalize, tuple):
            raise e.TypeError('`normalize` should be a tuple')

        self._normalize = normalize

    @property
    def shuffle(self):
        """bool: Whether data should be shuffled or not.

        """

        return self._shuffle

    @shuffle.setter
    def shuffle(self, shuffle):
        if not isinstance(shuffle, bool):
            raise e.TypeError('`shuffle` should be a boolean')

        self._shuffle = shuffle

    @property
    def batches(self):
        """tf.data.Dataset: Batches of data (samples, labels).

        """

        return self._batches

    @batches.setter
    def batches(self, batches):
        if not isinstance(batches, tf.data.Dataset):
            raise e.TypeError('`batches` should be a tf.data.Dataset')

        self._batches = batches

    def preprocess(self, data):
        """Pre-process the data by reshaping and normalizing, if necessary.

        Args:
            data (np.array): Array of data.

        Returns:
            Pre-processed data.

        """

        # If a shape is supplied
        if self.input_shape:
            # Reshapes the array and make sure that it is `float`
            data = data.reshape(self.input_shape).astype('float32')

        # If no shape is supplied
        else:
            # Just make sure that the array is `float`
            data = data.astype('float32')

        # If data should be normalized
        if self.normalize:
            # Gathers the lower and upper bounds of normalization
            low, high = self.normalize[0], self.normalize[1]

            # Gathers the minimum and maximum values of the data
            _min, _max = tf.math.reduce_min(data), tf.math.reduce_max(data)

            # Normalizes the data between `low` and `high`
            data = (high - low) * ((data - _min) / (_max - _min)) + low

        return data

    def _build(self):
        """Method that builds the class.

        Note that you need to implement this method directly on its child. Essentially,
        each Dataset has its building procedure.

        Raises:
            NotImplementedError.

        """

        raise NotImplementedError
