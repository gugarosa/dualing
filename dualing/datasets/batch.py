import tensorflow as tf

import dualing.utils.constants as c
import dualing.utils.exception as e
import dualing.utils.logging as l
from dualing.core import Dataset

logger = l.get_logger(__name__)


class BatchDataset(Dataset):
    """A BatchDataset class is responsible for implementing a standard dataset that uses input data and labels
    to provide batches.

    """

    def __init__(self, data, labels, batch_size=1, input_shape=None, normalize=(0, 1), shuffle=True, seed=0):
        """Initialization method.

        Args:
            data (np.array): Array of samples.
            labels (np.array): Array of labels.
            batch_size (int): Batch size.
            input_shape (tuple): Shape of the reshaped array.
            normalize (tuple): Normalization bounds.
            shuffle (bool): Whether data should be shuffled or not.
            seed (int): Provides deterministic traits when using `random` module.

        """

        logger.info('Overriding class: Dataset -> BatchDataset.')

        # Overrides its parent class with any custom arguments if needed
        super(BatchDataset, self).__init__(batch_size, input_shape, normalize, shuffle, seed)

        # Pre-processes the data
        data = self.preprocess(data)

        # Builds up the class
        self._build(data, labels)

        logger.info('Class overrided.')

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

    def _build(self, data, labels):
        """Builds the class.

        Args:
            data (np.array): Array of samples.
            labels (np.array): Array of labels.

        """

        # Checks if data should be shuffled
        if self.shuffle:
            # Creates dataset from shuffled and batched data
            self.batches = tf.data.Dataset.from_tensor_slices((data, labels)).shuffle(c.BUFFER_SIZE).batch(self.batch_size)

        # If data should not be shuffled
        else:
            # Creates dataset from batched data
            self.batches = tf.data.Dataset.from_tensor_slices((data, labels)).batch(self.batch_size)
