import tensorflow as tf

import dualing.utils.logging as l
from dualing.core import Dataset

logger = l.get_logger(__name__)


class RandomPairDataset(Dataset):
    """A RandomPairDataset class is responsible for implementing a dataset that randomly creates pairs of data,
    as well as their similarity (1) or dissimilarity (0).

    """

    def __init__(self, data, labels, batch_size=1, seed=0):
        """Initialization method.

        Args:
            data (np.array): Array of samples.
            labels (np.array): Array of labels.
            batch_size (int): Batch size.
            seed (int): Provides deterministic traits when using `random` module.

        """

        logger.info('Overriding class: Dataset -> RandomPairDataset.')

        # Overrides its parent class with any custom arguments if needed
        super(RandomPairDataset, self).__init__(seed)

        # Creates pairs of data and their labels
        pairs = self._create_pairs(data, labels)

        # Builds up the class
        self._build(pairs, batch_size)

        logger.info('Class overrided.')

    def _create_pairs(self, data, labels):
        """Creates random pairs from data and labels.

        Args:
            data (np.array): Array of samples.
            labels (np.array): Array of labels.

        Returns:
            Tuple containing (x1, x2, y) pairs.

        """

        # Defines the number of samples
        n_samples = data.shape[0]

        # Defines the number of possible pairs
        n_pairs = n_samples // 2

        # Randomly samples indexes
        indexes = tf.random.shuffle(tf.range(n_samples))

        # Gathers the `x1` and `x2` samples
        x1, x2 = tf.gather(data, indexes[:n_pairs]), tf.gather(data, indexes[n_pairs:])

        # Gathers the `y1` and `y2` samples
        y1, y2 = tf.gather(labels, indexes[:n_pairs]), tf.gather(labels, indexes[n_pairs:])

        # If `y1` and `y2` are equal, it means that samples are similar
        y = tf.cast(tf.equal(y1, y2), 'float32')

        return (x1, x2, y)

    def _build(self, pairs, batch_size):
        """Builds the class.

        Args:
            pairs (tuple): (x1, x2, y) pairs.
            batch_size (int): Batch size.

        """

        # Creates batches from tensor slices
        self.batches = tf.data.Dataset.from_tensor_slices(pairs).batch(batch_size)
