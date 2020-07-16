import numpy as np
import tensorflow as tf

import dualing.utils.constants as c
import dualing.utils.exception as e
import dualing.utils.logging as l
from dualing.core import Dataset

logger = l.get_logger(__name__)


class BalancedPairDataset(Dataset):
    """A BalancedPairDataset class is responsible for implementing a dataset that creates balanced pairs of data,
    as well as their similarity (1) or dissimilarity (0).

    """

    def __init__(self, data, labels, n_pairs=2, batch_size=1, input_shape=None, normalize=(0, 1), shuffle=True, seed=0):
        """Initialization method.

        Args:
            data (np.array): Array of samples.
            labels (np.array): Array of labels.
            n_pairs (int): Number of pairs.
            batch_size (int): Batch size.
            input_shape (tuple): Shape of the reshaped array.
            normalize (tuple): Normalization bounds.
            shuffle (bool): Whether data should be shuffled or not.
            seed (int): Provides deterministic traits when using `random` module.

        """

        logger.info('Overriding class: Dataset -> BalancedPairDataset.')

        # Overrides its parent class with any custom arguments if needed
        super(BalancedPairDataset, self).__init__(batch_size, input_shape, normalize, shuffle, seed)

        # Tries to assert the following statement
        try:
            # Checks if supplied labels are not equal
            assert not np.all(labels == labels[0])

        # If statement is not valid
        except:
            # Raises an error
            raise e.ValueError('`labels` should have distinct values')

        # Amount of pairs
        self.n_pairs = n_pairs

        # Pre-processes the data
        data = self.preprocess(data)

        # Creates pairs of data and labels
        pairs = self._create_pairs(data, labels)

        # Builds up the class
        self._build(pairs)

        logger.info('Class overrided.')

    @property
    def n_pairs(self):
        """int: Amount of pairs.

        """

        return self._n_pairs

    @n_pairs.setter
    def n_pairs(self, n_pairs):
        if not isinstance(n_pairs, int):
            raise e.TypeError('`n_pairs` should be a integer')
        
        self._n_pairs = n_pairs

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

    def _create_pairs(self, data, labels):
        """Creates balanced pairs from data and labels.

        Args:
            data (np.array): Array of samples.
            labels (np.array): Array of labels.

        Returns:
            Tuple containing pairs of samples along their labels.

        """

        logger.debug('Creating pairs ...')

        # Defines the number of samples
        n_samples = data.shape[0]

        # Divides equally the number of pairs
        n_pairs = self.n_pairs // 2

        # Defines the positive lists
        x1_p, x2_p, y_p = [], [], []

        # Defines the negative lists
        x1_n, x2_n, y_n = [], [], []

        # Iterates until both positive and negative pairs
        while len(y_p) < n_pairs or len(y_n) < n_pairs:
            # Samples two random indexes
            idx = tf.random.uniform([2], maxval=n_samples, dtype='int32')

            # If labels are equal on the particular indexes
            if tf.equal(tf.gather(labels, idx[0]), tf.gather(labels, idx[1])):
                # Appends positive data to `x1`
                x1_p.append(tf.gather(data, idx[0]))

                # Appends positive data to `x2`
                x2_p.append(tf.gather(data, idx[1]))

                # Appends positive label to `y`
                y_p.append(1.0)

            # If labels are not equal on the particular indexes
            else:
                # Appends negative data to `x1`
                x1_n.append(tf.gather(data, idx[0]))

                # Appends negative data to `x2`
                x2_n.append(tf.gather(data, idx[1]))

                # Appends negative label to `y`
                y_n.append(0.0)

        # Merges the positive and negative `x1`
        x1 = x1_p[:n_pairs] + x1_n[:n_pairs]

        # Merges the positive and negative `x2`
        x2 = x2_p[:n_pairs] + x2_n[:n_pairs]

        # Merges the positive and negative labels
        y = y_p[:n_pairs] + y_n[:n_pairs]

        logger.debug(f'Pairs: {self.n_pairs}.')

        return x1, x2, y

    def _build(self, pairs):
        """Builds the class.

        Args:
            pairs (tuple): Pairs of samples along their labels.

        """

        # Checks if data should be shuffled
        if self.shuffle:
            # Creates dataset from shuffled and batched data
            self.batches = tf.data.Dataset.from_tensor_slices(pairs).shuffle(c.BUFFER_SIZE).batch(self.batch_size)

        # If data should not be shuffled
        else:
            # Creates dataset from batched data
            self.batches = tf.data.Dataset.from_tensor_slices(pairs).batch(self.batch_size)


class RandomPairDataset(Dataset):
    """A RandomPairDataset class is responsible for implementing a dataset that randomly creates pairs of data,
    as well as their similarity (1) or dissimilarity (0).

    """

    def __init__(self, data, labels, batch_size=1, input_shape=None, normalize=(0, 1), seed=0):
        """Initialization method.

        Args:
            data (np.array): Array of samples.
            labels (np.array): Array of labels.
            batch_size (int): Batch size.
            input_shape (tuple): Shape of the reshaped array.
            normalize (tuple): Normalization bounds.
            seed (int): Provides deterministic traits when using `random` module.

        """

        logger.info('Overriding class: Dataset -> RandomPairDataset.')

        # Overrides its parent class with any custom arguments if needed
        super(RandomPairDataset, self).__init__(batch_size, input_shape, normalize, False, seed)

        # Pre-processes the data
        data = self.preprocess(data)

        # Creates pairs of data and labels
        pairs = self._create_pairs(data, labels)

        # Builds up the class
        self._build(pairs)

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

    def _create_pairs(self, data, labels):
        """Creates random pairs from data and labels.

        Args:
            data (np.array): Array of samples.
            labels (np.array): Array of labels.

        Returns:
            Tuple containing pairs of samples along their labels.

        """

        logger.debug('Creating pairs ...')

        # Defines the number of samples
        n_samples = data.shape[0]

        # Defines the number of possible pairs
        n_pairs = n_samples // 2

        # Randomly samples indexes
        indexes = tf.random.shuffle(tf.range(n_samples))

        # Gathers samples
        x1, x2 = tf.gather(data, indexes[:n_pairs]), tf.gather(data, indexes[n_pairs:])

        # Gathers samples
        y1, y2 = tf.gather(labels, indexes[:n_pairs]), tf.gather(labels, indexes[n_pairs:])

        # If labels are equal, it means that samples are similar
        y = tf.cast(tf.equal(y1, y2), 'float32')

        # Calculates the number of positive and negative pairs
        n_pos_pairs = tf.math.count_nonzero(y)
        n_neg_pairs = y.shape[0] - n_pos_pairs

        logger.debug(f'Positive pairs: {n_pos_pairs} | Negative pairs: {n_neg_pairs}')

        return (x1, x2, y)

    def _build(self, pairs):
        """Builds the class.

        Args:
            pairs (tuple): Pairs of samples along their labels.

        """

        # Creates batches from tensor slices
        self.batches = tf.data.Dataset.from_tensor_slices(pairs).batch(self.batch_size)
