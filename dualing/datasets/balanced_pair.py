import tensorflow as tf

import dualing.utils.logging as l
from dualing.core import Dataset

logger = l.get_logger(__name__)


class BalancedPairDataset(Dataset):
    """A BalancedPairDataset class is responsible for implementing a dataset that creates balanced pairs of data,
    as well as their similarity (1) or dissimilarity (0).

    """

    def __init__(self, data, labels, n_pairs=2, batch_size=1, shape=None, normalize=True, seed=0):
        """Initialization method.

        Args:
            data (np.array): Array of samples.
            labels (np.array): Array of labels.
            batch_size (int): Batch size.
            shape (tuple): A tuple containing the shape if the array should be forced to reshape.
            normalize (bool): Whether images should be normalized between -1 and 1.
            seed (int): Provides deterministic traits when using `random` module.

        """

        logger.info('Overriding class: Dataset -> BalancedPairDataset.')

        # Overrides its parent class with any custom arguments if needed
        super(BalancedPairDataset, self).__init__(seed)

        # Pre-processes the data
        data = self._preprocess(data, shape, normalize)

        # Creates pairs of data and their labels
        pairs = self._create_pairs(data, labels, n_pairs)

        # Builds up the class
        self._build(pairs, batch_size)

        logger.info('Class overrided.')
        logger.debug(f'Pairs: {n_pairs}.')

    def _create_pairs(self, data, labels, n_pairs):
        """Creates balanced pairs from data and labels.

        Args:
            data (np.array): Array of samples.
            labels (np.array): Array of labels.
            n_pairs (int): Number of total pairs.

        Returns:
            Tuple containing (x1, x2, y) pairs.

        """

        # Defines the number of samples
        n_samples = data.shape[0]

        # Divides equally the number of pairs
        n_pairs = n_pairs // 2

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

        return x1, x2, y

    def _build(self, pairs, batch_size):
        """Builds the class.

        Args:
            pairs (tuple): (x1, x2, y) pairs.
            batch_size (int): Batch size.

        """

        # Creates batches from tensor slices and shuffles them
        self.batches = tf.data.Dataset.from_tensor_slices(pairs).shuffle(1000000).batch(batch_size)
