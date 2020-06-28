import tensorflow as tf
from tensorflow import data

import dualing.utils.logging as l
from dualing.core import Dataset

logger = l.get_logger(__name__)


class PairDataset(Dataset):
    """An PairDataset class is responsible for creating a dataset that encodes data into pairs
    and provide a similarity label (0) if they belong to the same class.

    """

    def __init__(self, data, labels, batch_size=1, seed=0):
        """Initialization method.

        Args:
            data (np.array): An array of images.
            batch_size (int): Size of batches.
            shuffle (bool): Whether batches should be shuffled or not.

        """

        logger.info('Overriding class: Dataset -> PairDataset.')

        # Overrides its parent class with any custom arguments if needed
        super(PairDataset, self).__init__(seed)

        # Creating pairs of data and their labels.
        pairs = self._create_pairs(data, labels)

        # Building up the dataset class
        self._build(pairs, batch_size)

        logger.info('Class overrided.')

    def _create_pairs(self, data, labels):
        """
        """

        #
        n_samples = data.shape[0]

        #
        n_pairs = n_samples // 2

        #
        indexes = tf.random.shuffle(tf.range(n_samples))

        #
        x1, x2 = tf.gather(data, indexes[:n_pairs]), tf.gather(data, indexes[n_pairs:])

        #
        y1, y2 = tf.gather(labels, indexes[:n_pairs]), tf.gather(labels, indexes[n_pairs:])

        #
        y = tf.equal(y1, y2)
        
        return (x1, x2, y)


    def _build(self, pairs, batch_size):
        """Builds the batches based on the pre-processed images.

        Args:
            processed_images (np.array): An array of pre-processed images.
            batch_size (int): Size of batches.

        """

        # Creating the dataset from shuffled and batched data
        self.batches = data.Dataset.from_tensor_slices(pairs).batch(batch_size)
