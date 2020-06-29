import tensorflow as tf


class Dataset:
    """A Dataset class is responsible for receiving raw data, pre-processing it and
    persisting batches that will be feed as inputs to the networks.

    """

    def __init__(self, seed=0):
        """Initialization method.

        Args:
            seed (int): Provides deterministic traits when using `random` module.

        """

        # Creates a property to hold batches
        self.batches = None

        # Defines the tensorflow random seed
        tf.random.set_seed(seed)

    @property
    def batches(self):
        """tf.data.Dataset: Batches of data (samples, labels).

        """

        return self._batches

    @batches.setter
    def batches(self, batches):
        self._batches = batches

    def _build(self):
        """Method that builds the class.

        Note that you need to implement this method directly on its child. Essentially,
        each Dataset has its building procedure.

        Raises:
            NotImplementedError.

        """

        raise NotImplementedError
