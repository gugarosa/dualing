import tensorflow as tf

class Dataset:
    """A Dataset class is responsible for receiving raw data, pre-processing it and
    persisting batches that will be feed as inputs to the networks.

    """

    def __init__(self, seed=0):
        """Initialization method.

        Args:
            shuffle (bool): Whether batches should be shuffled or not.

        """

        # Creating a property to whether data should be shuffled or not
        self.seed = seed

        # Creating a property to hold the further batches
        self.batches = None

        #
        tf.random.set_seed(seed)
        

    @property
    def shuffle(self):
        """bool: Whether data should be shuffled or not.

        """

        return self._shuffle

    @shuffle.setter
    def shuffle(self, shuffle):
        self._shuffle = shuffle

    @property
    def batches(self):
        """tf.data.Dataset: An instance of tensorflow's dataset batches.

        """

        return self._batches

    @batches.setter
    def batches(self, batches):
        self._batches = batches

    def _build(self):
        """This method serves to build up the Dataset class. Note that for each child,
        you need to define your own building method.

        Raises:
            NotImplementedError

        """

        raise NotImplementedError
