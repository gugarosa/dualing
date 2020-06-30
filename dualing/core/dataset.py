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

    def _preprocess(self, data, shape, normalize):
        """Pre-process the data by reshaping and normalizing, if necessary.

        Args:
            data (np.array): Array of data.
            shape (tuple): Tuple containing the shape if the array should be forced to reshape.
            normalize (bool): Whether data should be normalized between -1 and 1.

        Returns:
            Array of pre-processed data.

        """

        # If a shape is supplied
        if shape:
            # Reshapes the array and make sure that it is float typed
            data = data.reshape(shape).astype('float32')

        # If no shape is supplied
        else:
            # Just make sure that the array is float typed
            data = data.astype('float32')

        # If data should be normalized
        if normalize:
            # Normalize the data between -1 and 1
            data = (data - 127.5) / 127.5

        return data

    def _build(self):
        """Method that builds the class.

        Note that you need to implement this method directly on its child. Essentially,
        each Dataset has its building procedure.

        Raises:
            NotImplementedError.

        """

        raise NotImplementedError
