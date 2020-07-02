import tensorflow as tf


class Base(tf.keras.Model):
    """A Base class is responsible for easily-implementing the base twin architecture of a Siamese Network.

    """

    def __init__(self, name=''):
        """Initialization method.

        Args:
            name (str): Naming identifier.

        """

        # Overrides its parent class with any custom arguments if needed
        super(Base, self).__init__(name=name)

    def call(self, x):
        """Method that holds vital information whenever this class is called.

        Note that you need to implement this method directly on its child. Essentially,
        each neural network has its own forward pass implementation.

        Args:
            x (tf.Tensor): Tensor containing the input sample.

        Raises:
            NotImplementedError.

        """

        raise NotImplementedError


class Siamese(tf.keras.Model):
    """An Siamese class is responsible for implementing the base of Siamese Neural Networks.

    """

    def __init__(self, base, name=''):
        """Initialization method.

        Args:
            base (Base): Twin architecture.
            name (str): Naming identifier.

        """

        # Overrides its parent class with any custom arguments if needed
        super(Siamese, self).__init__(name=name)

        # Defines the Siamese's base twin architecture
        self.B = base

    @property
    def B(self):
        """Base: Twin architecture.

        """

        return self._B

    @B.setter
    def B(self, B):
        self._B = B

    def compile(self, optimizer):
        """Method that builds the network by attaching optimizer, loss and metrics.

        Note that you need to implement this method directly on its child. Essentially,
        each type of Siamese has its own set of loss and metrics.

        Args:
            optimizer (tf.keras.optimizers): Optimization algorithm.

        Raises:
            NotImplementedError.

        """

        raise NotImplementedError

    @tf.function
    def step(self, x1, x2, y):
        """Method that performs a single batch optimization step.

        Note that you need to implement this method directly on its child. Essentially,
        each type of Siamese has an unique step.

        Args:
            x1 (tf.Tensor): Tensor containing first samples from input pairs.
            x2 (tf.Tensor): Tensor containing second samples from input pairs.
            y (tf.Tensor): Tensor containing labels (1 for similar, 0 for dissimilar).

        Raises:
            NotImplementedError.

        """

        raise NotImplementedError

    def fit(self, batches, epochs=100):
        """Method that trains the model over training batches.

        Note that you need to implement this method directly on its child. Essentially,
        each type of Siamese may use a distinct type of dataset.

        Args:
            batches (PairDataset | RandomPairDataset | TripletDataset): Batches of tuples holding training samples and labels.
            epochs (int): Maximum number of epochs.

        Raises:
            NotImplementedError.

        """

        raise NotImplementedError

    def evaluate(self, batches):
        """Method that evaluates the model over validation or testing batches.

        Note that you need to implement this method directly on its child. Essentially,
        each type of Siamese may use a distinct type of dataset.

        Args:
            batches (PairDataset | RandomPairDataset | TripletDataset): Batches of tuples
                holding validation / testing samples and labels.

        Raises:
            NotImplementedError.

        """

        raise NotImplementedError

    def predict(self, x1, x2):
        """Method that performs a forward pass over a set of samples and returns the network's output.

        Note that you need to implement this method directly on its child. Essentially,
        each type of Siamese may predict in a different way.

        Args:
            x1 (tf.Tensor): Tensor containing first samples from input pairs.
            x2 (tf.Tensor): Tensor containing second samples from input pairs.

        Raises:
            NotImplementedError.

        """

        raise NotImplementedError

    def extract_embeddings(self, x, input_shape=None):
        """Method that extracts embeddings by performing a forward pass over the base architecture (embedder).

        Args:
            x (np.array, tf.Tensor): Array or tensor containing the inputs to be embedded.
            input_shape (tuple): Shape of the input layer.

        Returns:
            A tensor containing the embedded inputs.

        """

        # Converts the input to a tensor
        x = tf.convert_to_tensor(x)

        # Reshapes the tensor
        x = tf.reshape(x, input_shape)

        # Passes down through the embedder
        x = self.B(x)

        return x
