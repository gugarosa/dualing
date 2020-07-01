import tensorflow as tf

import dualing.utils.logging as l
from dualing.core import Siamese

logger = l.get_logger(__name__)


class TripletSiamese(Siamese):
    """A TripletSiamese class is responsible for implementing the
    contrastive version of Siamese Neural Networks.

    """

    def __init__(self, base, distance='euclidean', margin=1.0, name=''):
        """Initialization method.

        Args:
            base (Base): Twin architecture.
            distance (str): Distance metric.
            margin (float): Radius around the embedding space.
            name (str): Naming identifier.

        """

        logger.info('Overriding class: Siamese -> TripletSiamese.')

        # Overrides its parent class with any custom arguments if needed
        super(TripletSiamese, self).__init__(base, name=name)

        # Defines the distance
        self.distance = distance

        # Defines the margin
        self.margin = margin

        logger.info('Class overrided.')

    @property
    def distance(self):
        """str: Distance metric.

        """

        return self._distance

    @distance.setter
    def distance(self, distance):
        self._distance = distance

    @property
    def margin(self):
        """float: Radius around the embedding space.

        """

        return self._margin

    @margin.setter
    def margin(self, margin):
        self._margin = margin

    def compile(self, optimizer):
        """Method that builds the network by attaching optimizer, loss and metrics.

        Args:
            optimizer (tf.keras.optimizers): Optimization algorithm.

        """

        # Creates an optimizer object
        self.optimizer = optimizer

        # Defines the loss function
        # self.loss = losses.batch_all_triplet_loss

        # Defines the loss metric
        self.loss_metric = tf.metrics.Mean(name='loss')

    @tf.function
    def step(self, x, y):
        """Method that performs a single batch optimization step.

        Args:
            x1 (tf.Tensor): Tensor containing first samples from input pairs.
            x2 (tf.Tensor): Tensor containing second samples from input pairs.
            y (tf.Tensor): Tensor containing labels (1 for similar, 0 for dissimilar).

        """

        # Uses tensorflow's gradient
        with tf.GradientTape() as tape:
            # Passes the first sample through the network
            z = self.B(x)

            # Calculates the loss
            loss = self.loss(y, z, self.margin)

        # Calculates the gradients for each training variable based on the loss function
        gradients = tape.gradient(loss, self.B.trainable_variables)

        # Applies the gradients using an optimizer
        self.optimizer.apply_gradients(zip(gradients, self.B.trainable_variables))

        # Updates the metrics' states
        self.loss_metric.update_state(loss)

    def fit(self, batches, epochs=100):
        """Method that trains the model over training batches.

        Args:
            batches (PairDataset | RandomPairDataset): Batches of tuples holding training samples and labels.
            epochs (int): Maximum number of epochs.

        """

        logger.info('Fitting model ...')

        # Gathers the amount of batches
        n_batches = tf.data.experimental.cardinality(batches).numpy()

        # Iterates through all epochs
        for e in range(epochs):
            logger.info(f'Epoch {e+1}/{epochs}')

            # Resets metrics' states
            self.loss_metric.reset_states()

            # Defines a customized progress bar
            b = tf.keras.utils.Progbar(n_batches, stateful_metrics=['loss'])

            # Iterates through all batches
            for (x_batch, y_batch) in batches:
                # Performs the optimization step
                self.step(x_batch, y_batch)

                # Adds corresponding values to the progress bar
                b.add(1, values=[('loss', self.loss_metric.result())])

            logger.file(f'Loss: {self.loss_metric.result()}')
