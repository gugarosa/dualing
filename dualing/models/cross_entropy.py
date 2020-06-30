import dualing.utils.logging as l
import dualing.utils.losses as losses
import tensorflow as tf
from dualing.core import Siamese

logger = l.get_logger(__name__)


class CrossEntropySiamese(Siamese):
    """A CrossEntropySiamese class is responsible for implementing the
    cross-entropy version of Siamese Neural Networks.

    """

    def __init__(self, base, name=''):
        """Initialization method.

        Args:
            base (Base): Twin architecture.
            name (str): Naming identifier.

        """

        # Overrides its parent class with any custom arguments if needed
        super(CrossEntropySiamese, self).__init__(base, name=name)

        #
        self.out = tf.keras.layers.Dense(1, activation='sigmoid')

    def compile(self, optimizer):
        """Method that builds the network by attaching optimizer, loss and metrics.

        Args:
            optimizer (tf.keras.optimizers): Optimization algorithm.

        """

        # Creates an optimizer object
        self.optimizer = optimizer

        # Defines the loss function
        self.loss = losses.binary_crossentropy

        # Defines the accuracy function
        self.acc = tf.keras.metrics.binary_accuracy

        # Defines the loss metric
        self.loss_metric = tf.metrics.Mean(name='loss')

        # Defines the accuracy metric
        self.acc_metric = tf.metrics.Mean(name='acc')

    @tf.function
    def step(self, x1, x2, y):
        """Method that performs a single batch optimization step.

        Args:
            x1 (tf.Tensor): Tensor containing first samples from input pairs.
            x2 (tf.Tensor): Tensor containing second samples from input pairs.
            y (tf.Tensor): Tensor containing labels (1 for similar, 0 for dissimilar).

        """

        # Uses tensorflow's gradient
        with tf.GradientTape() as tape:
            # Passes the first sample through the network
            z1 = self.B(x1)

            # Passes the second sample through the network
            z2 = self.B(x2)

            # Calculates their absolute distance (L1)
            dist = tf.abs(z1 - z2)

            # Passes the distance through sigmoid activation and removes last dimension
            score = tf.squeeze(self.out(dist), -1)

            # Calculates the loss
            loss = self.loss(y, score)

            # Calculates the accuracy
            acc = self.acc(y, score)

        # Calculates the gradients for each training variable based on the loss function
        gradients = tape.gradient(loss, self.B.trainable_variables)

        # Applies the gradients using an optimizer
        self.optimizer.apply_gradients(zip(gradients, self.B.trainable_variables))

        # Updates the metrics' states
        self.loss_metric.update_state(loss)
        self.acc_metric.update_state(acc)

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
            self.acc_metric.reset_states()

            # Defines a customized progress bar
            b = tf.keras.utils.Progbar(n_batches, stateful_metrics=['loss', 'acc'])

            # Iterates through all batches
            for (x1_batch, x2_batch, y_batch) in batches:
                # Performs the optimization step
                self.step(x1_batch, x2_batch, y_batch)

                # Adds corresponding values to the progress bar
                b.add(1, values=[('loss', self.loss_metric.result()), ('acc', self.acc_metric.result())])

            logger.file(f'Loss: {self.loss_metric.result()} | Accuracy: {self.acc_metric.result()}')

    def predict(self, x1, x2):
        """
        """
        
        #
        z1 = self.B(x1)

        #
        z2 = self.B(x2)

        #
        dist = tf.abs(z1 - z2)

        #
        score = tf.squeeze(self.out(dist), -1)

        return score
