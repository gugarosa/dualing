import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.utils import Progbar

import dualing.utils.logging as l

logger = l.get_logger(__name__)


class Network(Model):
    """A Network class is responsible for easily-implementing the base architecture of
    a Siamese Network, when custom training or additional sets are not needed.

    """

    def __init__(self, name=''):
        """Initialization method.

        Note that basic variables shared by all childs should be declared here, e.g., common layers.

        Args:
            name (str): The base network's identifier name.

        """

        # Overrides its parent class with any custom arguments if needed
        super(Network, self).__init__(name=name)

    def call(self, x):
        """Method that holds vital information whenever this class is called.

        Note that you need to implement this method directly on its child. Essentially,
        each neural network has its own forward pass implementation.

        Args:
            x (tf.Tensor): A tensorflow's tensor holding input data.

        Raises:
            NotImplementedError

        """

        raise NotImplementedError


class Siamese(Model):
    """An Siamese class is responsible for customly implementing Siamese Neural Networks.

    """

    def __init__(self, network, name=''):
        """Initialization method.

        Args:
            network (Network): Base architecture.
            name (str): The model's identifier string.

        """

        # Overrides its parent class with any custom arguments if needed
        super(Siamese, self).__init__(name=name)

        # Defining the base architecture
        self.N = network

        #
        self.out = tf.keras.layers.Dense(1, activation='sigmoid')

    @property
    def N(self):
        """Network: Base architecture.

        """

        return self._N

    @N.setter
    def N(self, N):
        self._N = N

    def compile(self, optimizer):
        """Main building method.

        Args:
            optimizer (tf.keras.optimizers): An optimizer instance.

        """

        # Creates an optimizer object
        self.optimizer = optimizer

        # Defining the loss function
        self.loss = tf.keras.losses.binary_crossentropy

        #
        self.loss_metric = tf.metrics.Mean(name='loss')

    @tf.function
    def step(self, x1, x2, y):
        """Performs a single batch optimization step.

        Args:
            x (tf.Tensor): A tensor containing the inputs.

        """

        # Using tensorflow's gradient
        with tf.GradientTape() as tape:
            #
            z1 = self.N(x1)

            #
            z2 = self.N(x2)

            #
            z = tf.concat([z1, z2], axis=0)

            #
            dist = tf.abs(z1 - z2)

            #
            score = tf.squeeze(self.out(dist), -1)

            #
            loss = self.loss(y, score)

        # Calculate the gradients based on loss for each training variable
        gradients = tape.gradient(loss, self.N.trainable_variables)

        # Applies the gradients using an optimizer
        self.optimizer.apply_gradients(zip(gradients, self.N.trainable_variables))

        # Updates the generator's loss state
        self.loss_metric.update_state(loss)

    def fit(self, batches, epochs=100):
        """Trains the model.

        Args:
            batches (Dataset): Training batches containing samples.
            epochs (int): The maximum number of training epochs.

        """

        logger.info('Fitting model ...')

        # Gathering the amount of batches
        n_batches = tf.data.experimental.cardinality(batches).numpy()

        # Iterate through all epochs
        for e in range(epochs):
            logger.info(f'Epoch {e+1}/{epochs}')

            # Resetting state to further append loss
            self.loss_metric.reset_states()

            # Defining a customized progress bar
            b = Progbar(n_batches, stateful_metrics=['loss'])

            # Iterate through all possible training batches
            for (x1_batch, x2_batch, y_batch) in batches:
                # Performs the optimization step
                self.step(x1_batch, x2_batch, y_batch)

                # Adding corresponding values to the progress bar
                b.add(1, values=[('loss', self.loss_metric.result())])

            logger.file(f'Loss: {self.loss_metric.result()}')