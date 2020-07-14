import tensorflow as tf

import dualing.utils.exception as e
import dualing.utils.logging as l
from dualing.core import Siamese, TripletHardLoss, TripletSemiHardLoss

logger = l.get_logger(__name__)


class TripletSiamese(Siamese):
    """A TripletSiamese class is responsible for implementing the
    triplet-loss version of Siamese Neural Networks.

    References:
        X. Dong and J. Shen. Triplet loss in siamese network for object tracking.
        Proceedings of the European Conference on Computer Vision (2018).

    """

    def __init__(self, base, loss='hard', margin=1.0, soft=False, distance_metric='L2', name=''):
        """Initialization method.

        Args:
            base (Base): Twin architecture.
            loss (str): Whether network should use hard or semi-hard negative mining.
            margin (float): Radius around the embedding space.
            soft (bool): Whether network should use soft margin or not.
            distance_metric (str): Distance metric.
            name (str): Naming identifier.

        """

        logger.info('Overriding class: Siamese -> TripletSiamese.')

        # Overrides its parent class with any custom arguments if needed
        super(TripletSiamese, self).__init__(base, name=name)

        # Type of loss
        self.loss_type = loss

        # Radius around embedding space
        self.margin = margin

        # Soft margin
        self.soft = soft

        # Distance metric
        self.distance = distance_metric

        logger.info('Class overrided.')

    @property
    def loss_type(self):
        """str: Type of loss (hard or semi-hard).

        """

        return self._loss_type

    @loss_type.setter
    def loss_type(self, loss_type):
        if loss_type not in ['hard', 'semi-hard']:
            raise e.ValueError('`loss_type` should be `hard` or `semi-hard`')

        self._loss_type = loss_type

    @property
    def soft(self):
        """bool: Whether soft margin should be used or not.

        """

        return self._soft

    @soft.setter
    def soft(self, soft):
        if not isinstance(soft, bool):
            raise e.TypeError('`soft` should be a boolean')
        
        self._soft = soft

    @property
    def margin(self):
        """float: Radius around the embedding space.

        """

        return self._margin

    @margin.setter
    def margin(self, margin):
        if not isinstance(margin, float):
            raise e.TypeError('`margin` should be a float')

        self._margin = margin

    @property
    def distance(self):
        """str: Distance metric.

        """

        return self._distance

    @distance.setter
    def distance(self, distance):
        if distance not in ['L1', 'L2', 'angular']:
            raise e.ValueError('`distance` should be `L1`, `L2` or `angular`')

        self._distance = distance

    def compile(self, optimizer):
        """Method that builds the network by attaching optimizer, loss and metrics.

        Args:
            optimizer (tf.keras.optimizers): Optimization algorithm.

        """

        # Creates an optimizer object
        self.optimizer = optimizer

        # Check it is supposed to use hard negative mining
        if self.loss_type == 'hard':
            # Creates the TripletHardLoss
            self.loss = TripletHardLoss()
        
        # If it is supposed to use semi-hard negative mining
        elif self.loss_type == 'semi-hard':
            # Creates the TripletSemiHardLoss
            self.loss = TripletSemiHardLoss()

        # Defines the loss metric
        self.loss_metric = tf.metrics.Mean(name='loss')

    @tf.function
    def step(self, x, y):
        """Method that performs a single batch optimization step.

        Args:
            x (tf.Tensor): Tensor containing samples.
            y (tf.Tensor): Tensor containing labels.

        """

        # Uses tensorflow's gradient
        with tf.GradientTape() as tape:
            # Passes the batch inputs through the network
            y_pred = self.B(x)

            # Performs the L2 normalization prior to the loss function
            y_pred = tf.math.l2_normalize(y_pred, axis=-1)

            # Calculates the loss
            loss = self.loss(y, y_pred, self.margin, self.soft, self.distance)

        # Calculates the gradients for each training variable based on the loss function
        gradients = tape.gradient(loss, self.B.trainable_variables)

        # Applies the gradients using an optimizer
        self.optimizer.apply_gradients(zip(gradients, self.B.trainable_variables))

        # Updates the metrics' states
        self.loss_metric.update_state(loss)

    def fit(self, batches, epochs=100):
        """Method that trains the model over training batches.

        Args:
            batches (BatchDataset): Batches of tuples holding training samples and labels.
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
