import tensorflow as tf

import dualing.utils.exception as e
import dualing.utils.logging as l
from dualing.core import ContrastiveLoss, Siamese

logger = l.get_logger(__name__)


class ContrastiveSiamese(Siamese):
    """A ContrastiveSiamese class is responsible for implementing the
    contrastive version of Siamese Neural Networks.

    References:
        I. Melekhov, J. Kannala and E. Rahtu. Siamese network features for image matching.
        23rd International Conference on Pattern Recognition (2016).

    """

    def __init__(self, base, margin=1.0, distance_metric='L2', name=''):
        """Initialization method.

        Args:
            base (Base): Twin architecture.
            margin (float): Radius around the embedding space.
            distance_metric (str): Distance metric.
            name (str): Naming identifier.

        """

        logger.info('Overriding class: Siamese -> ContrastiveSiamese.')

        # Overrides its parent class with any custom arguments if needed
        super(ContrastiveSiamese, self).__init__(base, name=name)

        # Radius around embedding space
        self.margin = margin

        # Distance metric
        self.distance = distance_metric

        logger.info('Class overrided.')

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

        # Defines the loss function
        self.loss = ContrastiveLoss()

        # Defines the loss metric
        self.loss_metric = tf.metrics.Mean(name='loss')

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

            # If distance is supposed to be L1
            if self.distance == 'L1':
                # Calculates the L1 distance
                y_pred = tf.math.sqrt(tf.linalg.norm(z1 - z2, axis=1))

            # If distance is supposed to be L2
            elif self.distance == 'L2':
                # Calculates the L2 distance
                y_pred = tf.linalg.norm(z1 - z2, axis=1)
            
            # If distance is supposed to be angular
            elif self.distance == 'angular':
                # Calculates the angular distance
                y_pred = tf.keras.losses.cosine_similarity(z1, z2)

            # Calculates the loss
            loss = self.loss(y, y_pred, self.margin)

        # Calculates the gradients for each training variable based on the loss function
        gradients = tape.gradient(loss, self.B.trainable_variables)

        # Applies the gradients using an optimizer
        self.optimizer.apply_gradients(zip(gradients, self.B.trainable_variables))

        # Updates the metrics' states
        self.loss_metric.update_state(loss)

    def fit(self, batches, epochs=100):
        """Method that trains the model over training batches.

        Args:
            batches (BalancedPairDataset, RandomPairDataset): Batches of tuples holding training samples and labels.
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
            for (x1_batch, x2_batch, y_batch) in batches:
                # Performs the optimization step
                self.step(x1_batch, x2_batch, y_batch)

                # Adds corresponding values to the progress bar
                b.add(1, values=[('loss', self.loss_metric.result())])

            logger.file(f'Loss: {self.loss_metric.result()}')

    def evaluate(self, batches):
        """Method that evaluates the model over validation or testing batches.

        Args:
            batches (BalancedPairDataset, RandomPairDataset): Batches of tuples holding
                validation / testing samples and labels.

        """

        logger.info('Evaluating model ...')

        # Gathers the amount of batches
        n_batches = tf.data.experimental.cardinality(batches).numpy()

        # Resets metrics' states
        self.loss_metric.reset_states()

        # Defines a customized progress bar
        b = tf.keras.utils.Progbar(n_batches, stateful_metrics=['val_loss'])

        # Iterates through all batches
        for (x1_batch, x2_batch, y_batch) in batches:
            # Performs the prediction
            y_pred = self.predict(x1_batch, x2_batch)

            # Calculates the loss
            loss = self.loss(y_batch, y_pred, self.margin)

            # Updates the metrics' states
            self.loss_metric.update_state(loss)

            # Adds corresponding values to the progress bar
            b.add(1, values=[('val_loss', self.loss_metric.result())])

        logger.file(f'Val Loss: {self.loss_metric.result()}')

    def predict(self, x1, x2):
        """Method that performs a forward pass over a set of samples and returns the network's output.

        Args:
            x1 (tf.Tensor): Tensor containing first samples from input pairs.
            x2 (tf.Tensor): Tensor containing second samples from input pairs.

        Returns:
            The distance between samples `x1` and `x2`.

        """

        # Passes the first sample through the network
        z1 = self.B(x1)

        # Passes the second sample through the network
        z2 = self.B(x2)

        # If distance is supposed to be L1
        if self.distance == 'L1':
            # Calculates the L1 distance
            y_pred = tf.math.sqrt(tf.linalg.norm(z1 - z2, axis=1))

        # If distance is supposed to be L2
        elif self.distance == 'L2':
            # Calculates the L2 distance
            y_pred = tf.linalg.norm(z1 - z2, axis=1)
        
        # If distance is supposed to be angular
        elif self.distance == 'angular':
            # Calculates the angular distance
            y_pred = tf.keras.losses.cosine_similarity(z1, z2)

        return y_pred
