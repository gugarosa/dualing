"""Cross-Entropy Siamese Network.
"""

from typing import Optional, Union

import tensorflow as tf

import dualing.utils.exception as e
from dualing.core import BinaryCrossEntropy, Siamese
from dualing.core.model import Base
from dualing.datasets.pair import BalancedPairDataset, RandomPairDataset
from dualing.utils import logging

logger = logging.get_logger(__name__)


class CrossEntropySiamese(Siamese):
    """A CrossEntropySiamese class is responsible for implementing the
    cross-entropy version of Siamese Neural Networks.

    References:
        G. Koch, R. Zemel and R. Salakhutdinov.
        Siamese neural networks for one-shot image recognition.
        ICML Deep Learning Workshop (2015).

    """

    def __init__(
        self,
        base: Base,
        distance_metric: Optional[str] = "concat",
        name: Optional[str] = "",
    ):
        """Initialization method.

        Args:
            base: Twin architecture.
            distance_metric: Distance metric.
            name: Naming identifier.

        """

        logger.info("Overriding class: Siamese -> CrossEntropySiamese.")

        super(CrossEntropySiamese, self).__init__(base, name=name)

        # Distance metric
        self.distance = distance_metric

        # Defines the output layer
        self.o = tf.keras.layers.Dense(1, activation="sigmoid")

        logger.info("Class overrided.")

    @property
    def distance(self) -> str:
        """Distance metric."""

        return self._distance

    @distance.setter
    def distance(self, distance: str) -> None:
        if distance not in ["concat", "diff"]:
            raise e.ValueError("`distance` should be `concat`, or `diff`")

        self._distance = distance

    def compile(self, optimizer: tf.keras.optimizers) -> None:
        """Method that builds the network by attaching optimizer, loss and metrics.

        Args:
            optimizer: Optimization algorithm.

        """

        # Creates an optimizer object
        self.optimizer = optimizer

        # Defines the loss function
        self.loss = BinaryCrossEntropy()

        # Defines the accuracy function
        self.acc = tf.keras.metrics.binary_accuracy

        # Defines the loss metric
        self.loss_metric = tf.metrics.Mean(name="loss")

        # Defines the accuracy metric
        self.acc_metric = tf.metrics.Mean(name="acc")

    @tf.function
    def step(self, x1: tf.Tensor, x2: tf.Tensor, y: tf.Tensor) -> None:
        """Method that performs a single batch optimization step.

        Args:
            x1: Tensor containing first samples from input pairs.
            x2: Tensor containing second samples from input pairs.
            y: Tensor containing labels (1 for similar, 0 for dissimilar).

        """

        # Uses tensorflow's gradient
        with tf.GradientTape() as tape:
            # Performs the prediction
            y_pred = self.predict(x1, x2)

            # Calculates the loss
            loss = self.loss(y, y_pred)

            # Calculates the accuracy
            acc = self.acc(y, y_pred)

        # Calculates the gradients for each training variable based on the loss function
        gradients = tape.gradient(loss, self.B.trainable_variables)

        # Applies the gradients using an optimizer
        self.optimizer.apply_gradients(zip(gradients, self.B.trainable_variables))

        # Updates the metrics' states
        self.loss_metric.update_state(loss)
        self.acc_metric.update_state(acc)

    def fit(
        self,
        batches: Union[BalancedPairDataset, RandomPairDataset],
        epochs: Optional[int] = 100,
    ) -> None:
        """Method that trains the model over training batches.

        Args:
            batches: Batches of tuples holding training samples and labels.
            epochs: Maximum number of epochs.

        """

        logger.info("Fitting model ...")

        # Gathers the amount of batches
        n_batches = tf.data.experimental.cardinality(batches).numpy()

        for epoch in range(epochs):
            logger.info("Epoch %d/%d", epoch + 1, epochs)

            # Resets metrics' states
            self.loss_metric.reset_states()
            self.acc_metric.reset_states()

            # Defines a customized progress bar
            b = tf.keras.utils.Progbar(n_batches, stateful_metrics=["loss", "acc"])

            for (x1_batch, x2_batch, y_batch) in batches:
                # Performs the optimization step
                self.step(x1_batch, x2_batch, y_batch)

                # Adds corresponding values to the progress bar
                b.add(
                    1,
                    values=[
                        ("loss", self.loss_metric.result()),
                        ("acc", self.acc_metric.result()),
                    ],
                )

            logger.to_file(
                f"Loss: {self.loss_metric.result()} | Accuracy: {self.acc_metric.result()}"
            )

    def evaluate(self, batches: Union[BalancedPairDataset, RandomPairDataset]) -> None:
        """Method that evaluates the model over validation or testing batches.

        Args:
            batches: Batches of tuples holding validation / testing samples and labels.

        """

        logger.info("Evaluating model ...")

        # Gathers the amount of batches
        n_batches = tf.data.experimental.cardinality(batches).numpy()

        # Resets metrics' states
        self.loss_metric.reset_states()
        self.acc_metric.reset_states()

        # Defines a customized progress bar
        b = tf.keras.utils.Progbar(n_batches, stateful_metrics=["val_loss", "val_acc"])

        for (x1_batch, x2_batch, y_batch) in batches:
            # Performs the prediction
            y_pred = self.predict(x1_batch, x2_batch)

            # Calculates the loss
            loss = self.loss(y_batch, y_pred)

            # Calculates the accuracy
            acc = self.acc(y_batch, y_pred)

            # Updates the metrics' states
            self.loss_metric.update_state(loss)
            self.acc_metric.update_state(acc)

            # Adds corresponding values to the progress bar
            b.add(
                1,
                values=[
                    ("val_loss", self.loss_metric.result()),
                    ("val_acc", self.acc_metric.result()),
                ],
            )

        logger.to_file(
            f"Val Loss: {self.loss_metric.result()} | Val Accuracy: {self.acc_metric.result()}"
        )

    def predict(self, x1: tf.Tensor, x2: tf.Tensor) -> tf.Tensor:
        """Method that performs a forward pass over samples and returns the network's output.

        Args:
            x1: Tensor containing first samples from input pairs.
            x2: Tensor containing second samples from input pairs.

        Returns:
            (tf.Tensor): The similarity score between samples `x1` and `x2`.

        """

        # Passes samples through the network
        z1 = self.B(x1)
        z2 = self.B(x2)

        if self.distance == "concat":
            y_pred = tf.squeeze(self.o(tf.concat([z1, z2], -1)), -1)

        elif self.distance == "diff":
            y_pred = tf.squeeze(self.o(tf.abs(z1 - z2)), -1)

        # Checks if rank of predictions is equal to two
        if tf.rank(y_pred) == 2:
            # If yes, reduces to an one-ranked tensor
            y_pred = tf.reduce_mean(y_pred, -1)

        return y_pred
