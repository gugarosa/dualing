"""Triplet Loss Siamese Network.
"""

from typing import Optional

import tensorflow as tf

import dualing.utils.exception as e
from dualing.core import Siamese, TripletHardLoss, TripletSemiHardLoss
from dualing.core.model import Base
from dualing.datasets.batch import BatchDataset
from dualing.utils import logging

logger = logging.get_logger(__name__)


class TripletSiamese(Siamese):
    """A TripletSiamese class is responsible for implementing the
    triplet-loss version of Siamese Neural Networks.

    References:
        X. Dong and J. Shen. Triplet loss in siamese network for object tracking.
        Proceedings of the European Conference on Computer Vision (2018).

    """

    def __init__(
        self,
        base: Base,
        loss: Optional[str] = "hard",
        margin: Optional[int] = 1.0,
        soft: Optional[bool] = False,
        distance_metric: Optional[str] = "L2",
        name: Optional[str] = "",
    ):
        """Initialization method.

        Args:
            base: Twin architecture.
            loss: Whether network should use hard or semi-hard negative mining.
            margin: Radius around the embedding space.
            soft: Whether network should use soft margin or not.
            distance_metric: Distance metric.
            name: Naming identifier.

        """

        logger.info("Overriding class: Siamese -> TripletSiamese.")

        super(TripletSiamese, self).__init__(base, name=name)

        # Type of loss
        self.loss_type = loss

        # Radius around embedding space
        self.margin = margin

        # Soft margin
        self.soft = soft

        # Distance metric
        if distance_metric == "L1":
            self.distance = "L2"
        elif distance_metric == "L2":
            self.distance = "squared-L2"
        else:
            self.distance = "angular"

        logger.info("Class overrided.")

    @property
    def loss_type(self) -> str:
        """Type of loss (hard or semi-hard)."""

        return self._loss_type

    @loss_type.setter
    def loss_type(self, loss_type: str) -> None:
        if loss_type not in ["hard", "semi-hard"]:
            raise e.ValueError("`loss_type` should be `hard` or `semi-hard`")

        self._loss_type = loss_type

    @property
    def soft(self) -> bool:
        """Whether soft margin should be used or not."""

        return self._soft

    @soft.setter
    def soft(self, soft: bool) -> None:
        if not isinstance(soft, bool):
            raise e.TypeError("`soft` should be a boolean")

        self._soft = soft

    @property
    def margin(self) -> float:
        """Radius around the embedding space."""

        return self._margin

    @margin.setter
    def margin(self, margin: float) -> None:
        if not isinstance(margin, float):
            raise e.TypeError("`margin` should be a float")
        if margin <= 0:
            raise e.ValueError("`margin` should be greater than 0")

        self._margin = margin

    @property
    def distance(self) -> str:
        """Distance metric."""

        return self._distance

    @distance.setter
    def distance(self, distance: str) -> None:
        if distance not in ["L1", "L2", "squared-L2", "angular"]:
            raise e.ValueError("`distance` should be `L1`, `L2` or `angular`")

        self._distance = distance

    def compile(self, optimizer: tf.keras.optimizers) -> None:
        """Method that builds the network by attaching optimizer, loss and metrics.

        Args:
            optimizer: Optimization algorithm.

        """

        # Creates an optimizer object
        self.optimizer = optimizer

        # Check it is supposed to use hard negative mining
        if self.loss_type == "hard":
            self.loss = TripletHardLoss()

        # If it is supposed to use semi-hard negative mining
        elif self.loss_type == "semi-hard":
            self.loss = TripletSemiHardLoss()

        # Defines the loss metric
        self.loss_metric = tf.metrics.Mean(name="loss")

    @tf.function
    def step(self, x: tf.Tensor, y: tf.Tensor) -> None:
        """Method that performs a single batch optimization step.

        Args:
            x: Tensor containing samples.
            y: Tensor containing labels.

        """

        # Uses tensorflow's gradient
        with tf.GradientTape() as tape:
            # Passes the batch inputs through the network
            y_pred = self.B(x)

            # Checks the rank of the output
            if tf.rank(y_pred) == 3:
                # If it is 3-rank, reduce its mean over the second dimension
                # This is purely to allow recurrrent-based models compatibility
                y_pred = tf.reduce_mean(y_pred, 1)

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

    def fit(self, batches: BatchDataset, epochs: Optional[int] = 100) -> None:
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

            # Defines a customized progress bar
            b = tf.keras.utils.Progbar(n_batches, stateful_metrics=["loss"])

            for (x_batch, y_batch) in batches:
                # Performs the optimization step
                self.step(x_batch, y_batch)

                # Adds corresponding values to the progress bar
                b.add(1, values=[("loss", self.loss_metric.result())])

            logger.to_file(f"Loss: {self.loss_metric.result()}")

    def evaluate(self, batches: BatchDataset) -> None:
        """Method that evaluates the model over validation or testing batches.

        Args:
            batches: Batches of tuples holding validation / test samples and labels.

        """

        logger.info("Evaluating model ...")

        # Gathers the amount of batches
        n_batches = tf.data.experimental.cardinality(batches).numpy()

        # Resets metrics' states
        self.loss_metric.reset_states()

        # Defines a customized progress bar
        b = tf.keras.utils.Progbar(n_batches, stateful_metrics=["val_loss"])

        for (x_batch, y_batch) in batches:
            # Passes the batch inputs through the network
            y_pred = self.B(x_batch)

            # Checks the rank of the output
            if tf.rank(y_pred) == 3:
                # If it is 3-rank, reduce its mean over the second dimension
                # This is purely to allow recurrrent-based models compatibility
                y_pred = tf.reduce_mean(y_pred, 1)

            # Performs the L2 normalization prior to the loss function
            y_pred = tf.math.l2_normalize(y_pred, axis=-1)

            # Calculates the loss
            loss = self.loss(y_batch, y_pred, self.margin)

            # Updates the metrics' states
            self.loss_metric.update_state(loss)

            # Adds corresponding values to the progress bar
            b.add(1, values=[("val_loss", self.loss_metric.result())])

        logger.to_file(f"Val Loss: {self.loss_metric.result()}")

    def predict(self, x1: tf.Tensor, x2: tf.Tensor) -> tf.Tensor:
        """Method that performs a forward pass over samples and returns the network's output.

        Args:
            x1: Tensor containing first samples from input pairs.
            x2: Tensor containing second samples from input pairs.

        Returns:
            (tf.Tensor): The distance between samples `x1` and `x2`.

        """

        # Passes samples through the network
        z1 = self.B(x1)
        z2 = self.B(x2)

        # Checks the rank of the output
        if tf.rank(z1) == 3:
            # If it is 3-rank, reduce its mean over the second dimension
            # This is purely to allow recurrrent-based models compatibility
            z1 = tf.reduce_mean(z1, 1)
            z2 = tf.reduce_mean(z2, 1)

        if self.distance == "L1":
            y_pred = tf.math.sqrt(tf.linalg.norm(z1 - z2, axis=1))

        elif self.distance == "L2":
            y_pred = tf.linalg.norm(z1 - z2, axis=1)

        elif self.distance == "angular":
            y_pred = tf.keras.losses.cosine_similarity(z1, z2)

        return y_pred
