"""Contrastive-loss Siamese model."""

import numbers

import tensorflow as tf

from dualing.core import ContrastiveLoss, Siamese
from dualing.losses import pair_distance
from dualing.models._utils import pair_dataset
from dualing.utils import exception


class ContrastiveSiamese(Siamese):
    """Train a shared embedder with contrastive loss."""

    def __init__(
        self,
        base: tf.keras.Model,
        margin: float = 1.0,
        distance_metric: str = "L2",
        name: str = "",
    ) -> None:
        super().__init__(base, name=name)

        self.margin = margin
        self.distance = distance_metric

    @property
    def margin(self) -> float:
        """Contrastive margin."""

        return self._margin

    @margin.setter
    def margin(self, margin: float) -> None:
        if not isinstance(margin, numbers.Real):
            raise exception.TypeError("`margin` should be a number")

        if margin <= 0:
            raise exception.ValueError("`margin` should be greater than 0")

        self._margin = float(margin)

    @property
    def distance(self) -> str:
        """Distance metric."""

        return self._distance

    @distance.setter
    def distance(self, distance: str) -> None:
        if distance not in {"L1", "L2", "squared-L2", "angular"}:
            raise exception.ValueError(
                "`distance` should be `L1`, `L2`, `squared-L2`, or `angular`"
            )

        self._distance = distance

    @property
    def distance_metric(self) -> str:
        """Distance metric."""

        return self.distance

    @distance_metric.setter
    def distance_metric(self, distance: str) -> None:
        self.distance = distance

    def call(self, inputs, training: bool = False) -> tf.Tensor:
        left, right = inputs

        return pair_distance(
            self.embed(left, training),
            self.embed(right, training),
            self.distance,
        )

    def compile(self, optimizer="rmsprop", **kwargs) -> None:
        """Compile the model with contrastive loss by default."""

        self.loss = ContrastiveLoss(self.margin)
        self.loss_metric = tf.keras.metrics.Mean(name="loss")

        kwargs.setdefault("loss", self.loss)

        tf.keras.Model.compile(self, optimizer=optimizer, **kwargs)

    def step(self, x1: tf.Tensor, x2: tf.Tensor, y: tf.Tensor) -> None:
        """Run one optimization step."""

        with tf.GradientTape() as tape:
            prediction = self((x1, x2), training=True)
            loss = self.loss(y, prediction)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_metric.update_state(loss)

    def fit(self, batches=None, y=None, epochs: int = 100, **kwargs):
        """Train with native or legacy pair datasets."""

        x = kwargs.pop("x", batches)

        if isinstance(x, tf.data.Dataset):
            x = pair_dataset(x)
            kwargs.setdefault("shuffle", False)

        return tf.keras.Model.fit(self, x=x, y=y, epochs=epochs, **kwargs)

    def evaluate(self, batches=None, y=None, **kwargs):
        """Evaluate with native or legacy pair datasets."""

        x = kwargs.pop("x", batches)

        return tf.keras.Model.evaluate(self, x=pair_dataset(x), y=y, **kwargs)

    def predict(self, x1, x2=None, **kwargs):
        """Predict natively or compare two sample batches."""

        if x2 is not None and not isinstance(x2, numbers.Integral):
            return self((x1, x2), training=False)

        batch_size = kwargs.pop("batch_size", x2)

        return tf.keras.Model.predict(self, x1, batch_size=batch_size, **kwargs)

    def compare(self, left, right) -> tf.Tensor:
        """Return distances between two batches of samples."""

        return self((left, right), training=False)
