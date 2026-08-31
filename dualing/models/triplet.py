"""Triplet-loss Siamese model."""

import numbers

import tensorflow as tf

from dualing.core import Siamese, TripletHardLoss, TripletSemiHardLoss
from dualing.losses import pair_distance
from dualing.utils import exception


class TripletSiamese(Siamese):
    """Train a shared embedder with hard or semi-hard triplet loss."""

    def __init__(
        self,
        base: tf.keras.Model,
        loss: str = "hard",
        margin: float = 1.0,
        soft: bool = False,
        distance_metric: str = "L2",
        name: str = "",
        *,
        mining: str | None = None,
    ) -> None:
        super().__init__(base, name=name)

        legacy_mode = mining is None

        self.loss_type = loss if mining is None else mining
        self.margin = margin
        self.soft = soft
        self.distance_metric = distance_metric

        if legacy_mode and distance_metric == "L1":
            self.distance = "L2"
        elif legacy_mode and distance_metric == "L2":
            self.distance = "squared-L2"
        elif legacy_mode:
            self.distance = "angular"
        else:
            self.distance = distance_metric

    @property
    def loss_type(self) -> str:
        """Triplet mining strategy."""

        return self._loss_type

    @loss_type.setter
    def loss_type(self, loss_type: str) -> None:
        if loss_type not in {"hard", "semi-hard"}:
            raise exception.ValueError("`loss_type` should be `hard` or `semi-hard`")

        self._loss_type = loss_type

    @property
    def mining(self) -> str:
        """Triplet mining strategy."""

        return self.loss_type

    @mining.setter
    def mining(self, mining: str) -> None:
        self.loss_type = mining

    @property
    def soft(self) -> bool:
        """Whether to use a soft margin."""

        return self._soft

    @soft.setter
    def soft(self, soft: bool) -> None:
        if not isinstance(soft, bool):
            raise exception.TypeError("`soft` should be a boolean")

        self._soft = soft

    @property
    def margin(self) -> float:
        """Triplet margin."""

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
        """Legacy pair and loss distance metric."""

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
        """Requested pair distance metric."""

        return self._distance_metric

    @distance_metric.setter
    def distance_metric(self, distance: str) -> None:
        if distance not in {"L1", "L2", "squared-L2", "angular"}:
            raise exception.ValueError(
                "`distance_metric` should be `L1`, `L2`, `squared-L2`, or `angular`"
            )

        self._distance_metric = distance

    def call(self, inputs, training: bool = False) -> tf.Tensor:
        return tf.math.l2_normalize(self.embed(inputs, training), axis=-1)

    def compile(self, optimizer="rmsprop", **kwargs) -> None:
        """Compile the model with the selected triplet loss."""

        loss_class = (
            TripletHardLoss if self.loss_type == "hard" else TripletSemiHardLoss
        )

        self.loss = loss_class(self.margin, self.soft, self.distance)
        self.loss_metric = tf.keras.metrics.Mean(name="loss")

        kwargs.setdefault("loss", self.loss)

        tf.keras.Model.compile(self, optimizer=optimizer, **kwargs)

    def step(self, x: tf.Tensor, y: tf.Tensor) -> None:
        """Run one optimization step."""

        with tf.GradientTape() as tape:
            embeddings = self(x, training=True)
            loss = self.loss(y, embeddings)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_metric.update_state(loss)

    def fit(self, batches=None, y=None, epochs: int = 100, **kwargs):
        """Train with native or legacy labeled datasets."""

        x = kwargs.pop("x", batches)

        if isinstance(x, tf.data.Dataset):
            kwargs.setdefault("shuffle", False)

        return tf.keras.Model.fit(self, x=x, y=y, epochs=epochs, **kwargs)

    def evaluate(self, batches=None, y=None, **kwargs):
        """Evaluate with native or legacy labeled datasets."""

        x = kwargs.pop("x", batches)

        return tf.keras.Model.evaluate(self, x=x, y=y, **kwargs)

    def predict(self, x1, x2=None, **kwargs):
        """Predict embeddings or compare two sample batches."""

        if x2 is not None and not isinstance(x2, numbers.Integral):
            return pair_distance(
                self.B(x1, training=False),
                self.B(x2, training=False),
                self.distance,
            )

        batch_size = kwargs.pop("batch_size", x2)

        return tf.keras.Model.predict(self, x1, batch_size=batch_size, **kwargs)

    def compare(self, left, right) -> tf.Tensor:
        """Return distances using the requested 2.x metric."""

        return pair_distance(
            self(left, training=False),
            self(right, training=False),
            self.distance_metric,
        )
