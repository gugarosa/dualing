"""Binary-cross-entropy Siamese model."""

import numbers

import tensorflow as tf

from dualing.core import BinaryCrossEntropy, Siamese
from dualing.models._utils import pair_dataset
from dualing.utils import exception


class CrossEntropySiamese(Siamese):
    """Train a shared embedder as a binary pair classifier."""

    def __init__(
        self,
        base: tf.keras.Model,
        distance_metric: str = "concat",
        name: str = "",
        *,
        merge: str | None = None,
    ) -> None:
        super().__init__(base, name=name)

        if merge is not None:
            distance_metric = "diff" if merge == "difference" else merge

        self.distance = distance_metric
        self.o = tf.keras.layers.Dense(1, activation="sigmoid")
        self.output_layer = self.o
        self.acc = tf.keras.metrics.binary_accuracy
        self.acc_metric = tf.keras.metrics.Mean(name="acc")

    @property
    def distance(self) -> str:
        """Pair merge strategy."""

        return self._distance

    @distance.setter
    def distance(self, distance: str) -> None:
        if distance not in {"concat", "diff"}:
            raise exception.ValueError("`distance` should be `concat` or `diff`")

        self._distance = distance

    @property
    def merge(self) -> str:
        """Pair merge strategy."""

        return "difference" if self.distance == "diff" else self.distance

    @merge.setter
    def merge(self, merge: str) -> None:
        self.distance = "diff" if merge == "difference" else merge

    def call(self, inputs, training: bool = False) -> tf.Tensor:
        left, right = inputs

        left = self.embed(left, training)
        right = self.embed(right, training)

        features = (
            tf.concat((left, right), axis=-1)
            if self.distance == "concat"
            else tf.abs(left - right)
        )

        return tf.squeeze(self.o(features), axis=-1)

    @staticmethod
    def _pair_loss(labels, predictions) -> tf.Tensor:
        """Keep one loss per scalar pair prediction for sample weighting."""

        return BinaryCrossEntropy()(
            tf.reshape(labels, [-1, 1]), tf.reshape(predictions, [-1, 1])
        )

    def compile(self, optimizer="rmsprop", **kwargs) -> None:
        """Compile the model with binary cross-entropy by default."""

        self.loss = self._pair_loss
        self.acc_metric.reset_state()

        kwargs.setdefault("loss", self.loss)
        kwargs.setdefault("metrics", [tf.keras.metrics.BinaryAccuracy(name="accuracy")])

        tf.keras.Model.compile(self, optimizer=optimizer, **kwargs)

    def compute_metrics(self, x, y, y_pred, sample_weight=None):
        """Update legacy accuracy alongside the configured Keras metrics."""

        self.acc_metric.update_state(self.acc(y, y_pred))

        return super().compute_metrics(x, y, y_pred, sample_weight)

    def step(self, x1: tf.Tensor, x2: tf.Tensor, y: tf.Tensor) -> None:
        """Run one optimization step."""

        with tf.GradientTape() as tape:
            prediction = self((x1, x2), training=True)
            loss = self.loss(y, prediction)
            accuracy = self.acc(y, prediction)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_metric.update_state(loss)
        self.acc_metric.update_state(accuracy)

    def fit(self, batches=None, y=None, epochs: int = 100, **kwargs):
        """Train with native or legacy pair datasets."""

        x = kwargs.pop("x", batches)

        if isinstance(x, tf.data.Dataset):
            x = pair_dataset(x)
            kwargs.setdefault("shuffle", False)

        if "validation_data" in kwargs:
            kwargs["validation_data"] = pair_dataset(kwargs["validation_data"])

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
        """Return similarity scores for two batches of samples."""

        return self((left, right), training=False)
