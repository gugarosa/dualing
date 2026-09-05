"""Base embedding and Siamese model classes."""

import numpy as np
import tensorflow as tf

from dualing.utils import exception


class Base(tf.keras.Model):
    """Base class for shared embedding models."""

    def __init__(self, name: str = "") -> None:
        super().__init__(name=name)

    def call(self, x):
        raise NotImplementedError


class Siamese(tf.keras.Model):
    """Base class for Siamese models."""

    def __init__(self, base: tf.keras.Model, name: str = "") -> None:
        super().__init__(name=name)

        self.B = base

    @property
    def B(self) -> tf.keras.Model:
        """Shared embedding model."""

        return self._B

    @B.setter
    def B(self, B: tf.keras.Model) -> None:
        if not isinstance(B, tf.keras.Model):
            raise exception.TypeError("`B` should be a Keras model")

        self._B = B

    @property
    def base(self) -> tf.keras.Model:
        """Shared embedding model."""

        return self.B

    @base.setter
    def base(self, base: tf.keras.Model) -> None:
        self.B = base

    @property
    def loss_metric(self) -> tf.keras.metrics.Metric:
        """Native Keras loss tracker, also used by the legacy step methods."""

        for metric in self.metrics:
            if metric.name == "loss":
                return metric

        raise AttributeError("compile the model before accessing loss_metric")

    def compile(self, optimizer="rmsprop", **kwargs) -> None:
        """Attach optimization configuration in a concrete model."""

        raise NotImplementedError

    def step(self, x, y) -> None:
        """Run one optimization step in a concrete model."""

        raise NotImplementedError

    def fit(self, batches, epochs: int = 100):
        """Train a concrete model."""

        raise NotImplementedError

    def evaluate(self, batches):
        """Evaluate a concrete model."""

        raise NotImplementedError

    def predict(self, x):
        """Run inference in a concrete model."""

        raise NotImplementedError

    def embed(self, inputs, training: bool = False) -> tf.Tensor:
        """Return fixed-size embeddings."""

        embeddings = self.B(inputs, training=training)

        if embeddings.shape.rank == 3:
            embeddings = tf.reduce_mean(embeddings, axis=1)

        return embeddings

    def extract_embeddings(self, x: np.ndarray | tf.Tensor) -> tf.Tensor:
        """Return the shared model output without reducing sequence dimensions."""

        return self.B(tf.convert_to_tensor(x), training=False)
