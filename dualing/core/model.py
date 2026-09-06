"""Base embedding and Siamese model classes."""

from typing import Self

import numpy as np
import tensorflow as tf

from dualing.utils import exception


class Base(tf.keras.Model):
    """Keras model base for embedders implementing ``call(inputs)``.

    ``name``, ``trainable``, ``dtype``, and other model options are delegated
    to Keras. Subclasses with constructor parameters should implement
    ``get_config`` to support cloning and persistence.
    """

    def __init__(self, name: str = "", **kwargs) -> None:
        super().__init__(name=name, **kwargs)

    def call(self, x):
        raise NotImplementedError


class Siamese(tf.keras.Model):
    """Own one shared Keras embedder and its serialization configuration.

    Args:
        base: Keras model reused for both branches. Sequence outputs of shape
            ``(batch, time, features)`` are mean-pooled by ``embed``.
        name: Model name.
        **kwargs: Standard Keras model options, including trainable and dtype.
    """

    def __init__(self, base: tf.keras.Model, name: str = "", **kwargs) -> None:
        super().__init__(name=name, **kwargs)

        self.B = base

    def get_config(self) -> dict:
        """Serialize the shared embedder alongside native Keras model options."""

        return {
            **super().get_config(),
            "base": tf.keras.utils.serialize_keras_object(self.B),
        }

    @classmethod
    def from_config(cls, config: dict) -> Self:
        """Reconstruct a model with its nested, shared embedder."""

        config = dict(config)
        base = config.pop("base")

        if isinstance(base, dict):
            base = tf.keras.utils.deserialize_keras_object(base)

        return cls(base=base, **config)

    def compile_from_config(self, config: dict) -> None:
        """Restore compilation and optimizer slots for resumed training."""

        self.compile(**tf.keras.utils.deserialize_keras_object(config))

        if self.built and self.optimizer is not None:
            self.optimizer.build(self.trainable_variables)

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
