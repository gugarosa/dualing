# Copyright (c) 2020-2026 Gustavo Rosa.
# Licensed under the Apache License, Version 2.0.

"""Base embedding and Siamese model classes."""

from typing import Self

import numpy as np
import tensorflow as tf

from dualing.utils import exception


class Base(tf.keras.Model):
    """Provide a Keras base for shared embedding models."""

    def __init__(self, name: str = "", **kwargs) -> None:
        """Initialize the embedding model base.

        Args:
            name: Model name.
            **kwargs: Native Keras model options such as trainable and dtype.

        """

        super().__init__(name=name, **kwargs)

    def call(self, x):
        raise NotImplementedError("`call` must be implemented by a subclass.")


class Siamese(tf.keras.Model):
    """Own one shared Keras embedder for Siamese learning."""

    def __init__(self, base: tf.keras.Model, name: str = "", **kwargs) -> None:
        """Initialize the Siamese model base.

        Args:
            base: Keras embedding model reused by both branches.
            name: Model name.
            **kwargs: Native Keras model options such as trainable and dtype.

        """

        super().__init__(name=name, **kwargs)

        self.B = base

    def get_config(self) -> dict:
        return {
            **super().get_config(),
            "base": tf.keras.utils.serialize_keras_object(self.B),
        }

    @classmethod
    def from_config(cls, config: dict) -> Self:
        config = dict(config)
        base = config.pop("base")

        if isinstance(base, dict):
            base = tf.keras.utils.deserialize_keras_object(base)

        return cls(base=base, **config)

    def compile_from_config(self, config: dict) -> None:
        self.compile(**tf.keras.utils.deserialize_keras_object(config))

        # Keras restores saved optimizer values after constructing their slots
        if self.built and self.optimizer is not None:
            self.optimizer.build(self.trainable_variables)

    @property
    def B(self) -> tf.keras.Model:
        """Shared embedding model."""

        return self._B

    @B.setter
    def B(self, B: tf.keras.Model) -> None:
        if not isinstance(B, tf.keras.Model):
            raise exception.TypeError("`B` must be a Keras model.")

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

        raise AttributeError("`loss_metric` is unavailable until the model is compiled.")

    def compile(self, optimizer="rmsprop", **kwargs) -> None:
        """Attach optimization configuration in a concrete model.

        Args:
            optimizer: Keras optimizer instance or identifier.
            **kwargs: Native Keras compilation options.

        Raises:
            NotImplementedError: The subclass has not implemented compilation.

        """

        raise NotImplementedError("`compile` must be implemented by a subclass.")

    def step(self, x, y) -> None:
        """Run one optimization step in a concrete model.

        Args:
            x: Input samples.
            y: Target labels.

        Raises:
            NotImplementedError: The subclass has not implemented optimization.

        """

        raise NotImplementedError("`step` must be implemented by a subclass.")

    def fit(self, batches, epochs: int = 100):
        """Train a concrete model.

        Args:
            batches: Training dataset.
            epochs: Number of training epochs.

        Raises:
            NotImplementedError: The subclass has not implemented training.

        """

        raise NotImplementedError("`fit` must be implemented by a subclass.")

    def evaluate(self, batches):
        """Evaluate a concrete model.

        Args:
            batches: Evaluation dataset.

        Raises:
            NotImplementedError: The subclass has not implemented evaluation.

        """

        raise NotImplementedError("`evaluate` must be implemented by a subclass.")

    def predict(self, x):
        """Run inference in a concrete model.

        Args:
            x: Input samples.

        Raises:
            NotImplementedError: The subclass has not implemented prediction.

        """

        raise NotImplementedError("`predict` must be implemented by a subclass.")

    def embed(self, inputs, training: bool = False) -> tf.Tensor:
        """Return shared embeddings with rank-three sequences mean-pooled over time.

        Args:
            inputs: Samples accepted by the shared embedding model.
            training: Whether the shared model runs in training mode.

        Returns:
            Embedding tensor with any rank-three time dimension reduced.

        """

        embeddings = self.B(inputs, training=training)

        if embeddings.shape.rank == 3:
            embeddings = tf.reduce_mean(embeddings, axis=1)

        return embeddings

    def extract_embeddings(self, x: np.ndarray | tf.Tensor) -> tf.Tensor:
        """Return the shared model output without reducing sequence dimensions.

        Args:
            x: Input samples converted to a tensor for inference.

        Returns:
            Raw embedding tensor with sequence dimensions retained.

        """

        return self.B(tf.convert_to_tensor(x), training=False)
