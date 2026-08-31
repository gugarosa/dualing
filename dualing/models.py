"""Siamese models using native Keras training loops."""

import tensorflow as tf

from dualing.losses import (
    contrastive_loss,
    pair_distance,
    triplet_hard_loss,
    triplet_semihard_loss,
)


class Siamese(tf.keras.Model):
    """Base model that owns a shared embedder."""

    def __init__(self, base: tf.keras.Model, **kwargs) -> None:
        if not isinstance(base, tf.keras.Model):
            raise TypeError("base must be a Keras model")
        super().__init__(**kwargs)
        self.base = base

    def embed(self, inputs, training: bool = False) -> tf.Tensor:
        """Return fixed-size embeddings from the shared base model."""

        embeddings = self.base(inputs, training=training)
        if embeddings.shape.rank == 3:
            embeddings = tf.reduce_mean(embeddings, axis=1)
        return embeddings


class ContrastiveSiamese(Siamese):
    """A Siamese model trained with contrastive loss."""

    def __init__(
        self,
        base: tf.keras.Model,
        margin: float = 1.0,
        distance_metric: str = "L2",
        **kwargs,
    ) -> None:
        if margin <= 0:
            raise ValueError("margin must be greater than zero")
        if distance_metric not in {"L1", "L2", "squared-L2", "angular"}:
            raise ValueError("distance_metric must be L1, L2, squared-L2, or angular")
        super().__init__(base, **kwargs)
        self.margin = float(margin)
        self.distance_metric = distance_metric

    def call(self, inputs, training: bool = False) -> tf.Tensor:
        left, right = inputs
        return pair_distance(
            self.embed(left, training),
            self.embed(right, training),
            self.distance_metric,
        )

    def compile(self, optimizer="rmsprop", **kwargs) -> None:
        """Compile with contrastive loss unless another loss is provided."""

        kwargs.setdefault("loss", self._loss)
        super().compile(optimizer=optimizer, **kwargs)

    def _loss(self, labels: tf.Tensor, distances: tf.Tensor) -> tf.Tensor:
        return contrastive_loss(labels, distances, self.margin)


class CrossEntropySiamese(Siamese):
    """A Siamese classifier trained with binary cross-entropy."""

    def __init__(
        self,
        base: tf.keras.Model,
        merge: str = "concat",
        **kwargs,
    ) -> None:
        if merge not in {"concat", "difference"}:
            raise ValueError("merge must be concat or difference")
        super().__init__(base, **kwargs)
        self.merge = merge
        self.output_layer = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs, training: bool = False) -> tf.Tensor:
        left, right = inputs
        left = self.embed(left, training)
        right = self.embed(right, training)
        features = (
            tf.concat((left, right), axis=-1)
            if self.merge == "concat"
            else tf.abs(left - right)
        )
        return tf.squeeze(self.output_layer(features), axis=-1)

    def compile(self, optimizer="rmsprop", **kwargs) -> None:
        """Compile with binary cross-entropy unless another loss is provided."""

        kwargs.setdefault("loss", tf.keras.losses.BinaryCrossentropy())
        kwargs.setdefault("metrics", [tf.keras.metrics.BinaryAccuracy(name="accuracy")])
        super().compile(optimizer=optimizer, **kwargs)


class TripletSiamese(Siamese):
    """A Siamese embedder trained with hard or semi-hard triplet loss."""

    def __init__(
        self,
        base: tf.keras.Model,
        mining: str = "hard",
        margin: float = 1.0,
        soft: bool = False,
        distance_metric: str = "L2",
        **kwargs,
    ) -> None:
        if mining not in {"hard", "semi-hard"}:
            raise ValueError("mining must be hard or semi-hard")
        if margin <= 0:
            raise ValueError("margin must be greater than zero")
        if distance_metric not in {"L1", "L2", "squared-L2", "angular"}:
            raise ValueError("distance_metric must be L1, L2, squared-L2, or angular")
        super().__init__(base, **kwargs)
        self.mining = mining
        self.margin = float(margin)
        self.soft = bool(soft)
        self.distance_metric = distance_metric

    def call(self, inputs, training: bool = False) -> tf.Tensor:
        return tf.math.l2_normalize(self.embed(inputs, training), axis=-1)

    def compile(self, optimizer="rmsprop", **kwargs) -> None:
        """Compile with the selected triplet loss unless another loss is provided."""

        kwargs.setdefault("loss", self._loss)
        super().compile(optimizer=optimizer, **kwargs)

    def _loss(self, labels: tf.Tensor, embeddings: tf.Tensor) -> tf.Tensor:
        loss = triplet_hard_loss if self.mining == "hard" else triplet_semihard_loss
        return loss(
            labels,
            embeddings,
            margin=self.margin,
            soft=self.soft,
            metric=self.distance_metric,
        )

    def compare(self, left, right) -> tf.Tensor:
        """Return distances between two batches of samples."""

        return pair_distance(
            self(left, training=False),
            self(right, training=False),
            self.distance_metric,
        )
