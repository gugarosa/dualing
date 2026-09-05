"""Callable loss classes retained for the original public API."""

import tensorflow as tf

from dualing.losses import (
    contrastive_loss,
    triplet_hard_loss,
    triplet_semihard_loss,
)


class BinaryCrossEntropy:
    """Binary cross-entropy loss averaged over the final axis."""

    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        hard_labels = tf.logical_or(tf.equal(y_true, 0), tf.equal(y_true, 1))
        exact_matches = tf.logical_and(hard_labels, tf.equal(y_true, y_pred))

        return tf.where(
            tf.reduce_all(exact_matches, axis=-1), tf.zeros_like(loss), loss
        )


class ContrastiveLoss:
    """Contrastive loss with a configurable margin."""

    def __init__(self, margin: float = 1.0) -> None:
        self.margin = margin

    def __call__(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        margin: float | None = None,
    ) -> tf.Tensor:
        return contrastive_loss(
            y_true,
            y_pred,
            self.margin if margin is None else margin,
        )


class TripletHardLoss:
    """Triplet loss with hard-negative mining."""

    def __init__(
        self,
        margin: float = 1.0,
        soft: bool = False,
        distance_metric: str = "L2",
    ) -> None:
        self.margin = margin
        self.soft = soft
        self.distance_metric = distance_metric

    def __call__(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        margin: float | None = None,
        soft: bool | None = None,
        distance_metric: str | None = None,
    ) -> tf.Tensor:
        margin = self.margin if margin is None else margin
        loss = triplet_hard_loss(
            y_true,
            y_pred,
            margin,
            self.soft if soft is None else soft,
            self.distance_metric if distance_metric is None else distance_metric,
        )

        return tf.where(
            tf.size(tf.unique(tf.reshape(y_true, [-1])).y) > 1,
            loss,
            tf.cast(margin, y_pred.dtype),
        )


class TripletSemiHardLoss:
    """Triplet loss with semi-hard-negative mining."""

    def __init__(
        self,
        margin: float = 1.0,
        soft: bool = False,
        distance_metric: str = "L2",
    ) -> None:
        self.margin = margin
        self.soft = soft
        self.distance_metric = distance_metric

    def __call__(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        margin: float | None = None,
        soft: bool | None = None,
        distance_metric: str | None = None,
    ) -> tf.Tensor:
        margin = self.margin if margin is None else margin
        loss = triplet_semihard_loss(
            y_true,
            y_pred,
            margin,
            self.soft if soft is None else soft,
            self.distance_metric if distance_metric is None else distance_metric,
        )

        return tf.where(
            tf.size(tf.unique(tf.reshape(y_true, [-1])).y) > 1,
            loss,
            tf.cast(margin, y_pred.dtype),
        )
