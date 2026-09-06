"""Callable loss classes retained for the original public API."""

from collections.abc import Callable
from typing import Self

import tensorflow as tf

from dualing.losses import (
    contrastive_loss,
    triplet_hard_loss,
    triplet_semihard_loss,
)


class _SerializableLoss:
    @classmethod
    def from_config(cls, config: dict) -> Self:
        """Restore constructor settings without changing the loss reduction."""

        return cls(**config)


@tf.keras.utils.register_keras_serializable(package="dualing")
class BinaryCrossEntropy(_SerializableLoss):
    """Binary cross-entropy averaged over the final axis.

    Inputs must have matching shapes. A vector produces a scalar; a tensor
    of shape ``(batch, features)`` produces one value per sample. Exact,
    correct hard-label predictions have zero loss; soft-label entropy is
    retained.
    """

    def get_config(self) -> dict:
        """Return the configuration of this parameter-free loss."""

        return {}

    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        hard_labels = tf.logical_or(tf.equal(y_true, 0), tf.equal(y_true, 1))
        exact_matches = tf.logical_and(hard_labels, tf.equal(y_true, y_pred))

        return tf.where(
            tf.reduce_all(exact_matches, axis=-1), tf.zeros_like(loss), loss
        )


@tf.keras.utils.register_keras_serializable(package="dualing")
class ContrastiveLoss(_SerializableLoss):
    """Per-pair contrastive loss, with 1 for similar and 0 for dissimilar pairs.

    Args:
        margin: Distance below which dissimilar pairs incur a penalty.

    The optional call-time margin overrides, but does not mutate, the
    constructor setting saved in the configuration.
    """

    def __init__(self, margin: float = 1.0) -> None:
        self.margin = margin

    def get_config(self) -> dict:
        """Return the default margin."""

        return {"margin": self.margin}

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


class _TripletLoss(_SerializableLoss):
    _loss_function: Callable[[tf.Tensor, tf.Tensor, float, bool, str], tf.Tensor]

    def __init__(
        self,
        margin: float = 1.0,
        soft: bool = False,
        distance_metric: str = "L2",
    ) -> None:
        self.margin = margin
        self.soft = soft
        self.distance_metric = distance_metric

    def get_config(self) -> dict:
        """Return the mining loss settings."""

        return {
            "margin": self.margin,
            "soft": self.soft,
            "distance_metric": self.distance_metric,
        }

    def __call__(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        margin: float | None = None,
        soft: bool | None = None,
        distance_metric: str | None = None,
    ) -> tf.Tensor:
        margin = self.margin if margin is None else margin
        loss = self._loss_function(
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


@tf.keras.utils.register_keras_serializable(package="dualing")
class TripletHardLoss(_TripletLoss):
    """Scalar triplet loss using the hardest positive and negative per anchor.

    Args:
        margin: Additive margin for the hinge loss.
        soft: Use softplus instead of the margin-based hinge.
        distance_metric: One of L1, L2, squared-L2, or angular.

    Labels have shape ``(batch,)`` or ``(batch, 1)`` and embeddings have shape
    ``(batch, features)``. Call-time settings override constructor defaults.
    For original-API compatibility, a single-class batch returns the margin.
    """

    _loss_function = staticmethod(triplet_hard_loss)


@tf.keras.utils.register_keras_serializable(package="dualing")
class TripletSemiHardLoss(_TripletLoss):
    """Scalar triplet loss using semi-hard negative mining.

    Args:
        margin: Additive margin for the hinge loss.
        soft: Use softplus instead of the margin-based hinge.
        distance_metric: One of L1, L2, squared-L2, or angular.

    Inputs and per-call overrides follow ``TripletHardLoss``. Each positive
    pair uses the nearest farther negative, or the farthest negative when
    none is farther. A single-class batch retains the legacy margin result.
    """

    _loss_function = staticmethod(triplet_semihard_loss)
