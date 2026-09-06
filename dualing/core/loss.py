# Copyright (c) 2020-2026 Gustavo Rosa.
# Licensed under the Apache License, Version 2.0.

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
        return cls(**config)


@tf.keras.utils.register_keras_serializable(package="dualing")
class BinaryCrossEntropy(_SerializableLoss):
    """Compute binary cross-entropy averaged over the final axis."""

    def get_config(self) -> dict:
        return {}

    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute cross-entropy while retaining soft-label entropy.

        A vector produces a scalar. A batch-by-feature tensor produces one value per sample.
        Exact, correct hard-label predictions have zero loss.

        Args:
            y_true: Target probabilities with the same shape as y_pred.
            y_pred: Predicted probabilities.

        Returns:
            Loss tensor reduced over the final axis.

        """

        loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        hard_labels = tf.logical_or(tf.equal(y_true, 0), tf.equal(y_true, 1))
        exact_matches = tf.logical_and(hard_labels, tf.equal(y_true, y_pred))

        return tf.where(tf.reduce_all(exact_matches, axis=-1), tf.zeros_like(loss), loss)


@tf.keras.utils.register_keras_serializable(package="dualing")
class ContrastiveLoss(_SerializableLoss):
    """Compute a configurable contrastive loss for sample pairs."""

    def __init__(self, margin: float = 1.0) -> None:
        """Initialize the default contrastive margin.

        Args:
            margin: Distance below which dissimilar pairs incur a penalty.

        """

        self.margin = margin

    def get_config(self) -> dict:
        return {"margin": self.margin}

    def __call__(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        margin: float | None = None,
    ) -> tf.Tensor:
        """Compute one contrastive loss per pair without mutating the configured margin.

        Args:
            y_true: Pair labels with 1 for similar and 0 for dissimilar.
            y_pred: Predicted pair distances.
            margin: Call-specific margin, or None to use the constructor setting.

        Returns:
            Loss tensor with the same shape as y_pred.

        """

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
    """Compute scalar triplet loss with hard-negative mining."""

    _loss_function = staticmethod(triplet_hard_loss)

    def __init__(self, margin: float = 1.0, soft: bool = False, distance_metric: str = "L2") -> None:
        """Initialize hard-negative triplet mining.

        Calls accept one class label per embedding and a batch-by-feature embedding tensor.
        Per-call settings override constructor settings. A single-class batch retains the legacy margin result.

        Args:
            margin: Additive margin for the hinge loss.
            soft: Whether to use softplus instead of the margin-based hinge.
            distance_metric: L1, L2, squared-L2, or angular distance.

        """

        super().__init__(margin, soft, distance_metric)


@tf.keras.utils.register_keras_serializable(package="dualing")
class TripletSemiHardLoss(_TripletLoss):
    """Compute scalar triplet loss with semi-hard negative mining."""

    _loss_function = staticmethod(triplet_semihard_loss)

    def __init__(self, margin: float = 1.0, soft: bool = False, distance_metric: str = "L2") -> None:
        """Initialize semi-hard negative triplet mining.

        Inputs and per-call overrides follow TripletHardLoss. Each positive pair selects the nearest farther negative.
        When none is farther, it uses the farthest negative. A single-class batch retains the legacy margin result.

        Args:
            margin: Additive margin for the hinge loss.
            soft: Whether to use softplus instead of the margin-based hinge.
            distance_metric: L1, L2, squared-L2, or angular distance.

        """

        super().__init__(margin, soft, distance_metric)
