"""Losses.
"""

from typing import Optional

import tensorflow as tf
import tensorflow_addons as tfa


class BinaryCrossEntropy:
    """A BinaryCrossEntropy class defines the binary cross-entropy loss."""

    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Method that holds vital information whenever this class is called.

        Args:
            y_true: Tensor containing the true labels.
            y_pred: Tensor containing the predictions, e.g., similar or dissimilar.

        Returns:
            (tf.Tensor): Binary cross-entropy loss.

        """

        return tf.keras.losses.binary_crossentropy(y_true, y_pred)


class ContrastiveLoss:
    """A ContrastiveEntropy class defines the contrastive loss."""

    def __call__(
        self, y_true: tf.Tensor, y_pred: tf.Tensor, margin: Optional[float] = 1.0
    ) -> tf.Tensor:
        """Method that holds vital information whenever this class is called.

        Args:
            y_true: Tensor containing the true labels.
            y_pred: Tensor containing the predictions, e.g., distance.
            margin: Radius around the embedding space.

        Returns:
            (tf.Tensor): Contrastive loss.

        """

        return tfa.losses.contrastive_loss(y_true, y_pred, margin)


class TripletHardLoss:
    """A TripletHardLoss class defines the triplet loss with hard negative mining."""

    def __call__(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        margin: Optional[float] = 1.0,
        soft: Optional[bool] = False,
        distance_metric: Optional[str] = "L2",
    ) -> tf.Tensor:
        """Method that holds vital information whenever this class is called.

        Args:
            y_true: Tensor containing the true labels.
            y_pred: Tensor containing the predictions, e.g., embeddings.
            margin: Radius around the embedding space.
            soft: Whether network should use soft margin or not.
            distance_metric: Distance metric.

        Returns:
            (tf.Tensor): Triplet loss with hard negative mining.

        """

        return tfa.losses.triplet_hard_loss(
            y_true, y_pred, margin, soft, distance_metric
        )


class TripletSemiHardLoss:
    """A TripletSemiHardLoss class defines the triplet loss with semi-hard negative mining."""

    def __call__(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        margin: Optional[float] = 1.0,
        soft: Optional[bool] = None,
        distance_metric: Optional[str] = "L2",
    ) -> tf.Tensor:
        """Method that holds vital information whenever this class is called.

        Args:
            y_true: Tensor containing the true labels.
            y_pred: Tensor containing the predictions, e.g., embeddings.
            margin: Radius around the embedding space.
            soft: Only for retro-compatibility.
            distance_metric: Distance metric.

        Returns:
            (tf.Tensor): Triplet loss with semi-hard negative mining.

        """

        return tfa.losses.triplet_semihard_loss(y_true, y_pred, margin, distance_metric)
