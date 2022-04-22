"""Losses.
"""

import tensorflow as tf
import tensorflow_addons as tfa


class BinaryCrossEntropy:
    """A BinaryCrossEntropy class defines the binary cross-entropy loss."""

    def __call__(self, y_true, y_pred):
        """Method that holds vital information whenever this class is called.

        Args:
            y_true (tf.Tensor): Tensor containing the true labels.
            y_pred (tf.Tensor): Tensor containing the predictions, e.g., similar or dissimilar.

        Returns:
            Binary cross-entropy loss.

        """

        return tf.keras.losses.binary_crossentropy(y_true, y_pred)


class ContrastiveLoss:
    """A ContrastiveEntropy class defines the contrastive loss."""

    def __call__(self, y_true, y_pred, margin=1.0):
        """Method that holds vital information whenever this class is called.

        Args:
            y_true (tf.Tensor): Tensor containing the true labels.
            y_pred (tf.Tensor): Tensor containing the predictions, e.g., distance.
            margin (float): Radius around the embedding space.

        Returns:
            Contrastive loss.

        """

        return tfa.losses.contrastive_loss(y_true, y_pred, margin)


class TripletHardLoss:
    """A TripletHardLoss class defines the triplet loss with hard negative mining."""

    def __call__(self, y_true, y_pred, margin=1.0, soft=False, distance_metric="L2"):
        """Method that holds vital information whenever this class is called.

        Args:
            y_true (tf.Tensor): Tensor containing the true labels.
            y_pred (tf.Tensor): Tensor containing the predictions, e.g., embeddings.
            margin (float): Radius around the embedding space.
            soft (bool): Whether network should use soft margin or not.
            distance_metric (str): Distance metric.

        Returns:
            Triplet loss with hard negative mining.

        """

        return tfa.losses.triplet_hard_loss(
            y_true, y_pred, margin, soft, distance_metric
        )


class TripletSemiHardLoss:
    """A TripletSemiHardLoss class defines the triplet loss with semi-hard negative mining."""

    def __call__(self, y_true, y_pred, margin=1.0, soft=None, distance_metric="L2"):
        """Method that holds vital information whenever this class is called.

        Args:
            y_true (tf.Tensor): Tensor containing the true labels.
            y_pred (tf.Tensor): Tensor containing the predictions, e.g., embeddings.
            margin (float): Radius around the embedding space.
            soft (None): Only for retro-compatibility.
            distance_metric (str): Distance metric.

        Returns:
            Triplet loss with semi-hard negative mining.

        """

        return tfa.losses.triplet_semihard_loss(y_true, y_pred, margin, distance_metric)
