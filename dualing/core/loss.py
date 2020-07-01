import tensorflow as tf
import tensorflow_addons as tfa


class BinaryCrossEntropy:
    """A BinaryCrossEntropy class defines the binary cross-entropy loss.

    """

    def __call__(self, y_true, y_pred):
        """Method that holds vital information whenever this class is called.

        Args:

        Returns:

        """

        return tf.keras.losses.binary_crossentropy(y_true, y_pred)


class ContrastiveLoss:
    """A ContrastiveEntropy class defines the contrastive loss.

    """

    def __call__(self, y_true, y_pred, margin=1.0):
        """Method that holds vital information whenever this class is called.

        Args:

        Returns:

        """

        return tfa.losses.contrastive_loss(y_true, y_pred, margin)


class TripletHardLoss:
    """A TripletHardLoss class defines the triplet loss with hard negative mining.

    """

    def __call__(self, y_true, y_pred, margin=1.0, soft=False, distance_metric='L2'):
        """Method that holds vital information whenever this class is called.

        Args:

        Returns:

        """

        return tfa.losses.triplet_hard_loss(y_true, y_pred, margin, soft, distance_metric)


class TripletSemiHardLoss:
    """A TripletSemiHardLoss class defines the triplet loss with semi-hard negative mining.

    """

    def __call__(self, y_true, y_pred, margin=1.0, distance_metric='L2'):
        """Method that holds vital information whenever this class is called.

        Args:

        Returns:

        """

        return tfa.losses.triplet_semihard_loss(y_true, y_pred, margin, distance)
