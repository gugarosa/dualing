import tensorflow as tf
import tensorflow_addons as tfa


class BinaryCrossEntropy:
    """
    """

    def __call__(self, y_true, y_pred):
        """
        """

        return tf.keras.losses.binary_crossentropy(y_true, y_pred)


class ContrastiveLoss:
    """
    """

    def __call__(self, y_true, y_pred, margin=1.0):
        """
        """

        return tfa.losses.contrastive_loss(y_true, y_pred, margin)


class TripletHardLoss:
    """
    """

    def __call__(self, y_true, y_pred, margin=1.0, soft=False, distance='L2'):
        """
        """

        return tfa.losses.triplet_hard_loss(y_true, y_pred, margin, soft, distance)
