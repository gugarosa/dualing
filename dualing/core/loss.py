import tensorflow as tf


class BinaryCrossEntropy:
    """
    """

    def __call__(self, y, pred):
        """
        """

        return tf.keras.losses.binary_crossentropy(y, pred)

class ContrastiveLoss:

    def __call__(self, y, pred, margin=1):
        """
        """

        return y * tf.math.square(pred) + (1.0 - y) * tf.math.square(tf.math.maximum(margin - pred, 0.0))
