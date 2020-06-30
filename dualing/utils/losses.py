import tensorflow as tf


def binary_crossentropy(y, pred):
    """Calculates the binary cross-entropy loss.

    Args:
        y (tf.Tensor): Tensor of true labels.
        pred (tf.Tensor): Tensor of predicted labels.

    Returns:
        Binary cross-entropy loss.

    """

    return tf.keras.losses.binary_crossentropy(y, pred)


def contrastive_loss(y, pred, margin=1):
    """Calculates the contrastive loss.

    Args:
        y (tf.Tensor): Tensor of true labels.
        pred (tf.Tensor): Tensor of predicted distances.
        margin (float): Radius around the space to be computed.

    Returns:
        Contrastive loss.

    """

    return y * tf.math.square(pred) + (1.0 - y) * tf.math.square(tf.math.maximum(margin - pred, 0.0))
