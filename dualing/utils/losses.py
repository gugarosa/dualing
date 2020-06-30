import tensorflow as tf


def binary_crossentropy(y_true, y_pred):
    """Calculates the binary cross-entropy loss.

    Args:
    
    Returns:

    """

    return tf.keras.losses.binary_crossentropy(y_true, y_pred)


def contrastive_loss(y_true, y_pred):
    """Calculates the contrastive loss.

    Args:
    
    Returns:
    
    """

    return tf.math.reduce_mean(y_true * tf.math.square(y_pred) + (1 - y_true) * tf.math.square(tf.math.maximum(1 - y_pred, 0)))
