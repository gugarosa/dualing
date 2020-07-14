import tensorflow as tf

from dualing.core import loss


def test_binary_cross_entropy():
    L = loss.BinaryCrossEntropy()

    y_true = tf.zeros(1)
    y_pred = tf.zeros(1)

    output = L(y_true, y_pred)

    assert output.numpy() == 0.0


def test_contrastive_loss():
    L = loss.ContrastiveLoss()

    y_true = tf.zeros(1)
    y_pred = tf.zeros(1)

    output = L(y_true, y_pred)

    assert output.numpy()[0] == 1.0


def test_triplet_hard_loss():
    L = loss.TripletHardLoss()

    y_true = tf.zeros((1, 1))
    y_pred = tf.zeros((1, 1))

    output = L(y_true, y_pred)

    assert output.numpy() == 1.0


def test_triplet_semi_hard_loss():
    L = loss.TripletSemiHardLoss()

    y_true = tf.zeros((10, 1))
    y_pred = tf.fill((10, 1), 1.5)

    output = L(y_true, y_pred)

    assert output.numpy() == 1.0
