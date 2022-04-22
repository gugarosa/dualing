import tensorflow as tf

from dualing.models.base import cnn


def test_cnn():
    new_base = cnn.CNN()

    assert new_base.name == "cnn"


def test_cnn_call():
    new_base = cnn.CNN()

    x = tf.ones((1, 1, 28, 28))

    y = new_base(x)

    assert y.shape == (1, 128)
