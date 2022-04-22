import tensorflow as tf

from dualing.models.base import rnn


def test_rnn():
    new_base = rnn.RNN()

    assert new_base.name == "rnn"


def test_rnn_call():
    new_base = rnn.RNN()

    x = tf.zeros((1, 10))

    y = new_base(x)

    assert y.shape == (1, 10, 1)
