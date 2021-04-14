import tensorflow as tf

from dualing.models.base import gru


def test_gru():
    new_base = gru.GRU()

    assert new_base.name == 'gru'


def test_gru_call():
    new_base = gru.GRU()

    x = tf.zeros((1, 10))

    y = new_base(x)

    assert y.shape == (1, 10, 1)
