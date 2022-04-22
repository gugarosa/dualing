import tensorflow as tf

from dualing.models.base import lstm


def test_lstm():
    new_base = lstm.LSTM()

    assert new_base.name == "lstm"


def test_lstm_call():
    new_base = lstm.LSTM()

    x = tf.zeros((1, 10))

    y = new_base(x)

    assert y.shape == (1, 10, 1)
