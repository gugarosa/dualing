import tensorflow as tf

from dualing.embedders import CNN, GRU, LSTM, MLP, RNN


def test_dense_and_convolutional_embedders():
    assert MLP((16, 8))(tf.ones((2, 4))).shape == (2, 8)
    assert CNN(blocks=2, embedding_dim=8)(tf.ones((2, 28, 28, 1))).shape == (
        2,
        8,
    )


def test_recurrent_embedders():
    inputs = tf.zeros((2, 5), dtype=tf.int32)
    for model in (RNN(10), GRU(10), LSTM(10)):
        assert model(inputs).shape == (2, 5, 10)
