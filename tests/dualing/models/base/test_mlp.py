import tensorflow as tf

from dualing.models.base import mlp


def test_mlp():
    new_base = mlp.MLP()

    assert new_base.name == "mlp"


def test_mlp_call():
    new_base = mlp.MLP()

    x = tf.ones((1, 784))

    y = new_base(x)

    assert y.shape == (1, 128)
