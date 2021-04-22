import tensorflow as tf

from dualing.datasets import pair
from dualing.models import cross_entropy
from dualing.models.base import mlp


def test_cross_entropy():
    new_base = mlp.Base()

    new_siamese = cross_entropy.CrossEntropySiamese(new_base)


def test_cross_entropy_distance():
    new_base = mlp.Base()
    new_siamese = cross_entropy.CrossEntropySiamese(new_base)

    assert new_siamese.distance == 'concat'


def test_cross_entropy_distance_setter():
    new_base = mlp.Base()
    new_siamese = cross_entropy.CrossEntropySiamese(new_base)

    try:
        new_siamese.distance = 'a'
    except:
        new_siamese.distance == 'concat'

    assert new_siamese.distance == 'concat'


def test_cross_entropy_step():
    (x, y), (_, _) = tf.keras.datasets.mnist.load_data()
    train = pair.BalancedPairDataset(
        x, y, n_pairs=10, input_shape=(x.shape[0], 784))

    new_base = mlp.MLP()
    new_siamese = cross_entropy.CrossEntropySiamese(new_base)
    new_siamese.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001))

    x1 = tf.ones((1, 784))
    x2 = tf.ones((1, 784))
    y = tf.zeros(1)

    new_siamese.step(x1, x2, y)


def test_cross_entropy_fit():
    (x, y), (_, _) = tf.keras.datasets.mnist.load_data()
    train = pair.BalancedPairDataset(
        x, y, n_pairs=10, input_shape=(x.shape[0], 784))

    new_base = mlp.MLP()
    new_siamese = cross_entropy.CrossEntropySiamese(new_base)
    new_siamese.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001))

    new_siamese.fit(train.batches, epochs=1)


def test_cross_entropy_evaluate():
    (x, y), (_, _) = tf.keras.datasets.mnist.load_data()
    train = pair.BalancedPairDataset(
        x, y, n_pairs=10, input_shape=(x.shape[0], 784))

    new_base = mlp.MLP()
    new_siamese = cross_entropy.CrossEntropySiamese(new_base)
    new_siamese.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001))

    new_siamese.fit(train.batches, epochs=1)
    new_siamese.evaluate(train.batches)


def test_cross_entropy_predict():
    (x, y), (_, _) = tf.keras.datasets.mnist.load_data()
    train = pair.BalancedPairDataset(
        x, y, n_pairs=10, input_shape=(x.shape[0], 784))

    new_base = mlp.MLP()
    new_siamese = cross_entropy.CrossEntropySiamese(new_base)
    new_siamese.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001))

    new_siamese.fit(train.batches, epochs=1)

    x1 = tf.ones((1, 784))
    x2 = tf.ones((1, 784))

    new_siamese.predict(x1, x2)
