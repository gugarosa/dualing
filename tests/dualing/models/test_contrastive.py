import tensorflow as tf

from dualing.datasets import pair
from dualing.models import contrastive
from dualing.models.base import mlp


def test_contrastive_margin():
    new_base = mlp.Base()
    new_siamese = contrastive.ContrastiveSiamese(new_base)

    assert new_siamese.margin == 1.0


def test_contrastive_margin_setter():
    new_base = mlp.Base()
    new_siamese = contrastive.ContrastiveSiamese(new_base)

    try:
        new_siamese.margin = -1
    except:
        new_siamese.margin = 1.0

    assert new_siamese.margin == 1.0


def test_contrastive_distance():
    new_base = mlp.Base()
    new_siamese = contrastive.ContrastiveSiamese(new_base)

    assert new_siamese.distance == "L2"


def test_contrastive_distance_setter():
    new_base = mlp.Base()
    new_siamese = contrastive.ContrastiveSiamese(new_base)

    try:
        new_siamese.distance = "a"
    except:
        new_siamese.distance == "L2"

    assert new_siamese.distance == "L2"


def test_contrastive_step():
    (x, y), (_, _) = tf.keras.datasets.mnist.load_data()

    new_base = mlp.MLP()
    new_siamese = contrastive.ContrastiveSiamese(new_base)
    new_siamese.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001))

    x1 = tf.ones((1, 784))
    x2 = tf.ones((1, 784))
    y = tf.zeros(1)

    new_siamese.step(x1, x2, y)


def test_contrastive_fit():
    (x, y), (_, _) = tf.keras.datasets.mnist.load_data()
    train = pair.BalancedPairDataset(x, y, n_pairs=10, input_shape=(x.shape[0], 784))

    new_base = mlp.MLP()
    new_siamese = contrastive.ContrastiveSiamese(new_base)
    new_siamese.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001))

    new_siamese.fit(train.batches, epochs=1)


def test_contrastive_evaluate():
    (x, y), (_, _) = tf.keras.datasets.mnist.load_data()
    train = pair.BalancedPairDataset(x, y, n_pairs=10, input_shape=(x.shape[0], 784))

    new_base = mlp.MLP()
    new_siamese = contrastive.ContrastiveSiamese(new_base)
    new_siamese.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001))

    new_siamese.fit(train.batches, epochs=1)
    new_siamese.evaluate(train.batches)


def test_contrastive_predict():
    (x, y), (_, _) = tf.keras.datasets.mnist.load_data()
    train = pair.BalancedPairDataset(x, y, n_pairs=10, input_shape=(x.shape[0], 784))

    new_base = mlp.MLP()
    new_siamese = contrastive.ContrastiveSiamese(new_base)
    new_siamese.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001))

    new_siamese.fit(train.batches, epochs=1)

    x1 = tf.ones((1, 784))
    x2 = tf.ones((1, 784))

    new_siamese.distance = "L1"
    new_siamese.predict(x1, x2)

    new_siamese.distance = "L2"
    new_siamese.predict(x1, x2)

    new_siamese.distance = "angular"
    new_siamese.predict(x1, x2)
