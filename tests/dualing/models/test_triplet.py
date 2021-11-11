import tensorflow as tf

from dualing.datasets import batch
from dualing.models import triplet
from dualing.models.base import mlp


def test_triplet():
    new_base = mlp.Base()
    new_siamese = triplet.TripletSiamese(new_base)


def test_triplet_loss_type():
    new_base = mlp.Base()
    new_siamese = triplet.TripletSiamese(new_base)

    assert new_siamese.loss_type == 'hard'


def test_triplet_loss_type_setter():
    new_base = mlp.Base()
    new_siamese = triplet.TripletSiamese(new_base)

    try:
        new_siamese.loss_type = 'a'
    except:
        new_siamese.loss_type = 'hard'

    assert new_siamese.loss_type == 'hard'


def test_triplet_soft():
    new_base = mlp.Base()
    new_siamese = triplet.TripletSiamese(new_base)

    assert new_siamese.soft == False


def test_triplet_soft_setter():
    new_base = mlp.Base()
    new_siamese = triplet.TripletSiamese(new_base)

    try:
        new_siamese.soft = -1
    except:
        new_siamese.soft = False

    assert new_siamese.soft == False


def test_triplet_margin():
    new_base = mlp.Base()
    new_siamese = triplet.TripletSiamese(new_base)

    assert new_siamese.margin == 1.0


def test_triplet_margin_setter():
    new_base = mlp.Base()
    new_siamese = triplet.TripletSiamese(new_base)

    try:
        new_siamese.margin = -1
    except:
        new_siamese.margin = 1.0

    assert new_siamese.margin == 1.0


def test_triplet_distance():
    new_base = mlp.Base()
    new_siamese = triplet.TripletSiamese(new_base)

    assert new_siamese.distance == 'squared-L2'


def test_triplet_distance_setter():
    new_base = mlp.Base()
    new_siamese = triplet.TripletSiamese(new_base)

    try:
        new_siamese.distance = 'a'
    except:
        new_siamese.distance == 'L2'

    assert new_siamese.distance == 'squared-L2'


def test_triplet_step():
    (x, y), (_, _) = tf.keras.datasets.mnist.load_data()
    train = batch.BatchDataset(x, y, input_shape=(x.shape[0], 784))

    new_base = mlp.MLP()
    new_siamese = triplet.TripletSiamese(new_base)
    new_siamese.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001))

    x = tf.ones((10, 784))
    y = tf.zeros(10)

    new_siamese.step(x, y)


def test_triplet_fit():
    (x, y), (_, _) = tf.keras.datasets.mnist.load_data()
    train = batch.BatchDataset(x[:10], y[:10], input_shape=(10, 784))

    new_base = mlp.MLP()

    new_siamese = triplet.TripletSiamese(new_base, loss='hard')
    new_siamese.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001))

    new_siamese = triplet.TripletSiamese(new_base, loss='semi-hard')
    new_siamese.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001))

    new_siamese.fit(train.batches, epochs=1)


def test_triplet_evaluate():
    (x, y), (_, _) = tf.keras.datasets.mnist.load_data()
    train = batch.BatchDataset(x[:10], y[:10], input_shape=(10, 784))

    new_base = mlp.MLP()
    new_siamese = triplet.TripletSiamese(new_base)
    new_siamese.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001))

    new_siamese.fit(train.batches, epochs=1)
    new_siamese.evaluate(train.batches)


def test_triplet_predict():
    (x, y), (_, _) = tf.keras.datasets.mnist.load_data()
    train = batch.BatchDataset(x[:10], y[:10], input_shape=(10, 784))

    new_base = mlp.MLP()
    new_siamese = triplet.TripletSiamese(new_base)
    new_siamese.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001))

    new_siamese.fit(train.batches, epochs=1)

    x1 = tf.ones((1, 784))
    x2 = tf.ones((1, 784))

    new_siamese.distance = 'L1'
    new_siamese.predict(x1, x2)

    new_siamese.distance = 'L2'
    new_siamese.predict(x1, x2)

    new_siamese.distance = 'angular'
    new_siamese.predict(x1, x2)
