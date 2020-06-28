import tensorflow as tf

from dualing.datasets.pair import PairDataset
from dualing.models.mlp import MLP

# Loading the MNIST dataset
(x, y), (_, _) = tf.keras.datasets.mnist.load_data()

#
train = PairDataset(x, y, batch_size=128)
