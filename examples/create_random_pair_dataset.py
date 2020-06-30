import tensorflow as tf

from dualing.datasets import RandomPairDataset
from dualing.models.mlp import MLP

# Loading the MNIST dataset
(x, y), (_, _) = tf.keras.datasets.mnist.load_data()

#
dataset = RandomPairDataset(x, y, batch_size=128)
