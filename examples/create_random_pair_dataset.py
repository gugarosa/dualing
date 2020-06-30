import tensorflow as tf

from dualing.datasets import RandomPairDataset

# Loads the MNIST dataset
(x, y), (_, _) = tf.keras.datasets.mnist.load_data()

# Creates a random-paired dataset
dataset = RandomPairDataset(x, y, batch_size=128)
