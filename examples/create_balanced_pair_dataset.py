import tensorflow as tf

from dualing.datasets import BalancedPairDataset

# Loads the MNIST dataset
(x, y), (_, _) = tf.keras.datasets.mnist.load_data()

# Creates a balanced-paired dataset
dataset = BalancedPairDataset(x, y, n_pairs=100, batch_size=128)
