import tensorflow as tf

from dualing.datasets import BalancedPairDataset

# Loads the MNIST dataset
(x, y), (_, _) = tf.keras.datasets.mnist.load_data()

# Creates a BalancedPairDataset
dataset = BalancedPairDataset(x, y, n_pairs=100, batch_size=128,
                              input_shape=(x.shape[0], 784), normalize=[-1, 1], shuffle=True, seed=0)
