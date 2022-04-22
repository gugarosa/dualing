import tensorflow as tf

from dualing.datasets import RandomPairDataset

# Loads the MNIST dataset
(x, y), (_, _) = tf.keras.datasets.mnist.load_data()

# Creates a RandomPairDataset
dataset = RandomPairDataset(
    x, y, batch_size=128, input_shape=(x.shape[0], 784), normalize=(-1, 1), seed=0
)
