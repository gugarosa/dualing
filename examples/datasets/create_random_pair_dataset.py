import tensorflow as tf

from dualing.datasets import RandomPairDataset

(x, y), _ = tf.keras.datasets.mnist.load_data()

dataset = RandomPairDataset(
    x,
    y,
    batch_size=128,
    input_shape=(x.shape[0], 784),
    normalize=(-1, 1),
    seed=0,
)
