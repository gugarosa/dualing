import tensorflow as tf

from dualing.datasets import BatchDataset

(x, y), _ = tf.keras.datasets.mnist.load_data()

dataset = BatchDataset(
    x,
    y,
    batch_size=128,
    input_shape=(x.shape[0], 784),
    normalize=(-1, 1),
    shuffle=True,
    seed=0,
)
