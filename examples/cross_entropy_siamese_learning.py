import tensorflow as tf

from dualing.models import CrossEntropySiamese
from dualing.datasets import RandomPairDataset
from dualing.models.base import MLP

# Loads the MNIST dataset
(x, y), (_, _) = tf.keras.datasets.mnist.load_data()

x = x / 255
x = x.astype('float32')

# Creates the training dataset
train = RandomPairDataset(x, y, batch_size=128)

# Creates the Cross-Entropy-based Siamese Network
s = CrossEntropySiamese(MLP(), name='cross_entropy_siamese')

# Compiles the network
s.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001))

# Fits the network
s.fit(train.batches, epochs=10)
