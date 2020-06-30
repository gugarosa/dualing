import tensorflow as tf

from dualing.datasets import RandomPairDataset
from dualing.models import ContrastiveSiamese
from dualing.models.base import MLP

# Loads the MNIST dataset
(x, y), (x_val, y_val) = tf.keras.datasets.mnist.load_data()

# Creates the training and validation datasets
train = RandomPairDataset(x, y, batch_size=128, shape=(x.shape[0], 784), normalize=True)
val = RandomPairDataset(x_val, y_val, batch_size=128, shape=(x_val.shape[0], 784), normalize=True)

# Creates the base architecture
mlp = MLP(n_hidden=[256, 128])

# Creates the contrastive siamese network
s = ContrastiveSiamese(mlp, distance='euclidean', margin=1.0, name='contrastive_siamese')

# Compiles the network
s.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001))

# Fits the network
s.fit(train.batches, epochs=10)

# Evaluates the network
s.evaluate(val.batches)
