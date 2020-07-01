import tensorflow as tf

from dualing.datasets import BalancedPairDataset
from dualing.models import ContrastiveSiamese
from dualing.models.base import MLP

# Loads the MNIST dataset
(x, y), (x_val, y_val) = tf.keras.datasets.mnist.load_data()

# Creates the training and validation datasets
train = BalancedPairDataset(x, y, n_pairs=1000, batch_size=64, shape=(x.shape[0], 784), normalize=[-1, 1])
val = BalancedPairDataset(x_val, y_val, n_pairs=100, batch_size=64, shape=(x_val.shape[0], 784), normalize=[-1, 1])

# Creates the base architecture
mlp = MLP(n_hidden=[512, 256, 128])

# Creates the contrastive siamese network
s = ContrastiveSiamese(mlp, margin=1.0, distance_metric='L2', name='contrastive_siamese')

# Compiles the network
s.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001))

# Fits the network
s.fit(train.batches, epochs=10)

# Evaluates the network
s.evaluate(val.batches)