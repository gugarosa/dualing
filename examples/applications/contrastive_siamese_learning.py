import tensorflow as tf

import dualing.utils.projector as p
from dualing.datasets import BalancedPairDataset
from dualing.models import ContrastiveSiamese
from dualing.models.base import CNN

# Loads the MNIST dataset
(x, y), (x_val, y_val) = tf.keras.datasets.mnist.load_data()

# Creates the training and validation datasets
train = BalancedPairDataset(x, y, n_pairs=1000, batch_size=64, input_shape=(x.shape[0], 28, 28, 1), normalize=(0, 1))
val = BalancedPairDataset(x_val, y_val, n_pairs=100, batch_size=64, input_shape=(x_val.shape[0], 28, 28, 1), normalize=(0, 1))

# Creates the base architecture
cnn = CNN(n_blocks=3, init_kernel=5, n_output=128)

# Creates the contrastive siamese network
s = ContrastiveSiamese(cnn, margin=1.0, distance_metric='L2', name='contrastive_siamese')

# Compiles the network
s.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001))

# Fits the network
s.fit(train.batches, epochs=10)

# Evaluates the network
s.evaluate(val.batches)

# Extract embeddings
embeddings = s.extract_embeddings(val.preprocess(x_val))

# Visualize embeddings
p.plot_embeddings(embeddings, y_val)
