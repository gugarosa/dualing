import tensorflow as tf

from dualing.datasets import BatchDataset
from dualing.models import OnContrastiveSiamese
from dualing.models.base import MLP

# Loads the MNIST dataset
(x, y), (x_val, y_val) = tf.keras.datasets.mnist.load_data()

# Creates the training and validation datasets
train = BatchDataset(x, y, batch_size=64, shape=(x.shape[0], 784), normalize=[-1, 1])
val = BatchDataset(x_val, y_val, batch_size=64, shape=(x_val.shape[0], 784), normalize=[-1, 1])

# Creates the base architecture
mlp = MLP(n_hidden=[512, 256, 128])

# Creates the online-based contrastive siamese network
s = OnContrastiveSiamese(mlp, distance='euclidean', margin=1.0, name='on_contrastive_siamese')

# Compiles the network
s.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001))

# Fits the network
s.fit(train.batches, epochs=10)

# Evaluates the network
s.evaluate(val.batches)
