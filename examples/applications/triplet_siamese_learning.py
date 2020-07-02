import tensorflow as tf

from dualing.datasets import BatchDataset
from dualing.models import TripletSiamese
from dualing.models.base import MLP

# Loads the MNIST dataset
(x, y), (x_val, y_val) = tf.keras.datasets.mnist.load_data()

# Creates the training and validation datasets
train = BatchDataset(x, y, batch_size=128, input_shape=(x.shape[0], 784), normalize=[-1, 1], shuffle=True)
val = BatchDataset(x_val, y_val, batch_size=128, input_shape=(x_val.shape[0], 784), normalize=[-1, 1], shuffle=True)

# Creates the base architecture
mlp = MLP(n_hidden=[512, 256, 128])

# Creates the triplet siamese network
s = TripletSiamese(mlp, loss='hard', margin=1.0, soft=False, distance_metric='L2', name='triplet_siamese')

# Compiles the network
s.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001))

# Fits the network
s.fit(train.batches, epochs=10)
