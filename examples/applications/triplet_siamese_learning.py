import tensorflow as tf

from dualing.models import TripletSiamese
from dualing.models.base import MLP

# Loads the MNIST dataset
(x, y), (x_val, y_val) = tf.keras.datasets.mnist.load_data()

x = x / 255
x = x.astype('float32')
x = tf.reshape(x, (x.shape[0], 784))
batches = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(1000000).batch(128)

# Creates the training and validation datasets
# train = BalancedPairDataset(x, y, n_pairs=1000, batch_size=64, shape=(x.shape[0], 784), normalize=True)
# val = BalancedPairDataset(x_val, y_val, n_pairs=100, batch_size=64, shape=(x_val.shape[0], 784), normalize=True)

# Creates the base architecture
mlp = MLP(n_hidden=[512, 256, 128])

# Creates the triplet siamese network
s = TripletSiamese(mlp, name='triplet_siamese')

# Compiles the network
s.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001))

# Fits the network
s.fit(batches, epochs=10)
