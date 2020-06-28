import tensorflow as tf
from dualing.core import Siamese
from dualing.datasets.pair import PairDataset
from dualing.models.mlp import MLP

# Loading the MNIST dataset
(x, y), (_, _) = tf.keras.datasets.mnist.load_data()

x = x.astype('float32')

#
train = PairDataset(x, y, batch_size=128)

# Creating the base architecture
mlp = MLP()

# Creating the Siamese Network
s = Siamese(mlp, name='siamese')

# Compiling the Siamese Network
s.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01))

# Fitting the Siamese Network
s.fit(train.batches, epochs=10)
