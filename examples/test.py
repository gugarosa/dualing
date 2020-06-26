import tensorflow as tf

from dualing.core import Siamese
from dualing.models.mlp import MLP

# Loading the MNIST dataset
(x, y), (_, _) = tf.keras.datasets.mnist.load_data()

x = x.astype('float32')

#
batches = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(1000000).batch(128)

# Creating the base architecture
mlp = MLP()

# Creating the Siamese Network
s = Siamese(mlp, name='siamese_mlp')

# Compiling the Siamese Network
s.compile(optimizer=tf.optimizers.Adam(learning_rate=0.0001))

# Fitting the Siamese Network
s.fit(batches, epochs=10)
