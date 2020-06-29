import matplotlib.pyplot as plt
import tensorflow as tf

from dualing.core import Siamese
from dualing.datasets import RandomPairDataset
from dualing.models.mlp import MLP

# Loading the MNIST dataset
(x, y), (x_val, y_val) = tf.keras.datasets.mnist.load_data()

x = x / 255
x = x.astype('float32')

#
train = RandomPairDataset(x, y, batch_size=128)
val = RandomPairDataset(x_val, y_val, batch_size=128)

# Creating the base architecture
mlp = MLP()

# Creating the Siamese Network
s = Siamese(mlp, name='siamese')

# Compiling the Siamese Network
s.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001))

# Fitting the Siamese Network
s.fit(train.batches, epochs=10)


x1, x2, y = next(iter(val.batches))
print(s.predict(x1, x2))
plt.imshow(x1.numpy()[9])
# plt.imshow(x2.numpy()[9])
plt.show()
