import tensorflow as tf

import dualing.utils.projector as p
from dualing.datasets import BatchDataset
from dualing.models import TripletSiamese
from dualing.models.base import CNN

# Loads the MNIST dataset
(x, y), (x_val, y_val) = tf.keras.datasets.mnist.load_data()

# Creates the training and validation datasets
train = BatchDataset(
    x,
    y,
    batch_size=128,
    input_shape=(x.shape[0], 28, 28, 1),
    normalize=(0, 1),
    shuffle=True,
)
val = BatchDataset(
    x_val,
    y_val,
    batch_size=128,
    input_shape=(x_val.shape[0], 28, 28, 1),
    normalize=(0, 1),
    shuffle=True,
)

# Creates the base architecture
cnn = CNN(n_blocks=3, init_kernel=5, n_output=128, activation="linear")

# Creates the triplet siamese network
s = TripletSiamese(
    cnn,
    loss="hard",
    margin=0.5,
    soft=False,
    distance_metric="L2",
    name="triplet_siamese",
)

# Compiles, fits and evaluates the network
s.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001))
s.fit(train.batches, epochs=10)
s.evaluate(val.batches)

# Extract embeddings
embeddings = s.extract_embeddings(val.preprocess(x_val))

# Visualize embeddings
p.plot_embeddings(embeddings, y_val)
