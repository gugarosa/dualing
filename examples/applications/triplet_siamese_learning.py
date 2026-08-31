import tensorflow as tf

from dualing import CNN, TripletSiamese, batch_dataset

(x, y), (x_val, y_val) = tf.keras.datasets.mnist.load_data()

train = batch_dataset(
    x,
    y,
    batch_size=128,
    input_shape=(x.shape[0], 28, 28, 1),
)
val = batch_dataset(
    x_val,
    y_val,
    batch_size=128,
    input_shape=(x_val.shape[0], 28, 28, 1),
)

model = TripletSiamese(
    CNN(blocks=3, kernel_size=5, embedding_dim=128, activation="linear"),
    mining="hard",
    margin=0.5,
    distance_metric="L2",
    name="triplet_siamese",
)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

model.fit(train, epochs=10, shuffle=False)

model.evaluate(val)
