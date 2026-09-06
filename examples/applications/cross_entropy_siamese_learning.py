# Copyright (c) 2020-2026 Gustavo Rosa.
# Licensed under the Apache License, Version 2.0.

import tensorflow as tf

from dualing import MLP, CrossEntropySiamese, balanced_pair_dataset

(x, y), (x_val, y_val) = tf.keras.datasets.mnist.load_data()

train = balanced_pair_dataset(x, y, n_pairs=1000, batch_size=64, input_shape=(x.shape[0], 784))
val = balanced_pair_dataset(
    x_val,
    y_val,
    n_pairs=100,
    batch_size=64,
    input_shape=(x_val.shape[0], 784),
)

model = CrossEntropySiamese(
    MLP(hidden_units=(512, 256, 128)),
    merge="concat",
    name="cross_entropy_siamese",
)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

model.fit(train, epochs=10, shuffle=False)

model.evaluate(val)
