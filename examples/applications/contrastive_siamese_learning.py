# Copyright (c) 2020-2026 Gustavo Rosa.
# Licensed under the Apache License, Version 2.0.

import tensorflow as tf

from dualing import CNN, ContrastiveSiamese, balanced_pair_dataset

(x, y), (x_val, y_val) = tf.keras.datasets.mnist.load_data()

train = balanced_pair_dataset(
    x,
    y,
    n_pairs=1000,
    batch_size=64,
    input_shape=(x.shape[0], 28, 28, 1),
)
val = balanced_pair_dataset(
    x_val,
    y_val,
    n_pairs=100,
    batch_size=64,
    input_shape=(x_val.shape[0], 28, 28, 1),
)

model = ContrastiveSiamese(
    CNN(blocks=3, kernel_size=5, embedding_dim=128),
    margin=1.0,
    distance_metric="L2",
    name="contrastive_siamese",
)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

model.fit(train, epochs=10, shuffle=False)

model.evaluate(val)
