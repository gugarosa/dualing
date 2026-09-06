# Copyright (c) 2020-2026 Gustavo Rosa.
# Licensed under the Apache License, Version 2.0.

import tensorflow as tf

from dualing.datasets import BatchDataset

(x, y), _ = tf.keras.datasets.mnist.load_data()

dataset = BatchDataset(
    x,
    y,
    batch_size=128,
    input_shape=(x.shape[0], 784),
    normalize=(-1, 1),
    shuffle=True,
    seed=0,
)
