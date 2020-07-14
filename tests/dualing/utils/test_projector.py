import numpy as np
import tensorflow as tf

from dualing.utils import projector


def test_tensor_to_numpy():
    tensor = tf.zeros(1)

    array = projector._tensor_to_numpy(tensor)

    assert type(array) == np.ndarray

    tensor = np.zeros(1)

    array = projector._tensor_to_numpy(tensor)

    assert type(array) == np.ndarray


def test_plot_embeddings():
    embeddings = tf.ones((5, 5))
    labels = tf.zeros(5, dtype='int32')
    dims = (0, 1)

    projector.plot_embeddings(embeddings, labels, dims)
