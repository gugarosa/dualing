# Copyright (c) 2020-2026 Gustavo Rosa.
# Licensed under the Apache License, Version 2.0.

"""Embedding projection helpers."""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def _tensor_to_numpy(tensor: tf.Tensor) -> np.ndarray:
    return tensor.numpy() if tf.is_tensor(tensor) else tensor


def plot_embeddings(
    embeddings: tf.Tensor | np.ndarray,
    labels: tf.Tensor | np.ndarray,
    dims: tuple[int, int] = (0, 1),
) -> None:
    """Create and display a two-dimensional projection grouped by integer labels.

    Display is delegated to Matplotlib and the function does not return the created figure.

    Args:
        embeddings: Tensor or array of sample-by-feature embeddings.
        labels: Integer class labels for the samples.
        dims: Indices of the two embedding dimensions to display.

    """

    embeddings = _tensor_to_numpy(embeddings)
    labels = _tensor_to_numpy(labels)

    _, axis = plt.subplots(figsize=(13, 7))

    axis.set(xlabel=f"$x_{dims[0]}$", ylabel=f"$x_{dims[1]}$")

    for label in range(int(np.max(labels)) + 1):
        indexes = np.where(labels == label)[0]

        axis.scatter(
            embeddings[indexes, dims[0]],
            embeddings[indexes, dims[1]],
            alpha=0.75,
            label=label,
        )

    axis.legend()

    plt.show()
