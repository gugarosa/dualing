"""Embedding projection helpers."""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def _tensor_to_numpy(tensor: tf.Tensor) -> np.ndarray:
    """Convert TensorFlow tensors to NumPy arrays."""

    return tensor.numpy() if tf.is_tensor(tensor) else tensor


def plot_embeddings(
    embeddings: tf.Tensor | np.ndarray,
    labels: tf.Tensor | np.ndarray,
    dims: tuple[int, int] = (0, 1),
) -> None:
    """Plot two embedding dimensions grouped by integer labels."""

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

    if "agg" not in plt.get_backend().lower():
        plt.show()
