"""Helpers and methods to project embedded data.
"""

from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def _tensor_to_numpy(tensor: tf.Tensor) -> np.ndarray:
    """Converts a tensor to a numpy array.

    Args:
        tensor: Tensor to be converted.

    Returns:
        (np.ndarray): Array with the same values as the tensor.

    """

    if tf.is_tensor(tensor):
        return tensor.numpy()

    return tensor


def plot_embeddings(
    embeddings: Union[tf.Tensor, np.array],
    labels: Union[tf.Tensor, np.array],
    dims: Optional[Tuple[int, ...]] = (0, 1),
) -> None:
    """Plots embedded data along their true labels.

    Args:
        embeddings: Tensor or array holding the embedded data.
        labels: Tensor or array holding the true labels.
        dims: Dimensions to be plotted.

    """

    # Makes sure that embeddings and labels are numpy arrays
    embeddings = _tensor_to_numpy(embeddings)
    labels = _tensor_to_numpy(labels)

    # Creates figure and axis subplots
    _, axis = plt.subplots(figsize=(13, 7))

    # Creates the axis labels strings
    x_label, y_label = f"$x_{dims[0]}$", f"$x_{dims[1]}$"

    # Defines some properties, such as axis labels
    axis.set(xlabel=r"{}".format(x_label), ylabel=r"{}".format(y_label))

    # Iterates through every possible labels
    for i in range(np.max(labels) + 1):
        # Gathers the indexes
        indexes = np.where(labels == i)[0]

        # Scatter plot the desired dimensions (2-D)
        plt.scatter(
            embeddings[indexes, dims[0]],
            embeddings[indexes, dims[1]],
            alpha=0.75,
            label=i,
        )

    plt.legend()
    plt.show()
