import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def _tensor_to_numpy(tensor):
    """Converts a tensor to a numpy array.

    Args:
        tensor (tf.Tensor): Tensor to be converted.

    Returns:
        A nunmpy array with the same values as the tensor.

    """

    # Checks if the inputted tensor is really a tensor
    if tf.is_tensor(tensor):
        # If yes, returns its numpy version
        return tensor.numpy()

    # If no, just returns it
    return tensor


def plot_embeddings(embeddings, labels, dims=(0, 1)):
    """Plots embedded data along their true labels.

    Args:
        embeddings (tf.Tensor, np.array): Tensor or array holding the embedded data.
        labels (tf.Tensor, np.array): Tensor or array holding the true labels.
        dims (tuple): Dimensions to be plotted.

    """

    # Makes sure that embeddings will be a numpy array
    embeddings = _tensor_to_numpy(embeddings)

    # Makes sure that labels will be a numpy array
    labels = _tensor_to_numpy(labels)

    # Creates figure and axis subplots
    fig, ax = plt.subplots(figsize=(7, 5))

    # Creates the axis labels strings
    x_label, y_label = f'$x_{dims[0]}$', f'$x_{dims[1]}$'

    # Defines some properties, such as axis labels
    ax.set(xlabel=r'{}'.format(x_label), ylabel=r'{}'.format(y_label))

    # Iterates through every possible labels
    for i in range(np.max(labels) + 1):
        # Gathers the indexes
        indexes = np.where(labels == i)[0]

        # Scatter plots the desired dimensions (2-D)
        plt.scatter(embeddings[indexes, dims[0]], embeddings[indexes, dims[1]], label=i)

    # Adds a legend to the plot
    plt.legend()

    # Shows the plot
    plt.show()
