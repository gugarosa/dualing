import matplotlib.pyplot as plt
import numpy as np


def plot_embeddings(embeddings, labels):
    """
    """
    embeddings = embeddings.numpy()

    plt.figure(figsize=(10,10))

    for i in range(10):
        inds = np.where(labels==i)[0]
        plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5, color=colors[i])

    plt.show()