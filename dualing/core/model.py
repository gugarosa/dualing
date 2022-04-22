"""Base architecture and Siamese Network.
"""

from typing import Optional, Union

import numpy as np
import tensorflow as tf

import dualing.utils.exception as e
from dualing.core.dataset import Dataset


class Base(tf.keras.Model):
    """A Base class is responsible for easily-implementing the
    base twin architecture of a Siamese Network.

    """

    def __init__(self, name: Optional[str] = "") -> None:
        """Initialization method.

        Args:
            name: Naming identifier.

        """

        super(Base, self).__init__(name=name)

    def call(self, x: tf.Tensor) -> None:
        """Method that holds vital information whenever this class is called.

        Note that you need to implement this method directly on its child. Essentially,
        each neural network has its own forward pass implementation.

        Args:
            x: Tensor containing the input sample.

        Raises:
            NotImplementedError.

        """

        raise NotImplementedError


class Siamese(tf.keras.Model):
    """An Siamese class is responsible for implementing the base of Siamese Neural Networks."""

    def __init__(self, base: Base, name: Optional[str] = "") -> None:
        """Initialization method.

        Args:
            base: Twin architecture.
            name: Naming identifier.

        """

        super(Siamese, self).__init__(name=name)

        # Defines the Siamese's base twin architecture
        self.B = base

    @property
    def B(self) -> Base:
        """Twin architecture."""

        return self._B

    @B.setter
    def B(self, B: Base) -> None:
        if not isinstance(B, Base):
            raise e.TypeError("`B` should be a child from Base class")

        self._B = B

    def compile(self, optimizer: tf.keras.optimizers) -> None:
        """Method that builds the network by attaching optimizer, loss and metrics.

        Note that you need to implement this method directly on its child. Essentially,
        each type of Siamese has its own set of loss and metrics.

        Args:
            optimizer: Optimization algorithm.

        Raises:
            NotImplementedError.

        """

        raise NotImplementedError

    def step(self, x: tf.Tensor, y: tf.Tensor) -> None:
        """Method that performs a single batch optimization step.

        Note that you need to implement this method directly on its child. Essentially,
        each type of Siamese has an unique step.

        Args:
            x: Tensor containing samples.
            y: Tensor containing labels.

        Raises:
            NotImplementedError.

        """

        raise NotImplementedError

    def fit(self, batches: Dataset, epochs: Optional[int] = 100) -> None:
        """Method that trains the model over training batches.

        Note that you need to implement this method directly on its child. Essentially,
        each type of Siamese may use a distinct type of dataset.

        Args:
            batches: Batches of tuples holding training samples and labels.
            epochs: Maximum number of epochs.

        Raises:
            NotImplementedError.

        """

        raise NotImplementedError

    def evaluate(self, batches: Dataset) -> None:
        """Method that evaluates the model over validation or testing batches.

        Note that you need to implement this method directly on its child. Essentially,
        each type of Siamese may use a distinct type of dataset.

        Args:
            batches: Batches of tuples holding validation / testing samples and labels.

        Raises:
            NotImplementedError.

        """

        raise NotImplementedError

    def predict(self, x: tf.Tensor) -> None:
        """Method that performs a forward pass over samples and returns the network's output.

        Note that you need to implement this method directly on its child. Essentially,
        each type of Siamese may predict in a different way.

        Args:
            x: Tensor containing samples.

        Raises:
            NotImplementedError.

        """

        raise NotImplementedError

    def extract_embeddings(self, x: Union[np.array, tf.Tensor]) -> tf.Tensor:
        """Method that extracts embeddings by performing a forward pass
        over the base architecture (embedder).

        Args:
            x: Array or tensor containing the inputs to be embedded.

        Returns:
            (tf.Tensor): A tensor containing the embedded inputs.

        """

        x = tf.convert_to_tensor(x)
        x = self.B(x)

        return x
