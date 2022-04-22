"""Convolutional Neural Network.
"""

from typing import Optional

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D

from dualing.core import Base
from dualing.utils import logging

logger = logging.get_logger(__name__)


class CNN(Base):
    """A CNN class stands for a standard Convolutional Neural Network implementation."""

    def __init__(
        self,
        n_blocks: Optional[int] = 3,
        init_kernel: Optional[int] = 5,
        n_output: Optional[int] = 128,
        activation: Optional[str] = "sigmoid",
    ):
        """Initialization method.

        Args:
            n_blocks: Number of convolutional/pooling blocks.
            init_kernel: Size of initial kernel.
            n_outputs: Number of output units.
            activation: Output activation function.

        """

        logger.info("Overriding class: Base -> CNN.")

        super(CNN, self).__init__(name="cnn")

        # Asserting that it will be possible to create the convolutional layers
        assert init_kernel - 2 * (n_blocks - 1) >= 1

        # Convolutional layers
        self.conv = [
            Conv2D(
                32 * (2**i), init_kernel - 2 * i, activation="relu", padding="same"
            )
            for i in range(n_blocks)
        ]

        # Pooling layers
        self.pool = [MaxPool2D() for _ in range(n_blocks)]

        # Flatenning layer
        self.flatten = Flatten()

        # Final fully-connected layer
        self.fc = Dense(n_output, activation=activation)

        logger.info("Class overrided.")
        logger.debug(
            "Blocks: %d | Initial Kernel: %d | Output (%s): %d.",
            n_blocks,
            init_kernel,
            activation,
            n_output,
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Method that holds vital information whenever this class is called.

        Args:
            x: Tensor containing the input sample.

        Returns:
            (tf.Tensor): The layer's outputs.

        """

        # Iterate through convolutional and poooling layers
        for (conv, pool) in zip(self.conv, self.pool):
            # Pass through convolutional layer
            x = conv(x)

            # Pass through pooling layer
            x = pool(x)

        # Flattens the outputs
        x = self.flatten(x)

        # Pass through the fully-connected layer
        x = self.fc(x)

        return x
