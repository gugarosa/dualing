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

        self.conv = [
            Conv2D(
                32 * (2**i), init_kernel - 2 * i, activation="relu", padding="same"
            )
            for i in range(n_blocks)
        ]
        self.pool = [MaxPool2D() for _ in range(n_blocks)]
        self.flatten = Flatten()
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

        for (conv, pool) in zip(self.conv, self.pool):
            x = conv(x)
            x = pool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x
