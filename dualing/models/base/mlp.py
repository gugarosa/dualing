"""Multi-Layer Perceptron.
"""

from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import Dense

from dualing.core import Base
from dualing.utils import logging

logger = logging.get_logger(__name__)


class MLP(Base):
    """An MLP class stands for a Multi-Layer Perceptron implementation."""

    def __init__(self, n_hidden: Optional[Tuple[int, ...]] = (128,)):
        """Initialization method.

        Args:
            n_hidden: Tuple containing the number of hidden units per layer.

        """

        logger.info("Overriding class: Base -> MLP.")

        super(MLP, self).__init__(name="mlp")

        # Fully-connected layers
        self.fc = [Dense(units) for units in n_hidden]

        logger.info("Class overrided.")
        logger.debug("Layers: %d | Units: %s.", len(n_hidden), n_hidden)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Method that holds vital information whenever this class is called.

        Args:
            x: Tensor containing the input sample.

        Returns:
            (tf.Tensor): The layer's outputs.

        """

        # Iterates through all fully-connected layers
        for fc in self.fc:
            # Passes down through the layer
            x = fc(x)

        return x
