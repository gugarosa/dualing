"""Multi-Layer Perceptron.
"""

from tensorflow.keras.layers import Dense

import dualing.utils.logging as l
from dualing.core import Base

logger = l.get_logger(__name__)


class MLP(Base):
    """An MLP class stands for a Multi-Layer Perceptron implementation.

    """

    def __init__(self, n_hidden=[128]):
        """Initialization method.

        """

        logger.info('Overriding class: Base -> MLP.')

        # Overrides its parent class with any custom arguments if needed
        super(MLP, self).__init__(name='mlp')

        # Fully-connected layers
        self.fc = [Dense(units) for units in n_hidden]

        logger.info('Class overrided.')
        logger.debug('Layers: %d | Units: %d', len(n_hidden), n_hidden)

    def call(self, x):
        """Method that holds vital information whenever this class is called.

        Args:
            x (tf.Tensor): Tensor containing the input sample.

        Returns:
            The layer's outputs.

        """

        # Iterates through all fully-connected layers
        for fc in self.fc:
            # Passes down through the layer
            x = fc(x)

        return x
