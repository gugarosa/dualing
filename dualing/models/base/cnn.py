"""Convolutional Neural Network.
"""

from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D

import dualing.utils.logging as l
from dualing.core import Base

logger = l.get_logger(__name__)


class CNN(Base):
    """A CNN class stands for a standard Convolutional Neural Network implementation.

    """

    def __init__(self, n_blocks=3, init_kernel=5, n_output=128, activation='sigmoid'):
        """Initialization method.

        """

        logger.info('Overriding class: Base -> CNN.')

        # Overrides its parent class with any custom arguments if needed
        super(CNN, self).__init__(name='cnn')

        # Asserting that it will be possible to create the convolutional layers
        assert init_kernel - 2 * (n_blocks - 1) >= 1

        # Convolutional layers
        self.conv = [Conv2D(32 * (2 ** i), init_kernel - 2 * i, activation='relu', padding='same')
                     for i in range(n_blocks)]

        # Pooling layers
        self.pool = [MaxPool2D() for _ in range(n_blocks)]

        # Flatenning layer
        self.flatten = Flatten()

        # Final fully-connected layer
        self.fc = Dense(n_output, activation=activation)

        logger.info('Class overrided.')
        logger.debug('Blocks: %d | Initial Kernel: %d | Output (%s): %d', n_blocks, init_kernel, activation, n_output)

    def call(self, x):
        """Method that holds vital information whenever this class is called.

        Args:
            x (tf.Tensor): Tensor containing the input sample.

        Returns:
            The layer's outputs.

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
