import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D

import dualing.utils.logging as l
from dualing.core import Base

logger = l.get_logger(__name__)


class CNN(Base):
    """A CNN class stands for a standard Convolutional Neural Network implementation.

    """

    def __init__(self, n_conv_units=[32, 64], n_dense_units=[256, 128]):
        """Initialization method.

        """

        logger.info('Overriding class: Base -> CNN.')

        # Overrides its parent class with any custom arguments if needed
        super(CNN, self).__init__(name='cnn')

        # Creates convolutional layers
        self.conv = [Conv2D(units, 5, strides=(2, 2), padding='same', activation='relu') for units in n_conv_units]

        # Creates pooling layers
        self.pool = [MaxPool2D(2, 2) for _ in range(len(n_conv_units))]

        # Creates the flattenning layer
        self.flatten = Flatten()

        # Creates fully-connected layers
        self.fc = [Dense(units, activation='relu') for units in n_dense_units]

        logger.info('Class overrided.')
        logger.debug(f'Convolutional: {n_conv_units} | Dense: {n_dense_units}')

    def call(self, x):
        """Method that holds vital information whenever this class is called.

        Args:
            x (tf.Tensor): Tensor containing the input sample.

        Returns:
            The layer's outputs.

        """

        # Iterates through all convolutional and pooling blocks
        for conv, pool in zip(self.conv, self.pool):
            # Passes down through convolutional
            x = conv(x)

            # Passes down through pooling
            x = pool(x)

        # Flattens the outputs
        x = self.flatten(x)
        
        # Iterates through all fully-connected layers
        for fc in self.fc:
            # Passes down through the layer
            x = fc(x)

        return x
