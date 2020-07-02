import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D

import dualing.utils.logging as l
from dualing.core import Base

logger = l.get_logger(__name__)


class CNN(Base):
    """A CNN class stands for a standard Convolutional Neural Network implementation.

    """

    def __init__(self):
        """Initialization method.

        """

        logger.info('Overriding class: Base -> CNN.')

        # Overrides its parent class with any custom arguments if needed
        super(CNN, self).__init__(name='cnn')

        # Creates convolutional layers
        self.conv1 = Conv2D(32, 7, activation='relu', padding='same')
        self.conv2 = Conv2D(64, 5, activation='relu', padding='same')
        self.conv3 = Conv2D(128, 3, activation='relu', padding='same')
        self.conv4 = Conv2D(256, 1, activation='relu', padding='same')
        self.conv5 = Conv2D(2, 1, padding='same')

        #
        self.pool1 = MaxPool2D(2, padding='same')
        self.pool2 = MaxPool2D(2, padding='same')
        self.pool3 = MaxPool2D(2, padding='same')
        self.pool4 = MaxPool2D(2, padding='same')
        self.pool5 = MaxPool2D(2, padding='same')

        #
        self.flatten = Flatten()

        logger.info('Class overrided.')

    def call(self, x):
        """Method that holds vital information whenever this class is called.

        Args:
            x (tf.Tensor): Tensor containing the input sample.

        Returns:
            The layer's outputs.

        """

        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.pool4(x)
        x = self.conv5(x)
        x = self.pool5(x)
        x = self.flatten(x)

        return x
