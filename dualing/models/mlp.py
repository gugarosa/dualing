from tensorflow.keras.layers import Dense, Flatten

import dualing.utils.logging as l
from dualing.core import Network

logger = l.get_logger(__name__)


class MLP(Network):
    """A MLP class stands for a Multi-Layer Perceptron.

    """

    def __init__(self):
        """Initialization method.

        """

        logger.info('Overriding class: Network -> MLP.')

        # Overrides its parent class with any custom arguments if needed
        super(MLP, self).__init__(name='mlp')

        #
        self.fc = Dense(150)

        #
        self.flatten = Flatten()

        logger.info('Class overrided.')

    def call(self, x):
        """Method that holds vital information whenever this class is called.

        Args:
            x (tf.Tensor): A tensorflow's tensor holding input data.

        Returns:
            The same tensor after passing through each defined layer.

        """

        #
        x = self.fc(x)

        #
        x = self.flatten(x)

        return x