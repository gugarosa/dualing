"""Gated Recurrent Unit.
"""

from tensorflow.keras.layers import RNN, Dense, Embedding, GRUCell

import dualing.utils.logging as l
from dualing.core import Base

logger = l.get_logger(__name__)


class GRU(Base):
    """A GRU class stands for a standard Gated Recurrent Unit implementation.

    """

    def __init__(self, vocab_size=1, embedding_size=32, hidden_size=64):
        """Initialization method.

        Args:
            vocab_size (int): Vocabulary size.
            embedding_size (int): Embedding layer units.
            hidden_size (int): Hidden layer units.

        """

        logger.info('Overriding class: Base -> GRU.')

        # Overrides its parent class with any custom arguments if needed
        super(GRU, self).__init__(name='gru')

        # Embedding layer
        self.embedding = Embedding(vocab_size, embedding_size, name='embedding')

        # GRU cell
        self.cell = GRUCell(hidden_size, name='gru_cell')

        # RNN layer
        self.rnn = RNN(self.cell, name='rnn_layer', return_sequences=True)

        # Linear (dense) layer
        self.fc = Dense(vocab_size, name='out')

        logger.info('Class overrided.')
        logger.debug('Embedding: %d | Hidden: %d | Output: %d.', embedding_size, hidden_size, vocab_size)

    def call(self, x):
        """Method that holds vital information whenever this class is called.

        Args:
            x (tf.Tensor): Tensor containing the input sample.

        Returns:
            The layer's outputs.

        """

        # Firstly, we apply the embedding layer
        x = self.embedding(x)

        # We need to apply the input into the first recurrent layer
        x = self.rnn(x)

        # The input also suffers a linear combination to output correct shape
        x = self.fc(x)

        return x
