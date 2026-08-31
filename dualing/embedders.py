"""Keras embedding models."""

import tensorflow as tf

from dualing.core.model import Base


class MLP(Base):
    """A stack of dense embedding layers."""

    def __init__(
        self,
        n_hidden: tuple[int, ...] = (128,),
        activation: str | None = None,
        name: str = "mlp",
        *,
        hidden_units: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__(name=name)

        if hidden_units is not None:
            n_hidden = hidden_units

        self.fc = [
            tf.keras.layers.Dense(units, activation=activation) for units in n_hidden
        ]

    def call(self, x):
        for layer in self.fc:
            x = layer(x)

        return x


class CNN(Base):
    """A convolutional embedder."""

    def __init__(
        self,
        n_blocks: int = 3,
        init_kernel: int = 5,
        n_output: int = 128,
        activation: str | None = "sigmoid",
        name: str = "cnn",
        *,
        blocks: int | None = None,
        kernel_size: int | None = None,
        embedding_dim: int | None = None,
    ) -> None:
        super().__init__(name=name)

        n_blocks = n_blocks if blocks is None else blocks
        init_kernel = init_kernel if kernel_size is None else kernel_size
        n_output = n_output if embedding_dim is None else embedding_dim

        if n_blocks < 1 or init_kernel - 2 * (n_blocks - 1) < 1:
            raise ValueError("blocks and kernel size produce an invalid convolution")

        self.conv = [
            tf.keras.layers.Conv2D(
                32 * 2**index,
                init_kernel - 2 * index,
                activation="relu",
                padding="same",
            )
            for index in range(n_blocks)
        ]
        self.pool = [tf.keras.layers.MaxPool2D() for _ in range(n_blocks)]
        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(n_output, activation=activation)

    def call(self, x):
        if (
            x.shape.rank == 4
            and x.shape[1] in {1, 3, 4}
            and x.shape[-1] not in {1, 3, 4}
        ):
            x = tf.transpose(x, (0, 2, 3, 1))

        for convolution, pooling in zip(self.conv, self.pool):
            x = convolution(x)
            x = pooling(x)

        x = self.flatten(x)

        return self.fc(x)


class RNN(Base):
    """A simple recurrent embedder."""

    def __init__(
        self,
        vocab_size: int = 1,
        embedding_size: int = 32,
        hidden_size: int = 64,
        name: str = "rnn",
        *,
        embedding_dim: int | None = None,
        hidden_units: int | None = None,
    ) -> None:
        super().__init__(name=name)

        embedding_size = embedding_size if embedding_dim is None else embedding_dim
        hidden_size = hidden_size if hidden_units is None else hidden_units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.cell = tf.keras.layers.SimpleRNNCell(hidden_size)
        self.rnn = tf.keras.layers.RNN(self.cell, return_sequences=True)
        self.fc = tf.keras.layers.Dense(vocab_size)

    def call(self, x):
        x = self.embedding(x)
        x = self.rnn(x)

        return self.fc(x)


class GRU(Base):
    """A gated recurrent embedder."""

    def __init__(
        self,
        vocab_size: int = 1,
        embedding_size: int = 32,
        hidden_size: int = 64,
        name: str = "gru",
        *,
        embedding_dim: int | None = None,
        hidden_units: int | None = None,
    ) -> None:
        super().__init__(name=name)

        embedding_size = embedding_size if embedding_dim is None else embedding_dim
        hidden_size = hidden_size if hidden_units is None else hidden_units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.cell = tf.keras.layers.GRUCell(hidden_size)
        self.rnn = tf.keras.layers.RNN(self.cell, return_sequences=True)
        self.fc = tf.keras.layers.Dense(vocab_size)

    def call(self, x):
        x = self.embedding(x)
        x = self.rnn(x)

        return self.fc(x)


class LSTM(Base):
    """A long short-term memory embedder."""

    def __init__(
        self,
        vocab_size: int = 1,
        embedding_size: int = 32,
        hidden_size: int = 64,
        name: str = "lstm",
        *,
        embedding_dim: int | None = None,
        hidden_units: int | None = None,
    ) -> None:
        super().__init__(name=name)

        embedding_size = embedding_size if embedding_dim is None else embedding_dim
        hidden_size = hidden_size if hidden_units is None else hidden_units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.cell = tf.keras.layers.LSTMCell(hidden_size)
        self.rnn = tf.keras.layers.RNN(self.cell, return_sequences=True)
        self.fc = tf.keras.layers.Dense(vocab_size)

    def call(self, x):
        x = self.embedding(x)
        x = self.rnn(x)

        return self.fc(x)
