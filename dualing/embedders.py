"""Small Keras embedding models."""

import tensorflow as tf


class MLP(tf.keras.Sequential):
    """A stack of dense embedding layers."""

    def __init__(
        self,
        hidden_units: tuple[int, ...] = (128,),
        activation: str | None = None,
        name: str = "mlp",
    ) -> None:
        super().__init__(
            [
                tf.keras.layers.Dense(units, activation=activation)
                for units in hidden_units
            ],
            name=name,
        )


class CNN(tf.keras.Sequential):
    """A convolutional embedder."""

    def __init__(
        self,
        blocks: int = 3,
        kernel_size: int = 5,
        embedding_dim: int = 128,
        activation: str | None = "sigmoid",
        name: str = "cnn",
    ) -> None:
        if blocks < 1 or kernel_size - 2 * (blocks - 1) < 1:
            raise ValueError("blocks and kernel_size produce an invalid convolution")

        layers = []
        for index in range(blocks):
            layers.extend(
                [
                    tf.keras.layers.Conv2D(
                        32 * 2**index,
                        kernel_size - 2 * index,
                        activation="relu",
                        padding="same",
                    ),
                    tf.keras.layers.MaxPool2D(),
                ]
            )
        layers.extend(
            [
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(embedding_dim, activation=activation),
            ]
        )
        super().__init__(layers, name=name)


def _recurrent_layers(layer, vocab_size: int, embedding_dim: int):
    return [
        tf.keras.layers.Embedding(vocab_size, embedding_dim),
        layer,
        tf.keras.layers.Dense(vocab_size),
    ]


class RNN(tf.keras.Sequential):
    """A simple recurrent embedder."""

    def __init__(
        self,
        vocab_size: int = 1,
        embedding_dim: int = 32,
        hidden_units: int = 64,
        name: str = "rnn",
    ) -> None:
        super().__init__(
            _recurrent_layers(
                tf.keras.layers.SimpleRNN(hidden_units, return_sequences=True),
                vocab_size,
                embedding_dim,
            ),
            name=name,
        )


class GRU(tf.keras.Sequential):
    """A gated recurrent embedder."""

    def __init__(
        self,
        vocab_size: int = 1,
        embedding_dim: int = 32,
        hidden_units: int = 64,
        name: str = "gru",
    ) -> None:
        super().__init__(
            _recurrent_layers(
                tf.keras.layers.GRU(hidden_units, return_sequences=True),
                vocab_size,
                embedding_dim,
            ),
            name=name,
        )


class LSTM(tf.keras.Sequential):
    """A long short-term memory embedder."""

    def __init__(
        self,
        vocab_size: int = 1,
        embedding_dim: int = 32,
        hidden_units: int = 64,
        name: str = "lstm",
    ) -> None:
        super().__init__(
            _recurrent_layers(
                tf.keras.layers.LSTM(hidden_units, return_sequences=True),
                vocab_size,
                embedding_dim,
            ),
            name=name,
        )
