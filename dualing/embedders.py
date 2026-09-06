# Copyright (c) 2020-2026 Gustavo Rosa.
# Licensed under the Apache License, Version 2.0.

"""Keras embedding models."""

from collections.abc import Callable

import tensorflow as tf

from dualing.core.model import Base

_Activation = str | Callable[[tf.Tensor], tf.Tensor] | None


@tf.keras.utils.register_keras_serializable(package="dualing")
class MLP(Base):
    """Apply dense layers to the final input dimension."""

    def __init__(
        self,
        n_hidden: tuple[int, ...] = (128,),
        activation: _Activation = None,
        name: str = "mlp",
        *,
        hidden_units: tuple[int, ...] | None = None,
        **kwargs,
    ) -> None:
        """Initialize a dense embedding model.

        Leading input dimensions are preserved. The output width is the final hidden width.
        An empty hidden-width tuple produces an identity map.

        Args:
            n_hidden: Width of each dense layer.
            activation: Keras activation name or callable, or None for linear activation.
            name: Model name.
            hidden_units: Alias that takes precedence over n_hidden when supplied.
            **kwargs: Native Keras model options such as trainable and dtype.

        """

        super().__init__(name=name, **kwargs)

        if hidden_units is not None:
            n_hidden = hidden_units

        self.fc = [tf.keras.layers.Dense(units, activation=activation, dtype=self.dtype_policy) for units in n_hidden]

    def get_config(self) -> dict:
        return {
            **super().get_config(),
            "n_hidden": tuple(layer.units for layer in self.fc),
            "activation": (tf.keras.activations.serialize(self.fc[0].activation) if self.fc else None),
        }

    def call(self, x):
        for layer in self.fc:
            x = layer(x)

        return x


@tf.keras.utils.register_keras_serializable(package="dualing")
class CNN(Base):
    """Embed images with convolutional blocks and a dense projection."""

    def __init__(
        self,
        n_blocks: int = 3,
        init_kernel: int = 5,
        n_output: int = 128,
        activation: _Activation = "sigmoid",
        name: str = "cnn",
        *,
        blocks: int | None = None,
        kernel_size: int | None = None,
        embedding_dim: int | None = None,
        **kwargs,
    ) -> None:
        """Initialize a convolutional embedding model.

        Prefer inputs shaped (batch, height, width, channels).
        Channels-first inputs are detected when dimension 1 is 1, 3, or 4 and the final dimension is not.
        Output shape is (batch, n_output). Kernels decrease by two per block and must remain positive.

        Args:
            n_blocks: Number of convolution and pooling blocks.
            init_kernel: First square convolution kernel size.
            n_output: Output embedding width.
            activation: Projection activation, with relu retained for the convolutions.
            name: Model name.
            blocks: Alias that takes precedence over n_blocks.
            kernel_size: Alias that takes precedence over init_kernel.
            embedding_dim: Alias that takes precedence over n_output.
            **kwargs: Native Keras model options such as trainable and dtype.

        Raises:
            ValueError: The block count and kernel size produce a nonpositive convolution size.

        """

        super().__init__(name=name, **kwargs)

        n_blocks = n_blocks if blocks is None else blocks
        init_kernel = init_kernel if kernel_size is None else kernel_size
        n_output = n_output if embedding_dim is None else embedding_dim

        if n_blocks < 1 or init_kernel - 2 * (n_blocks - 1) < 1:
            raise ValueError("`n_blocks` and `init_kernel` must produce positive convolution sizes.")

        self.conv = [
            tf.keras.layers.Conv2D(
                32 * 2**index,
                init_kernel - 2 * index,
                activation="relu",
                padding="same",
                dtype=self.dtype_policy,
            )
            for index in range(n_blocks)
        ]
        self.pool = [tf.keras.layers.MaxPool2D(dtype=self.dtype_policy) for _ in range(n_blocks)]
        self.flatten = tf.keras.layers.Flatten(dtype=self.dtype_policy)
        self.fc = tf.keras.layers.Dense(n_output, activation=activation, dtype=self.dtype_policy)

    def get_config(self) -> dict:
        return {
            **super().get_config(),
            "n_blocks": len(self.conv),
            "init_kernel": self.conv[0].kernel_size[0],
            "n_output": self.fc.units,
            "activation": tf.keras.activations.serialize(self.fc.activation),
        }

    def call(self, x):
        if x.shape.rank == 4 and x.shape[1] in {1, 3, 4} and x.shape[-1] not in {1, 3, 4}:
            x = tf.transpose(x, (0, 2, 3, 1))

        for convolution, pooling in zip(self.conv, self.pool):
            x = convolution(x)
            x = pooling(x)

        x = self.flatten(x)

        return self.fc(x)


class _RecurrentEmbedder(Base):
    _cell_class: type[tf.keras.layers.Layer]

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        hidden_size: int,
        name: str,
        *,
        embedding_dim: int | None = None,
        hidden_units: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(name=name, **kwargs)

        embedding_size = embedding_size if embedding_dim is None else embedding_dim
        hidden_size = hidden_size if hidden_units is None else hidden_units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size, dtype=self.dtype_policy)
        self.cell = self._cell_class(hidden_size, dtype=self.dtype_policy)
        self.rnn = tf.keras.layers.RNN(self.cell, return_sequences=True, dtype=self.dtype_policy)
        self.fc = tf.keras.layers.Dense(vocab_size, dtype=self.dtype_policy)

    def get_config(self) -> dict:
        return {
            **super().get_config(),
            "vocab_size": self.embedding.input_dim,
            "embedding_size": self.embedding.output_dim,
            "hidden_size": self.cell.units,
        }

    def call(self, x):
        x = self.embedding(x)
        x = self.rnn(x)

        return self.fc(x)


@tf.keras.utils.register_keras_serializable(package="dualing")
class RNN(_RecurrentEmbedder):
    """Embed token sequences with a simple recurrent cell and linear projection."""

    _cell_class = tf.keras.layers.SimpleRNNCell

    def __init__(
        self,
        vocab_size: int = 1,
        embedding_size: int = 32,
        hidden_size: int = 64,
        name: str = "rnn",
        *,
        embedding_dim: int | None = None,
        hidden_units: int | None = None,
        **kwargs,
    ) -> None:
        """Initialize a simple recurrent embedding model.

        Input token IDs lie in [0, vocab_size) and have shape (batch, time).
        Outputs have shape (batch, time, vocab_size), with time pooling performed by Siamese models.

        Args:
            vocab_size: Number of token IDs and output features.
            embedding_size: Token embedding width.
            hidden_size: Recurrent state width.
            name: Model name.
            embedding_dim: Alias that takes precedence over embedding_size.
            hidden_units: Alias that takes precedence over hidden_size.
            **kwargs: Native Keras model options such as trainable and dtype.

        """

        super().__init__(
            vocab_size,
            embedding_size,
            hidden_size,
            name,
            embedding_dim=embedding_dim,
            hidden_units=hidden_units,
            **kwargs,
        )


@tf.keras.utils.register_keras_serializable(package="dualing")
class GRU(_RecurrentEmbedder):
    """Embed token sequences with a gated recurrent cell and linear projection."""

    _cell_class = tf.keras.layers.GRUCell

    def __init__(
        self,
        vocab_size: int = 1,
        embedding_size: int = 32,
        hidden_size: int = 64,
        name: str = "gru",
        *,
        embedding_dim: int | None = None,
        hidden_units: int | None = None,
        **kwargs,
    ) -> None:
        """Initialize a gated recurrent embedding model.

        Token ranges and input/output shapes follow RNN.

        Args:
            vocab_size: Number of token IDs and output features.
            embedding_size: Token embedding width.
            hidden_size: Recurrent state width.
            name: Model name.
            embedding_dim: Alias that takes precedence over embedding_size.
            hidden_units: Alias that takes precedence over hidden_size.
            **kwargs: Native Keras model options such as trainable and dtype.

        """

        super().__init__(
            vocab_size,
            embedding_size,
            hidden_size,
            name,
            embedding_dim=embedding_dim,
            hidden_units=hidden_units,
            **kwargs,
        )


@tf.keras.utils.register_keras_serializable(package="dualing")
class LSTM(_RecurrentEmbedder):
    """Embed token sequences with an LSTM cell and linear projection."""

    _cell_class = tf.keras.layers.LSTMCell

    def __init__(
        self,
        vocab_size: int = 1,
        embedding_size: int = 32,
        hidden_size: int = 64,
        name: str = "lstm",
        *,
        embedding_dim: int | None = None,
        hidden_units: int | None = None,
        **kwargs,
    ) -> None:
        """Initialize an LSTM embedding model.

        Token ranges and input/output shapes follow RNN.

        Args:
            vocab_size: Number of token IDs and output features.
            embedding_size: Token embedding width.
            hidden_size: Recurrent state width.
            name: Model name.
            embedding_dim: Alias that takes precedence over embedding_size.
            hidden_units: Alias that takes precedence over hidden_size.
            **kwargs: Native Keras model options such as trainable and dtype.

        """

        super().__init__(
            vocab_size,
            embedding_size,
            hidden_size,
            name,
            embedding_dim=embedding_dim,
            hidden_units=hidden_units,
            **kwargs,
        )
