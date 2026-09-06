# Copyright (c) 2020-2026 Gustavo Rosa.
# Licensed under the Apache License, Version 2.0.

"""Contrastive-loss Siamese model."""

import numbers

import tensorflow as tf

from dualing.core import ContrastiveLoss, Siamese
from dualing.losses import pair_distance
from dualing.models._utils import (
    _legacy_fit_epochs,
    _prediction_input,
    pair_dataset,
)
from dualing.utils import exception


@tf.keras.utils.register_keras_serializable(package="dualing")
class ContrastiveSiamese(Siamese):
    """Train one shared embedder to separate dissimilar sample pairs."""

    def __init__(
        self,
        base: tf.keras.Model,
        margin: float = 1.0,
        distance_metric: str = "L2",
        name: str = "",
        **kwargs,
    ) -> None:
        """Initialize a contrastive Siamese model without cloning its shared embedder.

        Pair calls return shape (batch,), with targets 1 for similar and 0 for dissimilar pairs.
        Rank-three embeddings are mean-pooled over time.

        Args:
            base: Keras embedding model shared by both branches.
            margin: Positive margin for the default contrastive loss.
            distance_metric: L1, L2, squared-L2, or angular distance.
            name: Model name.
            **kwargs: Native Keras model options such as trainable and dtype.

        References:
            I. Melekhov, J. Kannala and E. Rahtu.
            Siamese network features for image matching.
            23rd International Conference on Pattern Recognition (2016).

        """

        super().__init__(base, name=name, **kwargs)

        self.margin = margin
        self.distance = distance_metric

    def get_config(self) -> dict:
        return {
            **super().get_config(),
            "margin": self.margin,
            "distance_metric": self.distance,
        }

    @property
    def margin(self) -> float:
        """Contrastive margin."""

        return self._margin

    @margin.setter
    def margin(self, margin: float) -> None:
        if not isinstance(margin, numbers.Real):
            raise exception.TypeError("`margin` must be a number.")

        if margin <= 0:
            raise exception.ValueError("`margin` must be greater than 0.")

        self._margin = float(margin)

    @property
    def distance(self) -> str:
        """Distance metric."""

        return self._distance

    @distance.setter
    def distance(self, distance: str) -> None:
        if distance not in {"L1", "L2", "squared-L2", "angular"}:
            raise exception.ValueError("`distance` must be L1, L2, squared-L2, or angular.")

        self._distance = distance

    @property
    def distance_metric(self) -> str:
        """Distance metric."""

        return self.distance

    @distance_metric.setter
    def distance_metric(self, distance: str) -> None:
        self.distance = distance

    def call(self, inputs, training: bool = False) -> tf.Tensor:
        left, right = inputs

        return pair_distance(
            self.embed(left, training),
            self.embed(right, training),
            self.distance,
        )

    def compile(self, optimizer="rmsprop", **kwargs) -> None:
        """Compile the model with contrastive loss unless a loss is supplied.

        Args:
            optimizer: Keras optimizer instance or identifier.
            **kwargs: Native Keras compilation options.

        """

        self.loss = ContrastiveLoss(self.margin)

        kwargs.setdefault("loss", self.loss)

        tf.keras.Model.compile(self, optimizer=optimizer, **kwargs)

    def step(self, x1: tf.Tensor, x2: tf.Tensor, y: tf.Tensor) -> None:
        """Update model weights and the legacy loss tracker for one batch.

        Args:
            x1: First sample batch.
            x2: Corresponding second sample batch.
            y: Pair labels with 1 for similar and 0 for dissimilar.

        """

        with tf.GradientTape() as tape:
            prediction = self((x1, x2), training=True)
            loss = self.loss(y, prediction)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_metric.update_state(loss)

    @_legacy_fit_epochs
    def fit(self, batches=None, y=None, epochs: int = 100, **kwargs) -> tf.keras.callbacks.History:
        """Train with native or legacy pair datasets.

        A positional integer after a dataset retains the original epoch argument.
        Array inputs use fit((left, right), labels, epochs=...).

        Args:
            batches: Pair dataset or array inputs, also accepted through the x keyword.
            y: Targets for array inputs.
            epochs: Number of training epochs.
            **kwargs: Native Keras fit options such as validation_data and callbacks.

        Returns:
            A Keras History object.

        """

        x = kwargs.pop("x", batches)

        if isinstance(x, tf.data.Dataset):
            x = pair_dataset(x)
            kwargs.setdefault("shuffle", False)

        if "validation_data" in kwargs:
            kwargs["validation_data"] = pair_dataset(kwargs["validation_data"])

        return tf.keras.Model.fit(self, x=x, y=y, epochs=epochs, **kwargs)

    def evaluate(self, batches=None, y=None, **kwargs):
        """Evaluate with native or legacy pair datasets without updating weights.

        Args:
            batches: Pair dataset or array inputs, also accepted through the x keyword.
            y: Targets for array inputs.
            **kwargs: Native Keras evaluation options.

        Returns:
            Keras loss and metric results as a scalar, list, or dictionary.

        """

        x = kwargs.pop("x", batches)

        return tf.keras.Model.evaluate(self, x=pair_dataset(x), y=y, **kwargs)

    def predict(self, x1=None, x2=None, **kwargs):
        """Predict natively or compare two sample batches without updating weights.

        Args:
            x1: Native inputs or the first sample batch, also accepted through the x keyword.
            x2: Second sample batch for direct comparison, or an integer native batch size.
            **kwargs: Native Keras prediction options.

        Returns:
            NumPy predictions for native calls or a tensor for direct comparisons.

        """

        x1 = _prediction_input(x1, kwargs)

        if x2 is not None and not isinstance(x2, numbers.Integral):
            return self((x1, x2), training=False)

        batch_size = kwargs.pop("batch_size", x2)

        return tf.keras.Model.predict(self, x1, batch_size=batch_size, **kwargs)

    def compare(self, left, right) -> tf.Tensor:
        """Return distances between two batches of samples.

        Args:
            left: First sample batch.
            right: Corresponding second sample batch.

        Returns:
            Distance tensor with shape (batch,).

        """

        return self((left, right), training=False)
