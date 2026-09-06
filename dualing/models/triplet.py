# Copyright (c) 2020-2026 Gustavo Rosa.
# Licensed under the Apache License, Version 2.0.

"""Triplet-loss Siamese model."""

import numbers
from typing import Self

import tensorflow as tf

from dualing.core import Siamese, TripletHardLoss, TripletSemiHardLoss
from dualing.losses import pair_distance
from dualing.models._utils import _legacy_fit_epochs, _prediction_input
from dualing.utils import exception


@tf.keras.utils.register_keras_serializable(package="dualing")
class TripletSiamese(Siamese):
    """Learn normalized embeddings using class labels and triplet mining."""

    def __init__(
        self,
        base: tf.keras.Model,
        loss: str = "hard",
        margin: float = 1.0,
        soft: bool = False,
        distance_metric: str = "L2",
        name: str = "",
        *,
        mining: str | None = None,
        **kwargs,
    ) -> None:
        """Initialize triplet learning without cloning its shared embedder.

        Model calls return normalized (batch, features) embeddings. Batches need positive and negative examples.
        Comparisons use the requested metric, while legacy pair prediction retains its effective unnormalized metric.
        Both metrics survive serialization. Explicit mining selects direct distance semantics.

        Args:
            base: Keras embedding model.
            loss: Legacy mining name, either hard or semi-hard.
            margin: Positive margin for the default hinge loss.
            soft: Whether to use softplus instead of the hinge.
            distance_metric: Requested L1, L2, squared-L2, or angular distance.
            name: Model name.
            mining: Explicit mining strategy, or None to retain the original distance mapping.
            **kwargs: Native Keras model options such as trainable and dtype.

        References:
            X. Dong and J. Shen.
            Triplet loss in siamese network for object tracking.
            Proceedings of the European Conference on Computer Vision (2018).

        """

        super().__init__(base, name=name, **kwargs)

        legacy_mode = mining is None

        self.loss_type = loss if mining is None else mining
        self.margin = margin
        self.soft = soft
        self.distance_metric = distance_metric

        if legacy_mode and distance_metric == "L1":
            self.distance = "L2"
        elif legacy_mode and distance_metric == "L2":
            self.distance = "squared-L2"
        elif legacy_mode:
            self.distance = "angular"
        else:
            self.distance = distance_metric

    def get_config(self) -> dict:
        return {
            **super().get_config(),
            "loss": self.loss_type,
            "margin": self.margin,
            "soft": self.soft,
            "distance_metric": self.distance_metric,
            "distance": self.distance,
        }

    @classmethod
    def from_config(cls, config: dict) -> Self:
        if "distance" not in config:
            return super().from_config(config)

        config = dict(config)
        distance = config.pop("distance")
        model = super().from_config(config)

        # Do not apply the legacy metric mapping again to an already resolved distance
        model.distance = distance

        return model

    @property
    def loss_type(self) -> str:
        """Triplet mining strategy."""

        return self._loss_type

    @loss_type.setter
    def loss_type(self, loss_type: str) -> None:
        if loss_type not in {"hard", "semi-hard"}:
            raise exception.ValueError("`loss_type` must be hard or semi-hard.")

        self._loss_type = loss_type

    @property
    def mining(self) -> str:
        """Triplet mining strategy."""

        return self.loss_type

    @mining.setter
    def mining(self, mining: str) -> None:
        self.loss_type = mining

    @property
    def soft(self) -> bool:
        """Whether to use a soft margin."""

        return self._soft

    @soft.setter
    def soft(self, soft: bool) -> None:
        if not isinstance(soft, bool):
            raise exception.TypeError("`soft` must be a boolean.")

        self._soft = soft

    @property
    def margin(self) -> float:
        """Triplet margin."""

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
        """Legacy pair and loss distance metric."""

        return self._distance

    @distance.setter
    def distance(self, distance: str) -> None:
        if distance not in {"L1", "L2", "squared-L2", "angular"}:
            raise exception.ValueError("`distance` must be L1, L2, squared-L2, or angular.")

        self._distance = distance

    @property
    def distance_metric(self) -> str:
        """Requested pair distance metric."""

        return self._distance_metric

    @distance_metric.setter
    def distance_metric(self, distance: str) -> None:
        if distance not in {"L1", "L2", "squared-L2", "angular"}:
            raise exception.ValueError("`distance_metric` must be L1, L2, squared-L2, or angular.")

        self._distance_metric = distance

    def call(self, inputs, training: bool = False) -> tf.Tensor:
        return tf.math.l2_normalize(self.embed(inputs, training), axis=-1)

    def compile(self, optimizer="rmsprop", **kwargs) -> None:
        """Compile the model with its selected triplet loss unless a loss is supplied.

        Args:
            optimizer: Keras optimizer instance or identifier.
            **kwargs: Native Keras compilation options.

        """

        loss_class = TripletHardLoss if self.loss_type == "hard" else TripletSemiHardLoss

        self.loss = loss_class(self.margin, self.soft, self.distance)

        kwargs.setdefault("loss", self.loss)

        tf.keras.Model.compile(self, optimizer=optimizer, **kwargs)

    def step(self, x: tf.Tensor, y: tf.Tensor) -> None:
        """Update model weights and the legacy loss tracker for one batch.

        Args:
            x: Input sample batch.
            y: Class labels for the samples.

        """

        with tf.GradientTape() as tape:
            embeddings = self(x, training=True)
            loss = self.loss(y, embeddings)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_metric.update_state(loss)

    @_legacy_fit_epochs
    def fit(self, batches=None, y=None, epochs: int = 100, **kwargs) -> tf.keras.callbacks.History:
        """Train with native or legacy labeled datasets.

        A positional integer after a dataset retains the original epoch argument.
        Array inputs use fit(samples, labels, epochs=...).

        Args:
            batches: Labeled dataset or array inputs, also accepted through the x keyword.
            y: Targets for array inputs.
            epochs: Number of training epochs.
            **kwargs: Native Keras fit options such as validation_data and callbacks.

        Returns:
            A Keras History object.

        """

        x = kwargs.pop("x", batches)

        if isinstance(x, tf.data.Dataset):
            kwargs.setdefault("shuffle", False)

        return tf.keras.Model.fit(self, x=x, y=y, epochs=epochs, **kwargs)

    def evaluate(self, batches=None, y=None, **kwargs):
        """Evaluate with native or legacy labeled datasets without updating weights.

        Args:
            batches: Labeled dataset or array inputs, also accepted through the x keyword.
            y: Targets for array inputs.
            **kwargs: Native Keras evaluation options.

        Returns:
            Keras loss and metric results as a scalar, list, or dictionary.

        """

        x = kwargs.pop("x", batches)

        return tf.keras.Model.evaluate(self, x=x, y=y, **kwargs)

    def predict(self, x1=None, x2=None, **kwargs):
        """Predict embeddings or compare two sample batches without updating weights.

        Args:
            x1: Native inputs or the first sample batch, also accepted through the x keyword.
            x2: Second sample batch for direct comparison, or an integer native batch size.
            **kwargs: Native Keras prediction options.

        Returns:
            NumPy embeddings for native calls or a tensor of legacy pair distances.

        """

        x1 = _prediction_input(x1, kwargs)

        if x2 is not None and not isinstance(x2, numbers.Integral):
            return pair_distance(
                self.embed(x1),
                self.embed(x2),
                self.distance,
            )

        batch_size = kwargs.pop("batch_size", x2)

        return tf.keras.Model.predict(self, x1, batch_size=batch_size, **kwargs)

    def compare(self, left, right) -> tf.Tensor:
        """Return normalized pair distances using the requested metric.

        Args:
            left: First sample batch.
            right: Corresponding second sample batch.

        Returns:
            Distance tensor with shape (batch,).

        """

        return pair_distance(
            self(left, training=False),
            self(right, training=False),
            self.distance_metric,
        )
