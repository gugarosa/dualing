# Copyright (c) 2020-2026 Gustavo Rosa.
# Licensed under the Apache License, Version 2.0.

from unittest.mock import Mock

import matplotlib
import numpy as np
import pytest
import tensorflow as tf

from dualing.core import (
    Base,
    BinaryCrossEntropy,
    ContrastiveLoss,
    Dataset,
    Siamese,
    TripletHardLoss,
    TripletSemiHardLoss,
)
from dualing.datasets import BalancedPairDataset, BatchDataset, RandomPairDataset
from dualing.models import ContrastiveSiamese, CrossEntropySiamese, TripletSiamese
from dualing.models.base import CNN, GRU, LSTM, MLP, RNN
from dualing.utils import constants, exception, logging, projector

matplotlib.use("Agg")


def _data():
    data = np.arange(32, dtype="float32").reshape(8, 4)
    labels = np.repeat([0, 1], 4)

    return data, labels


def test_original_dataset_classes():
    data, labels = _data()

    base = Dataset(batch_size=2, input_shape=(8, 4), normalize=None)

    assert base.preprocess(data).shape == (8, 4)

    with pytest.raises(NotImplementedError):
        base._build()

    batch = BatchDataset(data, labels, batch_size=2, normalize=None)
    balanced = BalancedPairDataset(
        data,
        labels,
        n_pairs=4,
        batch_size=2,
        normalize=None,
        shuffle=False,
    )
    random = RandomPairDataset(data, labels, batch_size=2, normalize=None)

    assert batch.batches.element_spec
    assert len(balanced.create_pairs(data, labels)) == 3
    assert len(random.create_pairs(data, labels)) == 3


@pytest.mark.parametrize("dataset_class", [BatchDataset, BalancedPairDataset, RandomPairDataset])
def test_original_datasets_normalize_constant_data(dataset_class):
    data = np.full((4, 2), 5.0)
    labels = np.array([0, 0, 1, 1])
    dataset = dataset_class(data, labels, batch_size=2, normalize=(-1.0, 1.0))

    for batch in dataset.batches:
        for samples in batch[:-1]:
            np.testing.assert_array_equal(samples, -np.ones(samples.shape))


def test_original_loss_classes():
    assert BinaryCrossEntropy()(tf.zeros(1), tf.zeros(1)).numpy() == 0.0
    assert ContrastiveLoss()(tf.zeros(1), tf.zeros(1)).numpy() == 1.0
    assert TripletHardLoss()(tf.zeros((1, 1)), tf.zeros((1, 1))).numpy() == 1.0
    assert TripletSemiHardLoss()(tf.zeros((2, 1)), tf.zeros((2, 1))).numpy() == 1.0


def test_original_base_models_and_paths():
    assert MLP(n_hidden=(8,))(tf.ones((2, 4))).shape == (2, 8)
    assert CNN(n_blocks=2, init_kernel=3, n_output=8)(tf.ones((2, 16, 16, 1))).shape == (2, 8)

    inputs = tf.zeros((2, 5), dtype=tf.int32)

    for model in (RNN(10), GRU(10), LSTM(10)):
        assert model(inputs).shape == (2, 5, 10)

    with pytest.raises(NotImplementedError):
        Base()(tf.ones((1, 4)))


def test_original_siamese_contract():
    base = MLP(n_hidden=(4,))
    model = Siamese(base)

    assert model.B is base
    assert model.extract_embeddings(tf.ones((2, 4))).shape == (2, 4)

    with pytest.raises(NotImplementedError):
        model.compile()


def test_original_concrete_model_methods():
    data, labels = _data()
    pairs = BalancedPairDataset(
        data,
        labels,
        n_pairs=8,
        batch_size=4,
        normalize=None,
        shuffle=False,
    )

    for model in (
        ContrastiveSiamese(MLP(n_hidden=(4,))),
        CrossEntropySiamese(MLP(n_hidden=(4,))),
    ):
        model.compile(optimizer="adam")
        model.fit(pairs.batches, epochs=1, verbose=0)
        model.evaluate(pairs.batches, verbose=0)

        left = tf.ones((2, 4))
        right = tf.zeros((2, 4))

        assert model.predict(left, right).shape == (2,)

    batches = BatchDataset(
        data,
        labels,
        batch_size=4,
        normalize=None,
        shuffle=False,
    )
    triplet = TripletSiamese(MLP(n_hidden=(4,)))

    assert triplet.distance == "squared-L2"

    triplet.compile(optimizer="adam")
    triplet.fit(batches.batches, epochs=1, verbose=0)
    triplet.evaluate(batches.batches, verbose=0)

    assert triplet.predict(data[:2], data[2:4]).shape == (2,)


def test_original_utilities(monkeypatch):
    assert constants.BUFFER_SIZE == 100000

    logger = logging.get_logger(__name__)

    assert logger.to_file("compatibility") is None

    for error in (
        exception.ArgumentError,
        exception.BuildError,
        exception.SizeError,
        exception.TypeError,
        exception.ValueError,
    ):
        with pytest.raises(error):
            raise error("`value` is invalid.")

    show = Mock()
    monkeypatch.setattr("matplotlib.pyplot.show", show)
    monkeypatch.setattr("matplotlib.pyplot.get_backend", lambda: "TkAgg")

    assert projector._tensor_to_numpy(tf.zeros(1)).shape == (1,)
    assert projector.plot_embeddings(tf.ones((2, 2)), tf.constant([0, 1])) is None
    show.assert_called_once_with()


def test_error_diagnostic_keeps_category_and_message(monkeypatch):
    diagnostic = Mock()
    monkeypatch.setattr(exception.logger, "error", diagnostic)
    message = "`value` is None."

    error = exception.ValueError(message)

    assert str(error) == f"ValueError: {message}"
    diagnostic.assert_called_once_with(f"`exception=ValueError` was raised with message {message!r}.")
