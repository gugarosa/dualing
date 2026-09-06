# Copyright (c) 2020-2026 Gustavo Rosa.
# Licensed under the Apache License, Version 2.0.

import json
import subprocess
import sys
from textwrap import dedent

import numpy as np
import pytest
import tensorflow as tf

from dualing import (
    CNN,
    GRU,
    LSTM,
    MLP,
    RNN,
    ContrastiveSiamese,
    CrossEntropySiamese,
    TripletSiamese,
)
from dualing.core import (
    BinaryCrossEntropy,
    ContrastiveLoss,
    TripletHardLoss,
    TripletSemiHardLoss,
)
from dualing.losses import (
    contrastive_loss,
    triplet_hard_loss,
    triplet_semihard_loss,
)

EMBEDDERS = [
    (MLP, {"n_hidden": (7, 3), "activation": "tanh"}, (2, 4)),
    (
        CNN,
        {"n_blocks": 2, "init_kernel": 3, "n_output": 3, "activation": "linear"},
        (2, 8, 8, 1),
    ),
    (RNN, {"vocab_size": 6, "embedding_size": 3, "hidden_size": 4}, (2, 5)),
    (GRU, {"vocab_size": 6, "embedding_size": 3, "hidden_size": 4}, (2, 5)),
    (LSTM, {"vocab_size": 6, "embedding_size": 3, "hidden_size": 4}, (2, 5)),
]


@pytest.mark.parametrize("model_class,options,shape", EMBEDDERS)
def test_embedders_clone_and_reload(model_class, options, shape, tmp_path):
    model = model_class(**options, name="saved_embedder")
    inputs = tf.ones(shape)
    expected = model(inputs, training=False).numpy()

    clone = tf.keras.models.clone_model(model)
    clone.set_weights(model.get_weights())

    assert type(clone) is model_class
    assert clone.get_config() == model.get_config()
    np.testing.assert_allclose(clone(inputs), expected, rtol=1e-6, atol=1e-7)

    path = tmp_path / "embedder.keras"
    model.save(path)
    restored = tf.keras.models.load_model(path)

    assert type(restored) is model_class
    assert restored.get_config() == model.get_config()
    np.testing.assert_allclose(restored(inputs), expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("model_class,options,shape", EMBEDDERS)
def test_embedders_accept_native_keras_options(model_class, options, shape):
    model = model_class(**options, name="configured_embedder", trainable=False, dtype="float64")

    assert model(tf.ones(shape)).dtype == tf.float64
    assert not model.trainable_variables

    clone = tf.keras.models.clone_model(model)

    assert clone.name == "configured_embedder"
    assert not clone.trainable
    assert clone.dtype_policy.name == "float64"


def test_aliases_have_one_canonical_configuration():
    mlp = MLP((1,), hidden_units=(7, 3))
    cnn = CNN(blocks=2, kernel_size=3, embedding_dim=4)
    rnn = RNN(6, embedding_dim=3, hidden_units=4)

    assert tuple(mlp.get_config()["n_hidden"]) == (7, 3)
    assert "hidden_units" not in mlp.get_config()
    assert cnn.get_config()["n_blocks"] == 2
    assert cnn.get_config()["init_kernel"] == 3
    assert cnn.get_config()["n_output"] == 4
    assert "blocks" not in cnn.get_config()
    assert rnn.get_config()["vocab_size"] == 6
    assert rnn.get_config()["embedding_size"] == 3
    assert rnn.get_config()["hidden_size"] == 4
    assert "embedding_dim" not in rnn.get_config()


@pytest.mark.parametrize(
    "loss,labels,predictions",
    [
        (BinaryCrossEntropy(), [[0.0], [1.0]], [[0.2], [0.8]]),
        (ContrastiveLoss(1.3), [1.0, 0.0], [0.2, 0.8]),
        (
            TripletHardLoss(0.7, True, "angular"),
            [0, 0, 1, 1],
            [[0.1, 0.3], [0.4, 0.2], [1.0, 0.1], [1.0, 0.3]],
        ),
        (
            TripletSemiHardLoss(0.7, True, "angular"),
            [0, 0, 1, 1],
            [[0.1, 0.3], [0.4, 0.2], [1.0, 0.1], [1.0, 0.3]],
        ),
    ],
)
def test_callable_loss_configuration_roundtrip(loss, labels, predictions):
    config = tf.keras.utils.serialize_keras_object(loss)
    restored = tf.keras.utils.deserialize_keras_object(config)
    labels = tf.constant(labels)
    predictions = tf.constant(predictions)

    assert type(restored) is type(loss)
    assert restored.get_config() == loss.get_config()
    np.testing.assert_allclose(restored(labels, predictions), loss(labels, predictions))


@pytest.mark.parametrize("loss", [contrastive_loss, triplet_hard_loss, triplet_semihard_loss])
def test_functional_losses_are_registered(loss):
    config = tf.keras.utils.serialize_keras_object(loss)

    assert tf.keras.utils.deserialize_keras_object(config) is loss


@pytest.mark.parametrize(
    "model_class,options",
    [
        (ContrastiveSiamese, {"margin": 1.3, "distance_metric": "L1"}),
        (CrossEntropySiamese, {"merge": "difference"}),
        (
            TripletSiamese,
            {"loss": "semi-hard", "margin": 0.7, "soft": True, "distance_metric": "L2"},
        ),
        (TripletSiamese, {"mining": "hard", "distance_metric": "L1"}),
    ],
)
def test_siamese_models_resume_training(model_class, options, tmp_path):
    model = model_class(MLP((5, 3), activation="tanh"), **options)
    left = tf.constant([[0.1, 0.3], [0.4, 0.2], [1.0, 0.1], [1.0, 0.3]])

    if model_class is TripletSiamese:
        inputs = left
        labels = tf.constant([0, 0, 1, 1])
    else:
        inputs = (left, left + 0.2)
        labels = tf.constant([1.0, 0.0, 1.0, 0.0])

    model.compile(optimizer=tf.keras.optimizers.Adam(0.01))
    model.train_on_batch(inputs, labels)
    expected = model(inputs, training=False).numpy()
    path = tmp_path / "siamese.keras"

    model.save(path)
    restored = tf.keras.models.load_model(path)

    assert type(restored) is model_class
    # Nested Keras build shapes cross a tuple/list boundary in JSON
    assert json.loads(restored.to_json()) == json.loads(model.to_json())
    assert restored.compiled
    assert int(restored.optimizer.iterations.numpy()) == 1
    np.testing.assert_allclose(restored(inputs), expected, rtol=1e-6, atol=1e-7)

    assert len(restored.optimizer.variables) == len(model.optimizer.variables)

    for original, loaded in zip(model.optimizer.variables, restored.optimizer.variables):
        np.testing.assert_allclose(original.numpy(), loaded.numpy(), rtol=1e-6, atol=1e-7)

    model.train_on_batch(inputs, labels)
    restored.train_on_batch(inputs, labels)

    assert int(restored.optimizer.iterations.numpy()) == 2
    np.testing.assert_allclose(restored(inputs), model(inputs), rtol=1e-6, atol=1e-7)


def test_custom_compile_configuration_is_preserved(tmp_path):
    base = tf.keras.Sequential(
        [
            tf.keras.layers.Input((4,)),
            tf.keras.layers.Dense(3, activation="tanh", kernel_regularizer="l2"),
        ]
    )
    model = ContrastiveSiamese(base, margin=1.3)
    inputs = (tf.ones((2, 4)), tf.zeros((2, 4)))
    model.compile(
        optimizer=tf.keras.optimizers.SGD(0.02, momentum=0.8),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="distance_error")],
        run_eagerly=True,
    )
    model.train_on_batch(inputs, tf.constant([1.0, 0.0]))
    path = tmp_path / "custom.keras"
    model.save(path)

    restored = tf.keras.models.load_model(path)

    assert isinstance(restored.optimizer, tf.keras.optimizers.SGD)
    assert isinstance(restored.loss, tf.keras.losses.MeanSquaredError)
    assert restored.run_eagerly
    assert restored.get_compile_config() == model.get_compile_config()


def test_saved_model_loads_in_a_fresh_process(tmp_path):
    model = CrossEntropySiamese(MLP((3,)))
    inputs = (tf.ones((2, 4)), tf.zeros((2, 4)))
    model.compile(optimizer="adam")
    model.train_on_batch(inputs, tf.constant([1.0, 0.0]))
    path = tmp_path / "fresh.keras"
    expected = tmp_path / "expected.npy"
    model.save(path)
    np.save(expected, model(inputs).numpy())
    script = dedent(
        """\
        import sys

        import dualing
        import numpy as np
        import tensorflow as tf

        model = tf.keras.models.load_model(sys.argv[1])
        assert model.compiled
        assert int(model.optimizer.iterations.numpy()) == 1
        actual = model((tf.ones((2, 4)), tf.zeros((2, 4))))
        np.testing.assert_allclose(
            actual, np.load(sys.argv[2]), rtol=1e-6, atol=1e-7
        )
        """
    )

    subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(path),
            str(expected),
        ],
        cwd=tmp_path,
        check=True,
    )
