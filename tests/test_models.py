import numpy as np
import pytest
import tensorflow as tf

from dualing.data import balanced_pair_dataset, batch_dataset
from dualing.embedders import GRU, LSTM, MLP, RNN
from dualing.models import ContrastiveSiamese, CrossEntropySiamese, TripletSiamese


def _pairs():
    data = np.arange(32, dtype="float32").reshape(8, 4)
    labels = np.repeat([0, 1], 4)

    return (
        data,
        labels,
        balanced_pair_dataset(
            data,
            labels,
            n_pairs=8,
            batch_size=4,
            normalize=None,
            shuffle=False,
        ),
    )


def test_pair_models_use_native_keras_training():
    _, _, pairs = _pairs()

    for model in (
        ContrastiveSiamese(MLP((4,))),
        CrossEntropySiamese(MLP((4,))),
    ):
        model.compile(optimizer="adam")

        history = model.fit(pairs, epochs=1, verbose=0, shuffle=False)
        pair_inputs, _ = next(iter(pairs))
        predictions = model.predict(pair_inputs, batch_size=4, verbose=0)

        assert "loss" in history.history
        assert predictions.shape == (4,)


def test_triplet_model_uses_native_keras_training():
    data, labels, _ = _pairs()

    dataset = batch_dataset(
        data,
        labels,
        batch_size=4,
        normalize=None,
        shuffle=False,
    )

    model = TripletSiamese(MLP((4,)), margin=1.0)

    model.compile(optimizer="adam")

    history = model.fit(dataset, epochs=1, verbose=0, shuffle=False)
    predictions = model.predict(data[:2], batch_size=2, verbose=0)

    assert "loss" in history.history
    assert predictions.shape == (2, 4)
    assert model.compare(data[:2], data[2:4]).shape == (2,)


@pytest.mark.parametrize("model_class", [ContrastiveSiamese, CrossEntropySiamese])
@pytest.mark.parametrize("as_dataset", [False, True])
def test_pair_models_respect_sample_weights(model_class, as_dataset):
    left = np.array([[0.0], [0.2], [0.4], [0.6]], dtype="float32")
    right = np.array([[0.1], [0.5], [0.8], [1.2]], dtype="float32")
    labels = np.array([1.0, 0.0, 1.0, 0.0], dtype="float32")
    weights = np.array([0.0, 0.0, 0.0, 4.0], dtype="float32")
    base = tf.keras.Sequential(
        [
            tf.keras.layers.Input((1,)),
            tf.keras.layers.Dense(1, use_bias=False, kernel_initializer="ones"),
        ]
    )
    model = model_class(base)
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.0))
    predictions = model((left, right)).numpy()

    if model_class is ContrastiveSiamese:
        losses = (
            labels * predictions**2 + (1 - labels) * np.maximum(1 - predictions, 0) ** 2
        )
    else:
        losses = -labels * np.log(predictions) - (1 - labels) * np.log1p(-predictions)

    expected = np.mean(losses * weights)

    if as_dataset:
        data = tf.data.Dataset.from_tensor_slices(
            ((left, right), labels, weights)
        ).batch(4)
        fit_options = {"x": data, "validation_data": data}
        evaluation_options = {"x": data}
    else:
        evaluation_options = {
            "x": (left, right),
            "y": labels,
            "sample_weight": weights,
            "batch_size": 4,
        }
        fit_options = {
            **evaluation_options,
            "validation_data": ((left, right), labels, weights),
        }

    history = model.fit(epochs=1, verbose=0, shuffle=False, **fit_options)
    result = model.evaluate(verbose=0, return_dict=True, **evaluation_options)

    np.testing.assert_allclose(history.history["loss"], [expected], rtol=1e-5)
    np.testing.assert_allclose(history.history["val_loss"], [expected], rtol=1e-5)
    np.testing.assert_allclose(result["loss"], expected, rtol=1e-5)

    model.reset_metrics()
    model.step(tf.constant(left), tf.constant(right), tf.constant(labels))

    np.testing.assert_allclose(model.loss_metric.result(), np.mean(losses), rtol=1e-5)

    if model_class is CrossEntropySiamese:
        expected_accuracy = np.mean(labels == (predictions > 0.5))

        np.testing.assert_allclose(history.history["acc"], [expected_accuracy])
        np.testing.assert_allclose(result["acc"], expected_accuracy)
        np.testing.assert_allclose(model.acc_metric.result(), expected_accuracy)


def test_triplet_model_reports_the_optimized_loss_after_recompiling():
    data = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]], "float32")
    labels = np.array([0, 0, 1, 1])
    base = tf.keras.Sequential(
        [
            tf.keras.layers.Input((2,)),
            tf.keras.layers.Dense(
                2, use_bias=False, kernel_initializer=tf.keras.initializers.Identity()
            ),
        ]
    )
    model = TripletSiamese(base, mining="hard")
    normalized = data / np.linalg.norm(data, axis=-1, keepdims=True)
    distances = np.linalg.norm(normalized[:, None] - normalized[None, :], axis=-1)
    expected = np.mean(
        [
            max(
                max(distances[anchor, labels == label])
                - min(distances[anchor, labels != label])
                + 1.0,
                0,
            )
            for anchor, label in enumerate(labels)
        ]
    )
    dataset = tf.data.Dataset.from_tensor_slices((data, labels)).batch(4)

    for _ in range(2):
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.0))
        history = model.fit(dataset, epochs=1, verbose=0)
        result = model.evaluate(dataset, verbose=0)

        np.testing.assert_allclose(history.history["loss"], [expected], rtol=1e-6)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

        model.reset_metrics()
        model.step(tf.constant(data), tf.constant(labels))

        np.testing.assert_allclose(model.loss_metric.result(), expected, rtol=1e-6)


@pytest.mark.parametrize("model_class", [ContrastiveSiamese, CrossEntropySiamese])
def test_pair_models_accept_legacy_validation_data(model_class):
    _, _, pairs = _pairs()
    (left, right), labels = next(iter(pairs))
    legacy = tf.data.Dataset.from_tensors((left, right, labels))
    model = model_class(MLP((4,)))
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.0))

    history = model.fit(legacy, validation_data=legacy, epochs=1, verbose=0)
    result = model.evaluate(legacy, verbose=0, return_dict=True)

    assert np.isfinite(history.history["val_loss"]).all()
    np.testing.assert_allclose(history.history["val_loss"], [result["loss"]])


@pytest.mark.parametrize("base_class", [RNN, GRU, LSTM])
def test_legacy_triplet_prediction_reduces_sequence_embeddings(base_class):
    left = tf.constant([[1, 2, 3], [2, 3, 4]])
    right = tf.constant([[3, 2, 1], [1, 4, 2]])
    base = base_class(5, embedding_size=3, hidden_size=4)
    model = TripletSiamese(base)
    left_embedding = np.mean(base(left).numpy(), axis=1)
    right_embedding = np.mean(base(right).numpy(), axis=1)
    expected = np.square(left_embedding - right_embedding).sum(axis=-1)

    predictions = model.predict(left, right)

    assert predictions.shape == (2,)
    np.testing.assert_allclose(predictions, expected, rtol=1e-5)


@pytest.mark.parametrize(
    "model_class", [ContrastiveSiamese, CrossEntropySiamese, TripletSiamese]
)
def test_siamese_argument_compatibility(model_class):
    data, labels, pairs = _pairs()

    if model_class is TripletSiamese:
        inputs = tf.constant(data)
        targets = tf.constant(labels)
        dataset = tf.data.Dataset.from_tensors((inputs, targets))
    else:
        inputs, targets = next(iter(pairs))
        dataset = tf.data.Dataset.from_tensors((*inputs, targets))

    model = model_class(MLP((4,)))
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.0))

    history = model.fit(dataset, 1, verbose=0)

    assert history.epoch == [0]

    history = model.fit(
        inputs, targets, epochs=1, batch_size=8, shuffle=False, verbose=0
    )
    predictions = model.predict(x=inputs, batch_size=8, verbose=0)

    assert history.epoch == [0]
    np.testing.assert_allclose(
        predictions, model(inputs, training=False), rtol=1e-6, atol=1e-7
    )

    with pytest.raises(TypeError, match="epochs"):
        model.fit(dataset, 1, epochs=100, verbose=0)

    with pytest.raises(ValueError):
        model.fit(dataset, y=1, epochs=1, verbose=0)

    with pytest.raises(TypeError, match="x"):
        model.predict(inputs, x=inputs, verbose=0)
