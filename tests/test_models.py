import numpy as np

from dualing.data import balanced_pair_dataset, batch_dataset
from dualing.embedders import MLP
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
