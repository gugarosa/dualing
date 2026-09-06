# Copyright (c) 2020-2026 Gustavo Rosa.
# Licensed under the Apache License, Version 2.0.

import numpy as np
import pytest

from dualing.data import (
    balanced_pair_dataset,
    batch_dataset,
    preprocess,
    random_pair_dataset,
)


def test_preprocess_and_batch_dataset():
    data = np.arange(12).reshape(4, 3)

    processed = preprocess(data, (2, 6), (-1.0, 1.0))

    assert processed.shape == (2, 6)
    assert np.isclose(processed.numpy().min(), -1.0)
    assert np.isclose(processed.numpy().max(), 1.0)

    samples, labels = next(iter(batch_dataset(data, np.arange(4), batch_size=2, shuffle=False)))

    assert samples.shape == (2, 3)
    assert labels.numpy().tolist() == [0, 1]


def test_balanced_pair_dataset():
    data = np.arange(24).reshape(8, 3)
    labels = np.repeat([0, 1], 4)

    (_, _), targets = next(
        iter(
            balanced_pair_dataset(
                data,
                labels,
                n_pairs=6,
                batch_size=6,
                normalize=None,
                shuffle=False,
            )
        )
    )

    assert targets.numpy().tolist() == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]

    with pytest.raises(ValueError):
        balanced_pair_dataset(data, np.zeros(8), normalize=None)


def test_random_pair_dataset():
    data = np.arange(15).reshape(5, 3)
    labels = np.arange(5)

    (left, right), targets = next(iter(random_pair_dataset(data, labels, batch_size=2, normalize=None, shuffle=False)))

    assert left.shape == right.shape == (2, 3)
    assert targets.shape == (2,)


@pytest.mark.parametrize("bounds", [(0.0, 1.0), (-1.0, 1.0), None])
def test_constant_preprocessing(bounds):
    data = np.full((4, 2), 5.0)
    expected = 5.0 if bounds is None else bounds[0]

    np.testing.assert_array_equal(preprocess(data, normalize=bounds), np.full((4, 2), expected))
