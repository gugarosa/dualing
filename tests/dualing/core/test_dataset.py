import numpy as np
import pytest

from dualing.core import dataset


def test_dataset_batch_size():
    new_dataset = dataset.Dataset()

    assert new_dataset.batch_size == 1


def test_dataset_batch_size_setter():
    try:
        new_dataset = dataset.Dataset(batch_size=1.5)
    except:
        new_dataset = dataset.Dataset()

    try:
        new_dataset = dataset.Dataset(batch_size=-1)
    except:
        new_dataset = dataset.Dataset()

    assert new_dataset.batch_size == 1


def test_dataset_input_shape():
    new_dataset = dataset.Dataset()

    assert new_dataset.input_shape is None


def test_dataset_input_shape_setter():
    try:
        new_dataset = dataset.Dataset(input_shape=1.5)
    except:
        new_dataset = dataset.Dataset()

    assert new_dataset.input_shape is None


def test_dataset_normalize():
    new_dataset = dataset.Dataset()

    assert new_dataset.normalize == (0, 1)


def test_dataset_normalize_setter():
    try:
        new_dataset = dataset.Dataset(normalize=1.5)
    except:
        new_dataset = dataset.Dataset()

    assert new_dataset.normalize == (0, 1)


def test_dataset_shuffle():
    new_dataset = dataset.Dataset()

    assert new_dataset.shuffle is True


def test_dataset_shuffle_setter():
    try:
        new_dataset = dataset.Dataset(shuffle=1.5)
    except:
        new_dataset = dataset.Dataset()

    assert new_dataset.shuffle is True


def test_dataset_preprocess():
    new_dataset = dataset.Dataset(input_shape=(2, 3))

    data = np.asarray([[1, 1, 1], [2, 2, 2]])

    proc_data = new_dataset.preprocess(data)

    assert proc_data.numpy()[0][0] == 0

    new_dataset = dataset.Dataset()

    proc_data = new_dataset.preprocess(data)

    assert proc_data.numpy()[0][0] == 0


def test_dataset_build():
    new_dataset = dataset.Dataset()

    with pytest.raises(NotImplementedError):
        new_dataset._build()
