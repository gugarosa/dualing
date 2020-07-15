import numpy as np
from dualing.datasets import batch


def test_batch_dataset_batches():
    x = np.zeros((1, 784))
    y = np.zeros(1)

    new_batch_dataset = batch.BatchDataset(x, y)

    assert new_batch_dataset.batches != None


def test_batch_dataset_batches_setter():
    x = np.zeros((1, 784))
    y = np.zeros(1)

    new_batch_dataset = batch.BatchDataset(x, y)

    try:
        new_batch_dataset.batches = 1
    except:
        pass

    assert new_batch_dataset.batches != 1


def test_batch_dataset_build():
    x = np.zeros((1, 784))
    y = np.zeros(1)

    new_batch_dataset = batch.BatchDataset(x, y, shuffle=True)

    assert new_batch_dataset.batches != None

    new_batch_dataset = batch.BatchDataset(x, y, shuffle=False)

    assert new_batch_dataset.batches != None
