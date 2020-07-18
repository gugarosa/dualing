import numpy as np

from dualing.datasets import pair


def test_balanced_pair_dataset_n_pairs():
    x = np.zeros((2, 784))

    try:
        y = np.zeros(2)
        new_balanced_pair_dataset = pair.BalancedPairDataset(x, y)
    except:
        y = np.asarray([0, 1])
        new_balanced_pair_dataset = pair.BalancedPairDataset(x, y)

    assert new_balanced_pair_dataset.n_pairs == 2


def test_balanced_pair_dataset_n_pairs_setter():
    x = np.zeros((2, 784))
    y = np.asarray([0, 1])

    new_balanced_pair_dataset = pair.BalancedPairDataset(x, y)

    try:
        new_balanced_pair_dataset.n_pairs = 1.5
    except:
        pass

    assert new_balanced_pair_dataset.n_pairs == 2


def test_balanced_pair_dataset_batches():
    x = np.zeros((2, 784))
    y = np.asarray([0, 1])

    new_balanced_pair_dataset = pair.BalancedPairDataset(x, y)

    assert new_balanced_pair_dataset.batches != None


def test_balanced_pair_dataset_batches_setter():
    x = np.zeros((2, 784))
    y = np.asarray([0, 1])

    new_balanced_pair_dataset = pair.BalancedPairDataset(x, y)

    try:
        new_balanced_pair_dataset.batches = 1
    except:
        pass

    assert new_balanced_pair_dataset.batches != 1


def test_balanced_pair_dataset_create_pairs():
    x = np.zeros((2, 784))
    y = np.asarray([0, 1])

    new_balanced_pair_dataset = pair.BalancedPairDataset(x, y)

    pairs = new_balanced_pair_dataset.create_pairs(x, y)

    assert len(pairs) == 3


def test_balanced_pair_dataset_build():
    x = np.zeros((2, 784))
    y = np.asarray([0, 1])

    new_balanced_pair_dataset = pair.BalancedPairDataset(x, y, shuffle=True)

    pairs = new_balanced_pair_dataset.create_pairs(x, y)

    new_balanced_pair_dataset._build(pairs)

    assert new_balanced_pair_dataset.batches != None

    new_balanced_pair_dataset = pair.BalancedPairDataset(x, y, shuffle=False)

    pairs = new_balanced_pair_dataset.create_pairs(x, y)

    new_balanced_pair_dataset._build(pairs)

    assert new_balanced_pair_dataset.batches != None


def test_random_pair_dataset_batches():
    x = np.zeros((2, 784))
    y = np.asarray([0, 1])

    new_random_pair_dataset = pair.RandomPairDataset(x, y)

    assert new_random_pair_dataset.batches != None


def test_random_pair_dataset_batches_setter():
    x = np.zeros((2, 784))
    y = np.asarray([0, 1])

    new_random_pair_dataset = pair.RandomPairDataset(x, y)

    try:
        new_random_pair_dataset.batches = 1
    except:
        pass

    assert new_random_pair_dataset.batches != 1


def test_random_pair_datasetcreate_pairs():
    x = np.zeros((2, 784))
    y = np.asarray([0, 1])

    new_random_pair_dataset = pair.RandomPairDataset(x, y)

    pairs = new_random_pair_dataset.create_pairs(x, y)

    assert len(pairs) == 3


def test_random_pair_dataset_build():
    x = np.zeros((2, 784))
    y = np.asarray([0, 1])

    new_random_pair_dataset = pair.RandomPairDataset(x, y)

    pairs = new_random_pair_dataset.create_pairs(x, y)

    new_random_pair_dataset._build(pairs)

    assert new_random_pair_dataset.batches != None
