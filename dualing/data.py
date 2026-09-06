# Copyright (c) 2020-2026 Gustavo Rosa.
# Licensed under the Apache License, Version 2.0.

"""Dataset helpers built on tf.data."""

import numpy as np
import tensorflow as tf


def _length(values: tf.Tensor) -> int:
    size = values.shape[0]

    return int(size) if size is not None else int(tf.shape(values)[0].numpy())


def _batch(values, size: int, batch_size: int, shuffle: bool, seed: int):
    if batch_size < 1:
        raise ValueError("`batch_size` must be greater than zero.")

    dataset = tf.data.Dataset.from_tensor_slices(values)

    if shuffle:
        dataset = dataset.shuffle(size, seed=seed)

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def preprocess(
    data,
    input_shape: tuple[int, ...] | None = None,
    normalize: tuple[float, float] | None = (0.0, 1.0),
) -> tf.Tensor:
    """Convert numeric data to a reshaped and globally normalized float tensor.

    Scaling uses one minimum and maximum for the entire input, not separate feature statistics.
    Constant data maps to the lower bound. TensorFlow conversion and reshape errors propagate.

    Args:
        data: Numeric array-like values or an eager tensor.
        input_shape: Full output shape with optional -1 inference, or None to preserve the input shape.
        normalize: Increasing lower/upper bounds, or None to disable scaling.

    Returns:
        Float32 tensor with the configured shape and global scaling.

    Raises:
        ValueError: Normalization bounds are not an increasing pair.

    """

    data = tf.cast(tf.convert_to_tensor(data), tf.float32)

    if input_shape is not None:
        data = tf.reshape(data, input_shape)

    if normalize is not None:
        if len(normalize) != 2 or normalize[0] >= normalize[1]:
            raise ValueError("`normalize` must contain increasing lower and upper bounds.")

        lower, upper = normalize
        minimum = tf.reduce_min(data)
        scaled = tf.math.divide_no_nan(data - minimum, tf.reduce_max(data) - minimum)

        data = scaled * (upper - lower) + lower

    return data


def batch_dataset(
    data,
    labels,
    batch_size: int = 1,
    input_shape: tuple[int, ...] | None = None,
    normalize: tuple[float, float] | None = (0.0, 1.0),
    shuffle: bool = True,
    seed: int = 0,
) -> tf.data.Dataset:
    """Create an eager dataset of preprocessed sample and label batches.

    The dataset yields (samples, labels), keeps the label dtype, and retains the final partial batch.

    Args:
        data: Numeric samples, with the sample dimension first.
        labels: Targets whose first dimension matches the preprocessed data.
        batch_size: Positive maximum number of samples per batch.
        input_shape: Full data shape passed to ``preprocess``, or None.
        normalize: Global scaling bounds passed to ``preprocess``, or None.
        shuffle: Shuffle before batching, reshuffling on each traversal.
        seed: Seed passed to TensorFlow shuffling.

    Returns:
        Prefetched tf.data.Dataset containing float32 samples and their labels.

    Raises:
        ValueError: Sample counts differ, batch_size is nonpositive, or normalization bounds are invalid.

    """

    data = preprocess(data, input_shape, normalize)

    if _length(data) != _length(tf.convert_to_tensor(labels)):
        raise ValueError("`data` and `labels` must contain the same number of samples.")

    return _batch((data, labels), _length(data), batch_size, shuffle, seed)


def balanced_pair_dataset(
    data,
    labels,
    n_pairs: int = 2,
    batch_size: int = 1,
    input_shape: tuple[int, ...] | None = None,
    normalize: tuple[float, float] | None = (0.0, 1.0),
    shuffle: bool = True,
    seed: int = 0,
) -> tf.data.Dataset:
    """Create an equal number of similar and dissimilar sample pairs.

    Targets are 1 for similar pairs and 0 for dissimilar pairs. Without shuffling, similar pairs come first.
    Classes are sampled uniformly. Sampling permits repeated samples and self-pairs.

    Args:
        data: Numeric samples, with the sample dimension first.
        labels: One class label per sample, flattened before pairing.
        n_pairs: Positive even number of pairs to generate.
        batch_size: Positive maximum number of pairs per batch.
        input_shape: Full data shape passed to ``preprocess``, or None.
        normalize: Global scaling bounds passed to ``preprocess``, or None.
        shuffle: Shuffle the generated pairs, reshuffling on each traversal.
        seed: Seed for local pair sampling and TensorFlow shuffling.

    Returns:
        Prefetched dataset yielding ((left, right), targets), with all components in float32.

    Raises:
        ValueError: Pair counts, sample counts, class diversity, preprocessing, or batching options are invalid.

    """

    if n_pairs < 2 or n_pairs % 2:
        raise ValueError("`n_pairs` must be a positive even number.")

    data = preprocess(data, input_shape, normalize)
    labels = np.asarray(labels).reshape(-1)

    if _length(data) != labels.size:
        raise ValueError("`data` and `labels` must contain the same number of samples.")

    classes = np.unique(labels)

    if classes.size < 2:
        raise ValueError("`labels` must contain at least two classes.")

    rng = np.random.default_rng(seed)
    indices = {label: np.flatnonzero(labels == label) for label in classes}

    first, second, targets = [], [], []

    for _ in range(n_pairs // 2):
        label = rng.choice(classes)
        left, right = rng.choice(indices[label], size=2, replace=True)
        first.append(left)
        second.append(right)
        targets.append(1.0)

    for _ in range(n_pairs // 2):
        left_label, right_label = rng.choice(classes, size=2, replace=False)
        first.append(rng.choice(indices[left_label]))
        second.append(rng.choice(indices[right_label]))
        targets.append(0.0)

    pairs = (tf.gather(data, first), tf.gather(data, second))

    return _batch(
        (pairs, tf.convert_to_tensor(targets, dtype=tf.float32)),
        n_pairs,
        batch_size,
        shuffle,
        seed,
    )


def random_pair_dataset(
    data,
    labels,
    batch_size: int = 1,
    input_shape: tuple[int, ...] | None = None,
    normalize: tuple[float, float] | None = (0.0, 1.0),
    shuffle: bool = False,
    seed: int = 0,
) -> tf.data.Dataset:
    """Randomly pair samples without reusing a sample within this dataset.

    There are floor(n_samples / 2) pairs, with one unused sample for odd input counts.
    Targets are 1 for equal labels and 0 otherwise. Pair counts are not balanced by class.

    Args:
        data: At least two numeric samples, with the sample dimension first.
        labels: One class label per sample, flattened before pairing.
        batch_size: Positive maximum number of pairs per batch.
        input_shape: Full data shape passed to ``preprocess``, or None.
        normalize: Global scaling bounds passed to ``preprocess``, or None.
        shuffle: Also shuffle the generated pairs on each traversal.
        seed: Seed for the sample permutation and optional shuffling.

    Returns:
        Prefetched dataset yielding ((left, right), targets), with all components in float32.

    Raises:
        ValueError: Sample counts differ, fewer than two samples exist, or preprocessing/batching is invalid.

    """

    data = preprocess(data, input_shape, normalize)
    labels = tf.reshape(tf.convert_to_tensor(labels), [-1])
    size = _length(data)

    if size != _length(labels):
        raise ValueError("`data` and `labels` must contain the same number of samples.")

    n_pairs = size // 2

    if not n_pairs:
        raise ValueError("`data` must contain at least two samples.")

    indices = np.random.default_rng(seed).permutation(size)[: 2 * n_pairs]
    left, right = indices[:n_pairs], indices[n_pairs:]

    pairs = (tf.gather(data, left), tf.gather(data, right))
    targets = tf.cast(tf.equal(tf.gather(labels, left), tf.gather(labels, right)), tf.float32)

    return _batch((pairs, targets), n_pairs, batch_size, shuffle, seed)
