"""Dataset helpers built on ``tf.data``."""

import numpy as np
import tensorflow as tf


def _length(values: tf.Tensor) -> int:
    size = values.shape[0]

    return int(size) if size is not None else int(tf.shape(values)[0].numpy())


def _batch(values, size: int, batch_size: int, shuffle: bool, seed: int):
    if batch_size < 1:
        raise ValueError("batch_size must be greater than zero")

    dataset = tf.data.Dataset.from_tensor_slices(values)

    if shuffle:
        dataset = dataset.shuffle(size, seed=seed)

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def preprocess(
    data,
    input_shape: tuple[int, ...] | None = None,
    normalize: tuple[float, float] | None = (0.0, 1.0),
) -> tf.Tensor:
    """Convert to float tensors, optionally reshaping and normalizing.

    Constant data maps to the lower normalization bound.
    """

    data = tf.cast(tf.convert_to_tensor(data), tf.float32)

    if input_shape is not None:
        data = tf.reshape(data, input_shape)

    if normalize is not None:
        if len(normalize) != 2 or normalize[0] >= normalize[1]:
            raise ValueError("normalize must contain increasing lower and upper bounds")

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
    """Create a batched dataset of samples and labels."""

    data = preprocess(data, input_shape, normalize)

    if _length(data) != _length(tf.convert_to_tensor(labels)):
        raise ValueError("data and labels must contain the same number of samples")

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
    """Create equal numbers of similar and dissimilar sample pairs."""

    if n_pairs < 2 or n_pairs % 2:
        raise ValueError("n_pairs must be a positive even number")

    data = preprocess(data, input_shape, normalize)
    labels = np.asarray(labels).reshape(-1)

    if _length(data) != labels.size:
        raise ValueError("data and labels must contain the same number of samples")

    classes = np.unique(labels)

    if classes.size < 2:
        raise ValueError("labels must contain at least two classes")

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
    """Create random disjoint sample pairs."""

    data = preprocess(data, input_shape, normalize)
    labels = tf.reshape(tf.convert_to_tensor(labels), [-1])
    size = _length(data)

    if size != _length(labels):
        raise ValueError("data and labels must contain the same number of samples")

    n_pairs = size // 2

    if not n_pairs:
        raise ValueError("at least two samples are required")

    indices = np.random.default_rng(seed).permutation(size)[: 2 * n_pairs]
    left, right = indices[:n_pairs], indices[n_pairs:]

    pairs = (tf.gather(data, left), tf.gather(data, right))
    targets = tf.cast(
        tf.equal(tf.gather(labels, left), tf.gather(labels, right)), tf.float32
    )

    return _batch((pairs, targets), n_pairs, batch_size, shuffle, seed)
