"""Contrastive and triplet losses."""

import tensorflow as tf


def pair_distance(left: tf.Tensor, right: tf.Tensor, metric: str = "L2") -> tf.Tensor:
    """Compute row-wise distances between two embedding tensors."""

    if metric == "L1":
        return tf.reduce_sum(tf.abs(left - right), axis=-1)
    if metric == "L2":
        return tf.linalg.norm(left - right, axis=-1)
    if metric == "squared-L2":
        return tf.reduce_sum(tf.square(left - right), axis=-1)
    if metric == "angular":
        left = tf.math.l2_normalize(left, axis=-1)
        right = tf.math.l2_normalize(right, axis=-1)
        return 1.0 - tf.reduce_sum(left * right, axis=-1)
    raise ValueError("metric must be L1, L2, squared-L2, or angular")


def pairwise_distances(embeddings: tf.Tensor, metric: str = "L2") -> tf.Tensor:
    """Compute all pairwise distances in an embedding batch."""

    if metric == "L1":
        return tf.reduce_sum(
            tf.abs(embeddings[:, tf.newaxis] - embeddings[tf.newaxis, :]), axis=-1
        )
    if metric == "angular":
        embeddings = tf.math.l2_normalize(embeddings, axis=-1)
        return 1.0 - tf.linalg.matmul(embeddings, embeddings, transpose_b=True)
    if metric not in {"L2", "squared-L2"}:
        raise ValueError("metric must be L1, L2, squared-L2, or angular")

    products = tf.linalg.matmul(embeddings, embeddings, transpose_b=True)
    squared_norms = tf.linalg.diag_part(products)
    distances = tf.maximum(
        squared_norms[:, tf.newaxis] - 2.0 * products + squared_norms[tf.newaxis, :],
        0.0,
    )
    if metric == "squared-L2":
        return distances
    return tf.where(
        distances > 0,
        tf.sqrt(tf.maximum(distances, tf.keras.backend.epsilon())),
        0.0,
    )


def contrastive_loss(
    y_true: tf.Tensor, y_pred: tf.Tensor, margin: float = 1.0
) -> tf.Tensor:
    """Return the standard contrastive loss for pair labels and distances."""

    y_true = tf.reshape(tf.cast(y_true, y_pred.dtype), tf.shape(y_pred))
    return y_true * tf.square(y_pred) + (1.0 - y_true) * tf.square(
        tf.nn.relu(margin - y_pred)
    )


def _triplet_masks(labels: tf.Tensor):
    labels = tf.reshape(labels, [-1])
    same = tf.equal(labels[:, tf.newaxis], labels[tf.newaxis, :])
    positive = tf.logical_and(
        same, tf.logical_not(tf.eye(tf.shape(labels)[0], dtype=tf.bool))
    )
    return positive, tf.logical_not(same)


def _masked_mean(values: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    values = tf.where(mask, values, 0.0)
    return tf.math.divide_no_nan(
        tf.reduce_sum(values), tf.reduce_sum(tf.cast(mask, values.dtype))
    )


def triplet_hard_loss(
    labels: tf.Tensor,
    embeddings: tf.Tensor,
    margin: float = 1.0,
    soft: bool = False,
    metric: str = "L2",
) -> tf.Tensor:
    """Use the hardest positive and negative for each valid anchor."""

    distances = pairwise_distances(embeddings, metric)
    positive, negative = _triplet_masks(labels)
    hardest_positive = tf.reduce_max(tf.where(positive, distances, 0.0), axis=1)
    fallback = tf.reduce_max(distances, axis=1, keepdims=True) + margin + 1.0
    hardest_negative = tf.reduce_min(tf.where(negative, distances, fallback), axis=1)
    valid = tf.logical_and(
        tf.reduce_any(positive, axis=1), tf.reduce_any(negative, axis=1)
    )
    difference = hardest_positive - hardest_negative
    losses = tf.nn.softplus(difference) if soft else tf.nn.relu(difference + margin)
    return _masked_mean(losses, valid)


def triplet_semihard_loss(
    labels: tf.Tensor,
    embeddings: tf.Tensor,
    margin: float = 1.0,
    soft: bool = False,
    metric: str = "L2",
) -> tf.Tensor:
    """Use the nearest negative farther away than each positive pair."""

    distances = pairwise_distances(embeddings, metric)
    positive, negative = _triplet_masks(labels)
    positive_distances = distances[:, :, tf.newaxis]
    negative_distances = distances[:, tf.newaxis, :]
    negative_candidates = negative[:, tf.newaxis, :]
    farther = tf.logical_and(
        negative_candidates, negative_distances > positive_distances
    )

    fallback_value = tf.reduce_max(distances) + margin + 1.0
    nearest_farther = tf.reduce_min(
        tf.where(farther, negative_distances, fallback_value), axis=2
    )
    farthest_negative = tf.reduce_max(
        tf.where(negative_candidates, negative_distances, 0.0), axis=2
    )
    selected_negative = tf.where(
        tf.reduce_any(farther, axis=2), nearest_farther, farthest_negative
    )

    difference = distances - selected_negative
    losses = tf.nn.softplus(difference) if soft else tf.nn.relu(difference + margin)
    valid = tf.logical_and(positive, tf.reduce_any(negative, axis=1)[:, tf.newaxis])
    return _masked_mean(losses, valid)
