import numpy as np
import pytest
import tensorflow as tf

from dualing.core import BinaryCrossEntropy
from dualing.losses import (
    contrastive_loss,
    pair_distance,
    pairwise_distances,
    triplet_hard_loss,
    triplet_semihard_loss,
)


def test_pair_distances():
    left = tf.constant([[0.0, 0.0], [1.0, 1.0]])
    right = tf.constant([[3.0, 4.0], [1.0, 1.0]])

    assert pair_distance(left, right).numpy().tolist() == [5.0, 0.0]

    distances = pairwise_distances(left)

    assert distances.shape == (2, 2)
    assert np.allclose(tf.linalg.diag_part(distances), 0.0)


def test_contrastive_loss():
    loss = contrastive_loss(
        tf.constant([1.0, 0.0]), tf.constant([0.5, 0.5]), margin=1.0
    )

    assert np.allclose(loss, [0.25, 0.25])


def test_triplet_losses_are_finite():
    labels = tf.constant([0, 0, 1, 1])
    embeddings = tf.constant([[0.0], [0.1], [1.0], [1.1]])

    for loss in (triplet_hard_loss, triplet_semihard_loss):
        assert np.isclose(loss(labels, embeddings, margin=1.0).numpy(), 0.15)


@pytest.mark.parametrize("shape", [(3,), (3, 1), (1, 3)])
def test_binary_crossentropy_reduces_only_the_feature_axis(shape):
    labels = np.array([0.0, 1.0, 0.5], dtype="float32").reshape(shape)
    predictions = np.array([0.2, 0.6, 0.5], dtype="float32").reshape(shape)
    expected = np.mean(
        -labels * np.log(predictions) - (1 - labels) * np.log1p(-predictions),
        axis=-1,
    )

    actual = BinaryCrossEntropy()(tf.constant(labels), tf.constant(predictions))

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_binary_crossentropy_preserves_exact_hard_label_matches():
    labels = tf.constant([[0.0], [1.0]])

    actual = BinaryCrossEntropy()(labels, labels)

    np.testing.assert_array_equal(actual, [0.0, 0.0])


@pytest.mark.parametrize("compiled", [False, True])
def test_pairwise_l2_preserves_small_distances_and_gradients(compiled):
    embeddings = tf.Variable([[0.0], [0.0], [1e-5]])
    distance = tf.function(pairwise_distances) if compiled else pairwise_distances

    with tf.GradientTape() as tape:
        distances = distance(embeddings)
        total = tf.reduce_sum(distances)

    gradient = tape.gradient(total, embeddings)

    np.testing.assert_allclose(
        distances,
        [[0.0, 0.0, 1e-5], [0.0, 0.0, 1e-5], [1e-5, 1e-5, 0.0]],
        rtol=1e-6,
        atol=0,
    )
    np.testing.assert_allclose(gradient, [[-2.0], [-2.0], [4.0]], rtol=1e-6)


@pytest.mark.parametrize("loss_function", [triplet_hard_loss, triplet_semihard_loss])
@pytest.mark.parametrize("metric", ["L1", "L2", "squared-L2", "angular"])
@pytest.mark.parametrize("soft", [False, True])
def test_triplet_losses_match_explicit_mining(loss_function, metric, soft):
    labels = np.array([0, 0, 1, 1, 1, 2])
    embeddings = np.array(
        [[0.1, 0.3], [0.4, 0.2], [1.0, 0.0], [1.0, 0.3], [1.3, 1.0], [2.0, 2.0]]
    )
    differences = embeddings[:, None] - embeddings[None, :]

    if metric == "L1":
        distances = np.abs(differences).sum(axis=-1)
    elif metric == "angular":
        normalized = embeddings / np.linalg.norm(embeddings, axis=-1, keepdims=True)
        distances = 1 - normalized @ normalized.T
    else:
        distances = np.square(differences).sum(axis=-1)

        if metric == "L2":
            distances = np.sqrt(distances)

    losses = []

    for anchor, label in enumerate(labels):
        positives = [
            distances[anchor, other]
            for other in range(len(labels))
            if other != anchor and labels[other] == label
        ]
        negatives = distances[anchor, labels != label]

        if not positives or not len(negatives):
            continue

        if loss_function is triplet_hard_loss:
            differences = [max(positives) - min(negatives)]
        else:
            differences = []

            for positive in positives:
                farther = negatives[negatives > positive]
                selected = min(farther) if len(farther) else max(negatives)
                differences.append(positive - selected)

        losses.extend(
            np.logaddexp(0, difference) if soft else max(difference + 0.5, 0)
            for difference in differences
        )

    actual = loss_function(
        tf.constant(labels),
        tf.constant(embeddings),
        margin=0.5,
        soft=soft,
        metric=metric,
    )

    np.testing.assert_allclose(actual, np.mean(losses), rtol=1e-6)
