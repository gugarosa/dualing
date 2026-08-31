import numpy as np
import tensorflow as tf

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
