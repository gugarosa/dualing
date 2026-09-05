"""Internal model helpers."""

import tensorflow as tf


@tf.autograph.experimental.do_not_convert
def _pack_pair(left, right, labels):
    return (left, right), labels


def pair_dataset(dataset):
    """Convert legacy three-item pair batches to Keras pair inputs."""

    if not isinstance(dataset, tf.data.Dataset):
        return dataset

    element_spec = dataset.element_spec

    if (
        isinstance(element_spec, tuple)
        and len(element_spec) == 3
        and not tf.nest.is_nested(element_spec[0])
    ):
        return dataset.map(_pack_pair)

    return dataset
