"""Internal model helpers."""

import numbers
from functools import wraps

import tensorflow as tf


def _legacy_fit_epochs(fit):
    """Adapt positional dataset epochs without reinterpreting keyword labels."""

    @wraps(fit)
    def wrapped(self, batches=None, *args, **kwargs):
        data = kwargs.get("x", batches)

        if (
            isinstance(data, tf.data.Dataset)
            and args
            and isinstance(args[0], numbers.Integral)
        ):
            if len(args) != 1 or "epochs" in kwargs:
                raise TypeError("epochs must be supplied only once")

            kwargs["epochs"] = args[0]
            args = ()

        return fit(self, batches, *args, **kwargs)

    return wrapped


def _prediction_input(x1, kwargs):
    """Resolve the native x keyword without silently discarding another input."""

    if "x" not in kwargs:
        return x1

    if x1 is not None:
        raise TypeError("pass only one of x and x1")

    return kwargs.pop("x")


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
