# Copyright (c) 2020-2026 Gustavo Rosa.
# Licensed under the Apache License, Version 2.0.

"""Internal model helpers."""

import numbers
from functools import wraps

import tensorflow as tf


def _legacy_fit_epochs(fit):
    @wraps(fit)
    def wrapped(self, batches=None, *args, **kwargs):
        data = kwargs.get("x", batches)

        # Resolve legacy positional epochs before Python binds them to native labels
        if isinstance(data, tf.data.Dataset) and args and isinstance(args[0], numbers.Integral):
            if len(args) != 1 or "epochs" in kwargs:
                raise TypeError("`epochs` must be supplied only once.")

            kwargs["epochs"] = args[0]
            args = ()

        return fit(self, batches, *args, **kwargs)

    return wrapped


def _prediction_input(x1, kwargs):
    if "x" not in kwargs:
        return x1

    if x1 is not None:
        raise TypeError("`x` cannot be combined with `x1`.")

    return kwargs.pop("x")


def pair_dataset(dataset):
    if not isinstance(dataset, tf.data.Dataset):
        return dataset

    element_spec = dataset.element_spec

    if isinstance(element_spec, tuple) and len(element_spec) == 3 and not tf.nest.is_nested(element_spec[0]):
        return dataset.map(lambda left, right, labels: ((left, right), labels))

    return dataset
