# Dualing

[![CI](https://github.com/gugarosa/dualing/actions/workflows/ci.yml/badge.svg)](https://github.com/gugarosa/dualing/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/dualing.svg)](https://pypi.org/project/dualing/)
[![Python](https://img.shields.io/pypi/pyversions/dualing.svg)](https://pypi.org/project/dualing/)
[![License](https://img.shields.io/github/license/gugarosa/dualing.svg)](https://github.com/gugarosa/dualing/blob/main/LICENSE)

Dualing provides small TensorFlow building blocks for contrastive,
cross-entropy, and triplet Siamese networks.

## Install

```bash
pip install dualing
```

Dualing requires Python 3.11 or newer. In a uv-managed project, use
`uv add dualing` instead.

## Quick start

```python
import numpy as np

from dualing import MLP, ContrastiveSiamese, balanced_pair_dataset

samples = np.random.default_rng(0).normal(size=(100, 16))
labels = np.repeat([0, 1], 50)
dataset = balanced_pair_dataset(samples, labels, n_pairs=100, batch_size=16)

model = ContrastiveSiamese(MLP((32, 8)))
model.compile(optimizer="adam")
model.fit(dataset, epochs=5)
```

Dataset helpers return native `tf.data.Dataset` objects and all models use
standard Keras `compile`, `fit`, and `evaluate` behavior. The original
`dualing.core`, `dualing.datasets`, `dualing.models.base`, and `dualing.utils`
APIs remain available.

The [examples directory](https://github.com/gugarosa/dualing/tree/main/examples)
contains dataset/model construction examples and MNIST training scripts for
all three Siamese variants.

Native pair datasets yield `((left, right), labels)` or
`((left, right), labels, sample_weights)`. The original pair dataset classes
yield `(left, right, labels)` through their `.batches` attribute. Both forms
work with pair-model `fit`, `evaluate`, and `fit(validation_data=...)`.
Default pair losses retain one value per pair so Keras can apply sample weights.

Native and original dataset APIs share preprocessing: normalization maps
constant data to the lower bound instead of producing NaNs. Use `normalize=None`
to leave values unscaled.

## Save and restore models

Dualing models support native Keras cloning and `.keras` persistence:

```python
import dualing  # Registers Dualing models and losses with Keras.
import tensorflow as tf

model.save("siamese.keras")
restored = tf.keras.models.load_model("siamese.keras")
restored.fit(dataset, epochs=1)
```

Saved trained models retain their weights, loss configuration, and optimizer
state. Constructor aliases remain accepted but serialize to one canonical
configuration. Custom embedders and activations follow Keras's own registration
or `custom_objects` mechanism.

See the [usage guide](https://dualing.readthedocs.io/en/latest/usage.html)
for tensor shapes, triplet distance semantics, and extension conventions.

## Development

From a cloned checkout, use the existing development tools:

```bash
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run --group docs sphinx-build -W -b html docs docs/_build/html
uv build
```

For an editable runtime-only installation, `pip install -e .` remains supported.

API documentation is available at
[dualing.readthedocs.io](https://dualing.readthedocs.io).
