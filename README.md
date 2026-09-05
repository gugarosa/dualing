# Dualing

[![CI](https://github.com/gugarosa/dualing/actions/workflows/ci.yml/badge.svg)](https://github.com/gugarosa/dualing/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/dualing.svg)](https://pypi.org/project/dualing/)
[![Python](https://img.shields.io/pypi/pyversions/dualing.svg)](https://pypi.org/project/dualing/)
[![License](https://img.shields.io/github/license/gugarosa/dualing.svg)](LICENSE)

Dualing provides small TensorFlow building blocks for contrastive,
cross-entropy, and triplet Siamese networks.

## Install

```bash
uv add dualing
```

Dualing requires Python 3.11 or newer.

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

Native pair datasets yield `((left, right), labels)` or
`((left, right), labels, sample_weights)`. The original pair dataset classes
yield `(left, right, labels)` through their `.batches` attribute. Both forms
work with pair-model `fit`, `evaluate`, and `fit(validation_data=...)`.
Default pair losses retain one value per pair so Keras can apply sample weights.

Native and original dataset APIs share preprocessing: normalization maps
constant data to the lower bound instead of producing NaNs. Use `normalize=None`
to leave values unscaled.

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run --group docs sphinx-build -W -b html docs docs/_build/html
uv build
```

API documentation is available at
[dualing.readthedocs.io](https://dualing.readthedocs.io).
