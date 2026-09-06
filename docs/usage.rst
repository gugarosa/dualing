Usage and extension guide
=========================

Dualing composes TensorFlow datasets and Keras models; it does not own a
separate training engine, configuration format, or checkpoint loader.
The original classes and import paths remain available alongside the native
dataset functions.

Data contracts
--------------

``preprocess`` converts numeric inputs to ``float32``. Its ``input_shape`` is
the shape of the entire tensor, including the sample dimension, not the shape
of an individual sample. Normalization uses one minimum and maximum across
the entire tensor. Constant inputs map to the lower bound; ``normalize=None``
disables scaling.

``batch_dataset`` yields ``(samples, labels)``. Pair helpers yield
``((left, right), targets)``, where 1 means similar and 0 means dissimilar.
The original pair classes expose ``(left, right, targets)`` through
``.batches``. Pair models accept both formats for training, evaluation, and
``validation_data``. Native weighted pair datasets use
``((left, right), targets, sample_weights)``.

The original ``fit(dataset, epochs)`` positional form is supported alongside
``fit(inputs, labels, epochs=...)`` for arrays. Keyword ``y`` still means
targets and is invalid with a dataset. Native ``predict(x=...)`` is supported;
use ``compare(left, right)`` when explicitly comparing two batches.

Training now uses Keras return values: ``fit`` returns ``History``, and
``evaluate`` returns loss/metric results. It is not the original logging-only
training loop, even when the input uses an original dataset layout.

Balanced sampling produces equal positive and negative pair counts, but
allows repeated samples and self-pairs. Random pairing is disjoint and
leaves one sample unused when the sample count is odd. Neither helper
stratifies triplet-learning batches.

Embedding and distance contracts
--------------------------------

* ``MLP`` preserves leading dimensions and transforms the final feature axis.
* ``CNN`` prefers channels-last images and produces one vector per image.
  Its original channels-first detection remains heuristic; use channels-last
  inputs when the shape is ambiguous.
* ``RNN``, ``GRU``, and ``LSTM`` consume token IDs of shape ``(batch, time)``
  and produce ``(batch, time, vocab_size)`` tensors. Token embedding width
  and recurrent hidden width are not the final projection width.

Token IDs are identifiers, not continuous features: do not normalize them.
Use a native ``tf.data.Dataset`` to preserve integer tensors rather than
the numeric preprocessing helpers, which convert inputs to ``float32``.

Siamese models reuse one shared embedder. ``extract_embeddings`` returns its
raw output; ``embed`` mean-pools rank-three sequence outputs over time.
Contrastive and cross-entropy models return one value per pair.

For new triplet code, specify ``mining`` explicitly so ``distance_metric``
is used directly:

.. code-block:: python

    from dualing import MLP, TripletSiamese

    model = TripletSiamese(
        MLP((32, 8)),
        mining="hard",
        distance_metric="L2",
        margin=0.5,
    )

Calls without ``mining`` retain the original distance mapping. Triplet
``compare`` uses normalized embeddings and the requested metric; the
original ``predict(left, right)`` uses unnormalized pooled embeddings and
the effective ``distance``. Serialization preserves this distinction.

Triplet mining operates within each batch. A valid anchor needs another
sample of its class and at least one negative class. Functional triplet
losses exclude invalid anchors/pairs and return zero when none are valid.
The original callable loss classes retain their single-class margin result.
Use meaningful class mixtures rather than interpreting degenerate batches
as evidence of learning.

Native Keras lifecycle
----------------------

The models accept normal Keras options such as ``name``, ``trainable``, and
``dtype``. Constructor aliases remain supported. Configuration output uses
canonical constructor names and the effective layer dimensions, rather
than storing contradictory old/new aliases.

Use ``clone_model`` for a new model with the same configuration. Cloning
does not copy learned weights:

.. code-block:: python

    import tensorflow as tf
    from dualing import MLP

    embedder = MLP((16, 8), activation="tanh", name="encoder")
    embedder(tf.ones((2, 4)))

    clone = tf.keras.models.clone_model(embedder)
    clone.set_weights(embedder.get_weights())

Use the native ``.keras`` format when resuming training:

.. code-block:: python

    import dualing
    import numpy as np
    import tensorflow as tf

    samples = np.array(
        [[0.0, 0.1], [0.2, 0.3], [0.8, 0.9], [1.0, 1.1]],
        dtype="float32",
    )
    labels = np.array([0, 0, 1, 1])
    pairs = dualing.balanced_pair_dataset(
        samples, labels, n_pairs=8, batch_size=4
    )

    model = dualing.ContrastiveSiamese(dualing.MLP((8, 3)))
    model.compile(optimizer="adam")
    model.fit(pairs, epochs=1, verbose=0)
    model.save("siamese.keras")

    restored = tf.keras.models.load_model("siamese.keras")
    restored.fit(pairs, epochs=1, verbose=0)

Import ``dualing`` before loading in a fresh process so Keras can find its
registered models and losses. Trained model weights, configured losses and
metrics, and optimizer state are restored through Keras. The inherited
build and compile-configuration mechanisms are reused; only reconstruction
of nested embedders and custom compile defaults needs a Dualing hook.

Archive compatibility follows Keras's format and version requirements.
Support for multiple Keras versions does not imply that a newer archive
can be loaded by an older version. Load archives only from trusted sources.

As with other Keras models, recompile after changing training configuration.
A model's constructor settings and an explicitly supplied compiled loss
are separate configurations; neither is silently substituted for the other
during loading.

Extending the library
---------------------

Keep responsibilities local:

* Layers own tensor operations and weights; embedders compose those layers.
* Siamese models own the shared embedder and loss/training defaults.
* Dataset functions return ``tf.data.Dataset`` rather than custom iterators.
* Public callable losses retain their documented reductions and per-call
  overrides; serialization must not change numerical behavior.

Custom Keras embedders can be passed directly as ``base``. Implement
``get_config`` for constructor state and ``from_config`` when nested objects
need reconstruction. Register custom classes/activations with Keras, or
supply ``custom_objects`` on load. Private shared recurrent and loss bases
are implementation details, not new extension interfaces.

For public APIs, document the input/output shapes, label meanings, defaults,
alias precedence, normalization, relevant exceptions, and ownership of
state. Use Google-style docstrings consistently, with ``Args:``, ``Returns:``,
``Raises:``, and ``Attributes:`` where applicable. Do not mix in NumPy-style
underlined sections. Sphinx Napoleon is configured for Google-style parsing
only. A new configuration system, abstract factory hierarchy, or scikit-learn
estimator facade is unnecessary for these contracts.

Review conventions
------------------

The canonical code-style and review rules are in
`CONVENTIONS.md <https://github.com/gugarosa/dualing/blob/main/CONVENTIONS.md>`_.
They adopt cpmux's Google-style constructor documentation, import ordering,
diagnostic wording, restrained comments, and logical phase separation while
retaining Dualing's published APIs and interpreter support.

Design references
-----------------

These conventions follow Keras for layer ownership and persistence, and
borrow precise public-contract documentation rather than copying another
framework's API:

* `Keras subclassing guide
  <https://keras.io/guides/making_new_layers_and_models_via_subclassing/>`_
* `Keras serialization guide
  <https://keras.io/guides/serialization_and_saving/>`_
* `scikit-learn developer guide
  <https://scikit-learn.org/stable/developers/develop.html>`_
* `Google Python style guide: comments and docstrings
  <https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings>`_
