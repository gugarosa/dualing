# Dualing conventions

The code style follows [cpmux](https://github.com/gugarosa/cpmux/blob/main/CONVENTIONS.md)
and its phitrain rules, adapted to Dualing's existing APIs and TensorFlow/Keras domain.

## Compatibility and ownership

- Keep the published import paths, constructor arguments, defaults, and original calling forms.
- Keep the declared Python >=3.11 support. The requested modern annotation idioms work on both 3.11 and 3.12+.
  Do not introduce 3.12-only syntax or raise the interpreter floor without an explicit compatibility decision.
- Keep the Apache-2.0 license, existing Ruff/pytest/Sphinx tooling, and native Keras model lifecycle.
- Keep package re-exports: they are published Dualing APIs, unlike cpmux's deliberately empty package facades.
- Layers own weights, embedders compose layers, and Siamese models own one shared embedder.
  Do not introduce a parallel configuration, persistence, or training framework.

## Code style

- Use `X | None` and builtin generics. Import ABCs such as `Callable` and `Iterable` from `collections.abc`.
  Import only typing-specific constructs such as `Any`, `Literal`, `Annotated`, and `Self` from `typing`. (R2)
- Start every Python source, test, example, and documentation configuration file with:

  ```python
  # Copyright (c) 2020-2026 Gustavo Rosa.
  # Licensed under the Apache License, Version 2.0.
  ```

- Use top-level, absolute imports. Separate stdlib, third-party, and local imports with blank lines.
- Use Google-style docstrings for public APIs. Keep a regular class summary on one line and put constructor
  arguments on `__init__`. Keep `Args:`, `Returns:`, and `Raises:` entries on one line each.
  Do not use semicolons or `defaults to <X>` tails in entries. (R3, R13)
- Multiline docstrings have one blank line before their closing `"""` and one blank line afterward.
  Short class and property summaries may remain single-line, as in cpmux.
- Private helpers have no docstrings. Keras-dispatched hooks such as `call`, `get_config`, `from_config`,
  `compile_from_config`, and `compute_metrics` have no docstrings. User-facing training methods and callable
  loss APIs remain documented even when they also serve Keras.
- Explain mutation, persistence, ownership, and failure behavior in public I/O contracts.
  Put a property's documentation on its getter rather than duplicating it on the setter.
- Data classes without explicit constructors document their fields in `Attributes:`, one entry per field.
  Do not introduce data-class machinery solely to satisfy this convention.
- Use `get_logger(__name__)` from `dualing.utils.logging`. Do not use `print()` in library code.
  Warning/error diagnostics identify a backticked offender and end with a period.
  Info/debug messages remain plain. Keep the existing logger and exception import paths. (R14)
- Raised messages identify a backticked name and end with a period, optionally explaining the offending value.
  Describe sentinel conditions as `is None` or `is True`. Existing exception types and category prefixes remain. (R1)
- Validate with `if` and a specific exception, never `assert`. Do not add bare exception handlers.
- Comments explain why, not what. Prefer none or one line, with a three-line maximum, no banners, and no
  trailing period. Copyright/license notices are the required header exception. (R8)
- In function bodies of at least 12 lines, separate logical phases with one blank line.
  Keep short cohesive assignments together. (R11)
- Inline first. Extract a helper, constant, or parameter only for a second real consumer.
  Preserve published constants and named callbacks required by Keras serialization. (R16)
- Use double-quoted strings. Keep readable source and docstring prose within 120 characters. (R9)

## Tests and review

- Keep pytest functions plain, without docstrings or type hints. Assertions belong in tests, not input validation.
- Preserve regression expectations and actionable failure information. Do not weaken them to accommodate formatting.
- Use the existing commands in the README for pytest, Ruff, strict Sphinx builds, and package builds.
- Check actual calling forms, numerical behavior, and installed artifacts when relevant.
  A formatter cannot determine whether logical phases or API documentation are correct.
- Retain useful scientific references and explain intentional compatibility changes separately from style cleanup.
