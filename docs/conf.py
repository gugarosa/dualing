# Copyright (c) 2020-2026 Gustavo Rosa.
# Licensed under the Apache License, Version 2.0.

from importlib.metadata import version as distribution_version

project = "Dualing"
author = "Gustavo Rosa"
copyright = "2020-2026, Gustavo Rosa"
version = release = distribution_version("dualing")

extensions = ["sphinx.ext.autodoc", "sphinx.ext.autosummary", "sphinx.ext.napoleon"]
napoleon_google_docstring = True
napoleon_numpy_docstring = False
autoclass_content = "both"
autodoc_inherit_docstrings = False
autosummary_generate = True
autodoc_member_order = "bysource"
html_theme = "alabaster"
