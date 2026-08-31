from importlib.metadata import version as distribution_version

project = "Dualing"
author = "Gustavo Rosa"
copyright = "2020, Gustavo Rosa"
version = release = distribution_version("dualing")

extensions = ["sphinx.ext.autodoc", "sphinx.ext.autosummary", "sphinx.ext.napoleon"]
autosummary_generate = True
autodoc_member_order = "bysource"
html_theme = "alabaster"
