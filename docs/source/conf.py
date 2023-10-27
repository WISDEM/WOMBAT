"""Configuration file for the Sphinx documentation builder."""


# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

import wombat


sys.path.insert(0, os.path.abspath("../.."))


# -- Project information -----------------------------------------------------

project = "WOMBAT"
copyright = "2021 Alliance for Sustainable Energy, LLC"
author = (
    "Rob Hammond,"
    " Aubryn Cooperman,"
    " Aaron Barker,"
    " Alicia Key,"
    " Matt Shields,"
    " Annika Eberle"
)

# The full version, including alpha/beta/rc tags
release = wombat.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.
extensions = [
    "myst_nb",
    "sphinx_book_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "sphinxcontrib.bibtex",
]

# Make sure the target is unique
autosectionlabel_prefix_document = True

master_doc = "index"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
nb_execution_excludepatterns = ["_build", "Thumbs.db", ".DS_Store"]

bibtex_bibfiles = ["refs.bib"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.10", None),
    "sphinx": ("https://www.sphinx-doc.org/en/3.x", None),
}

suppress_warnings = ["myst.domains"]

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    # "html_admonition",
    # "html_image",
    "colon_fence",
    # "smartquotes",
    # "replacements",
    "linkify",
    # "substitution",
]
myst_url_schemes = ["http", "https", "mailto"]

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
}

# toggle this between auto/off to rerun full documentation build
nb_execution_mode = "auto"
nb_execution_timeout = -1
nb_execution_allow_errors = True
nb_execution_excludepatterns.extend(
    [
        "how_to.md",
        "metrics_demonstration.md",
        "strategy_demonstration.md",
    ]
)

myst_heading_anchors = 3

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"
html_title = f"WOMBAT v{release}"
html_theme_options = {
    "github_url": "https://github.com/WISDEM/WOMBAT",
    "repository_url": "https://github.com/WISDEM/WOMBAT",
    "repository_branch": "main",
    "use_repository_button": True,
    "use_issues_button": True,
    "path_to_docs": "docs/",
    "navigation_depth": 2,
    "home_page_in_toc": True,
    "show_toc_level": 2,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]

# Napoleon options
napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_ivar = True
