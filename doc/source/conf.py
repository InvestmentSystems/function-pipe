#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# function-pipe documentation build configuration file, created by
# sphinx-quickstart on Fri Jan  6 16:49:22 2017.
#
# This file is execfile()d with the current directory set to its containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.


# -- General configuration -----------------------------------------------------

import os
import sys

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath("../.."))

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named "sphinx.ext.*") or your custom ones.
extensions = [
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    # "sphinx.ext.viewcode",
    # "sphinx.ext.graphviz",
    "sphinx_jinja",
    # "matplotlib.sphinxext.only_directives",
    # "matplotlib.sphinxext.plot_directive",
    # "sphinx.ext.autosectionlabel",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix of source filenames.
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "function-pipe"
copyright = "2017, Christopher Ariza"

# The version info for the project you"re documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.

# The short X.Y version.
version = "2.1"
# The full version, including alpha/beta/rc tags.

release = "2.1.2"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for autodoc -------------------------------------------------------
add_module_names = False

# -- Options for HTML output ---------------------------------------------------

html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["images"]

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = "%b %d, %Y"

# Output file base name for HTML help builder.
htmlhelp_basename = "function-pipedoc"


# -- Options for LaTeX output --------------------------------------------------

latex_elements = {}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [
    (
        "index",
        "function-pipe.tex",
        "function-pipe Documentation",
        "Christopher Ariza",
        "manual",
    ),
]


# -- Options for manual page output --------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (
        "index",
        "function-pipe",
        "function-pipe Documentation",
        ["Christopher Ariza", "Charles Burkland"],
        1,
    )
]


# -- Options for Texinfo output ------------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        "index",
        "function-pipe",
        "function-pipe Documentation",
        "Christopher Ariza",
        "function-pipe",
        "One line description of project.",
        "Miscellaneous",
    ),
]

copybutton_selector = "div.copy-button pre"
