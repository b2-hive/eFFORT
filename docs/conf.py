# Configuration file for the Sphinx documentation builder.
#
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
import pathlib
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'eFFORT'
copyright = '2019, Markus Tobias Prim, Maximilian Welsch'
author = 'Markus Tobias Prim, Maximilian Welsch'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.napoleon',
    "sphinx.ext.intersphinx",

]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("http://pandas-docs.github.io/pandas-docs-travis/", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "matplotlib": ("https://matplotlib.org/", None),
    # 'wilson': ('https://wilson-eft.github.io/wilson/', None)
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autodoc_default_options = {
    "special-members": "__init__",
    "undoc-members": True,
    "show-inheritance": True,
}

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#

this_dir = pathlib.Path(__file__).resolve().parent
with (this_dir / ".." / "eFFORT" / "version.txt").open() as vf:
    version = vf.read()
print("Version as read from version.txt: '{}'".format(version))

# The short X.Y version.
# version = 'dev'
# The full version, including alpha/beta/rc tags.
release = version

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
try:
    import importlib
    theme = importlib.import_module("sphinx_rtd_theme")
    html_theme = "sphinx_rtd_theme"
    html_theme_path = [theme.get_html_theme_path()]
except ImportError:
    print("Run pip install sphinx_rtd_theme to get the RTD theming.")
    html_theme = "alabaster"
print("html_theme='{}'".format(html_theme))

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']
