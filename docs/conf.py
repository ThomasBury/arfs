# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from arfs import __version__

# -- Project information -----------------------------------------------------
project = "arfs"
copyright = "2024, Thomas Bury"
author = "Thomas Bury"
release = __version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "nbsphinx",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = []
autosummary_generate = True

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "build", "Thumbs.db", ".DS_Store"]
autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 1
# These sklearn-generated helpers link to sklearn's own metadata-routing guide.
autodoc_default_options = {
    "exclude-members": "set_fit_request,set_score_request,set_transform_request,set_inverse_transform_request"
}

# Execution is checked separately; optional and large examples must not run on build.
nbsphinx_execute = "never"


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = "sphinx_rtd_theme"
html_permalinks_icon = "<span>#</span>"
html_theme = "sphinxawesome_theme"

# If not None, a 'Last updated on:' timestamp is inserted at every page
# bottom, using the given strftime format.
# The empty string is equivalent to '%b %d, %Y'.
html_last_updated_fmt = "%B %d, %Y at %H:%M"
today_fmt = "%B %d, %Y at %H:%M"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]
html_title = "ARFS Documentation"
html_show_sourcelink = True
html_logo = "logo.png"

# -- Napoleon settings (for numpydoc parsing) --------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html#configuration
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = False
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True
