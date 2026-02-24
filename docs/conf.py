"""Sphinx configuration for TN-Jax documentation."""

import warnings

project = "TN-Jax"
copyright = "2025, TN-Jax Contributors"
author = "TN-Jax Contributors"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "myst_parser",
    "sphinx_design",
]

# Autodoc
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

# Napoleon (Google-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_rtype = False

# MyST (Markdown support)
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

# Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# Theme
html_theme = "furo"
html_title = "TN-Jax"

# Source
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
exclude_patterns = ["_build"]

# Suppress third-party deprecation warnings during build
warnings.filterwarnings("ignore", category=DeprecationWarning,
                        module="sphinx_autodoc_typehints")
