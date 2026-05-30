"""
Sphinx configuration for pism-terra.

Layout mirrors xDEM (https://xdem.readthedocs.io/) — pydata-sphinx-theme +
sphinx-book-theme overrides, sphinx-design, sphinx-gallery for runnable
``plot_*.py`` examples, and myst-nb for notebook-native narrative pages.
"""

from __future__ import annotations

from importlib import metadata
from pathlib import Path

# -- Project information -----------------------------------------------------

project = "pism-terra"
author = "Andy Aschwanden"
copyright = f"2025-2026, {author}"  # pylint: disable=redefined-builtin

try:
    release = metadata.version("pism-terra")
except metadata.PackageNotFoundError:
    release = "0.0.0+unknown"
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
    "myst_nb",
    "numpydoc",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# myst-nb registers parsers for both .md and .ipynb on its own; .rst stays
# the default Sphinx parser. No explicit ``source_suffix`` mapping needed.

# -- Theme -------------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Placeholder logo until we have real branding.
html_logo = "_static/logo_placeholder.svg"
html_favicon = None
html_title = "pism-terra"

html_theme_options = {
    "logo": {
        "alt_text": "pism-terra - Home",
    },
    "github_url": "https://github.com/pism/pism-terra",
    "icon_links": [],
    "navigation_depth": 2,
    "show_toc_level": 2,
    "use_edit_page_button": False,
    "header_links_before_dropdown": 6,
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
    "navbar_align": "left",
    "secondary_sidebar_items": ["page-toc", "sourcelink"],
}

# -- autodoc / autosummary / numpydoc ---------------------------------------

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

# numpydoc validation lives in pyproject.toml; only suppress the noisy
# class-attribute table here, which is hostile to dataclasses/pydantic.
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

# -- MyST / MyST-NB ---------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3

# Don't try to execute notebooks at build time — data paths aren't portable yet.
nb_execution_mode = "off"

# -- sphinx-gallery ---------------------------------------------------------

sphinx_gallery_conf = {
    "examples_dirs": [str(Path(__file__).parent / "../../examples")],
    "gallery_dirs": ["auto_examples"],
    "filename_pattern": r"plot_",
    "remove_config_comments": True,
    "show_signature": False,
    "download_all_examples": False,
}

# -- intersphinx ------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "xarray": ("https://docs.xarray.dev/en/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "geopandas": ("https://geopandas.org/en/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "rasterio": ("https://rasterio.readthedocs.io/en/stable", None),
    "rioxarray": ("https://corteva.github.io/rioxarray/stable", None),
    "pyproj": ("https://pyproj4.github.io/pyproj/stable", None),
}

# -- copybutton -------------------------------------------------------------

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# -- Warning suppression for placeholder pages ------------------------------

# autosummary :toctree: will warn until generated/ exists at first build.
suppress_warnings = ["autosummary"]
