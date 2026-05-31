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
    "sphinxcontrib.bibtex",
    "myst_nb",
    "numpydoc",
]

# -- sphinxcontrib-bibtex ---------------------------------------------------

bibtex_bibfiles = ["refs.bib"]
bibtex_default_style = "alpha"
bibtex_reference_style = "author_year"

templates_path = ["_templates"]
exclude_patterns: list[str] = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    # Sphinx-gallery generates .rst, .py, .ipynb, .codeobj.json, .py.md5 and
    # .zip side-by-side under ``auto_examples/``. The .ipynb collides with
    # myst-nb's parser for the same document, so let sphinx-gallery's .rst
    # be the canonical source.
    "auto_examples/**/*.ipynb",
]

# myst-nb registers parsers for both .md and .ipynb on its own; .rst stays
# the default Sphinx parser. No explicit ``source_suffix`` mapping needed.

# -- Theme -------------------------------------------------------------------

html_theme = "sphinx_book_theme"
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
    "repository_url": "https://github.com/pism/pism-terra",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_download_button": False,
    "use_edit_page_button": False,
    # Always show the full nav tree in the left sidebar, expanded.
    "navigation_depth": 4,
    "show_navbar_depth": 2,
    "collapse_navbar": False,
    "show_toc_level": 2,
    "home_page_in_toc": True,
    "path_to_docs": "docs/source",
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

# -- Warning suppression ----------------------------------------------------

suppress_warnings = [
    # autosummary :toctree: warns until generated/ exists at first build.
    "autosummary",
    # myst-nb tolerated warnings on unconfigured cell-metadata keys.
    "mystnb.unknown_mime_type",
]


# -- Pydantic v2 dedup -------------------------------------------------------
# Pydantic models emit each field twice from a single ``autoclass`` call —
# once as the typed class attribute, once through ``model_fields`` — which
# Sphinx flags as "duplicate object description". Track the exact objects
# we've documented and skip the second registration.
def _skip_duplicate_pydantic_members(_app, _what, _name, obj, skip, _options):
    """Drop the second registration of an identical object from autodoc."""
    if skip:
        return True
    seen = _skip_duplicate_pydantic_members.__dict__.setdefault("_seen", set())
    key = id(obj)
    if key in seen:
        return True
    seen.add(key)
    return None


def setup(app):
    """Sphinx extension entry point."""
    app.connect("autodoc-skip-member", _skip_duplicate_pydantic_members)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
