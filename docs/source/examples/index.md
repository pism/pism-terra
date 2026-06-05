# Gallery of examples

Curated, end-to-end notebooks that demonstrate pism-terra workflows.

Two flavours of example are supported in this site:

- **Narrative notebooks** (this page's children) — `.ipynb` files rendered by
  [myst-nb](https://myst-nb.readthedocs.io). Build-time execution is
  currently disabled (`nb_execution_mode = "off"`); rendered output is
  whatever was saved into the notebook.
- **Runnable scripts** — `plot_*.py` files under `examples/` at the repo
  root, picked up by
  [sphinx-gallery](https://sphinx-gallery.github.io) and rendered into
  `auto_examples/`.

```{toctree}
:maxdepth: 1

quick_start
pism_cloud_intro
geometry
kitp_analysis
kitp_debm_calibration
```

```{admonition} TODO
- Curate and copy the substantive notebooks into this directory.
- Decide whether to enable build-time notebook execution against pinned test data.
- Add a sphinx-gallery `plot_quick_start.py` so the Auto-examples section
  also has content.
```
