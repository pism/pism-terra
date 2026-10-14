# Building the documentation

The documentation you are reading is Sphinx + MyST, published on
[Read the Docs](https://pism-terra.readthedocs.io/). It builds from
`docs/source/`, and you can build the same thing locally before pushing.

## Environment

The docs build needs the **development** environment — the runtime
`environment.yml` deliberately leaves out everything that is not needed to run
a simulation:

```bash
conda env create -f environment-dev.yml
conda activate pism-terra
```

Or with [Mamba](https://mamba.readthedocs.io/), which resolves considerably
faster:

```bash
mamba env create -f environment-dev.yml
mamba activate pism-terra
```

Then install the package together with the `docs` extras, which pull in Sphinx,
the theme and the MyST/autosummary machinery:

```bash
python -m pip install -e ".[docs]"
```

## Build

```bash
cd docs
make html
open _build/html/index.html
```

While editing, `make livehtml` rebuilds on save and serves the result at
<http://127.0.0.1:8000>:

```bash
cd docs
make livehtml
```

## Starting from scratch

`make clean` removes the build tree **and** two generated source trees:
`source/reference/generated/` (the autosummary API stubs) and
`source/auto_examples/` (the rendered gallery). Reach for it when the API
reference disagrees with the code — a renamed or deleted module leaves its
stub behind, and Sphinx keeps happily rendering the stale page:

```bash
cd docs
make clean html
```

```{admonition} Warnings are not failures
:class: note

`.readthedocs.yaml` sets `fail_on_warning: false`, so a build that emits
warnings still publishes. A local build behaves the same way — read the
warnings rather than relying on the exit status.
```

## What gets executed

Fenced code blocks are inert by default. A block immediately preceded by a
`<!-- pism-terra: test -->` comment is collected by `pytest` (see
`tests/conftest.py`) and actually run, so those blocks have to keep working.
The {doc}`../resources/cheatsheet` explains the convention.
