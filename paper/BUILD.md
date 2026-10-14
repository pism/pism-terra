# Building the paper, step by step

How to go from a fresh checkout to `paper/build/main.pdf`. Commands are run from
the repository root unless they start with `cd`.

## 1. One-time setup

1. **The pism-terra environment.** The figures are made with pism-terra itself:

   ```bash
   conda env create -f environment.yml
   conda activate pism-terra
   python -m pip install -e .
   ```

   `make figures` also needs jupytext. It is in the development environment
   (`environment-dev.yml`) and in the `docs` extras
   (`python -m pip install -e ".[docs]"`); in the plain environment, add it
   with `conda install -c conda-forge jupytext`.

2. **pandoc 3 or later** (`pandoc --version`), e.g. `brew install pandoc`. The
   pism-terra environment may already have one.

3. **LaTeX with `latexmk`**: MacTeX on macOS (`brew install --cask mactex`), TeX
   Live elsewhere. Check with `latexmk --version`.

4. **The Copernicus LaTeX package** is part of the repository, in
   `paper/copernicus/` (version 7.16, see `README_copernicus_package_*.txt`
   there). When Copernicus publishes a new version (GMD website, *Submission* →
   *Manuscript preparation*, the LaTeX template), replace the files there with
   the new ones and commit them.

## 2. The case-study data

The figures are made from the small postprocessed products of the case-study
runs, not from the runs themselves. They are read from the directory named by
`PISM_TERRA_PAPER_DATA`, one subdirectory per case study, each keeping the
layout of the run's directory. The files each case study needs are listed on
its page in `docs/source/paper/`.

Until the data are archived with a DOI, assemble the directory from the runs.
For case study 1 (Wrangell Mountains, run `2026_10_s4f_baseline`):

```bash
export PISM_TERRA_PAPER_DATA=$HOME/pism_terra_paper_data
RGI=RGI2000-v7.0-C-01-04374
RUN=2026_10_s4f_baseline/$RGI/output
CASE=$PISM_TERRA_PAPER_DATA/case_study_1
mkdir -p $CASE/output/dh $CASE/output/processed_scalar $CASE/input
cp $RUN/dh/dh_${RGI}_id_0_2000-01-01_2020-01-01.nc $CASE/output/dh/
cp $RUN/processed_scalar/scalar_C_g200m_${RGI}_id_0_1986-01-01_2025-01-01.nc $CASE/output/processed_scalar/
cp glacier_s4f_input/$RGI/input/obs_$RGI.nc $CASE/input/
```

Put the `export PISM_TERRA_PAPER_DATA=...` line in your shell profile so you
need not set it every time.

## 3. Make the figures

```bash
cd paper
make figures
```

This runs every case-study page (`docs/source/paper/case_study_*.md`) in a
Jupyter kernel with jupytext, in the page's own directory, so the figures land
in `docs/source/paper/figures/`, where `main.tex` looks for them. The executed
notebooks are kept as `paper/build/case_study_*.ipynb`; open one to see the
numbers its page prints, the ones quoted in the text. The figures are
generated files and are not under version control; rerun `make figures`
whenever a case study or its data changes.

## 4. Build the PDF

```bash
cd paper
make
open build/main.pdf
```

`make` converts every `sections/*.md` into `build/*.tex` with pandoc, fetches
PISM's bibliography (below) the first time, and compiles `main.tex` with latexmk, running BibTeX and LaTeX as often as the
citations and cross-references need. While writing, `make watch` rebuilds on
every change to a section, `main.tex` or the bibliography (needs `fswatch`,
`brew install fswatch`). `make clean` removes `build/`.

## 5. Writing

- The prose is in `paper/sections/*.md`; the journal structure (title, authors,
  section headings, availability statements) is in `paper/main.tex`. A new
  section is a new Markdown file and an `\input{build/<name>}` in `main.tex`.
- The paper cites from two bibliographies:
  - `docs/source/refs.bib`, pism-terra's, shared with the documentation. New
    references go here.
  - PISM's `doc/ice-bib.bib`, which `make` downloads from GitHub into
    `paper/build/ice-bib.bib`, at the commit pinned by `PISM_BIB_COMMIT` in the
    Makefile (now the commit of `aaschwanden/ismip7` that last changed it). Look
    up its keys there; a reference missing from it can go into PISM's file (and
    the pin moved to that commit once it is on GitHub) or into `refs.bib`.

  Cite with `[@key]` (in parentheses) or `@key` (in the text). A key in both
  files makes BibTeX warn and take the first, from `refs.bib`.
- The conventions for headings, figures, equations and cross-references are in
  `paper/README.md`.
- The case-study pages are part of the documentation too, under *Model
  description paper*; build it with `make -C docs html` to see them.

## 6. When something goes wrong

| Symptom | Fix |
|---|---|
| `copernicus/copernicus.cls is missing` | The package is missing from `paper/copernicus/` (step 1.4); `git status` shows whether it was deleted. |
| `set PISM_TERRA_PAPER_DATA ...` from `make figures` | Step 2. |
| `FileNotFoundError` from `make figures` | A file of the case study is missing from `$PISM_TERRA_PAPER_DATA`; compare with the table on its page. |
| `File 'case_study_1_dh.png' not found` | Run `make figures` first (step 3). |
| A citation shows as `(?)` | The key is in neither `docs/source/refs.bib` nor `paper/build/ice-bib.bib`, or is spelled differently. |
| `curl: (6) Could not resolve host` or `(22) ... 404` | `make` needs the network the first time, to fetch PISM's bibliography; a 404 means `PISM_BIB_COMMIT` names a commit that is not on GitHub. |
| A reference shows as `??` | The `\label` is missing or misspelled; `build/main.log` lists the undefined ones. |
| `keyval Error: alt undefined` | The LaTeX installation predates the `alt` key pandoc writes for figures; update TeX Live or MacTeX. |
| A section is missing from the PDF | Its `\input{build/<name>}` is missing from `main.tex`. |

The full LaTeX log is `paper/build/main.log`.

## 7. Before submitting

- Release the pism-terra version the paper describes (`v1.0.0`) and archive it
  with a DOI (Zenodo); the version goes into the title.
- Archive the case-study data with a DOI, and let the pages fetch them instead
  of reading `PISM_TERRA_PAPER_DATA`.
- Fill in the code and data availability, author contributions and
  acknowledgements in `main.tex`.
- Check that `paper/copernicus/` holds the current Copernicus package, and the
  PDF against the GMD manuscript guidelines.
