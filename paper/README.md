# Model description paper

The pism-terra paper for [Geoscientific Model Development](https://www.geoscientific-model-development.net/).
Write the sections in Markdown in `sections/`; `make` converts them with pandoc
and compiles `main.tex`, the Copernicus template, into `build/main.pdf`.

**[BUILD.md](BUILD.md) walks through every step**, from installing the tools
and the Copernicus package to the case-study data, the figures and the PDF.

## Requirements

- [pandoc](https://pandoc.org) 3 or later and a LaTeX installation with `latexmk`
  (MacTeX on macOS).
- [jupytext](https://jupytext.readthedocs.io), for `make figures`.
- The Copernicus LaTeX package, version 7.16, which is included in
  `copernicus/`.

## Writing

- One file per section in `sections/`. `main.tex` holds the section headings and
  inputs `build/<name>.tex` for each `sections/<name>.md`; a new section needs a
  file here and an `\input` there.
- `#` headings inside a section file become subsections.
- Cite with `[@key]` for `\citep` and `@key` for `\citet`; the keys are those of
  `docs/source/refs.bib`, shared with the documentation.
- Figures: `![Caption.](file.png){#fig:name width=100%}` gives a figure with
  `\label{fig:name}`; refer to it with `Fig. \ref{fig:name}`. Figures are looked
  up in `paper/figures/` and in `docs/source/paper/figures/`, where the
  case-study pages of the documentation write them.
- Equations are LaTeX (`$...$`, `$$...$$`). A display equation with a `\label`
  is numbered (`filters/equations.lua`) and referred to with `Eq. (\ref{eq:name})`.

## Building

```bash
make figures  # the figures, from the case-study data (see BUILD.md)
make          # build/main.pdf
make tex      # only convert the sections, e.g. to check pandoc's output
make watch    # rebuild on every change (needs fswatch)
make clean
```
