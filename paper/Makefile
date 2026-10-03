# Build the pism-terra model description paper for Geoscientific Model Development.
#
# The sections are written in Markdown (sections/*.md). Pandoc turns each into a
# LaTeX fragment in build/, main.tex (the Copernicus template) inputs them, and
# latexmk compiles the PDF. The bibliography is the one of the documentation.
#
#   make          build build/main.pdf
#   make figures  make the figures from the case-study pages (needs PISM_TERRA_PAPER_DATA)
#   make tex      convert the sections only
#   make watch    rebuild whenever a section or main.tex changes (needs fswatch)
#   make clean    remove build/

PANDOC ?= pandoc
LATEXMK ?= latexmk
JUPYTEXT ?= jupytext

# PISM's bibliography, fetched from GitHub at a pinned commit so every build
# cites the same entries: the commit on aaschwanden/ismip7 of pism/pism that
# last changed doc/ice-bib.bib (it adds the plume references to dev's). To move
# to newer references, set this to a later commit that changed the file, pushed
# to GitHub, and run `make clean`.
PISM_BIB_COMMIT ?= cf758bb4b0b274c6be99207dd3c73d4ea640b8ad
PISM_BIB_URL := https://raw.githubusercontent.com/pism/pism/$(PISM_BIB_COMMIT)/doc/ice-bib.bib

# The case-study pages of the documentation, whose code makes the figures.
PAGES := $(wildcard ../docs/source/paper/case_study_*.md)

SECTIONS := $(wildcard sections/*.md)
FRAGMENTS := $(patsubst sections/%.md,build/%.tex,$(SECTIONS))

# The Copernicus LaTeX package (copernicus.cls, copernicus.bst) is in
# copernicus/; see BUILD.md. The bibliographies are docs/source/refs.bib
# (pism-terra's) and build/ice-bib.bib (PISM's, fetched below).
# The trailing colons keep TeX's default search paths.
export TEXINPUTS := $(CURDIR)/copernicus:$(CURDIR):$(TEXINPUTS):
export BSTINPUTS := $(CURDIR)/copernicus:$(BSTINPUTS):
export BIBINPUTS := $(CURDIR)/../docs/source:$(CURDIR)/build:$(BIBINPUTS):

.PHONY: all pdf tex figures watch clean

all: pdf

pdf: build/main.pdf

tex: $(FRAGMENTS)

# One fragment per section. main.tex holds the section headings (Copernicus has
# its own \introduction and \conclusions), so a section's "#" headings become
# subsections. --natbib turns [@key] into \citep{key} and @key into \citet{key};
# the filter numbers the display equations that carry a \label.
build/%.tex: sections/%.md filters/equations.lua | build
	$(PANDOC) $< --from markdown --to latex --natbib --shift-heading-level-by=1 --wrap=preserve \
		--lua-filter=filters/equations.lua -o $@

build/main.pdf: main.tex $(FRAGMENTS) ../docs/source/refs.bib build/ice-bib.bib $(wildcard ../docs/source/paper/figures/*) | build
	@test -f copernicus/copernicus.cls || { echo "copernicus/copernicus.cls is missing: see BUILD.md"; exit 1; }
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex

build:
	mkdir -p build

build/ice-bib.bib: | build
	curl -fsSL $(PISM_BIB_URL) -o $@.tmp && mv $@.tmp $@

# Runs every case-study page in a Jupyter kernel, in the page's directory, so
# the figures land in ../docs/source/paper/figures/, where main.tex looks for
# them. The executed notebooks, with the numbers the pages print, are kept in
# build/ to look at.
figures: | build
	@test -n "$$PISM_TERRA_PAPER_DATA" || { echo "set PISM_TERRA_PAPER_DATA to the case-study data: see BUILD.md"; exit 1; }
	@for page in $(PAGES); do \
		echo "Running $$page"; \
		$(JUPYTEXT) --to ipynb --execute --run-path $$(dirname $$page) --output build/$$(basename $$page .md).ipynb $$page || exit 1; \
	done

watch:
	@command -v fswatch >/dev/null || { echo "make watch needs fswatch (brew install fswatch)"; exit 1; }
	$(MAKE) pdf; fswatch -o sections main.tex ../docs/source/refs.bib | while read -r _; do $(MAKE) pdf; done

clean:
	rm -rf build
