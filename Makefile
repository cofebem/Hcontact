# Makefile — build the LaTeX documents (slide deck + PCG theory note).
#
# Both PDFs are generated artifacts and are not tracked in git.
# Override the tool paths if they are not on your PATH, e.g.:
#   make slides PDFLATEX=/home/vyastrebov/DISTR/TEXLIVE2020/bin/x86_64-linux/pdflatex
#
# `make figures` needs the conda build env (numpy + the compiled
# hmatrix_contact module); see CLAUDE.md for the build instructions.

PDFLATEX ?= pdflatex
BIBTEX   ?= bibtex
PYTHON   ?= python

SLIDES_DIR := doc/slides
THEORY_DIR := doc/theory

.PHONY: docs slides theory figures clean help

help:
	@echo "Targets:"
	@echo "  make docs     - build slides + theory note"
	@echo "  make slides   - regenerate figures, then build the Beamer deck"
	@echo "  make theory   - build the PCG theory note (doc/theory/pcg.pdf)"
	@echo "  make figures  - (re)generate slide figures (needs conda build env)"
	@echo "  make clean    - remove LaTeX build artifacts"

## docs: build every document
docs: slides theory

## figures: (re)generate slide figures into doc/slides/figures/
figures:
	cd $(SLIDES_DIR) && $(PYTHON) generate_figures.py

## slides: build the Beamer slide deck (two passes for refs/nav)
slides: figures
	cd $(SLIDES_DIR) && $(PDFLATEX) -interaction=nonstopmode slides.tex \
	                 && $(PDFLATEX) -interaction=nonstopmode slides.tex

## theory: build every theory note in doc/theory (pdflatex/bibtex/pdflatex x2)
theory:
	cd $(THEORY_DIR) && for f in *.tex; do \
	    base=$${f%.tex}; \
	    $(PDFLATEX) -interaction=nonstopmode $$f && \
	    $(BIBTEX) $$base && \
	    $(PDFLATEX) -interaction=nonstopmode $$f && \
	    $(PDFLATEX) -interaction=nonstopmode $$f; \
	done

## clean: remove LaTeX build artifacts (keeps the .pdf output)
clean:
	rm -f $(SLIDES_DIR)/slides.aux $(SLIDES_DIR)/slides.log $(SLIDES_DIR)/slides.nav \
	      $(SLIDES_DIR)/slides.out $(SLIDES_DIR)/slides.snm $(SLIDES_DIR)/slides.toc \
	      $(SLIDES_DIR)/slides.vrb
	rm -f $(THEORY_DIR)/*.aux $(THEORY_DIR)/*.log $(THEORY_DIR)/*.out \
	      $(THEORY_DIR)/*.toc $(THEORY_DIR)/*.bbl $(THEORY_DIR)/*.blg
