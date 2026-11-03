PYTHON ?= python3

.PHONY: reproduce figures reports
reproduce:
	$(PYTHON) analysis/reproduce.py
figures: reproduce
	$(PYTHON) analysis/make_figures.py
reports: figures
	cd report && pdflatex -interaction=nonstopmode -halt-on-error milestone.tex
	cd report && pdflatex -interaction=nonstopmode -halt-on-error milestone.tex
	cd report && pdflatex -interaction=nonstopmode -halt-on-error final_report.tex
	cd report && pdflatex -interaction=nonstopmode -halt-on-error final_report.tex
