PAPERS := sft_paper rl_paper eval_paper
PDFS := $(addsuffix /main.pdf,$(PAPERS))
STYLE_FILES := \
	template/icml2026/algorithm.sty \
	template/icml2026/algorithmic.sty \
	template/icml2026/fancyhdr.sty \
	template/icml2026/icml2026.bst \
	template/icml2026/icml2026.sty \
	template/preamble.tex

.PHONY: all clean

all: $(PDFS)

%/main.pdf: %/main.tex %/references.bib $(STYLE_FILES)
	cd $* && TEXINPUTS="../template/icml2026//:" \
		latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

clean:
	@for paper in $(PAPERS); do \
		(cd $$paper && latexmk -C main.tex); \
	done
