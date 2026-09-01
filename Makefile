PAPERS := sft_paper rl_paper eval_paper
PDFS := $(addsuffix /main.pdf,$(PAPERS))
STYLE_FILES := \
	template/icml2026/algorithm.sty \
	template/icml2026/algorithmic.sty \
	template/icml2026/fancyhdr.sty \
	template/icml2026/icml2026.bst \
	template/icml2026/icml2026.sty \
	template/preamble.tex
BUILD := scripts/latex_build.sh

.PHONY: all clean check-deps

all: $(PDFS)

%/main.pdf: %/main.tex %/references.bib $(STYLE_FILES) $(BUILD)
	$(BUILD) $*

clean:
	@for paper in $(PAPERS); do \
		$(BUILD) $$paper clean; \
	done

check-deps:
	@$(BUILD) sft_paper deps
