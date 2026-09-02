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

.PHONY: all clean check-deps protocol-audits

all: $(PDFS)

%/main.pdf: %/main.tex %/references.bib $(STYLE_FILES) $(BUILD)
	$(BUILD) $*

clean:
	@for paper in $(PAPERS); do \
		$(BUILD) $$paper clean; \
	done

check-deps:
	@$(BUILD) sft_paper deps

protocol-audits:
	@for paper in $(PAPERS); do \
		name=$${paper%_paper}; \
		python3 scripts/run_experiment_pipeline.py \
			--plan experiments/$${name}_pipeline.json \
			--execute \
			--commands experiments/synthetic_commands.json \
			--work-dir artifacts/$${name}-synthetic-audit \
			--force || exit 1; \
	done
