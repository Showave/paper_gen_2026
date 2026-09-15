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

.PHONY: all clean check-deps source-audit provenance-audit content-gates-audit external-overlap-audit sft-reference-audit sft-factorial-audit rl-estimator-audit eval-simulation-design-audit design-audits protocol-audits

all: $(PDFS)

%/main.pdf: %/main.tex %/references.bib $(STYLE_FILES) $(BUILD)
	$(BUILD) $*

clean:
	@for paper in $(PAPERS); do \
		$(BUILD) $$paper clean; \
	done

check-deps:
	@$(BUILD) sft_paper deps

source-audit:
	python3 scripts/audit_public_sources.py

provenance-audit:
	python3 scripts/materialize_public_data.py --self-test

content-gates-audit:
	python3 scripts/run_content_gates.py --self-test

external-overlap-audit:
	python3 scripts/validate_external_overlap_audit.py --self-test

sft-reference-audit:
	python3 scripts/validate_sft_reference_contract.py --self-test

sft-factorial-audit:
	python3 scripts/aggregate_sft_factorial.py --self-test

rl-estimator-audit:
	python3 scripts/aggregate_rl_estimator.py --self-test

eval-simulation-design-audit:
	python3 scripts/validate_eval_simulation_design.py --self-test

design-audits: sft-reference-audit sft-factorial-audit rl-estimator-audit eval-simulation-design-audit

protocol-audits:
	@for paper in $(PAPERS); do \
		name=$${paper%_paper}; \
		python3 scripts/run_experiment_pipeline.py \
			--plan experiments/$${name}_pipeline.json \
			--execute \
			--synthetic-audit \
			--commands experiments/synthetic_commands.json \
			--work-dir artifacts/$${name}-synthetic-audit \
			--force || exit 1; \
	done
