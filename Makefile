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

.PHONY: all clean check-deps pdf-check source-audit provenance-audit content-gates-audit external-overlap-audit overlap-projection-audit sft-reference-audit sft-factorial-audit sft-ledger-audit rl-estimator-audit rl-fault-injection-audit rl-reconstruction-audit eval-simulation-design-audit eval-bundle-audit design-audits protocol-audits

all: $(PDFS)

%/main.pdf: %/main.tex %/references.bib $(STYLE_FILES) $(BUILD)
	$(BUILD) $*

clean:
	@for paper in $(PAPERS); do \
		$(BUILD) $$paper clean; \
	done

check-deps:
	@$(BUILD) sft_paper deps

pdf-check: $(PDFS)
	python3 scripts/check_paper_pdfs.py $(PDFS)

source-audit:
	python3 scripts/audit_public_sources.py

provenance-audit:
	python3 scripts/materialize_public_data.py --self-test

content-gates-audit:
	python3 scripts/run_content_gates.py --self-test

external-overlap-audit:
	python3 scripts/validate_external_overlap_audit.py --self-test

overlap-projection-audit:
	python3 scripts/validate_overlap_projection_contract.py --self-test

sft-reference-audit:
	python3 scripts/validate_sft_reference_contract.py --self-test

sft-factorial-audit:
	python3 scripts/aggregate_sft_factorial.py --self-test

sft-ledger-audit:
	python3 scripts/validate_sft_train_ledger.py --self-test

rl-estimator-audit:
	python3 scripts/aggregate_rl_estimator.py --self-test

rl-fault-injection-audit:
	python3 scripts/rl_gate0_fault_injections.py --self-test

rl-reconstruction-audit:
	python3 scripts/reconstruct_rl_trajectories.py --self-test

eval-simulation-design-audit:
	python3 scripts/validate_eval_simulation_design.py --self-test

eval-bundle-audit:
	python3 scripts/validate_eval_sequential_bundle.py --self-test

design-audits: overlap-projection-audit sft-reference-audit sft-factorial-audit sft-ledger-audit rl-estimator-audit rl-fault-injection-audit rl-reconstruction-audit eval-simulation-design-audit eval-bundle-audit

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
