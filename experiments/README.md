# Experiment Protocols

The three JSON plans turn each paper's experimental section into the same five
stage contract:

1. `acquire`: freeze data, provenance, licenses, and sealed resources;
2. `process`: deduplicate, normalize, split, and run leakage checks;
3. `build`: freeze model, method, software, and hardware configurations;
4. `train`: run the estimator audit or matched-budget experiment; and
5. `evaluate`: open sealed resources and aggregate the preregistered endpoints.

Inspect and validate a plan without executing external code:

```bash
python3 scripts/run_experiment_pipeline.py \
  --plan experiments/sft_pipeline.json
```

Execution is intentionally site-local. Provide a JSON object mapping each stage
to an argument array; no shell interpolation is used:

```json
{
  "acquire": ["python", "/path/to/acquire.py", "--output", "{stage_dir}"],
  "process": ["python", "/path/to/process.py", "--output", "{stage_dir}"],
  "build": ["python", "/path/to/build.py", "--output", "{stage_dir}"],
  "train": ["python", "/path/to/train.py", "--output", "{stage_dir}"],
  "evaluate": ["python", "/path/to/evaluate.py", "--output", "{stage_dir}"]
}
```

Available placeholders are `{repo}`, `{work_dir}`, `{stage_dir}`, and
`{paper}`. Each stage command must produce every artifact declared in its plan
and a `{stage_dir}/costs.json` containing the declared cost keys. The runner
hashes the plan, referenced preregistration, command mapping, runner, and
outputs and updates
`run_manifest.json` atomically. It refuses to resume a work directory when an
immutable hash or Git revision changed unless `--force` starts a replacement
run. Never place access tokens or private data values in command arguments or
committed plans.

## Public-source preflight

`public_source_registry.json` pins full Hugging Face revisions, admitted
configurations, intended roles, license evidence, and source-specific manual
review requirements. The committed `public_source_snapshot.json` records the
metadata observed on 2026-09-03. Validate it without network access:

```bash
python3 scripts/audit_public_sources.py
```

To deliberately inspect upstream drift and regenerate the snapshot:

```bash
python3 scripts/audit_public_sources.py --refresh
```

Neither command downloads rows or weights. Automated license tags and pinned
card text are triage evidence, not legal or ethical clearance. Real pipeline
execution fails unless every registry entry referenced by that paper is
approved. The standalone `--require-manual-clearance` flag intentionally checks
the entire registry. Do not change a pending status to `approved` without
linking the site-local review record in the resulting acquisition artifact.

## Reproducible design audits

The repository includes deterministic, standard-library-only commands that
exercise all five stages without downloading data, training a language model,
or collecting human judgments:

```bash
for paper in sft rl eval; do
  python3 scripts/run_experiment_pipeline.py \
    --plan "experiments/${paper}_pipeline.json" \
    --execute \
    --synthetic-audit \
    --commands experiments/synthetic_commands.json \
    --work-dir "artifacts/${paper}-synthetic-audit"
done
```

The SFT audit enumerates feasible packed subsets and checks the profiled-cost
identity and actual-change gate. The RL audit exactly enumerates a finite
trajectory law, then tests the augmented estimator, mixture-weight bound, and
deliberately incorrect ratios. The evaluation audit simulates predictable
adaptive acquisition with misspecified outcome predictions and checks the
sequential augmented Hansen--Hurwitz identity, positivity, and fixed-budget
coverage.

Every synthetic artifact is marked `empirical_evidence: false`. These audits
are executable tests of equations, stage interfaces, and logging; their
numbers must not be copied into the papers' model-training or human-study
result tables.
