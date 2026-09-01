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
hashes outputs and updates `run_manifest.json` atomically. Never place access
tokens or private data values in command arguments or committed plans.
