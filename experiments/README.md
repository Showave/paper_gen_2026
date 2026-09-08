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

## Row-level materialization and deduplication

After clearance, a site-local exporter must write JSONL at the registry's exact
revision and provide an input specification containing the export byte hash,
exporter name/version, an admitted configuration, the recorded upstream split,
stable-ID fields, per-row license strings, and an exact role already admitted
by the registry. Every source-specific schema must project one nonempty text
field to the paper-wide canonical `dedup-text` slot; this shared projection is
what makes normalized hashes comparable across sources. The repository adapter
never fetches an arbitrary URL:

```bash
python3 scripts/materialize_public_data.py \
  --paper sft \
  --input-spec /site/reviewed/sft_exports.json \
  --output artifacts/sft-materialized
```

Real materialization fails unless every transitive dataset and model review is
`approved` and names a nonempty `review_record`. The adapter records the source
revision and stable row identifier, hashes the canonical raw row and selected
content, applies a versioned Unicode/case/whitespace normalization, removes
within-role duplicate rows, and quarantines every normalized-content cluster
that crosses candidate-source, reference, benchmark, or other registered
roles. Downstream splitters must assign a complete retained duplicate cluster
to one of score, gate, audit, or training rather than splitting its members.
It writes hash-verified acquisition and processing manifests plus retained
role-specific JSONL; the five-stage acquisition artifact must embed those
manifest and role-file hashes. Input bytes are hashed and parsed from the same
read, and verification covers both acquisition and processed files.
Before publishing a materialization, the adapter now runs the frozen
`experiments/content_gates.json` contract. It uses canonical projected text for
lexical comparison and scans every retained raw dictionary key and string value
for named PII and secret patterns. It records no matched spans and rejects
retained character-5-gram near-duplicates within or across roles.
Small artifacts use exhaustive pair comparison; larger ones use deterministic
one-permutation MinHash LSH candidate generation followed by exact Jaccard
scoring. The emitted `process/content_readiness_ledger.json` binds the gate
specification and implementation to the acquisition, content, and retained-file
hashes. Because LSH is not recall-complete, any artifact above the exhaustive
limit is recorded as `inconclusive`.

For a larger reviewed export, `--prepare-external-overlap-audit` may retain an
otherwise valid materialization whose sole unresolved failure is approximate
candidate generation. This is an explicitly unready artifact: every real
pipeline stage continues to reject it. A site-local exact engine must then
write `process/external_overlap_audit/manifest.json` and its shard files under
the materialization. The frozen
`experiments/external_overlap_audit.json` contract orders records by identifier,
indexes the complete upper triangle, and requires contiguous half-open pair
ranges that cover exactly `n*(n-1)/2` comparisons. The validator binds every
shard to the materialization, gate specification, reviewed engine source,
runtime image, and review record; rejects gaps, overlaps, approximation,
errors, and early termination; and independently checks every reported pair's
identity, role, threshold, and score. It also reconstructs a complete
shared-5-gram pair index in temporary SQLite storage and exactly scores every
such pair. Since every pair with positive Jaccard similarity shares a 5-gram
and all frozen thresholds are positive, this independently detects an omitted
qualifying pair without materializing all `n*(n-1)/2` outcomes in the bundle.
Thus shard coverage is retained as an engine-governance attestation, while
readiness depends on the independently reproduced qualifying-pair set. It
records no matched text.

The committed engine fields are intentionally null and
`execution_status` is `unfrozen_blocker`, so no large artifact can currently
pass through this route. After independent engine review and replacement
preregistration, rerun the gate; only a zero-finding, fully covered exact bundle
can replace the approximate blocker. Re-run the scanner or verify an existing
passing ledger with:

```bash
python3 scripts/run_content_gates.py \
  --materialization artifacts/sft-materialized
python3 scripts/run_content_gates.py \
  --materialization artifacts/sft-materialized \
  --verify-ledger
python3 scripts/validate_external_overlap_audit.py \
  --materialization artifacts/sft-materialized \
  --bundle artifacts/sft-materialized/process/external_overlap_audit
```

For real five-stage execution, the work directory must retain this
`acquire/` and `process/` materialization layout. The runner recomputes the
readiness ledger after both acquisition and processing, and
`splits_manifest.json`, `trajectory_schema.json`, or `cell_manifest.json`
(depending on the paper) must record its exact
`content_readiness_ledger_sha256`. Its `content_lineage` list must account for
every scanned record exactly once as either `included` with one output
partition or `excluded` with one frozen reason. The `content_partitions`
declarations name every preregistered partition and bind JSONL containing exact
retained materialization records, plus its row count, file hash, and record-ID
hash. A scanned ID cannot occur twice or carry substituted content; included
lineage must reproduce those files exactly. Every later real stage revalidates
the ledger, lineage, partition artifacts, and immutable predecessors before it
runs. A declarative check string alone is not accepted.

This is a deterministic lexical and pattern-based preflight, not a semantic,
privacy, legal, or human-subjects clearance. The detector families do not cover
multilingual paraphrase, images, or audio. Reviewed site-local semantic and
privacy audits therefore remain mandatory; a passing exhaustive ledger cannot
change a registry review from `pending` to `approved`.

The only clearance bypass is a committed, generated fixture:

```bash
make provenance-audit
make content-gates-audit
make external-overlap-audit
```

That fixture deliberately contains both a within-role duplicate and a collision
between two registered candidate-source roles. Successful execution means both
were detected, the cross-role rows were quarantined, and the retained rows pass
the content-readiness contract. The separate gate self-test uses only generated
strings to exercise each detector family and near-duplicate rejection. Every
output is marked non-empirical. The external-overlap self-test uses generated
identifiers only and verifies engine binding, complete pair-space sharding,
shard hashes, and fail-closed gap and early-termination handling.

The SFT factorial aggregator is independently executable:

```bash
make sft-factorial-audit
```

For real endpoints, `scripts/aggregate_sft_factorial.py` requires one
run-manifest-linked row per frozen seed, target, budget, cell, and tolerance.
It applies the registered failed/missing-run rule, computes lower-is-better
dominated hypervolume, evaluates the paired cost-by-packer
difference-in-differences, and forms simultaneous max-$t$ intervals by
resampling complete seed blocks. Fixture endpoints cannot be consumed in
`real` mode.

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
