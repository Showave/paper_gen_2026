#!/usr/bin/env python3
"""Validate, display, or execute a paper experiment protocol.

Dry-run planning is the default. Execution requires a separate JSON file that
maps each stage name to an argv array, which keeps site-specific launchers and
private dataset locations out of the anonymous paper artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_STAGES = ("acquire", "process", "build", "train", "evaluate")
REQUIRED_STAGE_FIELDS = ("objective", "inputs", "artifacts", "checks", "costs")


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        fail(f"file not found: {path}")
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON in {path}: {exc}")


def canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_plan(plan: Any, source: Path) -> None:
    if not isinstance(plan, dict):
        fail(f"{source} must contain a JSON object")
    for field in ("schema_version", "paper", "purpose", "stages"):
        if field not in plan:
            fail(f"{source} is missing {field!r}")
    if plan["schema_version"] != 1:
        fail(f"{source} has unsupported schema_version")
    if not isinstance(plan["stages"], list):
        fail(f"{source} stages must be a list")
    preregistration = plan.get("preregistration")
    if not isinstance(preregistration, dict) or not all(
        isinstance(preregistration.get(field), str) and preregistration[field]
        for field in ("path", "section")
    ):
        fail(f"{source} needs preregistration path and section")
    names = tuple(stage.get("name") for stage in plan["stages"])
    if names != REQUIRED_STAGES:
        fail(f"{source} stages must be ordered as {REQUIRED_STAGES}")
    for stage in plan["stages"]:
        for field in REQUIRED_STAGE_FIELDS:
            value = stage.get(field)
            if field == "objective":
                if not isinstance(value, str) or not value.strip():
                    fail(f"stage {stage['name']} needs a nonempty objective")
            elif not isinstance(value, list) or not value:
                fail(f"stage {stage['name']} field {field!r} must be a nonempty list")
        for artifact in stage["artifacts"]:
            path = Path(artifact)
            if path.is_absolute() or ".." in path.parts:
                fail(f"artifact must be work-directory relative: {artifact}")


def load_preregistration(
    plan: dict[str, Any], repo: Path
) -> tuple[dict[str, Any], Path]:
    declaration = plan["preregistration"]
    path = Path(declaration["path"])
    if path.is_absolute() or ".." in path.parts:
        fail("preregistration path must be repository relative")
    source = repo / path
    preregistration = load_json(source)
    if not isinstance(preregistration, dict) or preregistration.get("schema_version") != 1:
        fail(f"{source} must contain a schema_version 1 object")
    section = preregistration.get(declaration["section"])
    if not isinstance(section, dict) or not section:
        fail(f"{source} has no nonempty section {declaration['section']!r}")
    if declaration["section"] != plan["paper"]:
        fail("preregistration section must match the paper")
    validate_preregistration_section(section, plan["paper"], preregistration, repo)
    return preregistration, source


def nested(value: dict[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            fail(f"preregistration is missing {path!r}")
        current = current[part]
    return current


def referenced_registry_keys(section: dict[str, Any]) -> set[str]:
    references: set[str] = set()
    for field, value in section.items():
        if field == "source_keys" or field.endswith("_model_key") or field.endswith(
            "_model_keys"
        ):
            values = value if isinstance(value, list) else [value]
            if not all(isinstance(item, str) and item for item in values):
                fail(f"preregistration field {field!r} must contain registry keys")
            references.update(values)
    return references


def validate_preregistration_section(
    section: dict[str, Any],
    paper: str,
    preregistration: dict[str, Any],
    repo: Path,
) -> None:
    source_keys = nested(section, "source_keys")
    if not isinstance(source_keys, list) or not source_keys:
        fail(f"{paper} preregistration needs source_keys")
    registry_path = Path(str(preregistration.get("source_registry", "")))
    if registry_path.is_absolute() or ".." in registry_path.parts:
        fail("source_registry must be repository relative")
    registry = load_json(repo / registry_path)
    entries = {
        item.get("key"): item
        for item in registry.get("sources", [])
        if isinstance(item, dict) and isinstance(item.get("key"), str)
    }
    references = referenced_registry_keys(section)
    if not references.issubset(entries):
        fail(f"{paper} preregistration refers to an unregistered source or model")
    for key in source_keys:
        if entries[key].get("kind") != "dataset":
            fail(f"{paper} source key {key!r} is not a dataset")
    for key in references - set(source_keys):
        if entries[key].get("kind") != "model":
            fail(f"{paper} model key {key!r} is not a model")
    for key in references:
        usage = entries[key].get("usage")
        if not isinstance(usage, list) or not any(
            isinstance(role, str) and role.startswith(paper + ":") for role in usage
        ):
            fail(f"{paper} registry key {key!r} has no declared role for this paper")

    seeds = nested(section, "seeds") if paper != "eval" else None
    if seeds is not None and (
        not isinstance(seeds, list)
        or len(seeds) != 5
        or len(set(seeds)) != len(seeds)
        or not all(isinstance(seed, int) for seed in seeds)
    ):
        fail(f"{paper} preregistration needs five unique integer seeds")

    if paper == "sft":
        window = nested(section, "optimizer_window")
        if window["physical_packs_per_update"] != window["microbatches_per_update"]:
            fail("SFT packs and microbatches per update must match")
        capacity = (
            window["physical_packs_per_update"] * window["maximum_sequence_length"]
        )
        if not (
            0
            < window["response_tokens_minimum"]
            <= window["response_tokens_maximum"]
            <= capacity
        ):
            fail("SFT response-token window exceeds physical capacity")
        selector = nested(section, "selector")
        if selector["shortlist_size"] > selector["candidate_pool_size"]:
            fail("SFT shortlist exceeds candidate pool")
        gate = nested(section, "actual_update_gate")
        if gate.get("mode") == "formal_hoeffding":
            required = math.ceil(
                2
                * gate["clipped_example_nll_bound"] ** 2
                * math.log(
                    gate["protected_slices"]
                    * len(gate["scale_grid"])
                    * gate["maximum_rounds"]
                    / gate["family_failure_probability"]
                )
                / gate["per_round_tolerance"] ** 2
            )
            if gate["samples_per_slice_round"] != required:
                fail("SFT gate sample count does not reproduce the Hoeffding formula")
        else:
            if gate.get("population_coverage_claim") is not False:
                fail("heuristic SFT gate cannot claim population coverage")
            if gate.get("sampling") != "with replacement from the frozen gate distribution":
                fail("SFT gate sampling must match the stated IID law")
            if not math.isclose(
                gate["cumulative_tolerance_budget"],
                gate["maximum_rounds"] * gate["per_round_tolerance"],
            ):
                fail("SFT cumulative tolerance budget does not reproduce")
        total_draws = (
            gate["protected_slices"]
            * gate["maximum_rounds"]
            * gate["samples_per_slice_round"]
        )
        if gate.get("total_gate_draws", total_draws) != total_draws:
            fail("SFT total gate draws do not reproduce")
    elif paper == "rl":
        audit = nested(section, "estimator_audit")
        alphas = audit["alpha_grid"]
        if not isinstance(alphas, list) or not alphas or not all(
            isinstance(alpha, (int, float)) and 0 < alpha <= 1 for alpha in alphas
        ):
            fail("RL alpha_grid must lie in (0, 1]")
        count_rows = audit["stratum_counts_by_alpha"]
        restriction = audit.get("allocation_restriction_mode")
        if not isinstance(count_rows, list) or not count_rows:
            fail("RL stratum counts must be nonempty")
        if restriction == "alpha_equals_sampling_fraction" and len(
            count_rows
        ) != len(alphas):
            fail("restricted RL pilot needs exactly one allocation per alpha")
        for row in count_rows:
            if not isinstance(row.get("allocation_id"), str) or not row[
                "allocation_id"
            ].strip():
                fail("RL allocation cells need stable allocation_id values")
            total = row["current"] + row["stale"]
            if total <= 0:
                fail("RL stratum counts must have positive totals")
            if (
                restriction == "alpha_equals_sampling_fraction"
                and not math.isclose(row["current"] / total, row["alpha"])
            ):
                fail("RL pilot alpha must equal current count divided by total count")
        if {row["alpha"] for row in count_rows} != set(alphas):
            fail("RL stratum-count alpha values differ from alpha_grid")
        allocation_cells = {
            (row["alpha"], row["allocation_id"], row["current"], row["stale"])
            for row in count_rows
        }
        if len(allocation_cells) != len(count_rows):
            fail("RL alpha-allocation cells must be unique")
        if restriction not in {
            "alpha_equals_sampling_fraction",
            "crossed_alpha_allocation",
        }:
            fail("RL allocation_restriction_mode is unsupported")
        if restriction == "alpha_equals_sampling_fraction" and len(
            {row["allocation_id"] for row in count_rows}
        ) != len(count_rows):
            fail("restricted RL allocation_id values must be unique")
        if restriction == "crossed_alpha_allocation":
            allocations: dict[str, tuple[int, int]] = {}
            for row in count_rows:
                counts = (row["current"], row["stale"])
                previous = allocations.setdefault(row["allocation_id"], counts)
                if previous != counts:
                    fail("crossed RL allocation_id must have fixed source counts")
            if len(allocations) < 2 or {
                (row["alpha"], row["allocation_id"]) for row in count_rows
            } != {
                (alpha, allocation_id)
                for alpha in alphas
                for allocation_id in allocations
            }:
                fail("crossed RL design must be a complete alpha-allocation product")
        fractions = [
            audit["nuisance_fit_prompt_fraction"],
            audit["alpha_selection_prompt_fraction"],
            audit["untouched_confirmation_prompt_fraction"],
        ]
        if any(fraction <= 0 for fraction in fractions) or not math.isclose(
            sum(fractions), 1.0
        ):
            fail("RL prompt audit fractions must be positive and sum to one")
        decoder = nested(section, "decoder")
        for field in (
            "top_k",
            "top_p",
            "repetition_penalty",
            "minimum_length",
            "forced_tokens",
        ):
            if decoder[field] is not None:
                fail(f"RL theorem-facing decoder requires null {field}")
        if nested(section, "failure_semantics")["refill_failed_or_slow_trajectories"]:
            fail("RL theorem-facing batches cannot refill failed trajectories")
        if audit.get("reference_replicas_per_target_law") != 2:
            fail("RL cross-reference MSE requires exactly two reference replicas")
        if audit.get("reference_current_policy_trajectories_per_replica", 0) <= 0:
            fail("RL reference replicas must contain positive trajectory counts")
    else:
        acquisition = nested(section, "acquisition")
        epsilon = acquisition["exploration_epsilon"]
        if not isinstance(epsilon, (int, float)) or not 0 < epsilon <= 1:
            fail("evaluation exploration epsilon must lie in (0, 1]")
        if acquisition["adaptive_fixed_budget_attempts"] != (
            acquisition["uniform_pilot_attempts"]
            + acquisition["adaptive_attempts"]
        ):
            fail("evaluation adaptive fixed budget does not reproduce")
        if acquisition["adaptive_attempts"] % acquisition["batch_size"] != 0:
            fail("evaluation adaptive attempts must be a whole number of batches")
        loss = nested(section, "decision.loss_matrix")
        if len(loss) != 4 or any(not isinstance(row, list) or len(row) != 3 for row in loss):
            fail("evaluation loss matrix must be 4 by 3")
        frame = nested(section, "frame")
        counts = frame.get("root_items_by_source")
        if not isinstance(counts, dict) or sum(counts.values()) != frame["root_items"]:
            fail("evaluation per-source root counts must reproduce root_items")
        if counts.get("livebench-coding-eval") != 128:
            fail("evaluation pinned LiveBench coding frame contains 128 roots")
        if nested(section, "working_posterior").get("common_monte_carlo_draws", 0) <= 0:
            fail("evaluation EVI needs positive frozen common Monte Carlo draws")


def require_manual_source_clearance(
    preregistration: dict[str, Any], section: dict[str, Any], repo: Path
) -> None:
    registry_path = Path(str(preregistration.get("source_registry", "")))
    if (
        not registry_path.parts
        or registry_path.is_absolute()
        or ".." in registry_path.parts
    ):
        fail("preregistration needs a repository-relative source_registry")
    registry = load_json(repo / registry_path)
    sources = registry.get("sources") if isinstance(registry, dict) else None
    if not isinstance(sources, list) or not sources:
        fail("public source registry has no sources")
    references = referenced_registry_keys(section)
    by_key = {
        item.get("key"): item for item in sources if isinstance(item, dict)
    }
    blocked = []
    for key in sorted(references):
        item = by_key.get(key)
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("manual_review"), dict)
            or item["manual_review"].get("status") != "approved"
            or not isinstance(item["manual_review"].get("review_record"), str)
            or not item["manual_review"]["review_record"].strip()
        ):
            blocked.append(key)
    if blocked:
        fail(
            "real execution is blocked by pending manual source clearance: "
            + ", ".join(blocked)
        )


def require_execution_ready(
    section: dict[str, Any], paper: str, source_registry: dict[str, Any]
) -> None:
    def sha256(value: Any) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        )

    if paper == "rl":
        stale_actor = nested(section, "stale_actor")
        if stale_actor.get("construction_status") != "frozen":
            fail(
                "real RL execution is blocked until the current/stale policy pair, "
                "KL law, decoder, and verifier are frozen"
            )
        pair = stale_actor.get("frozen_policy_pair")
        pair_fields = {
            "current_tensor_sha256",
            "behavior_tensor_sha256",
            "current_construction_manifest_sha256",
            "tokenizer_sha256",
            "chat_template_sha256",
            "decoder_specification_sha256",
        }
        if (
            not isinstance(pair, dict)
            or set(pair) != pair_fields
            or not all(sha256(value) for value in pair.values())
        ):
            fail("real RL execution needs every frozen policy-pair SHA-256")
        kl_audit = stale_actor.get("kl_audit")
        if (
            not isinstance(kl_audit, dict)
            or not sha256(kl_audit.get("prompt_law_manifest_sha256"))
            or not sha256(kl_audit.get("estimator_implementation_sha256"))
            or not isinstance(kl_audit.get("sample_size"), int)
            or kl_audit["sample_size"] <= 0
            or not isinstance(kl_audit.get("estimator"), str)
            or not kl_audit["estimator"].strip()
        ):
            fail("real RL execution needs a fully frozen KL audit")
        verifier = stale_actor.get("verifier")
        registry_entries = {
            item.get("key"): item
            for item in source_registry.get("sources", [])
            if isinstance(item, dict)
        }
        verifier_key = verifier.get("registry_key") if isinstance(verifier, dict) else None
        verifier_entry = registry_entries.get(verifier_key)
        revision = (
            verifier.get("implementation_revision")
            if isinstance(verifier, dict)
            else None
        )
        if (
            not isinstance(verifier, dict)
            or not isinstance(verifier_key, str)
            or not isinstance(verifier_entry, dict)
            or "rl:verifier" not in verifier_entry.get("usage", [])
            or verifier_entry.get("manual_review", {}).get("status") != "approved"
            or not isinstance(revision, str)
            or len(revision) != 40
            or any(character not in "0123456789abcdef" for character in revision)
            or revision != verifier_entry.get("revision")
            or not sha256(verifier.get("sealed_answer_manifest_sha256"))
        ):
            fail("real RL execution needs a registered frozen verifier")
    if paper == "eval":
        if section.get("execution_status") != "frozen":
            fail(
                "real evaluation execution is blocked until estimand-generating "
                "prompts, decoders, responses, and simulation DGPs are frozen"
            )
        hashes = section.get("frozen_artifact_hashes")
        hash_fields = {
            "prompt_generator_sha256",
            "rendered_prompt_bundle_sha256",
            "decoder_and_coupling_sha256",
            "paired_response_manifest_sha256",
            "simulation_implementation_sha256",
            "posterior_configuration_sha256",
            "frame_functional_sha256",
            "newton_update_implementation_sha256",
        }
        if (
            not isinstance(hashes, dict)
            or set(hashes) != hash_fields
            or not all(sha256(value) for value in hashes.values())
        ):
            fail("real evaluation execution needs every estimand artifact SHA-256")
        if nested(section, "simulation").get("execution_status") != "frozen":
            fail("real evaluation execution needs a frozen executable simulation")
        human = nested(section, "human_protocol")
        audit = human.get("adaptive_audit")
        audit_fields = {
            "equivalence_margin",
            "covariance_implementation_sha256",
            "small_cluster_correction",
            "failure_rule",
        }
        if (
            human.get("adaptive_audit_equivalence_status") != "frozen"
            or not isinstance(audit, dict)
            or set(audit) != audit_fields
            or not isinstance(audit["equivalence_margin"], (int, float))
            or audit["equivalence_margin"] <= 0
            or not sha256(audit["covariance_implementation_sha256"])
            or not isinstance(audit["small_cluster_correction"], str)
            or not audit["small_cluster_correction"].strip()
            or not isinstance(audit["failure_rule"], str)
            or not audit["failure_rule"].strip()
        ):
            fail("real evaluation execution needs a frozen audit contrast")


def print_plan(
    plan: dict[str, Any],
    selected: set[str],
    preregistration_source: Path,
    preregistration_digest: str,
) -> None:
    print(f"{plan['paper']}: {plan['purpose']}")
    print(
        f"preregistration: {preregistration_source} "
        f"(sha256 {preregistration_digest})"
    )
    for stage in plan["stages"]:
        if stage["name"] not in selected:
            continue
        print(f"\n[{stage['name']}] {stage['objective']}")
        for field in ("inputs", "artifacts", "checks", "costs"):
            print(f"  {field}:")
            for item in stage[field]:
                print(f"    - {item}")


def git_revision(repo: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def load_committed_json(repo: Path, relative_path: str) -> Any:
    result = subprocess.run(
        ["git", "show", f"HEAD:{relative_path}"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        fail(f"could not read committed {relative_path}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        fail(f"committed {relative_path} is invalid JSON: {exc}")


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_artifact(path: Path) -> dict[str, Any]:
    if path.is_file():
        return {"path": str(path), "sha256": digest_file(path), "bytes": path.stat().st_size}
    if path.is_dir():
        files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
        members = [
            {
                "path": str(candidate.relative_to(path)),
                "sha256": digest_file(candidate),
                "bytes": candidate.stat().st_size,
            }
            for candidate in files
        ]
        return {"path": str(path), "tree_sha256": canonical_digest(members), "members": members}
    fail(f"declared artifact was not produced: {path}")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    fail(f"{path}:{line_number}: invalid JSONL: {exc}")
                if not isinstance(row, dict):
                    fail(f"{path}:{line_number}: expected an object")
                rows.append(row)
    except FileNotFoundError:
        fail(f"declared ledger was not produced: {path}")
    if not rows:
        fail(f"declared ledger is empty: {path}")
    return rows


def validate_real_stage(
    paper: str,
    stage_name: str,
    work_dir: Path,
    section: dict[str, Any],
    run_manifest: dict[str, Any],
) -> dict[str, Any]:
    if paper == "sft" and stage_name == "evaluate":
        summary = load_json(work_dir / "evaluate" / "summary.json")
        factorial = nested(section, "primary_factorial")
        if (
            summary.get("execution_class") != "real"
            or summary.get("empirical_evidence") is not True
            or summary.get("bootstrap_replicates")
            != factorial["bootstrap_replicates"]
            or summary.get("bootstrap_seed") != factorial["bootstrap_seed"]
            or summary.get("preregistration_section_sha256")
            != canonical_digest(section)
            or not isinstance(summary.get("contrasts"), list)
            or len(summary["contrasts"]) != 6
        ):
            fail("SFT summary does not match the frozen factorial aggregation")
        return {"factorial_contrasts": 6, "execution_class": "real"}

    if paper == "eval" and stage_name == "acquire":
        frame = load_json(work_dir / "acquire" / "frame_manifest.json")
        expected = nested(section, "frame.root_items_by_source")
        if frame.get("root_items_by_source") != expected:
            fail("evaluation frame manifest has incorrect per-source root counts")
        masses = frame.get("target_source_mass")
        if (
            not isinstance(masses, dict)
            or set(masses) != set(expected)
            or any(not math.isclose(value, 1 / 3) for value in masses.values())
        ):
            fail("evaluation frame manifest must assign one-third mass per source")
        if not math.isclose(frame.get("target_root_mass_sum", 0), 1.0):
            fail("evaluation frame target root masses must sum to one")
        roots = frame.get("roots")
        if not isinstance(roots, list) or len(roots) != sum(expected.values()):
            fail("evaluation frame manifest must list every frozen root")
        row_uids: set[str] = set()
        observed_counts = {key: 0 for key in expected}
        observed_mass = 0.0
        for root in roots:
            if not isinstance(root, dict):
                fail("evaluation frame root must be an object")
            source = root.get("source_key")
            row_uid = root.get("row_uid")
            mass = root.get("target_mass")
            if (
                source not in expected
                or not isinstance(row_uid, str)
                or row_uid in row_uids
                or not isinstance(mass, (int, float))
                or not math.isclose(mass, 1 / (3 * expected[source]))
            ):
                fail("evaluation frame has an invalid, duplicate, or misweighted root")
            row_uids.add(row_uid)
            observed_counts[source] += 1
            observed_mass += mass
        if observed_counts != expected or not math.isclose(observed_mass, 1.0):
            fail("evaluation root records do not reproduce counts and total mass")
        return {"root_items": sum(expected.values()), "source_mass_valid": True}

    if paper == "eval" and stage_name == "build":
        model = load_json(
            work_dir / "build" / "evaluation_model_manifest.json"
        )
        evi = model.get("evi") if isinstance(model, dict) else None
        posterior = nested(section, "working_posterior")
        required_hashes = (
            "posterior_configuration_sha256",
            "frame_functional_sha256",
            "newton_update_implementation_sha256",
        )
        frozen_hashes = section["frozen_artifact_hashes"]
        if (
            not isinstance(evi, dict)
            or evi.get("common_monte_carlo_draws")
            != posterior["common_monte_carlo_draws"]
            or evi.get("standard_normal_common_random_seed")
            != posterior["standard_normal_common_random_seed"]
            or not all(
                isinstance(evi.get(field), str)
                and len(evi[field]) == 64
                and all(
                    character in "0123456789abcdef"
                    for character in evi[field]
                )
                and evi[field] == frozen_hashes[field]
                for field in required_hashes
            )
        ):
            fail("evaluation model manifest does not freeze the EVI computation")
        return {"evi_configuration_valid": True}

    if paper == "eval" and stage_name == "train":
        rows = load_jsonl(work_dir / "train" / "acquisition_ledger.jsonl")
        expected_slots = nested(section, "acquisition.adaptive_fixed_budget_attempts")
        slot_ids = {row.get("slot_id") for row in rows}
        if (
            len(rows) != expected_slots
            or slot_ids != set(range(expected_slots))
            or any(row.get("replacement_attempted") is not False for row in rows)
            or any(not isinstance(row.get("response_observed"), bool) for row in rows)
        ):
            fail("evaluation acquisition ledger violates the fixed-slot contract")
        missing = sum(not row["response_observed"] for row in rows)
        return {"fixed_slots": expected_slots, "nonresponses": missing}

    if paper == "eval" and stage_name == "evaluate":
        summary = load_json(work_dir / "evaluate" / "summary.json")
        audit_rows = load_jsonl(work_dir / "evaluate" / "audit_ledger.jsonl")
        human = nested(section, "human_protocol")
        audit = human["adaptive_audit"]
        audit_slots = nested(
            section, "acquisition.sealed_target_random_audit_attempts"
        )
        if (
            len(audit_rows) != audit_slots
            or {row.get("slot_id") for row in audit_rows}
            != set(range(audit_slots))
            or any(
                row.get("replacement_attempted") is not False
                or not isinstance(row.get("response_observed"), bool)
                or row.get("target_to_draw_weight") != 1.0
                for row in audit_rows
            )
        ):
            fail("evaluation audit ledger violates the fixed-slot target draw")
        lower_values = []
        upper_values = []
        for row in audit_rows:
            if row["response_observed"]:
                if row.get("outcome") not in {-1, 0, 1}:
                    fail("observed audit outcome must be -1, 0, or 1")
                lower_values.append(row["outcome"])
                upper_values.append(row["outcome"])
            else:
                lower_values.append(-1)
                upper_values.append(1)
        alpha = 1 - nested(section, "decision.confidence_level")
        radius = math.sqrt(2 * math.log(2 / alpha) / audit_slots)
        expected_interval = [
            max(-1.0, sum(lower_values) / audit_slots - radius),
            min(1.0, sum(upper_values) / audit_slots + radius),
        ]
        train_validation = run_manifest["stages"].get("train", {}).get(
            "semantic_validation", {}
        )
        reported = summary.get("adaptive_audit_contrast")
        if (
            train_validation.get("fixed_slots")
            != nested(section, "acquisition.adaptive_fixed_budget_attempts")
            or not isinstance(reported, dict)
            or reported.get("equivalence_margin") != audit["equivalence_margin"]
            or reported.get("covariance_implementation_sha256")
            != audit["covariance_implementation_sha256"]
            or reported.get("small_cluster_correction")
            != audit["small_cluster_correction"]
            or reported.get("audit_fixed_slots") != audit_slots
            or not isinstance(reported.get("audit_interval"), list)
            or len(reported["audit_interval"]) != 2
            or any(
                not math.isclose(observed, expected)
                for observed, expected in zip(
                    reported["audit_interval"], expected_interval
                )
            )
        ):
            fail("evaluation summary does not match the frozen audit contrast")
        return {
            "audit_contrast_valid": True,
            "audit_interval_recomputed": expected_interval,
        }

    if paper == "rl" and stage_name == "train":
        rows = load_jsonl(work_dir / "train" / "trajectory_ledger.jsonl")
        audit = nested(section, "estimator_audit")
        dimension = audit["projection_dimension"]
        count_by_cell = {
            (row["alpha"], row["allocation_id"]): (row["current"], row["stale"])
            for row in audit["stratum_counts_by_alpha"]
        }
        attempt_ids: set[str] = set()
        trajectory_ids: set[str] = set()
        reference_attempts: dict[tuple[str, str], int] = {}
        candidate_attempts: dict[tuple[str, float, str, int, str], int] = {}
        confirmation_cells: set[tuple[float, str]] = set()
        failures = 0
        for row in rows:
            attempt_id = row.get("attempt_id")
            law = row.get("target_law")
            role = row.get("estimator_role")
            failed = row.get("infrastructure_failure")
            source = row.get("source_component")
            if (
                not isinstance(attempt_id, str)
                or attempt_id in attempt_ids
                or law not in {"select", "confirm"}
                or role not in {"candidate", "r1", "r2"}
                or not isinstance(failed, bool)
                or source not in {"current", "stale"}
            ):
                fail("RL trajectory ledger has an invalid or duplicate attempt")
            attempt_ids.add(attempt_id)
            if role in {"r1", "r2"}:
                if source != "current":
                    fail("RL reference trajectories must use the current policy")
                key = (law, role)
                reference_attempts[key] = reference_attempts.get(key, 0) + 1
            else:
                alpha = row.get("alpha")
                allocation_id = row.get("allocation_id")
                replication = row.get("replication_id")
                if (
                    (alpha, allocation_id) not in count_by_cell
                    or not isinstance(replication, int)
                    or replication < 0
                ):
                    fail("RL candidate attempt has invalid alpha or replication")
                if law == "select" and replication >= audit["selection_replications"]:
                    fail("RL selection replication is outside the frozen range")
                if law == "confirm":
                    confirmation_cells.add((alpha, allocation_id))
                    if replication >= audit["confirmation_replications"]:
                        fail("RL confirmation replication is outside the frozen range")
                key = (law, alpha, allocation_id, replication, source)
                candidate_attempts[key] = candidate_attempts.get(key, 0) + 1
            if failed:
                failures += 1
                continue
            trajectory_id = row.get("trajectory_id")
            projected = row.get("projected_gradient")
            if (
                not isinstance(trajectory_id, str)
                or trajectory_id in trajectory_ids
                or row.get("use_count") != 1
                or row.get("scored_outcome") is not True
                or not isinstance(projected, list)
                or len(projected) != dimension
                or not all(
                    isinstance(value, (int, float)) and math.isfinite(value)
                    for value in projected
                )
            ):
                fail(
                    "RL successful trajectory must be unique, scored, single-use, "
                    "and have a finite projected gradient"
                )
            trajectory_ids.add(trajectory_id)
        reference_count = audit["reference_current_policy_trajectories_per_replica"]
        for law in ("select", "confirm"):
            for role in ("r1", "r2"):
                if reference_attempts.get((law, role), 0) != reference_count:
                    fail(f"RL {law} {role} reference count is incorrect")
        for (alpha, allocation_id), (current, stale) in count_by_cell.items():
            for replication in range(audit["selection_replications"]):
                if candidate_attempts.get(
                    ("select", alpha, allocation_id, replication, "current"), 0
                ) != current or candidate_attempts.get(
                    ("select", alpha, allocation_id, replication, "stale"), 0
                ) != stale:
                    fail("RL selection candidate allocation is incomplete")
        if len(confirmation_cells) != 1:
            fail("RL confirmation ledger must contain exactly one selected cell")
        selected_alpha, selected_allocation_id = next(iter(confirmation_cells))
        current, stale = count_by_cell[(selected_alpha, selected_allocation_id)]
        for replication in range(audit["confirmation_replications"]):
            if candidate_attempts.get(
                (
                    "confirm",
                    selected_alpha,
                    selected_allocation_id,
                    replication,
                    "current",
                ),
                0,
            ) != current or candidate_attempts.get(
                (
                    "confirm",
                    selected_alpha,
                    selected_allocation_id,
                    replication,
                    "stale",
                ),
                0,
            ) != stale:
                fail("RL confirmation candidate allocation is incomplete")
        return {
            "attempts": len(rows),
            "infrastructure_failures": failures,
            "trajectory_ids_disjoint": True,
            "selected_alpha": selected_alpha,
            "selected_allocation_id": selected_allocation_id,
            "selection_replications_per_alpha": audit["selection_replications"],
            "confirmation_replications": audit["confirmation_replications"],
        }

    if paper == "rl" and stage_name == "evaluate":
        summary = load_json(work_dir / "evaluate" / "summary.json")
        train_validation = run_manifest["stages"].get("train", {}).get(
            "semantic_validation", {}
        )
        failures = train_validation.get("infrastructure_failures")
        selected_alpha = train_validation.get("selected_alpha")
        selected_allocation_id = train_validation.get("selected_allocation_id")
        audit = nested(section, "estimator_audit")
        projection_specification_sha256 = canonical_digest(
            {
                "dimension": audit["projection_dimension"],
                "seed": audit["projection_seed"],
                "distribution": audit["projection_distribution"],
                "parameter_order": audit["parameter_order"],
            }
        )
        expected_cells = {
            f"{row['alpha']}|{row['allocation_id']}"
            for row in audit["stratum_counts_by_alpha"]
        }
        reported_mse = summary.get("projected_mse_by_cell", {})
        cell_results = summary.get("cell_results")
        if summary.get("projected_mse_formula") != "(g_hat-R1)^T(g_hat-R2)":
            fail("RL summary uses the wrong projected MSE formula")
        if summary.get("three_way_reference_bootstrap") is not True:
            fail("RL summary must resample candidate, R1, and R2 sources separately")
        if (
            summary.get("bootstrap_replicates")
            != audit["mse_bootstrap_replicates"]
            or summary.get("bootstrap_seed") != audit["mse_bootstrap_seed"]
            or summary.get("projection_dimension") != audit["projection_dimension"]
            or summary.get("projection_seed") != audit["projection_seed"]
            or summary.get("projection_specification_sha256")
            != projection_specification_sha256
            or summary.get("selected_alpha") != selected_alpha
            or summary.get("selected_allocation_id") != selected_allocation_id
            or set(reported_mse) != expected_cells
            or not all(
                isinstance(value, (int, float)) and math.isfinite(value)
                for value in reported_mse.values()
            )
            or not isinstance(cell_results, list)
            or len(cell_results) != len(expected_cells)
        ):
            fail("RL summary does not match the frozen MSE audit")
        objectives: list[tuple[float, float, str]] = []
        seen_cells: set[str] = set()
        for result in cell_results:
            if not isinstance(result, dict):
                fail("RL cell result must be an object")
            cell = f"{result.get('alpha')}|{result.get('allocation_id')}"
            mse = result.get("projected_mse")
            accelerator_seconds = result.get("accelerator_seconds")
            interval = result.get("three_way_bootstrap_interval")
            if (
                cell not in expected_cells
                or cell in seen_cells
                or mse != reported_mse[cell]
                or not isinstance(accelerator_seconds, (int, float))
                or accelerator_seconds <= 0
                or not isinstance(interval, list)
                or len(interval) != 2
                or not all(
                    isinstance(value, (int, float)) and math.isfinite(value)
                    for value in interval
                )
            ):
                fail("RL cell result is incomplete or inconsistent")
            seen_cells.add(cell)
            objectives.append(
                (
                    mse * accelerator_seconds,
                    -float(result["alpha"]),
                    str(result["allocation_id"]),
                )
            )
        selected = min(objectives)
        selected_result = next(
            result
            for result in cell_results
            if float(result["alpha"]) == -selected[1]
            and str(result["allocation_id"]) == selected[2]
        )
        if (
            selected_result["alpha"] != selected_alpha
            or selected_result["allocation_id"] != selected_allocation_id
        ):
            fail("RL selected cell is not the frozen MSE-times-cost argmin")
        if failures is None or summary.get("h1_eligible") != (failures == 0):
            fail("RL H1 eligibility must be false after any infrastructure failure")
        return {"h1_eligible": failures == 0}

    return {"declared_checks_require_site_validator": True}


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def render_argv(argv: Any, variables: dict[str, str], stage_name: str) -> list[str]:
    if not isinstance(argv, list) or not argv or not all(isinstance(arg, str) for arg in argv):
        fail(f"command for {stage_name} must be a nonempty JSON array of strings")
    rendered = []
    for arg in argv:
        for name, value in variables.items():
            arg = arg.replace("{" + name + "}", value)
        rendered.append(arg)
    return rendered


def repository_command_digests(
    commands: dict[str, Any], repo: Path, work_dir: Path, paper: str
) -> list[dict[str, str]]:
    files: dict[str, str] = {}
    for stage_name, command in commands.items():
        variables = {
            "repo": str(repo),
            "work_dir": str(work_dir),
            "stage_dir": str(work_dir / stage_name),
            "paper": paper,
        }
        for argument in render_argv(command, variables, stage_name):
            candidate = Path(argument)
            if not candidate.is_absolute():
                candidate = repo / candidate
            try:
                resolved = candidate.resolve(strict=True)
            except FileNotFoundError:
                continue
            if not resolved.is_file() or not resolved.is_relative_to(repo):
                continue
            if resolved.is_relative_to(work_dir):
                continue
            relative = str(resolved.relative_to(repo))
            files[relative] = digest_file(resolved)
    return [
        {"path": path, "sha256": digest}
        for path, digest in sorted(files.items())
    ]


def execute(
    plan: dict[str, Any],
    commands: Any,
    selected: set[str],
    repo: Path,
    work_dir: Path,
    force: bool,
    preregistration_digest: str,
    source_registry_digest: str,
    preregistration_section: dict[str, Any],
    synthetic_audit: bool,
) -> None:
    if not isinstance(commands, dict):
        fail("commands file must map stage names to argv arrays")
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = work_dir / "run_manifest.json"
    command_files = repository_command_digests(
        commands, repo, work_dir, plan["paper"]
    )
    manifest = {
        "schema_version": 1,
        "paper": plan["paper"],
        "plan_sha256": canonical_digest(plan),
        "preregistration_sha256": preregistration_digest,
        "source_registry_sha256": source_registry_digest,
        "synthetic_audit": synthetic_audit,
        "commands_sha256": canonical_digest(commands),
        "repository_command_files": command_files,
        "repository_command_files_sha256": canonical_digest(command_files),
        "runner_sha256": digest_file(Path(__file__).resolve()),
        "git_revision": git_revision(repo),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stages": {},
    }
    if manifest_path.exists() and not force:
        previous = load_json(manifest_path)
        immutable_fields = (
            "plan_sha256",
            "preregistration_sha256",
            "source_registry_sha256",
            "synthetic_audit",
            "commands_sha256",
            "repository_command_files_sha256",
            "runner_sha256",
            "git_revision",
        )
        changed = [
            field
            for field in immutable_fields
            if previous.get(field) != manifest[field]
        ]
        if changed:
            fail(
                "existing run has different immutable inputs "
                f"({', '.join(changed)}); choose a new work directory or --force"
            )
        manifest = previous

    for stage in plan["stages"]:
        name = stage["name"]
        if name not in selected:
            continue
        previous_stage = manifest["stages"].get(name, {})
        if previous_stage.get("status") == "completed" and not force:
            print(f"[{name}] already completed; skipping")
            continue
        stage_dir = work_dir / name
        stage_dir.mkdir(parents=True, exist_ok=True)
        variables = {
            "repo": str(repo),
            "work_dir": str(work_dir),
            "stage_dir": str(stage_dir),
            "paper": plan["paper"],
        }
        if name not in commands:
            fail(f"commands file has no command for selected stage {name!r}")
        argv = render_argv(commands[name], variables, name)
        record = {
            "status": "running",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "argv": argv,
        }
        manifest["stages"][name] = record
        atomic_json(manifest_path, manifest)
        print(f"[{name}] running")
        with (stage_dir / "stdout.txt").open("w", encoding="utf-8") as stdout, (
            stage_dir / "stderr.txt"
        ).open("w", encoding="utf-8") as stderr:
            result = subprocess.run(argv, cwd=repo, stdout=stdout, stderr=stderr, check=False)
        record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        record["return_code"] = result.returncode
        if result.returncode != 0:
            record["status"] = "failed"
            atomic_json(manifest_path, manifest)
            fail(f"stage {name} failed; inspect {stage_dir / 'stderr.txt'}")
        try:
            record["artifacts"] = [
                digest_artifact(work_dir / relative) for relative in stage["artifacts"]
            ]
            costs_path = stage_dir / "costs.json"
            costs = load_json(costs_path)
            if not isinstance(costs, dict):
                fail(f"stage {name} costs.json must contain an object")
            missing_costs = sorted(set(stage["costs"]) - set(costs))
            if missing_costs:
                fail(f"stage {name} costs.json is missing: {', '.join(missing_costs)}")
            invalid_costs = [
                key
                for key in stage["costs"]
                if not isinstance(costs[key], (int, float)) or costs[key] < 0
            ]
            if invalid_costs:
                fail(f"stage {name} has invalid nonnegative costs: {', '.join(invalid_costs)}")
            record["costs"] = costs
            if not synthetic_audit:
                record["semantic_validation"] = validate_real_stage(
                    plan["paper"],
                    name,
                    work_dir,
                    preregistration_section,
                    manifest,
                )
        except SystemExit as exc:
            record["status"] = "failed_validation"
            record["validation_error"] = str(exc)
            atomic_json(manifest_path, manifest)
            raise
        record["status"] = "completed"
        atomic_json(manifest_path, manifest)
        print(f"[{name}] completed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True, help="protocol JSON file")
    parser.add_argument(
        "--stage",
        action="append",
        choices=REQUIRED_STAGES,
        help="stage to display or run; repeat as needed (default: all)",
    )
    parser.add_argument("--execute", action="store_true", help="execute instead of dry-run")
    parser.add_argument("--commands", type=Path, help="site-local stage-to-argv JSON mapping")
    parser.add_argument("--work-dir", type=Path, help="artifact directory")
    parser.add_argument("--force", action="store_true", help="rerun completed stages")
    parser.add_argument(
        "--synthetic-audit",
        action="store_true",
        help="permit generated design audits while manual source reviews are pending",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    plan_path = args.plan.resolve()
    plan = load_json(plan_path)
    validate_plan(plan, plan_path)
    preregistration, preregistration_source = load_preregistration(plan, repo)
    section = preregistration[plan["preregistration"]["section"]]
    preregistration_digest = canonical_digest(section)
    source_registry = load_json(repo / preregistration["source_registry"])
    source_registry_digest = canonical_digest(source_registry)
    selected = set(args.stage or REQUIRED_STAGES)
    if not args.execute:
        print_plan(
            plan,
            selected,
            preregistration_source,
            preregistration_digest,
        )
        return
    if args.commands is None:
        fail("--execute requires --commands")
    commands = load_json(args.commands.resolve())
    if args.synthetic_audit:
        trusted_synthetic = load_committed_json(
            repo, "experiments/synthetic_commands.json"
        )
        if canonical_digest(commands) != canonical_digest(trusted_synthetic):
            fail("--synthetic-audit requires the committed synthetic command mapping")
    else:
        require_manual_source_clearance(preregistration, section, repo)
        require_execution_ready(section, plan["paper"], source_registry)
    work_dir = (args.work_dir or repo / "artifacts" / f"{plan['paper']}-run").resolve()
    execute(
        plan,
        commands,
        selected,
        repo,
        work_dir,
        args.force,
        preregistration_digest,
        source_registry_digest,
        section,
        args.synthetic_audit,
    )


if __name__ == "__main__":
    main()
