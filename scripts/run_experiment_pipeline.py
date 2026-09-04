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
        if not isinstance(count_rows, list) or len(count_rows) != len(alphas):
            fail("RL stratum counts must cover every alpha")
        for row in count_rows:
            total = row["current"] + row["stale"]
            if total <= 0 or not math.isclose(row["current"] / total, row["alpha"]):
                fail("RL alpha must equal current count divided by total count")
        if {row["alpha"] for row in count_rows} != set(alphas):
            fail("RL stratum-count alpha values differ from alpha_grid")
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


def execute(
    plan: dict[str, Any],
    commands: Any,
    selected: set[str],
    repo: Path,
    work_dir: Path,
    force: bool,
    preregistration_digest: str,
    source_registry_digest: str,
    synthetic_audit: bool,
) -> None:
    if not isinstance(commands, dict):
        fail("commands file must map stage names to argv arrays")
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = work_dir / "run_manifest.json"
    manifest = {
        "schema_version": 1,
        "paper": plan["paper"],
        "plan_sha256": canonical_digest(plan),
        "preregistration_sha256": preregistration_digest,
        "source_registry_sha256": source_registry_digest,
        "synthetic_audit": synthetic_audit,
        "commands_sha256": canonical_digest(commands),
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
        trusted_synthetic = load_json(repo / "experiments" / "synthetic_commands.json")
        if canonical_digest(commands) != canonical_digest(trusted_synthetic):
            fail("--synthetic-audit requires the committed synthetic command mapping")
    else:
        require_manual_source_clearance(preregistration, section, repo)
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
        args.synthetic_audit,
    )


if __name__ == "__main__":
    main()
