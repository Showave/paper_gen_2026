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


def print_plan(plan: dict[str, Any], selected: set[str]) -> None:
    print(f"{plan['paper']}: {plan['purpose']}")
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
) -> None:
    if not isinstance(commands, dict):
        fail("commands file must map stage names to argv arrays")
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = work_dir / "run_manifest.json"
    manifest = {
        "schema_version": 1,
        "paper": plan["paper"],
        "plan_sha256": canonical_digest(plan),
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    plan_path = args.plan.resolve()
    plan = load_json(plan_path)
    validate_plan(plan, plan_path)
    selected = set(args.stage or REQUIRED_STAGES)
    if not args.execute:
        print_plan(plan, selected)
        return
    if args.commands is None:
        fail("--execute requires --commands")
    commands = load_json(args.commands.resolve())
    work_dir = (args.work_dir or repo / "artifacts" / f"{plan['paper']}-run").resolve()
    execute(plan, commands, selected, repo, work_dir, args.force)


if __name__ == "__main__":
    main()
