#!/usr/bin/env python3
"""Aggregate the frozen SFT 2x2 factorial without filling paper tables.

Input is one JSONL endpoint per seed, target, budget, cell, and tolerance.
Failed or missing runs receive the frozen hypervolume reference point.  The
script computes lower-is-better dominated hypervolume, the paired
difference-in-differences, and a simultaneous max-t bootstrap interval across
target-by-budget contrasts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import tempfile
from pathlib import Path
from statistics import mean, stdev
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREREGISTRATION = ROOT / "experiments" / "pilot_preregistration.json"


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except FileNotFoundError:
        fail(f"run manifest not found: {path}")
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        fail(f"could not read {path}: {exc}")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    fail(f"{path}:{line_number}: invalid JSON: {exc}")
                if not isinstance(value, dict):
                    fail(f"{path}:{line_number}: expected an object")
                rows.append(value)
    except FileNotFoundError:
        fail(f"file not found: {path}")
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def dominated_hypervolume(
    points: list[tuple[float, float]], reference: tuple[float, float]
) -> float:
    rx, ry = reference
    clipped = sorted(
        {
            (min(float(x), rx), min(float(y), ry))
            for x, y in points
            if math.isfinite(float(x)) and math.isfinite(float(y))
        }
    )
    area = 0.0
    best_y = ry
    for x, y in clipped:
        if y < best_y:
            area += (rx - x) * (best_y - y)
            best_y = y
    return area


def standard_error(values: list[float]) -> float:
    return stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = math.ceil(probability * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def aggregate(
    rows: list[dict[str, Any]],
    preregistration: dict[str, Any],
    target_settings: list[str],
    execution_class: str,
    manifest_root: Path,
) -> dict[str, Any]:
    section = preregistration["sft"]
    factorial = section["primary_factorial"]
    if target_settings != factorial["target_settings"]:
        fail("target settings must exactly match the frozen factorial order")
    cells = list(factorial["cells"])
    seeds = section["seeds"]
    budgets = section["cost_design"]["budget_fractions"]
    tolerances = factorial["protected_tolerance_grid"]
    reference = tuple(factorial["hypervolume_reference_box"])
    required_fields = {
        "seed",
        "target_setting",
        "budget_fraction",
        "cell",
        "protected_tolerance",
        "status",
        "target_delta_nll",
        "protected_delta_nll",
        "run_manifest_path",
        "run_manifest_sha256",
        "empirical_evidence",
    }
    indexed: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        if not required_fields.issubset(row):
            fail("endpoint row is missing required fields")
        key = (
            row["seed"],
            row["target_setting"],
            row["budget_fraction"],
            row["cell"],
            row["protected_tolerance"],
        )
        if key in indexed:
            fail(f"duplicate endpoint row: {key}")
        if (
            row["seed"] not in seeds
            or row["target_setting"] not in target_settings
            or row["budget_fraction"] not in budgets
            or row["cell"] not in cells
            or row["protected_tolerance"] not in tolerances
            or row["status"] not in {"completed", "failed"}
        ):
            fail(f"endpoint row is outside the frozen factorial: {key}")
        if not (
            isinstance(row["run_manifest_sha256"], str)
            and len(row["run_manifest_sha256"]) == 64
            and all(
                character in "0123456789abcdef"
                for character in row["run_manifest_sha256"]
            )
        ):
            fail("endpoint row needs a run-manifest SHA-256")
        if not isinstance(row["run_manifest_path"], str):
            fail("run_manifest_path must be a string")
        relative_manifest = Path(row["run_manifest_path"])
        if relative_manifest.is_absolute() or ".." in relative_manifest.parts:
            fail("run_manifest_path must stay under --manifest-root")
        manifest_path = manifest_root / relative_manifest
        if file_digest(manifest_path) != row["run_manifest_sha256"]:
            fail(f"run manifest hash mismatch: {manifest_path}")
        run_manifest = load_json(manifest_path)
        if run_manifest.get("preregistration_sha256") != canonical_digest(section):
            fail(f"run manifest has stale SFT preregistration: {manifest_path}")
        if execution_class == "real" and run_manifest.get("synthetic_audit") is not False:
            fail("real aggregation cannot consume a synthetic run manifest")
        if execution_class == "real" and row["empirical_evidence"] is not True:
            fail("real aggregation cannot consume non-empirical endpoints")
        if execution_class == "fixture" and row["empirical_evidence"] is not False:
            fail("fixture aggregation accepts only non-empirical endpoints")
        indexed[key] = row

    hypervolumes: dict[tuple[int, str, float, str], float] = {}
    missing = 0
    for seed in seeds:
        for target in target_settings:
            for budget in budgets:
                for cell in cells:
                    points: list[tuple[float, float]] = []
                    for tolerance in tolerances:
                        key = (seed, target, budget, cell, tolerance)
                        row = indexed.get(key)
                        if row is None or row["status"] == "failed":
                            points.append(reference)
                            missing += 1
                        else:
                            values = (
                                row["target_delta_nll"],
                                row["protected_delta_nll"],
                            )
                            if not all(
                                isinstance(value, (int, float))
                                and math.isfinite(value)
                                for value in values
                            ):
                                fail(f"completed endpoint has invalid values: {key}")
                            points.append(values)
                    hypervolumes[(seed, target, budget, cell)] = (
                        dominated_hypervolume(points, reference)
                    )

    did_by_contrast: dict[tuple[str, float], list[float]] = {}
    for target in target_settings:
        for budget in budgets:
            values = []
            for seed in seeds:
                hv = {
                    cell: hypervolumes[(seed, target, budget, cell)]
                    for cell in cells
                }
                values.append(
                    (hv["measured_nonlinear__fixed_budget_beam"]
                    - hv["measured_nonlinear__deterministic_best_fit"])
                    - (hv["additive_token__fixed_budget_beam"]
                    - hv["additive_token__deterministic_best_fit"])
                )
            did_by_contrast[(target, budget)] = values

    observed = {
        key: {"mean": mean(values), "se": standard_error(values)}
        for key, values in did_by_contrast.items()
    }
    rng = random.Random(factorial["bootstrap_seed"])
    max_statistics: list[float] = []
    for _ in range(factorial["bootstrap_replicates"]):
        selected = [rng.randrange(len(seeds)) for _ in seeds]
        statistics = []
        for key, values in did_by_contrast.items():
            sample = [values[index] for index in selected]
            se = standard_error(sample)
            if se > 0:
                statistics.append(abs((mean(sample) - observed[key]["mean"]) / se))
        max_statistics.append(max(statistics, default=0.0))
    critical = quantile(max_statistics, 0.95)

    contrasts = []
    for (target, budget), values in did_by_contrast.items():
        estimate = observed[(target, budget)]["mean"]
        se = observed[(target, budget)]["se"]
        contrasts.append(
            {
                "target_setting": target,
                "budget_fraction": budget,
                "paired_seed_values": values,
                "estimate": estimate,
                "standard_error": se,
                "simultaneous_95_interval": [
                    estimate - critical * se,
                    estimate + critical * se,
                ],
            }
        )
    return {
        "schema_version": 1,
        "execution_class": execution_class,
        "empirical_evidence": execution_class == "real",
        "preregistration_section_sha256": canonical_digest(section),
        "input_rows": len(rows),
        "failed_or_missing_cells": missing,
        "hypervolume_reference_box": list(reference),
        "bootstrap_replicates": factorial["bootstrap_replicates"],
        "bootstrap_seed": factorial["bootstrap_seed"],
        "simultaneous_max_t_critical_value": critical,
        "contrasts": contrasts,
    }


def self_test() -> None:
    preregistration = load_json(DEFAULT_PREREGISTRATION)
    section = preregistration["sft"]
    with tempfile.TemporaryDirectory(prefix="sft-factorial-") as directory:
        manifest_root = Path(directory)
        run_manifest_path = manifest_root / "fixture-run-manifest.json"
        write_json(
            run_manifest_path,
            {
                "preregistration_sha256": canonical_digest(section),
                "synthetic_audit": True,
            },
        )
        run_manifest_sha256 = file_digest(run_manifest_path)
        rows = []
        for seed_index, seed in enumerate(section["seeds"]):
            for target in ("math", "code"):
                for budget in section["cost_design"]["budget_fractions"]:
                    for cell_index, cell in enumerate(
                        section["primary_factorial"]["cells"]
                    ):
                        for tolerance in section["primary_factorial"][
                            "protected_tolerance_grid"
                        ]:
                            rows.append(
                                {
                                    "seed": seed,
                                    "target_setting": target,
                                    "budget_fraction": budget,
                                    "cell": cell,
                                    "protected_tolerance": tolerance,
                                    "status": "completed",
                                    "target_delta_nll": (
                                        0.8
                                        - 0.05 * cell_index
                                        - 0.01 * seed_index
                                    ),
                                    "protected_delta_nll": (
                                        0.15 - 0.01 * cell_index + tolerance
                                    ),
                                    "run_manifest_path": (
                                        "fixture-run-manifest.json"
                                    ),
                                    "run_manifest_sha256": run_manifest_sha256,
                                    "empirical_evidence": False,
                                }
                            )
        summary = aggregate(
            rows,
            preregistration,
            ["math", "code"],
            "fixture",
            manifest_root,
        )
        if (
            len(summary["contrasts"]) != 6
            or summary["failed_or_missing_cells"] != 0
        ):
            fail("factorial fixture did not produce six complete contrasts")
        output = Path(directory) / "summary.json"
        write_json(output, summary)
        load_json(output)
    print("validated SFT hypervolume, paired DID, and max-t aggregation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--target-settings", nargs="+", default=["math", "code"])
    parser.add_argument(
        "--manifest-root",
        type=Path,
        help="root for endpoint run_manifest_path values (defaults to input parent)",
    )
    parser.add_argument("--execution-class", choices=("real", "fixture"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test and (
        args.input is None or args.output is None or args.execution_class is None
    ):
        parser.error("--input, --output, and --execution-class are required")
    return args


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    preregistration = load_json(args.preregistration)
    summary = aggregate(
        load_jsonl(args.input),
        preregistration,
        args.target_settings,
        args.execution_class,
        (args.manifest_root or args.input.parent).resolve(),
    )
    write_json(args.output, summary)
    print(f"wrote {args.output} with {len(summary['contrasts'])} contrasts")


if __name__ == "__main__":
    main()
