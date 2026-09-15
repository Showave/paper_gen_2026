#!/usr/bin/env python3
"""Validate and enumerate the evaluation paper's fractional simulation design.

The design audit checks the fraction, condition count, preregistration values,
and deterministic seeds.  It does not execute a DGP, query a model, or collect
human labels, and all emitted summaries are non-empirical.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESIGN = ROOT / "experiments" / "eval_simulation_design.json"
DEFAULT_PREREGISTRATION = ROOT / "experiments" / "pilot_preregistration.json"


class DesignError(ValueError):
    """Raised when the fractional simulation design is inconsistent."""


def fail(message: str) -> None:
    raise DesignError(message)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        fail(f"file not found: {path}")
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON in {path}: {exc}")


def canonical_bytes(value: Any) -> bytes:
    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        fail(f"value is not canonical JSON: {exc}")
    return payload.encode("utf-8")


def digest_value(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def multiply_columns(columns: list[tuple[int, ...]]) -> tuple[int, ...]:
    if not columns:
        fail("a generated factor must depend on at least one base factor")
    return tuple(math.prod(values) for values in zip(*columns))


def fractional_rows(design: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, tuple[int, ...]]]:
    screen = design.get("fractional_screen")
    if not isinstance(screen, dict):
        fail("simulation design has no fractional_screen")
    base = screen.get("base_factors")
    generators = screen.get("generators")
    factors = screen.get("factors")
    if (
        not isinstance(base, list)
        or len(base) != 4
        or len(set(base)) != len(base)
        or not all(isinstance(column, str) and column for column in base)
        or not isinstance(generators, dict)
        or not isinstance(factors, list)
        or len(factors) != 8
    ):
        fail("fractional screen must define four bases and eight factors")
    sign_rows = list(itertools.product((-1, 1), repeat=len(base)))
    columns: dict[str, tuple[int, ...]] = {
        column: tuple(row[index] for row in sign_rows)
        for index, column in enumerate(base)
    }
    for column, parents in generators.items():
        if (
            column in columns
            or not isinstance(column, str)
            or not isinstance(parents, list)
            or len(parents) < 2
            or not all(parent in columns for parent in parents)
        ):
            fail("fractional generator is invalid or not base-defined")
        columns[column] = multiply_columns([columns[parent] for parent in parents])
    factor_by_column: dict[str, dict[str, Any]] = {}
    factor_names: set[str] = set()
    for factor in factors:
        if not isinstance(factor, dict):
            fail("fractional factors must be objects")
        column = factor.get("column")
        name = factor.get("name")
        low = factor.get("low")
        high = factor.get("high")
        if (
            column not in columns
            or column in factor_by_column
            or not isinstance(name, str)
            or not name
            or name in factor_names
            or not isinstance(low, (int, float))
            or not isinstance(high, (int, float))
            or not math.isfinite(low)
            or not math.isfinite(high)
            or low >= high
        ):
            fail("fractional factor is duplicated or has invalid levels")
        factor_by_column[column] = factor
        factor_names.add(name)
    if set(factor_by_column) != set(columns):
        fail("fractional factors and generated columns differ")

    intercept = (1,) * len(sign_rows)
    for name, column in columns.items():
        if column == intercept or column == tuple(-value for value in intercept):
            fail(f"main-effect column {name} aliases the intercept")
        if sum(column) != 0:
            fail(f"main-effect column {name} is not balanced")
    for left, right in itertools.combinations(columns, 2):
        if abs(sum(a * b for a, b in zip(columns[left], columns[right]))) != 0:
            fail(f"main-effect columns {left} and {right} are not orthogonal")
    pair_columns = {
        (left, right): multiply_columns([columns[left], columns[right]])
        for left, right in itertools.combinations(columns, 2)
    }
    for name, column in columns.items():
        for pair, interaction in pair_columns.items():
            if column == interaction or column == tuple(-value for value in interaction):
                fail(f"main effect {name} aliases two-factor interaction {pair}")

    rows: list[dict[str, Any]] = []
    for row_index in range(len(sign_rows)):
        values = {
            factor["name"]: (
                factor["high"]
                if columns[column][row_index] == 1
                else factor["low"]
            )
            for column, factor in factor_by_column.items()
        }
        rows.append(
            {
                "fractional_row": row_index,
                "signs": {
                    column: columns[column][row_index] for column in sorted(columns)
                },
                "factors": values,
            }
        )
    return rows, columns


def effect_values(design: dict[str, Any]) -> tuple[list[float], list[float]]:
    decision = design.get("decision")
    if not isinstance(decision, dict):
        fail("simulation design has no decision block")
    margin = decision.get("practical_margin")
    budget = decision.get("fixed_budget_attempts")
    boundaries = decision.get("boundary_effects")
    constants = decision.get("local_alternative_c")
    if (
        not isinstance(margin, (int, float))
        or margin <= 0
        or not isinstance(budget, int)
        or budget <= 0
        or boundaries != [-margin, 0.0, margin]
        or not isinstance(constants, list)
        or not constants
        or not all(isinstance(value, (int, float)) and value > 0 for value in constants)
    ):
        fail("simulation decision effects are invalid")
    local = sorted(
        {
            sign * float(margin) + direction * float(constant) / math.sqrt(budget)
            for sign in (-1, 1)
            for direction in (-1, 1)
            for constant in constants
        }
    )
    if len(local) != 4 * len(constants):
        fail("local alternatives collide under the frozen constants")
    return [float(value) for value in boundaries], local


def preregistration_eval(preregistration: Any) -> dict[str, Any]:
    section = preregistration.get("eval") if isinstance(preregistration, dict) else None
    if not isinstance(section, dict):
        fail("preregistration has no evaluation section")
    return section


def validate_preregistered_values(
    design: dict[str, Any], preregistration: dict[str, Any]
) -> None:
    section = preregistration_eval(preregistration)
    decision = design["decision"]
    if (
        decision["practical_margin"] != section.get("decision", {}).get(
            "practical_margin"
        )
        or decision["fixed_budget_attempts"]
        != section.get("acquisition", {}).get("adaptive_fixed_budget_attempts")
    ):
        fail("simulation decision values differ from the evaluation protocol")
    simulation = section.get("simulation")
    if not isinstance(simulation, dict):
        fail("evaluation preregistration has no simulation block")
    if (
        design.get("monte_carlo", {}).get("trials_per_condition")
        != simulation.get("trials_per_condition")
        or design.get("monte_carlo", {}).get("base_seed") != simulation.get("seed")
        or decision["local_alternative_c"] != simulation.get("local_alternative_c")
    ):
        fail("simulation replication, seed, or local alternatives differ")
    factors = design["fractional_screen"]["factors"]
    center = design.get("center_condition")
    preregistration_fields = {
        "exploration_epsilon": "exploration_epsilon_sensitivity",
        "temporal_drift_gamma": "temporal_drift_gamma",
    }
    if not isinstance(center, dict):
        fail("fractional simulation design needs a center condition")
    for factor in factors:
        name = factor["name"]
        registered = simulation.get(preregistration_fields.get(name, name))
        if (
            not isinstance(registered, list)
            or factor["low"] not in registered
            or factor["high"] not in registered
            or name not in center
            or center[name] not in registered
        ):
            fail(f"factor {name} levels differ from the preregistration")


def enumerate_conditions(
    design: dict[str, Any], preregistration: dict[str, Any]
) -> list[dict[str, Any]]:
    if (
        not isinstance(design, dict)
        or design.get("schema_version") != 1
        or design.get("artifact_type")
        != "evaluation_fractional_simulation_design"
        or design.get("design_status") != "frozen_screening_layout"
        or design.get("execution_status") != "unfrozen_dgp_blocker"
    ):
        fail("evaluation simulation design has an unsupported status or schema")
    construction = design.get("condition_construction")
    regimes = design.get("model_regimes")
    policies = design.get("acquisition_policies")
    blockers = design.get("blocking_requirements")
    if (
        not isinstance(construction, dict)
        or construction.get("full_cartesian_product") is not False
        or not isinstance(regimes, list)
        or len(regimes) != 3
        or len(set(regimes)) != len(regimes)
        or not isinstance(policies, list)
        or len(policies) != 8
        or len(set(policies)) != len(policies)
        or not isinstance(blockers, list)
        or not blockers
    ):
        fail("simulation design must remain fractional, blocked, and explicit")
    validate_preregistered_values(design, preregistration)
    rows, _ = fractional_rows(design)
    boundaries, local = effect_values(design)
    if construction.get("expected_fractional_rows") != len(rows):
        fail("declared fractional-row count does not reproduce")

    conditions: list[dict[str, Any]] = []
    for row, effect, regime in itertools.product(rows, boundaries, regimes):
        conditions.append(
            {
                "block": "fractional-boundary",
                "effect": effect,
                "model_regime": regime,
                "fractional_row": row["fractional_row"],
                "factors": row["factors"],
            }
        )
    center = design["center_condition"]
    for effect, regime in itertools.product(boundaries, regimes):
        conditions.append(
            {
                "block": "center-boundary",
                "effect": effect,
                "model_regime": regime,
                "fractional_row": None,
                "factors": center,
            }
        )
    for effect, regime in itertools.product(local, regimes):
        conditions.append(
            {
                "block": "center-local-alternative",
                "effect": effect,
                "model_regime": regime,
                "fractional_row": None,
                "factors": center,
            }
        )
    if construction.get("expected_conditions") != len(conditions):
        fail("declared simulation-condition count does not reproduce")

    monte_carlo = design["monte_carlo"]
    base_seed = monte_carlo.get("base_seed")
    identifiers: set[str] = set()
    enumerated: list[dict[str, Any]] = []
    for condition in conditions:
        condition_id = digest_value(condition)
        if condition_id in identifiers:
            fail("simulation conditions are duplicated")
        identifiers.add(condition_id)
        seed = int(condition_id[:16], 16) ^ base_seed
        enumerated.append(
            {
                **condition,
                "condition_id": condition_id,
                "condition_seed": seed,
                "trials": monte_carlo["trials_per_condition"],
            }
        )
    return enumerated


def validate_design(
    design: dict[str, Any], preregistration: dict[str, Any]
) -> dict[str, Any]:
    conditions = enumerate_conditions(design, preregistration)
    rows, columns = fractional_rows(design)
    block_counts: dict[str, int] = {}
    for condition in conditions:
        block = condition["block"]
        block_counts[block] = block_counts.get(block, 0) + 1
    return {
        "design_status": design["design_status"],
        "execution_status": design["execution_status"],
        "empirical_evidence": False,
        "fractional_rows": len(rows),
        "main_effect_columns": len(columns),
        "main_effects_orthogonal": True,
        "main_to_two_factor_aliases": 0,
        "conditions": len(conditions),
        "conditions_by_block": block_counts,
        "acquisition_policies": len(design["acquisition_policies"]),
        "trials_per_condition": design["monte_carlo"]["trials_per_condition"],
        "design_sha256": digest_value(design),
        "condition_manifest_sha256": digest_value(conditions),
    }


def expect_rejection(
    design: dict[str, Any],
    preregistration: dict[str, Any],
    description: str,
) -> None:
    try:
        validate_design(design, preregistration)
    except DesignError:
        return
    fail(f"self-test accepted {description}")


def self_test(
    design: dict[str, Any], preregistration: dict[str, Any]
) -> dict[str, Any]:
    summary = validate_design(design, preregistration)

    aliased = copy.deepcopy(design)
    aliased["fractional_screen"]["generators"]["E"] = ["A", "B"]
    expect_rejection(aliased, preregistration, "a main--two-factor alias")

    wrong_count = copy.deepcopy(design)
    wrong_count["condition_construction"]["expected_conditions"] += 1
    expect_rejection(wrong_count, preregistration, "a stale condition count")

    full_cross = copy.deepcopy(design)
    full_cross["condition_construction"]["full_cartesian_product"] = True
    expect_rejection(full_cross, preregistration, "a full Cartesian crossing")
    summary["fault_injections_rejected"] = 3
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument(
        "--preregistration", type=Path, default=DEFAULT_PREREGISTRATION
    )
    parser.add_argument("--write-manifest", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    design = read_json(args.design)
    preregistration = read_json(args.preregistration)
    result = (
        self_test(design, preregistration)
        if args.self_test
        else validate_design(design, preregistration)
    )
    if args.write_manifest:
        manifest = {
            "schema_version": 1,
            "artifact_type": "evaluation_simulation_condition_manifest",
            "execution_class": "design-audit",
            "empirical_evidence": False,
            "design_sha256": digest_value(design),
            "conditions": enumerate_conditions(design, preregistration),
        }
        args.write_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.write_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except DesignError as exc:
        raise SystemExit(f"error: {exc}") from exc
