#!/usr/bin/env python3
"""Recompute the RL estimator audit from its trajectory ledger.

Successful candidate rows store the projected per-trajectory term before its
stratum mean: ``A=(h-b)s+alpha*w*(U-h)s`` for current rows and
``B=w*(U-h)s`` for stale rows.  This script reconstructs
``mean(A)+(1-alpha)*mean(B)`` and the cross-reference MSE.  Its generated
self-test is equation and artifact validation, not empirical evidence.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREREGISTRATION = ROOT / "experiments" / "pilot_preregistration.json"


class AggregationError(ValueError):
    """Raised when a trajectory ledger or summary violates the contract."""


def fail(message: str) -> None:
    raise AggregationError(message)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        fail(f"file not found: {path}")
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON in {path}: {exc}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    fail(f"{path}:{line_number}: blank rows are forbidden")
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    fail(f"{path}:{line_number}: invalid JSON: {exc}")
                if not isinstance(row, dict):
                    fail(f"{path}:{line_number}: expected an object")
                rows.append(row)
    except FileNotFoundError:
        fail(f"file not found: {path}")
    if not rows:
        fail(f"trajectory ledger is empty: {path}")
    return rows


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except FileNotFoundError:
        fail(f"file not found: {path}")
    return digest.hexdigest()


def finite_vector(value: Any, dimension: int, name: str) -> list[float]:
    if (
        not isinstance(value, list)
        or len(value) != dimension
        or not all(
            isinstance(component, (int, float)) and math.isfinite(component)
            for component in value
        )
    ):
        fail(f"{name} must be a finite projected vector of dimension {dimension}")
    return [float(component) for component in value]


def vector_mean(vectors: list[list[float]], dimension: int) -> list[float]:
    if not vectors:
        fail("cannot average an empty vector stratum")
    totals = [0.0] * dimension
    for vector in vectors:
        for index, value in enumerate(vector):
            totals[index] += value
    return [value / len(vectors) for value in totals]


def dot(left: list[float], right: list[float]) -> float:
    return math.fsum(a * b for a, b in zip(left, right))


def cross_mse(
    estimate: list[float], reference_one: list[float], reference_two: list[float]
) -> float:
    return dot(
        [value - reference for value, reference in zip(estimate, reference_one)],
        [value - reference for value, reference in zip(estimate, reference_two)],
    )


def audit_configuration(section: dict[str, Any]) -> dict[str, Any]:
    audit = section.get("estimator_audit")
    if not isinstance(audit, dict):
        fail("RL preregistration has no estimator_audit")
    required = (
        "projection_dimension",
        "stratum_counts_by_alpha",
        "selection_replications",
        "confirmation_replications",
        "reference_current_policy_trajectories_per_replica",
    )
    if any(field not in audit for field in required):
        fail("RL estimator_audit is incomplete")
    return audit


def expected_cells(audit: dict[str, Any]) -> dict[tuple[float, str], tuple[int, int]]:
    cells: dict[tuple[float, str], tuple[int, int]] = {}
    rows = audit["stratum_counts_by_alpha"]
    if not isinstance(rows, list) or not rows:
        fail("RL stratum_counts_by_alpha must be nonempty")
    for row in rows:
        if not isinstance(row, dict):
            fail("RL allocation cells must be objects")
        alpha = row.get("alpha")
        allocation_id = row.get("allocation_id")
        current = row.get("current")
        stale = row.get("stale")
        key = (alpha, allocation_id)
        if (
            not isinstance(alpha, (int, float))
            or not 0 < alpha <= 1
            or not isinstance(allocation_id, str)
            or not allocation_id
            or not isinstance(current, int)
            or not isinstance(stale, int)
            or current <= 0
            or stale < 0
            or key in cells
        ):
            fail("RL allocation cell is invalid or duplicated")
        if alpha == 1 and stale != 0:
            fail("the alpha=1 cell cannot contain stale trajectories")
        if alpha < 1 and stale <= 0:
            fail("a stale-source cell needs positive stale count")
        cells[(float(alpha), allocation_id)] = (current, stale)
    return cells


def candidate_estimate(
    current: list[list[float]],
    stale: list[list[float]],
    alpha: float,
    dimension: int,
) -> list[float]:
    direct = vector_mean(current, dimension)
    if alpha == 1:
        if stale:
            fail("alpha=1 candidate unexpectedly contains stale rows")
        return direct
    residual = vector_mean(stale, dimension)
    return [
        current_value + (1 - alpha) * stale_value
        for current_value, stale_value in zip(direct, residual)
    ]


def analyze_rows(
    rows: Iterable[dict[str, Any]], section: dict[str, Any]
) -> dict[str, Any]:
    audit = audit_configuration(section)
    dimension = audit["projection_dimension"]
    if not isinstance(dimension, int) or dimension <= 0:
        fail("RL projection dimension must be positive")
    cells = expected_cells(audit)
    references: dict[tuple[str, str], list[list[float]]] = defaultdict(list)
    candidates: dict[
        tuple[str, float, str, int, str], list[list[float]]
    ] = defaultdict(list)
    candidate_costs: dict[tuple[str, float, str, int], float] = defaultdict(float)
    attempt_ids: set[str] = set()
    trajectory_ids: set[str] = set()
    failures = 0
    reference_accelerator_seconds = 0.0

    for row in rows:
        attempt_id = row.get("attempt_id")
        law = row.get("target_law")
        role = row.get("estimator_role")
        source = row.get("source_component")
        failed = row.get("infrastructure_failure")
        cost = row.get("allocated_accelerator_seconds")
        if (
            not isinstance(attempt_id, str)
            or not attempt_id
            or attempt_id in attempt_ids
            or law not in {"select", "confirm"}
            or role not in {"candidate", "r1", "r2"}
            or source not in {"current", "stale"}
            or not isinstance(failed, bool)
            or not isinstance(cost, (int, float))
            or not math.isfinite(cost)
            or cost < 0
        ):
            fail("RL trajectory row has invalid common fields")
        attempt_ids.add(attempt_id)
        if failed:
            failures += 1
            continue
        trajectory_id = row.get("trajectory_id")
        if (
            not isinstance(trajectory_id, str)
            or not trajectory_id
            or trajectory_id in trajectory_ids
            or row.get("use_count") != 1
            or row.get("scored_outcome") is not True
        ):
            fail("successful RL trajectories must be unique, scored, and single-use")
        trajectory_ids.add(trajectory_id)
        vector = finite_vector(
            row.get("projected_gradient"), dimension, "projected_gradient"
        )
        if role in {"r1", "r2"}:
            if (
                source != "current"
                or row.get("projected_term_semantics") != "target=U*s"
            ):
                fail("reference rows must store current-policy target terms")
            references[(law, role)].append(vector)
            reference_accelerator_seconds += float(cost)
            continue

        alpha = row.get("alpha")
        allocation_id = row.get("allocation_id")
        replication = row.get("replication_id")
        key = (float(alpha), allocation_id) if isinstance(alpha, (int, float)) else None
        if (
            key not in cells
            or not isinstance(replication, int)
            or replication < 0
        ):
            fail("candidate row has an invalid allocation or replication")
        expected_semantics = (
            "A=(h-b)s+alpha*w*(U-h)s"
            if source == "current"
            else "B=w*(U-h)s"
        )
        if row.get("projected_term_semantics") != expected_semantics:
            fail("candidate projected term has ambiguous or incorrect semantics")
        expected_replications = (
            audit["selection_replications"]
            if law == "select"
            else audit["confirmation_replications"]
        )
        if replication >= expected_replications:
            fail("candidate replication is outside the frozen range")
        candidates[(law, key[0], key[1], replication, source)].append(vector)
        candidate_costs[(law, key[0], key[1], replication)] += float(cost)

    if failures:
        return {
            "aggregation_status": "blocked_infrastructure_failure",
            "empirical_evidence": False,
            "h1_eligible": False,
            "infrastructure_failures": failures,
            "attempts": len(attempt_ids),
        }

    reference_count = audit["reference_current_policy_trajectories_per_replica"]
    reference_means: dict[tuple[str, str], list[float]] = {}
    for law in ("select", "confirm"):
        for role in ("r1", "r2"):
            vectors = references.get((law, role), [])
            if len(vectors) != reference_count:
                fail(f"{law} {role} reference count does not reproduce")
            reference_means[(law, role)] = vector_mean(vectors, dimension)

    law_results: dict[str, list[dict[str, Any]]] = {"select": [], "confirm": []}
    for law in ("select", "confirm"):
        replications = (
            audit["selection_replications"]
            if law == "select"
            else audit["confirmation_replications"]
        )
        present_cells = {
            (alpha, allocation_id)
            for candidate_law, alpha, allocation_id, _, _ in candidates
            if candidate_law == law
        }
        if law == "select" and present_cells != set(cells):
            fail("selection law must contain every frozen alpha/allocation cell")
        if law == "confirm" and len(present_cells) != 1:
            fail("confirmation law must contain exactly one selected cell")
        for alpha, allocation_id in sorted(present_cells):
            current_count, stale_count = cells[(alpha, allocation_id)]
            replication_mse: list[float] = []
            replication_cost: list[float] = []
            for replication in range(replications):
                current = candidates.get(
                    (law, alpha, allocation_id, replication, "current"), []
                )
                stale = candidates.get(
                    (law, alpha, allocation_id, replication, "stale"), []
                )
                if len(current) != current_count or len(stale) != stale_count:
                    fail("candidate stratum counts do not reproduce")
                estimate = candidate_estimate(current, stale, alpha, dimension)
                replication_mse.append(
                    cross_mse(
                        estimate,
                        reference_means[(law, "r1")],
                        reference_means[(law, "r2")],
                    )
                )
                cost = candidate_costs.get(
                    (law, alpha, allocation_id, replication), 0.0
                )
                if cost <= 0:
                    fail("candidate replication must have positive measured cost")
                replication_cost.append(cost)
            point_mse = statistics.fmean(replication_mse)
            point_cost = statistics.fmean(replication_cost)
            law_results[law].append(
                {
                    "alpha": alpha,
                    "allocation_id": allocation_id,
                    "projected_mse": point_mse,
                    "accelerator_seconds": point_cost,
                    "mse_times_accelerator_seconds": point_mse * point_cost,
                    "replications": replications,
                }
            )

    selected = min(
        law_results["select"],
        key=lambda result: (
            result["mse_times_accelerator_seconds"],
            -result["alpha"],
            result["allocation_id"],
        ),
    )
    confirmation = law_results["confirm"][0]
    if (
        confirmation["alpha"] != selected["alpha"]
        or confirmation["allocation_id"] != selected["allocation_id"]
    ):
        fail("confirmation cell is not the selection MSE-times-cost argmin")
    return {
        "aggregation_status": "complete",
        "empirical_evidence": True,
        "h1_eligible": True,
        "infrastructure_failures": 0,
        "attempts": len(attempt_ids),
        "reference_accelerator_seconds": reference_accelerator_seconds,
        "projection_dimension": dimension,
        "projected_mse_formula": "(g_hat-R1)^T(g_hat-R2)",
        "candidate_estimator_formula": "mean(A_current)+(1-alpha)*mean(B_stale)",
        "selection_cell_results": law_results["select"],
        "selected_alpha": selected["alpha"],
        "selected_allocation_id": selected["allocation_id"],
        "confirmation_result": confirmation,
    }


def close(observed: Any, expected: Any) -> bool:
    return (
        isinstance(observed, (int, float))
        and math.isfinite(observed)
        and math.isclose(float(observed), float(expected), rel_tol=1e-12, abs_tol=1e-12)
    )


def validate_summary(
    summary: Any,
    analysis: dict[str, Any],
    ledger_sha256: str,
    section: dict[str, Any],
) -> dict[str, Any]:
    if (
        not isinstance(summary, dict)
        or summary.get("ledger_sha256") != ledger_sha256
        or summary.get("aggregation_status") != analysis["aggregation_status"]
        or summary.get("h1_eligible") is not analysis["h1_eligible"]
    ):
        fail("RL summary does not bind the analyzed trajectory ledger")
    if analysis["aggregation_status"] == "blocked_infrastructure_failure":
        forbidden = {
            "projected_mse_by_cell",
            "cell_results",
            "selected_alpha",
            "selected_allocation_id",
            "confirmation_result",
        }
        if forbidden.intersection(summary):
            fail("an infrastructure-blocked RL summary reports estimator results")
        return {
            "aggregation_status": analysis["aggregation_status"],
            "h1_eligible": False,
            "infrastructure_failures": analysis["infrastructure_failures"],
        }

    audit = audit_configuration(section)
    if (
        summary.get("empirical_evidence") is not True
        or summary.get("projected_mse_formula") != analysis["projected_mse_formula"]
        or summary.get("candidate_estimator_formula")
        != analysis["candidate_estimator_formula"]
        or summary.get("selected_alpha") != analysis["selected_alpha"]
        or summary.get("selected_allocation_id")
        != analysis["selected_allocation_id"]
        or not close(
            summary.get("reference_accelerator_seconds"),
            analysis["reference_accelerator_seconds"],
        )
        or summary.get("bootstrap_replicates")
        != audit["mse_bootstrap_replicates"]
        or summary.get("bootstrap_seed") != audit["mse_bootstrap_seed"]
        or summary.get("three_way_reference_bootstrap") is not True
    ):
        fail("RL summary metadata does not match the frozen estimator audit")
    reported_by_cell = summary.get("projected_mse_by_cell")
    reported_results = summary.get("cell_results")
    expected_results = analysis["selection_cell_results"]
    if (
        not isinstance(reported_by_cell, dict)
        or not isinstance(reported_results, list)
        or len(reported_results) != len(expected_results)
    ):
        fail("RL summary has incomplete selection-cell results")
    by_key = {
        f"{result['alpha']}|{result['allocation_id']}": result
        for result in expected_results
    }
    if set(reported_by_cell) != set(by_key):
        fail("RL summary selection cells differ from the ledger")
    seen: set[str] = set()
    for reported in reported_results:
        if not isinstance(reported, dict):
            fail("RL summary cell results must be objects")
        key = f"{reported.get('alpha')}|{reported.get('allocation_id')}"
        expected = by_key.get(key)
        interval = reported.get("three_way_bootstrap_interval")
        if (
            expected is None
            or key in seen
            or not close(reported_by_cell[key], expected["projected_mse"])
            or not close(reported.get("projected_mse"), expected["projected_mse"])
            or not close(
                reported.get("accelerator_seconds"),
                expected["accelerator_seconds"],
            )
            or not close(
                reported.get("mse_times_accelerator_seconds"),
                expected["mse_times_accelerator_seconds"],
            )
            or not isinstance(interval, list)
            or len(interval) != 2
            or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in interval)
        ):
            fail("RL summary selection result does not recompute from the ledger")
        seen.add(key)
    reported_confirmation = summary.get("confirmation_result")
    expected_confirmation = analysis["confirmation_result"]
    if not isinstance(reported_confirmation, dict):
        fail("RL summary must report the untouched confirmation result")
    for field in (
        "projected_mse",
        "accelerator_seconds",
        "mse_times_accelerator_seconds",
    ):
        if not close(reported_confirmation.get(field), expected_confirmation[field]):
            fail("RL confirmation result does not recompute from the ledger")
    if (
        reported_confirmation.get("alpha") != expected_confirmation["alpha"]
        or reported_confirmation.get("allocation_id")
        != expected_confirmation["allocation_id"]
        or not isinstance(
            reported_confirmation.get("three_way_bootstrap_interval"), list
        )
        or len(reported_confirmation["three_way_bootstrap_interval"]) != 2
    ):
        fail("RL confirmation result is incomplete")
    return {
        "aggregation_status": "complete",
        "h1_eligible": True,
        "selection_cells_recomputed": len(expected_results),
        "confirmation_cell_recomputed": True,
        "ledger_sha256": ledger_sha256,
    }


def make_row(
    *,
    attempt: str,
    law: str,
    role: str,
    source: str,
    vector: list[float],
    cost: float = 0.0,
    alpha: float | None = None,
    allocation_id: str | None = None,
    replication: int | None = None,
) -> dict[str, Any]:
    row = {
        "attempt_id": attempt,
        "target_law": law,
        "estimator_role": role,
        "source_component": source,
        "infrastructure_failure": False,
        "allocated_accelerator_seconds": cost,
        "trajectory_id": f"trajectory-{attempt}",
        "use_count": 1,
        "scored_outcome": True,
        "projected_gradient": vector,
        "projected_term_semantics": (
            "target=U*s"
            if role in {"r1", "r2"}
            else (
                "A=(h-b)s+alpha*w*(U-h)s"
                if source == "current"
                else "B=w*(U-h)s"
            )
        ),
    }
    if role == "candidate":
        row.update(
            {
                "alpha": alpha,
                "allocation_id": allocation_id,
                "replication_id": replication,
            }
        )
    return row


def generated_fixture() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    section = {
        "estimator_audit": {
            "projection_dimension": 2,
            "stratum_counts_by_alpha": [
                {
                    "alpha": 0.5,
                    "allocation_id": "half",
                    "current": 1,
                    "stale": 1,
                },
                {
                    "alpha": 1.0,
                    "allocation_id": "on-policy",
                    "current": 2,
                    "stale": 0,
                },
            ],
            "selection_replications": 2,
            "confirmation_replications": 2,
            "reference_current_policy_trajectories_per_replica": 2,
            "mse_bootstrap_replicates": 17,
            "mse_bootstrap_seed": 2026091501,
        }
    }
    rows: list[dict[str, Any]] = []
    for law in ("select", "confirm"):
        for role, vectors in (
            ("r1", ([1.0, 0.0], [-1.0, 0.0])),
            ("r2", ([0.0, 1.0], [0.0, -1.0])),
        ):
            for index, vector in enumerate(vectors):
                rows.append(
                    make_row(
                        attempt=f"{law}-{role}-{index}",
                        law=law,
                        role=role,
                        source="current",
                        vector=vector,
                    )
                )
    for replication, (current, stale) in enumerate(
        (([1.0, 0.0], [0.0, 2.0]), ([0.0, 1.0], [2.0, 0.0]))
    ):
        rows.append(
            make_row(
                attempt=f"select-half-current-{replication}",
                law="select",
                role="candidate",
                source="current",
                vector=current,
                cost=1.0,
                alpha=0.5,
                allocation_id="half",
                replication=replication,
            )
        )
        rows.append(
            make_row(
                attempt=f"select-half-stale-{replication}",
                law="select",
                role="candidate",
                source="stale",
                vector=stale,
                cost=1.0,
                alpha=0.5,
                allocation_id="half",
                replication=replication,
            )
        )
    for law in ("select", "confirm"):
        for replication in range(2):
            for index in range(2):
                rows.append(
                    make_row(
                        attempt=f"{law}-on-policy-{replication}-{index}",
                        law=law,
                        role="candidate",
                        source="current",
                        vector=[0.5, 0.5],
                        cost=1.0,
                        alpha=1.0,
                        allocation_id="on-policy",
                        replication=replication,
                    )
                )
    return rows, section


def generated_summary(analysis: dict[str, Any], ledger_sha256: str, section: dict[str, Any]) -> dict[str, Any]:
    results = []
    by_cell = {}
    for result in analysis["selection_cell_results"]:
        item = dict(result)
        item["three_way_bootstrap_interval"] = [
            result["projected_mse"],
            result["projected_mse"],
        ]
        results.append(item)
        by_cell[f"{result['alpha']}|{result['allocation_id']}"] = result[
            "projected_mse"
        ]
    confirmation = dict(analysis["confirmation_result"])
    confirmation["three_way_bootstrap_interval"] = [
        confirmation["projected_mse"],
        confirmation["projected_mse"],
    ]
    audit = section["estimator_audit"]
    return {
        "aggregation_status": "complete",
        "empirical_evidence": True,
        "h1_eligible": True,
        "ledger_sha256": ledger_sha256,
        "projected_mse_formula": analysis["projected_mse_formula"],
        "candidate_estimator_formula": analysis["candidate_estimator_formula"],
        "projected_mse_by_cell": by_cell,
        "cell_results": results,
        "selected_alpha": analysis["selected_alpha"],
        "selected_allocation_id": analysis["selected_allocation_id"],
        "reference_accelerator_seconds": analysis[
            "reference_accelerator_seconds"
        ],
        "confirmation_result": confirmation,
        "bootstrap_replicates": audit["mse_bootstrap_replicates"],
        "bootstrap_seed": audit["mse_bootstrap_seed"],
        "three_way_reference_bootstrap": True,
    }


def self_test() -> dict[str, Any]:
    rows, section = generated_fixture()
    analysis = analyze_rows(rows, section)
    if (
        analysis["selected_alpha"] != 1.0
        or analysis["selected_allocation_id"] != "on-policy"
        or not close(analysis["selection_cell_results"][0]["projected_mse"], 2.0)
        or not close(analysis["selection_cell_results"][1]["projected_mse"], 0.5)
    ):
        fail("generated fixture does not reproduce hand-computed estimator values")
    ledger_sha256 = hashlib.sha256(b"generated-ledger-v1").hexdigest()
    summary = generated_summary(analysis, ledger_sha256, section)
    validate_summary(summary, analysis, ledger_sha256, section)

    tampered = copy.deepcopy(summary)
    tampered["cell_results"][0]["projected_mse"] += 0.25
    try:
        validate_summary(tampered, analysis, ledger_sha256, section)
    except AggregationError:
        pass
    else:
        fail("self-test accepted a tampered MSE")

    incomplete = copy.deepcopy(rows)
    incomplete.pop(
        next(
            index
            for index, row in enumerate(incomplete)
            if row.get("attempt_id") == "select-half-stale-0"
        )
    )
    try:
        analyze_rows(incomplete, section)
    except AggregationError:
        pass
    else:
        fail("self-test accepted an incomplete source stratum")

    failed = copy.deepcopy(rows)
    failed[0]["infrastructure_failure"] = True
    for field in ("trajectory_id", "use_count", "scored_outcome", "projected_gradient"):
        failed[0].pop(field)
    blocked = analyze_rows(failed, section)
    if blocked["aggregation_status"] != "blocked_infrastructure_failure":
        fail("self-test did not block an infrastructure failure")
    return {
        "execution_class": "synthetic-audit",
        "empirical_evidence": False,
        "candidate_estimator_formula": analysis["candidate_estimator_formula"],
        "projected_mse_formula": analysis["projected_mse_formula"],
        "selection_cells_recomputed": len(analysis["selection_cell_results"]),
        "confirmation_cell_recomputed": True,
        "hand_computed_values_reproduced": True,
        "fault_injections_rejected": 3,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument(
        "--preregistration", type=Path, default=DEFAULT_PREREGISTRATION
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        result = self_test()
    else:
        if args.ledger is None or args.summary is None:
            fail("--ledger and --summary are required outside --self-test")
        preregistration = read_json(args.preregistration)
        section = (
            preregistration.get("rl")
            if isinstance(preregistration, dict)
            else None
        )
        if not isinstance(section, dict):
            fail("preregistration has no RL section")
        analysis = analyze_rows(read_jsonl(args.ledger), section)
        result = validate_summary(
            read_json(args.summary), analysis, digest_file(args.ledger), section
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except AggregationError as exc:
        raise SystemExit(f"error: {exc}") from exc
