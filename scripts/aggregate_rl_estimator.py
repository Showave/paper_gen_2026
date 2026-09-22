#!/usr/bin/env python3
"""Recompute the RL estimator audit from its trajectory ledger.

Successful candidate rows store the projected per-trajectory term before its
stratum mean: ``A=(h-b)s+alpha*w*(U-h)s`` for current rows and
``B=w*(U-h)s`` for stale rows.  This script reconstructs
``mean(A)+(1-alpha)*mean(B)``, the cross-reference MSE, and the registered
three-way bootstrap by independently resampling candidate replications and
both reference-trajectory samples.  Its generated self-test is equation and
artifact validation, not empirical evidence.
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

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised by deployment preflight
    np = None


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
            isinstance(component, (int, float))
            and not isinstance(component, bool)
            and math.isfinite(component)
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


def nearest_rank_quantile(values: list[float], probability: float) -> float:
    if not values or not 0 <= probability <= 1:
        fail("bootstrap quantile requires nonempty values and a probability")
    ordered = sorted(values)
    index = math.ceil(probability * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def bootstrap_seed(base_seed: int, law: str) -> int:
    payload = f"rl-three-way-bootstrap-v1|{base_seed}|{law}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def three_way_bootstrap_intervals(
    *,
    law: str,
    candidate_estimates: dict[tuple[float, str], list[list[float]]],
    reference_one: list[list[float]],
    reference_two: list[list[float]],
    replicates: int,
    seed: int,
    dimension: int,
    chunk_size: int,
    numpy_version: str,
    rng_name: str,
) -> dict[tuple[float, str], list[float]]:
    """Chunked exact multinomial resampling of all three empirical sources."""
    if (
        np is None
        or np.__version__ != numpy_version
        or rng_name != "numpy.random.PCG64"
        or not isinstance(replicates, int)
        or replicates <= 0
        or not isinstance(seed, int)
        or isinstance(seed, bool)
        or not isinstance(chunk_size, int)
        or chunk_size <= 0
        or not reference_one
        or not reference_two
        or any(not estimates for estimates in candidate_estimates.values())
    ):
        fail("three-way bootstrap inputs are incomplete")
    ordered_cells = sorted(candidate_estimates)
    r1 = np.asarray(reference_one, dtype=np.float64)
    r2 = np.asarray(reference_two, dtype=np.float64)
    candidate_arrays = {
        cell: np.asarray(candidate_estimates[cell], dtype=np.float64)
        for cell in ordered_cells
    }
    if (
        r1.shape != (len(reference_one), dimension)
        or r2.shape != (len(reference_two), dimension)
        or any(
            values.shape != (len(candidate_estimates[cell]), dimension)
            for cell, values in candidate_arrays.items()
        )
    ):
        fail("three-way bootstrap arrays do not match the projection dimension")

    def uniform_probabilities(count: int) -> Any:
        probabilities = np.full(count, 1.0 / count, dtype=np.float64)
        probabilities[-1] = 1.0 - float(probabilities[:-1].sum())
        return probabilities

    r1_probabilities = uniform_probabilities(len(r1))
    r2_probabilities = uniform_probabilities(len(r2))
    candidate_probabilities = {
        cell: uniform_probabilities(len(values))
        for cell, values in candidate_arrays.items()
    }
    candidate_norms = {
        cell: np.einsum("ij,ij->i", values, values)
        for cell, values in candidate_arrays.items()
    }
    draws = np.empty((replicates, len(ordered_cells)), dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(bootstrap_seed(seed, law)))
    for start in range(0, replicates, chunk_size):
        stop = min(start + chunk_size, replicates)
        count = stop - start
        r1_weights = rng.multinomial(
            len(r1), r1_probabilities, size=count
        )
        r2_weights = rng.multinomial(
            len(r2), r2_probabilities, size=count
        )
        r1_mean = r1_weights @ r1 / len(r1)
        r2_mean = r2_weights @ r2 / len(r2)
        reference_cross = np.einsum("ij,ij->i", r1_mean, r2_mean)
        for column, cell in enumerate(ordered_cells):
            values = candidate_arrays[cell]
            weights = rng.multinomial(
                len(values), candidate_probabilities[cell], size=count
            )
            candidate_mean = weights @ values / len(values)
            candidate_norm_mean = (
                weights @ candidate_norms[cell] / len(values)
            )
            draws[start:stop, column] = (
                candidate_norm_mean
                - np.einsum("ij,ij->i", candidate_mean, r1_mean)
                - np.einsum("ij,ij->i", candidate_mean, r2_mean)
                + reference_cross
            )
    return {
        cell: [
            nearest_rank_quantile(draws[:, column].tolist(), 0.025),
            nearest_rank_quantile(draws[:, column].tolist(), 0.975),
        ]
        for column, cell in enumerate(ordered_cells)
    }


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
        "mse_bootstrap_replicates",
        "mse_bootstrap_seed",
        "mse_bootstrap_chunk_size",
        "mse_bootstrap_numpy_version",
        "mse_bootstrap_rng",
    )
    if any(field not in audit for field in required):
        fail("RL estimator_audit is incomplete")
    if (
        not isinstance(audit["mse_bootstrap_replicates"], int)
        or audit["mse_bootstrap_replicates"] <= 0
        or not isinstance(audit["mse_bootstrap_seed"], int)
        or isinstance(audit["mse_bootstrap_seed"], bool)
        or not isinstance(audit["mse_bootstrap_chunk_size"], int)
        or audit["mse_bootstrap_chunk_size"] <= 0
        or not isinstance(audit["mse_bootstrap_numpy_version"], str)
        or not audit["mse_bootstrap_numpy_version"]
        or audit["mse_bootstrap_rng"] != "numpy.random.PCG64"
    ):
        fail("RL bootstrap replicate count or seed is invalid")
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
    references: dict[tuple[str, str], list[tuple[str, list[float]]]] = defaultdict(
        list
    )
    candidates: dict[
        tuple[str, float, str, int, str], list[tuple[str, list[float]]]
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
            or isinstance(cost, bool)
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
            references[(law, role)].append((trajectory_id, vector))
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
        candidates[(law, key[0], key[1], replication, source)].append(
            (trajectory_id, vector)
        )
        candidate_costs[(law, key[0], key[1], replication)] += float(cost)

    if failures:
        return {
            "aggregation_status": "blocked_infrastructure_failure",
            "empirical_evidence": False,
            "estimator_claim_eligible": False,
            "infrastructure_failures": failures,
            "attempts": len(attempt_ids),
        }

    reference_count = audit["reference_current_policy_trajectories_per_replica"]
    reference_vectors: dict[tuple[str, str], list[list[float]]] = {}
    reference_means: dict[tuple[str, str], list[float]] = {}
    for law in ("select", "confirm"):
        for role in ("r1", "r2"):
            identified_vectors = references.get((law, role), [])
            if len(identified_vectors) != reference_count:
                fail(f"{law} {role} reference count does not reproduce")
            vectors = [
                vector
                for _, vector in sorted(
                    identified_vectors, key=lambda item: item[0]
                )
            ]
            reference_vectors[(law, role)] = vectors
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
        if law == "confirm" and not 1 <= len(present_cells) <= 2:
            fail("confirmation law must contain one or two frozen cells")
        estimates_by_cell: dict[tuple[float, str], list[list[float]]] = {}
        for alpha, allocation_id in sorted(present_cells):
            current_count, stale_count = cells[(alpha, allocation_id)]
            replication_mse: list[float] = []
            replication_cost: list[float] = []
            replication_estimates: list[list[float]] = []
            for replication in range(replications):
                identified_current = candidates.get(
                    (law, alpha, allocation_id, replication, "current"), []
                )
                identified_stale = candidates.get(
                    (law, alpha, allocation_id, replication, "stale"), []
                )
                if (
                    len(identified_current) != current_count
                    or len(identified_stale) != stale_count
                ):
                    fail("candidate stratum counts do not reproduce")
                current = [
                    vector
                    for _, vector in sorted(
                        identified_current, key=lambda item: item[0]
                    )
                ]
                stale = [
                    vector
                    for _, vector in sorted(
                        identified_stale, key=lambda item: item[0]
                    )
                ]
                estimate = candidate_estimate(current, stale, alpha, dimension)
                replication_estimates.append(estimate)
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
            estimates_by_cell[(alpha, allocation_id)] = replication_estimates
            point_mse = statistics.fmean(replication_mse)
            point_cost = statistics.fmean(replication_cost)
            law_results[law].append(
                {
                    "alpha": alpha,
                    "allocation_id": allocation_id,
                    "projected_mse": point_mse,
                    "accelerator_seconds": point_cost,
                    "mse_times_accelerator_seconds": point_mse * point_cost,
                    "selection_risk": max(point_mse, 0.0) * point_cost,
                    "replications": replications,
                }
            )
        intervals = three_way_bootstrap_intervals(
            law=law,
            candidate_estimates=estimates_by_cell,
            reference_one=reference_vectors[(law, "r1")],
            reference_two=reference_vectors[(law, "r2")],
            replicates=audit["mse_bootstrap_replicates"],
            seed=audit["mse_bootstrap_seed"],
            dimension=dimension,
            chunk_size=audit["mse_bootstrap_chunk_size"],
            numpy_version=audit["mse_bootstrap_numpy_version"],
            rng_name=audit["mse_bootstrap_rng"],
        )
        for result in law_results[law]:
            result["three_way_bootstrap_interval"] = intervals[
                (result["alpha"], result["allocation_id"])
            ]

    selected = min(
        law_results["select"],
        key=lambda result: (
            result["selection_risk"],
            result["accelerator_seconds"],
            -result["alpha"],
            result["allocation_id"],
        ),
    )
    on_policy_cells = {
        key for key in cells if math.isclose(key[0], 1.0)
    }
    if len(on_policy_cells) != 1:
        fail("RL audit must define exactly one alpha=1 control cell")
    selected_key = (selected["alpha"], selected["allocation_id"])
    control_key = next(iter(on_policy_cells))
    expected_confirmation_cells = {selected_key, control_key}
    observed_confirmation_cells = {
        (result["alpha"], result["allocation_id"])
        for result in law_results["confirm"]
    }
    if observed_confirmation_cells != expected_confirmation_cells:
        fail("confirmation must contain the selected and alpha=1 control cells")
    confirmation_by_key = {
        (result["alpha"], result["allocation_id"]): result
        for result in law_results["confirm"]
    }
    selected_confirmation = confirmation_by_key[selected_key]
    control_confirmation = confirmation_by_key[control_key]
    confirmation_contrast = {
        "selected_minus_on_policy_projected_mse": (
            selected_confirmation["projected_mse"]
            - control_confirmation["projected_mse"]
        ),
        "selected_minus_on_policy_selection_risk": (
            selected_confirmation["selection_risk"]
            - control_confirmation["selection_risk"]
        ),
    }
    return {
        "aggregation_status": "complete",
        "empirical_evidence": True,
        "estimator_claim_eligible": True,
        "infrastructure_failures": 0,
        "attempts": len(attempt_ids),
        "reference_accelerator_seconds": reference_accelerator_seconds,
        "projection_dimension": dimension,
        "projected_mse_formula": "(g_hat-R1)^T(g_hat-R2)",
        "candidate_estimator_formula": "mean(A_current)+(1-alpha)*mean(B_stale)",
        "three_way_bootstrap_rule": (
            "independently resample candidate replications, R1 trajectories, "
            "and R2 trajectories within target law"
        ),
        "bootstrap_seed_schedule": (
            "first 64 bits of sha256("
            "'rl-three-way-bootstrap-v1'|base_seed|target_law)"
        ),
        "bootstrap_engine": (
            f"numpy-{audit['mse_bootstrap_numpy_version']}:"
            f"{audit['mse_bootstrap_rng']}"
        ),
        "bootstrap_chunk_size": audit["mse_bootstrap_chunk_size"],
        "selection_cell_results": law_results["select"],
        "selected_alpha": selected["alpha"],
        "selected_allocation_id": selected["allocation_id"],
        "confirmation_results": law_results["confirm"],
        "confirmation_contrast": confirmation_contrast,
    }


def close(observed: Any, expected: Any) -> bool:
    return (
        isinstance(observed, (int, float))
        and not isinstance(observed, bool)
        and math.isfinite(observed)
        and math.isclose(float(observed), float(expected), rel_tol=1e-12, abs_tol=1e-12)
    )


def validate_summary(
    summary: Any,
    analysis: dict[str, Any],
    ledger_sha256: str,
    section: dict[str, Any],
    *,
    execution_class: str,
) -> dict[str, Any]:
    if (
        not isinstance(summary, dict)
        or execution_class not in {"real", "synthetic-audit"}
        or summary.get("execution_class") != execution_class
        or summary.get("empirical_evidence") is not (execution_class == "real")
        or summary.get("ledger_sha256") != ledger_sha256
        or summary.get("aggregation_status") != analysis["aggregation_status"]
        or summary.get("estimator_claim_eligible")
        is not analysis["estimator_claim_eligible"]
    ):
        fail("RL summary does not bind the analyzed trajectory ledger")
    if analysis["aggregation_status"] == "blocked_infrastructure_failure":
        forbidden = {
            "projected_mse_by_cell",
            "cell_results",
            "selected_alpha",
            "selected_allocation_id",
            "confirmation_results",
            "confirmation_contrast",
        }
        if forbidden.intersection(summary):
            fail("an infrastructure-blocked RL summary reports estimator results")
        return {
            "aggregation_status": analysis["aggregation_status"],
            "estimator_claim_eligible": False,
            "infrastructure_failures": analysis["infrastructure_failures"],
        }

    audit = audit_configuration(section)
    if (
        summary.get("projected_mse_formula") != analysis["projected_mse_formula"]
        or summary.get("candidate_estimator_formula")
        != analysis["candidate_estimator_formula"]
        or summary.get("three_way_bootstrap_rule")
        != analysis["three_way_bootstrap_rule"]
        or summary.get("bootstrap_seed_schedule")
        != analysis["bootstrap_seed_schedule"]
        or summary.get("bootstrap_engine") != analysis["bootstrap_engine"]
        or summary.get("bootstrap_chunk_size")
        != analysis["bootstrap_chunk_size"]
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
            or not close(reported.get("selection_risk"), expected["selection_risk"])
            or not isinstance(interval, list)
            or len(interval) != 2
            or not all(
                close(observed, expected_value)
                for observed, expected_value in zip(
                    interval, expected["three_way_bootstrap_interval"]
                )
            )
            or interval[0] > interval[1]
        ):
            fail("RL summary selection result does not recompute from the ledger")
        seen.add(key)
    reported_confirmation = summary.get("confirmation_results")
    expected_confirmation = analysis["confirmation_results"]
    if (
        not isinstance(reported_confirmation, list)
        or len(reported_confirmation) != len(expected_confirmation)
    ):
        fail("RL summary must report selected and on-policy confirmation results")
    expected_confirmation_by_key = {
        f"{result['alpha']}|{result['allocation_id']}": result
        for result in expected_confirmation
    }
    seen_confirmation: set[str] = set()
    for reported in reported_confirmation:
        key = f"{reported.get('alpha')}|{reported.get('allocation_id')}"
        expected = expected_confirmation_by_key.get(key)
        interval = reported.get("three_way_bootstrap_interval")
        if (
            expected is None
            or key in seen_confirmation
            or any(
                not close(reported.get(field), expected[field])
                for field in (
                    "projected_mse",
                    "accelerator_seconds",
                    "mse_times_accelerator_seconds",
                    "selection_risk",
                )
            )
            or not isinstance(interval, list)
            or len(interval) != 2
            or not all(
                close(observed, expected_value)
                for observed, expected_value in zip(
                    interval, expected["three_way_bootstrap_interval"]
                )
            )
            or interval[0] > interval[1]
        ):
            fail("RL confirmation result does not recompute from the ledger")
        seen_confirmation.add(key)
    if seen_confirmation != set(expected_confirmation_by_key):
        fail("RL summary omits a required confirmation cell")
    reported_contrast = summary.get("confirmation_contrast")
    if not isinstance(reported_contrast, dict) or any(
        not close(reported_contrast.get(field), value)
        for field, value in analysis["confirmation_contrast"].items()
    ):
        fail("RL selected-minus-on-policy confirmation contrast is incorrect")
    return {
        "aggregation_status": "complete",
        "estimator_claim_eligible": True,
        "selection_cells_recomputed": len(expected_results),
        "confirmation_cells_recomputed": len(expected_confirmation),
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
            "mse_bootstrap_chunk_size": 4,
            "mse_bootstrap_numpy_version": "2.5.3",
            "mse_bootstrap_rng": "numpy.random.PCG64",
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
    for law in ("select", "confirm"):
        for replication, (current, stale) in enumerate(
            (([0.1, 0.0], [0.0, 0.2]), ([0.0, 0.1], [0.2, 0.0]))
        ):
            rows.append(
                make_row(
                    attempt=f"{law}-half-current-{replication}",
                    law=law,
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
                    attempt=f"{law}-half-stale-{replication}",
                    law=law,
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
        results.append(item)
        by_cell[f"{result['alpha']}|{result['allocation_id']}"] = result[
            "projected_mse"
        ]
    confirmation_results = []
    for result in analysis["confirmation_results"]:
        item = dict(result)
        confirmation_results.append(item)
    audit = section["estimator_audit"]
    return {
        "execution_class": "synthetic-audit",
        "aggregation_status": "complete",
        "empirical_evidence": False,
        "estimator_claim_eligible": True,
        "ledger_sha256": ledger_sha256,
        "projected_mse_formula": analysis["projected_mse_formula"],
        "candidate_estimator_formula": analysis["candidate_estimator_formula"],
        "three_way_bootstrap_rule": analysis["three_way_bootstrap_rule"],
        "bootstrap_seed_schedule": analysis["bootstrap_seed_schedule"],
        "bootstrap_engine": analysis["bootstrap_engine"],
        "bootstrap_chunk_size": analysis["bootstrap_chunk_size"],
        "projected_mse_by_cell": by_cell,
        "cell_results": results,
        "selected_alpha": analysis["selected_alpha"],
        "selected_allocation_id": analysis["selected_allocation_id"],
        "reference_accelerator_seconds": analysis[
            "reference_accelerator_seconds"
        ],
        "confirmation_results": confirmation_results,
        "confirmation_contrast": analysis["confirmation_contrast"],
        "bootstrap_replicates": audit["mse_bootstrap_replicates"],
        "bootstrap_seed": audit["mse_bootstrap_seed"],
        "three_way_reference_bootstrap": True,
    }


def self_test() -> dict[str, Any]:
    rows, section = generated_fixture()
    analysis = analyze_rows(rows, section)
    if (
        analysis["selected_alpha"] != 0.5
        or analysis["selected_allocation_id"] != "half"
        or not close(analysis["selection_cell_results"][0]["projected_mse"], 0.02)
        or not close(analysis["selection_cell_results"][1]["projected_mse"], 0.5)
    ):
        fail("generated fixture does not reproduce hand-computed estimator values")
    ledger_sha256 = hashlib.sha256(b"generated-ledger-v1").hexdigest()
    summary = generated_summary(analysis, ledger_sha256, section)
    validate_summary(
        summary,
        analysis,
        ledger_sha256,
        section,
        execution_class="synthetic-audit",
    )

    tampered = copy.deepcopy(summary)
    tampered["cell_results"][0]["projected_mse"] += 0.25
    try:
        validate_summary(
            tampered,
            analysis,
            ledger_sha256,
            section,
            execution_class="synthetic-audit",
        )
    except AggregationError:
        pass
    else:
        fail("self-test accepted a tampered MSE")

    tampered_interval = copy.deepcopy(summary)
    tampered_interval["cell_results"][0]["three_way_bootstrap_interval"][0] -= 0.25
    try:
        validate_summary(
            tampered_interval,
            analysis,
            ledger_sha256,
            section,
            execution_class="synthetic-audit",
        )
    except AggregationError:
        pass
    else:
        fail("self-test accepted a fabricated three-way bootstrap interval")

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

    production_section = read_json(DEFAULT_PREREGISTRATION)["rl"]
    production_audit = audit_configuration(production_section)
    production_dimension = production_audit["projection_dimension"]
    zero_reference = [
        [0.0] * production_dimension
        for _ in range(
            production_audit[
                "reference_current_policy_trajectories_per_replica"
            ]
        )
    ]
    zero_candidates = {
        (float(cell["alpha"]), cell["allocation_id"]): [
            [0.0] * production_dimension
            for _ in range(production_audit["selection_replications"])
        ]
        for cell in production_audit["stratum_counts_by_alpha"]
    }
    production_intervals = three_way_bootstrap_intervals(
        law="select",
        candidate_estimates=zero_candidates,
        reference_one=zero_reference,
        reference_two=zero_reference,
        replicates=production_audit["mse_bootstrap_replicates"],
        seed=production_audit["mse_bootstrap_seed"],
        dimension=production_dimension,
        chunk_size=production_audit["mse_bootstrap_chunk_size"],
        numpy_version=production_audit["mse_bootstrap_numpy_version"],
        rng_name=production_audit["mse_bootstrap_rng"],
    )
    if (
        set(production_intervals) != set(zero_candidates)
        or any(interval != [0.0, 0.0] for interval in production_intervals.values())
    ):
        fail("production-shape vectorized bootstrap fixture did not reproduce")
    return {
        "execution_class": "synthetic-audit",
        "empirical_evidence": False,
        "candidate_estimator_formula": analysis["candidate_estimator_formula"],
        "projected_mse_formula": analysis["projected_mse_formula"],
        "selection_cells_recomputed": len(analysis["selection_cell_results"]),
        "confirmation_cells_recomputed": len(analysis["confirmation_results"]),
        "bootstrap_replicates_recomputed": section["estimator_audit"][
            "mse_bootstrap_replicates"
        ],
        "production_shape": {
            "projection_dimension": production_dimension,
            "candidate_replications": production_audit[
                "selection_replications"
            ],
            "reference_trajectories_per_replica": production_audit[
                "reference_current_policy_trajectories_per_replica"
            ],
            "bootstrap_replicates": production_audit[
                "mse_bootstrap_replicates"
            ],
            "chunk_size": production_audit["mse_bootstrap_chunk_size"],
            "selection_cells": len(zero_candidates),
        },
        "production_shape_vectorized_bootstrap_validated": True,
        "hand_computed_values_reproduced": True,
        "fault_injections_rejected": 4,
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
            read_json(args.summary),
            analysis,
            digest_file(args.ledger),
            section,
            execution_class="real",
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except AggregationError as exc:
        raise SystemExit(f"error: {exc}") from exc
