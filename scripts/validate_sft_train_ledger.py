#!/usr/bin/env python3
"""Close SFT factorial endpoint rows against their per-round train ledgers.

Each factorial endpoint row names one run manifest.  The manifest binds a
hash-verified JSONL train ledger with one record per scheduling round.  The
validator replays the ledger under the frozen SFT preregistration and requires

* a contiguous parameter-state and optimizer-step chain from the common
  initial state, with exactly one step per accepted update and none for a
  rejected update or null action;
* admission of every round only when its reserve fits the remaining
  accelerator-second budget, complete component charging, and a terminal
  reason that reproduces the stop rule (maximum rounds, exhausted reserve, or
  a retained reserve overrun marked as a protocol failure);
* the frozen response-token window, pack capacity, per-round physical pack
  assignment and capacity, per-example source-slice coverage floor
  (at least ``source_coverage_floor`` of selected examples per capability
  slice), deterministic length-descending best-fit packing for best-fit cells,
  reserve recomputation from the 200-plan cost-calibration bundle (0.995
  residual quantile margin plus worst-case gate reserve), fresh per-slice gate
  blocks, per-draw clipped gate-loss recomputation across paired scale
  candidates, the heuristic upper statistic, largest-passing-scale commitment
  to that scale's candidate state, and the cumulative protected-tolerance
  ledger;
* each non-null round's 16-plan shortlist re-scoring, per-example sketch
  aggregation, and selection rule: candidate-independent ``u_c^T v`` across the
  shortlist, global-``N_B``-normalized selected-plan sketch inner products
  ``u_c^T q_B = (1/N_B) sum_{i in B} u_c^T q_i`` and
  ``v^T q_B = (1/N_B) sum_{i in B} v^T q_i``, estimated sketch scores
  ``widehat S_c = -u_c^T v + eta u_c^T q_B``, penalized lower scores
  ``underline S_c = widehat S_c - e_c`` across target and protected slices,
  sketched step-norm quadratic
  ``widehat q = ||v||_2^2 - 2 eta v^T q_B + eta^2 ||q_B||_2^2`` with
  Cauchy--Schwarz check ``(v^T q_B)^2 <= ||v||_2^2 ||q_B||_2^2``, upper
  step-norm bound ``overline q = widehat q + e^q``, protected
  curvature/tolerance and trust-radius feasibility, and selection of
  the feasible shortlist plan maximizing ``underline S_0`` (or a rejected/null
  round if no shortlist plan is feasible);
* audit endpoints evaluated on the final ledger state, from which the row's
  target and worst-protected NLL changes are recomputed;
* every declared initial and final audit NLL recomputed from hash-bound
  per-example loss records as the frozen row-uniform mean of per-example
  response-token-mean NLL, with records evaluated on the exact initial and
  final states and covering exactly the hash-identified audit partition;
* sealed audits for protocol-failure runs, which enter only as failures.

Across rows, paired runs must share the split-manifest hash, the reference
cost per target setting, the initial state, cost-calibration bundle, and
initial audit NLLs per seed and target setting, and one disjoint gate-partition
and audit-partition identity per slice.  The per-example forward passes and
hardware timing measurements themselves are still produced by the build,
training, and evaluation stages; this closes their shortlist selection, source
coverage, pack plans, reserve calibration, clipping, pairing, aggregation, and
state binding, not the hardware execution that measured them.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREREGISTRATION = ROOT / "experiments" / "pilot_preregistration.json"
TOLERANCE = 1e-9
CHARGE_COMPONENTS = (
    "packing",
    "reference_gradients",
    "candidate_scoring",
    "solving",
    "gate_forwards",
    "training",
    "rejected_update",
)
TERMINAL_REASONS = ("maximum_rounds", "reserve_exhausted", "protocol_failure")
DECISIONS = ("accepted", "rejected", "null_action")


class LedgerError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LedgerError(message)


def canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def close(a: Any, b: float) -> bool:
    return isinstance(a, (int, float)) and math.isfinite(a) and abs(a - b) <= TOLERANCE * max(
        1.0, abs(a), abs(b)
    )


def protected_slices(repo: Path, section: dict[str, Any]) -> dict[str, list[str]]:
    contract_path = repo / section["reference_contract"]["specification_path"]
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    return {
        setting["setting_id"]: list(setting["protected_capabilities"])
        for setting in contract["target_settings"]
    }


def target_slices(repo: Path, section: dict[str, Any]) -> dict[str, str]:
    contract_path = repo / section["reference_contract"]["specification_path"]
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    return {
        setting["setting_id"]: setting["target_capability"]
        for setting in contract["target_settings"]
    }


def safe_child(root: Path, relative: Any) -> Path:
    require(isinstance(relative, str), "ledger path must be a string")
    path = Path(relative)
    require(
        path.parts and not path.is_absolute() and ".." not in path.parts,
        "ledger path must stay under the manifest root",
    )
    return root / path


def load_ledger(
    path: Path, expected_sha256: str, kind: str = "train ledger"
) -> list[dict[str, Any]]:
    require(path.is_file(), f"{kind} not found: {path}")
    payload = path.read_bytes()
    require(
        hashlib.sha256(payload).hexdigest() == expected_sha256,
        f"{kind} hash mismatch: {path}",
    )
    rows = []
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LedgerError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        require(isinstance(value, dict), f"{path}:{line_number}: expected an object")
        rows.append(value)
    return rows


def load_json_verified(
    path: Path, expected_sha256: str, kind: str = "JSON artifact"
) -> dict[str, Any]:
    require(path.is_file(), f"{kind} not found: {path}")
    payload = path.read_bytes()
    require(
        hashlib.sha256(payload).hexdigest() == expected_sha256,
        f"{kind} hash mismatch: {path}",
    )
    try:
        value = json.loads(payload.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise LedgerError(f"{path}: invalid JSON: {exc}") from exc
    require(isinstance(value, dict), f"{path}: expected an object")
    return value


def predicted_pack_cost(pack_loads: list[int], cost_model: dict[str, Any]) -> float:
    c0 = float(cost_model["launch_overhead_seconds"])
    c1 = float(cost_model["linear_token_coeff"])
    c2 = float(cost_model["quadratic_load_coeff"])
    max_len = float(cost_model["max_sequence_length"])
    return math.fsum(
        [c0 * len(pack_loads)]
        + [c1 * load + c2 * (load * load) / max_len for load in pack_loads]
    )


def deterministic_best_fit_packing(
    example_ids: list[str],
    packed_lengths: list[int],
    max_sequence_length: int,
    max_packs: int,
) -> list[int]:
    """Canonical length-descending (tie: example_id) best-fit pack assignment."""
    ordered = sorted(
        range(len(example_ids)),
        key=lambda idx: (-packed_lengths[idx], example_ids[idx], idx),
    )
    pack_loads: list[int] = []
    assignments = [-1] * len(example_ids)
    for idx in ordered:
        length = packed_lengths[idx]
        fitting = [
            pack_id
            for pack_id, current in enumerate(pack_loads)
            if current + length <= max_sequence_length
        ]
        if fitting:
            best_pack = min(
                fitting,
                key=lambda pack_id: (
                    max_sequence_length - (pack_loads[pack_id] + length),
                    pack_id,
                ),
            )
            pack_loads[best_pack] += length
            assignments[idx] = best_pack
        else:
            best_pack = len(pack_loads)
            require(
                best_pack < max_packs,
                "deterministic best-fit packing exceeds physical pack capacity",
            )
            pack_loads.append(length)
            assignments[idx] = best_pack
    return assignments


def validate_cost_calibration_bundle(
    bundle: dict[str, Any],
    section: dict[str, Any],
    *,
    expected_seed: int | None = None,
    expected_target_setting: str | None = None,
    require_real: bool = False,
) -> dict[str, Any]:
    """Validate the 200-plan hardware cost-calibration bundle and reserve margin."""
    require(bundle.get("schema_version") == 1, "unsupported cost-calibration bundle schema")
    if require_real:
        require(
            bundle.get("synthetic_audit") is False
            and bundle.get("empirical_evidence") is True,
            "real SFT execution cannot use a synthetic cost-calibration bundle",
        )
    else:
        require(
            isinstance(bundle.get("synthetic_audit"), bool)
            and isinstance(bundle.get("empirical_evidence"), bool),
            "cost-calibration bundle needs synthetic_audit and empirical_evidence flags",
        )
    require(
        bundle.get("model_key") == section["primary_factorial_model_key"],
        "cost-calibration bundle model_key differs from primary_factorial_model_key",
    )
    if expected_seed is not None:
        require(
            bundle.get("seed") == expected_seed,
            "cost-calibration bundle seed differs from the run seed",
        )
    if expected_target_setting is not None:
        require(
            bundle.get("target_setting") == expected_target_setting,
            "cost-calibration bundle target_setting differs from the run target_setting",
        )
    window = section["optimizer_window"]
    gate = section["actual_update_gate"]
    cost_design = section["cost_design"]
    max_seq_len = window["maximum_sequence_length"]
    max_packs = window["physical_packs_per_update"]
    cost_model = bundle.get("cost_model")
    require(
        isinstance(cost_model, dict)
        and set(cost_model)
        == {
            "launch_overhead_seconds",
            "linear_token_coeff",
            "quadratic_load_coeff",
            "max_sequence_length",
        }
        and cost_model["max_sequence_length"] == max_seq_len
        and all(
            isinstance(cost_model[k], (int, float))
            and not isinstance(cost_model[k], bool)
            and math.isfinite(cost_model[k])
            and cost_model[k] > 0
            for k in (
                "launch_overhead_seconds",
                "linear_token_coeff",
                "quadratic_load_coeff",
            )
        ),
        "cost-calibration bundle cost_model is invalid",
    )
    c_fwd = bundle.get("gate_forward_cost_per_example_seconds")
    require(
        isinstance(c_fwd, (int, float))
        and not isinstance(c_fwd, bool)
        and math.isfinite(c_fwd)
        and c_fwd > 0,
        "cost-calibration bundle needs positive gate_forward_cost_per_example_seconds",
    )
    expected_gate_reserve = (
        float(c_fwd)
        * (len(gate["scale_grid"]) + 1)
        * gate["protected_slices"]
        * gate["samples_per_slice_round"]
    )
    require(
        close(bundle.get("worst_case_gate_reserve_seconds"), expected_gate_reserve),
        "cost-calibration bundle worst_case_gate_reserve_seconds does not reproduce",
    )
    plans = bundle.get("calibration_plans")
    expected_plan_count = cost_design["independent_cost_calibration_plans"]
    require(
        isinstance(plans, list) and len(plans) == expected_plan_count,
        f"cost-calibration bundle must contain exactly {expected_plan_count} independent plans",
    )
    plan_ids: set[str] = set()
    calibration_example_ids: set[str] = set()
    residuals: list[float] = []
    for index, plan in enumerate(plans):
        label = f"cost-calibration plan {index}"
        require(
            isinstance(plan, dict)
            and set(plan)
            == {
                "plan_id",
                "example_ids",
                "pack_loads",
                "predicted_cost_seconds",
                "measured_cost_seconds",
                "residual_seconds",
            },
            f"{label} has an unexpected schema",
        )
        plan_id = plan["plan_id"]
        require(
            isinstance(plan_id, str) and plan_id and plan_id not in plan_ids,
            f"{label} has a missing or duplicate plan_id",
        )
        plan_ids.add(plan_id)
        ex_ids = plan["example_ids"]
        require(
            isinstance(ex_ids, list)
            and ex_ids
            and all(isinstance(x, str) and x for x in ex_ids)
            and len(set(ex_ids)) == len(ex_ids)
            and not (set(ex_ids) & calibration_example_ids),
            f"{label} reuses or omits calibration example IDs",
        )
        calibration_example_ids.update(ex_ids)
        pack_loads = plan["pack_loads"]
        require(
            isinstance(pack_loads, list)
            and 1 <= len(pack_loads) <= max_packs
            and all(
                isinstance(load, int)
                and not isinstance(load, bool)
                and 1 <= load <= max_seq_len
                for load in pack_loads
            ),
            f"{label} has invalid pack_loads",
        )
        pred = predicted_pack_cost(pack_loads, cost_model)
        meas = plan["measured_cost_seconds"]
        res = plan["residual_seconds"]
        require(
            isinstance(meas, (int, float))
            and not isinstance(meas, bool)
            and math.isfinite(meas)
            and meas > 0
            and close(plan["predicted_cost_seconds"], pred)
            and close(res, float(meas) - pred),
            f"{label} predicted_cost_seconds or residual_seconds does not reproduce",
        )
        residuals.append(float(meas) - pred)

    quantile_prob = float(cost_design["reserve_quantile"])
    require(
        close(bundle.get("reserve_quantile"), quantile_prob),
        "cost-calibration bundle reserve_quantile differs from the preregistration",
    )
    ordered_residuals = sorted(residuals)
    q_index = max(0, min(math.ceil(quantile_prob * len(ordered_residuals)) - 1, len(ordered_residuals) - 1))
    expected_margin = ordered_residuals[q_index]
    require(
        expected_margin >= 0 and close(bundle.get("reserve_margin_seconds"), expected_margin),
        "cost-calibration bundle reserve_margin_seconds does not reproduce the 0.995 residual quantile",
    )
    return {
        "cost_model": cost_model,
        "worst_case_gate_reserve_seconds": expected_gate_reserve,
        "reserve_margin_seconds": expected_margin,
        "calibration_example_ids": calibration_example_ids,
    }


def recompute_round_plans_and_reserves(
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    section: dict[str, Any],
    calibration: dict[str, Any],
    target_slice: str,
    protected: list[str],
) -> tuple[set[str], dict[str, str]]:
    """Verify every round's physical pack plan, source coverage, shortlist selection, and reserve."""
    window = section["optimizer_window"]
    selector = section["selector"]
    gate = section["actual_update_gate"]
    max_seq_len = window["maximum_sequence_length"]
    max_packs = window["physical_packs_per_update"]
    shortlist_size = selector["shortlist_size"]
    source_floor = float(section["data_roles"]["source_coverage_floor"])
    require(0 < source_floor <= 1.0 / 3.0, "invalid source_coverage_floor")
    all_slices = {target_slice, *protected}
    cell = manifest["run_identity"]["cell"]
    require_best_fit = cell.endswith("deterministic_best_fit")
    cost_model = calibration["cost_model"]
    gate_reserve = calibration["worst_case_gate_reserve_seconds"]
    margin = calibration["reserve_margin_seconds"]
    calibration_ids = calibration["calibration_example_ids"]
    candidate_ids: set[str] = set()
    candidate_slices_by_id: dict[str, str] = {}
    for row in rows:
        label = f"round {row['round']}"
        decision = row["decision"]
        charged = row["charged"]
        sunk_cost = math.fsum(
            charged[k]
            for k in ("packing", "reference_gradients", "candidate_scoring", "solving")
        )
        plan = row.get("selected_plan")
        shortlist = row.get("shortlist")
        if decision == "null_action":
            require(
                plan is None and shortlist is None,
                f"{label} null action cannot carry a selected_plan or shortlist",
            )
            continue
        require(
            isinstance(plan, dict)
            and set(plan)
            == {
                "plan_id",
                "example_ids",
                "source_slices",
                "packed_lengths",
                "response_tokens_by_example",
                "pack_assignments",
                "u_dot_q_by_slice",
                "v_dot_q_by_example",
            },
            f"{label} selected_plan has an invalid schema",
        )
        plan_id = plan["plan_id"]
        ex_ids = plan["example_ids"]
        source_slices = plan["source_slices"]
        lengths = plan["packed_lengths"]
        resp_tokens = plan["response_tokens_by_example"]
        assignments = plan["pack_assignments"]
        u_dot_q_by_slice = plan["u_dot_q_by_slice"]
        v_dot_q_by_example = plan["v_dot_q_by_example"]
        require(
            isinstance(plan_id, str)
            and plan_id
            and isinstance(ex_ids, list)
            and ex_ids
            and all(isinstance(x, str) and x for x in ex_ids)
            and len(set(ex_ids)) == len(ex_ids)
            and not (set(ex_ids) & calibration_ids),
            f"{label} selected_plan has invalid, duplicate, or calibration-overlapping example_ids",
        )
        candidate_ids.update(ex_ids)
        count = len(ex_ids)
        require(
            isinstance(source_slices, list)
            and len(source_slices) == count
            and all(isinstance(s, str) and s in all_slices for s in source_slices),
            f"{label} selected_plan source_slices are invalid",
        )
        for ex_id, s_name in zip(ex_ids, source_slices):
            previous_slice = candidate_slices_by_id.setdefault(ex_id, s_name)
            require(
                previous_slice == s_name,
                f"{label} example {ex_id} changes capability slice across rounds",
            )
        for s_name in sorted(all_slices):
            s_count = sum(1 for s in source_slices if s == s_name)
            require(
                s_count >= source_floor * count - TOLERANCE,
                f"{label} selected_plan violates source_coverage_floor for {s_name}",
            )
        require(
            isinstance(lengths, list)
            and len(lengths) == count
            and isinstance(resp_tokens, list)
            and len(resp_tokens) == count
            and isinstance(assignments, list)
            and len(assignments) == count
            and all(
                isinstance(n_i, int)
                and not isinstance(n_i, bool)
                and isinstance(r_i, int)
                and not isinstance(r_i, bool)
                and 1 <= r_i <= n_i <= max_seq_len
                for n_i, r_i in zip(lengths, resp_tokens)
            ),
            f"{label} selected_plan lengths or response tokens are invalid",
        )
        n_b = sum(resp_tokens)
        require(
            n_b == row["response_tokens"],
            f"{label} selected_plan response_tokens_by_example do not sum to response_tokens",
        )
        require(
            isinstance(u_dot_q_by_slice, dict)
            and set(u_dot_q_by_slice) == all_slices
            and all(
                isinstance(u_dot_q_by_slice[s_name], list)
                and len(u_dot_q_by_slice[s_name]) == count
                and all(
                    isinstance(val, (int, float))
                    and not isinstance(val, bool)
                    and math.isfinite(val)
                    for val in u_dot_q_by_slice[s_name]
                )
                for s_name in all_slices
            )
            and isinstance(v_dot_q_by_example, list)
            and len(v_dot_q_by_example) == count
            and all(
                isinstance(val, (int, float))
                and not isinstance(val, bool)
                and math.isfinite(val)
                for val in v_dot_q_by_example
            ),
            f"{label} selected_plan per-example sketch inner products are invalid",
        )
        recomputed_u_dot_q = {
            s_name: math.fsum(float(val) for val in u_dot_q_by_slice[s_name]) / n_b
            for s_name in all_slices
        }
        recomputed_v_dot_q = math.fsum(float(val) for val in v_dot_q_by_example) / n_b
        packs = row["physical_packs"]
        require(
            all(
                isinstance(p, int) and not isinstance(p, bool) and 0 <= p < packs
                for p in assignments
            )
            and set(assignments) == set(range(packs)),
            f"{label} pack_assignments do not partition the declared physical_packs",
        )
        pack_loads = [0] * packs
        for length, pack_id in zip(lengths, assignments):
            pack_loads[pack_id] += length
        require(
            all(1 <= load <= max_seq_len for load in pack_loads),
            f"{label} physical pack load exceeds maximum_sequence_length {max_seq_len}",
        )
        if require_best_fit:
            expected_assignments = deterministic_best_fit_packing(
                ex_ids, lengths, max_seq_len, max_packs
            )
            require(
                assignments == expected_assignments,
                f"{label} deterministic_best_fit cell does not use the canonical best-fit packing",
            )
        expected_reserve = math.fsum(
            [
                sunk_cost,
                predicted_pack_cost(pack_loads, cost_model),
                margin,
                gate_reserve,
            ]
        )
        require(
            close(row["reserve_accelerator_seconds"], expected_reserve),
            f"{label} reserve_accelerator_seconds does not reproduce from calibrated pack cost + margin + gate reserve",
        )

        require(
            isinstance(shortlist, dict)
            and set(shortlist)
            == {
                "learning_rate",
                "v_norm_sq",
                "u_dot_v",
                "trust_radius_sq",
                "curvature_upper",
                "candidates",
            },
            f"{label} shortlist has an invalid schema",
        )
        eta = shortlist["learning_rate"]
        v_norm_sq = shortlist["v_norm_sq"]
        round_u_dot_v = shortlist["u_dot_v"]
        rho_sq = shortlist["trust_radius_sq"]
        beta_upper = shortlist["curvature_upper"]
        candidates = shortlist["candidates"]
        require(
            close(eta, float(section["adamw"]["learning_rate"]))
            and isinstance(v_norm_sq, (int, float))
            and not isinstance(v_norm_sq, bool)
            and math.isfinite(v_norm_sq)
            and v_norm_sq >= 0
            and isinstance(round_u_dot_v, dict)
            and set(round_u_dot_v) == all_slices
            and all(
                isinstance(round_u_dot_v[s], (int, float))
                and not isinstance(round_u_dot_v[s], bool)
                and math.isfinite(round_u_dot_v[s])
                for s in all_slices
            )
            and isinstance(rho_sq, (int, float))
            and not isinstance(rho_sq, bool)
            and math.isfinite(rho_sq)
            and rho_sq > 0
            and isinstance(beta_upper, dict)
            and set(beta_upper) == set(protected)
            and all(
                isinstance(beta_upper[s], (int, float))
                and not isinstance(beta_upper[s], bool)
                and math.isfinite(beta_upper[s])
                and beta_upper[s] > 0
                for s in protected
            )
            and isinstance(candidates, list)
            and len(candidates) == shortlist_size,
            f"{label} shortlist parameters or candidate count are invalid",
        )
        seen_shortlist_ids: set[str] = set()
        feasible_candidates: list[tuple[float, float, str]] = []
        selected_entry: dict[str, Any] | None = None
        remaining_budget = float(row["remaining_budget_before"])
        round_tau = float(gate["per_round_tolerance"])
        for c_idx, cand in enumerate(candidates):
            c_label = f"{label} shortlist candidate {c_idx}"
            require(
                isinstance(cand, dict)
                and set(cand)
                == {
                    "plan_id",
                    "pack_loads",
                    "reserve_accelerator_seconds",
                    "v_dot_q",
                    "q_norm_sq",
                    "step_norm_sq_hat",
                    "step_norm_sq_margin",
                    "step_norm_sq_upper",
                    "target_score",
                    "protected_scores",
                    "feasible",
                },
                f"{c_label} has an unexpected schema",
            )
            c_id = cand["plan_id"]
            require(
                isinstance(c_id, str) and c_id and c_id not in seen_shortlist_ids,
                f"{c_label} has a missing or duplicate plan_id",
            )
            seen_shortlist_ids.add(c_id)
            c_loads = cand["pack_loads"]
            require(
                isinstance(c_loads, list)
                and 1 <= len(c_loads) <= max_packs
                and all(
                    isinstance(ld, int) and not isinstance(ld, bool) and 1 <= ld <= max_seq_len
                    for ld in c_loads
                ),
                f"{c_label} has invalid pack_loads",
            )
            c_reserve = math.fsum(
                [
                    sunk_cost,
                    predicted_pack_cost(c_loads, cost_model),
                    margin,
                    gate_reserve,
                ]
            )
            require(
                close(cand["reserve_accelerator_seconds"], c_reserve),
                f"{c_label} reserve_accelerator_seconds does not reproduce",
            )
            v_dot_q = cand["v_dot_q"]
            q_norm_sq = cand["q_norm_sq"]
            q_hat = cand["step_norm_sq_hat"]
            q_margin = cand["step_norm_sq_margin"]
            q_upper = cand["step_norm_sq_upper"]
            require(
                isinstance(v_dot_q, (int, float))
                and not isinstance(v_dot_q, bool)
                and math.isfinite(v_dot_q)
                and isinstance(q_norm_sq, (int, float))
                and not isinstance(q_norm_sq, bool)
                and math.isfinite(q_norm_sq)
                and q_norm_sq >= 0
                and isinstance(q_hat, (int, float))
                and not isinstance(q_hat, bool)
                and isinstance(q_margin, (int, float))
                and not isinstance(q_margin, bool)
                and q_hat >= 0
                and q_margin >= 0
                and float(v_dot_q) ** 2
                <= float(v_norm_sq) * float(q_norm_sq)
                + TOLERANCE * max(1.0, float(v_norm_sq) * float(q_norm_sq))
                and close(
                    q_hat,
                    float(v_norm_sq)
                    - 2.0 * float(eta) * float(v_dot_q)
                    + (float(eta) ** 2) * float(q_norm_sq),
                )
                and close(q_upper, float(q_hat) + float(q_margin)),
                f"{c_label} step_norm_sq_hat or step_norm_sq_upper violates the sketch quadratic or Cauchy-Schwarz bound",
            )
            q_bar = float(q_hat) + float(q_margin)

            def check_score_block(
                block: Any, block_label: str, slice_name: str
            ) -> float:
                require(
                    isinstance(block, dict)
                    and set(block)
                    == {"u_dot_v", "u_dot_q", "margin", "hat_s", "lower_s"}
                    and all(
                        isinstance(block[k], (int, float))
                        and not isinstance(block[k], bool)
                        and math.isfinite(block[k])
                        for k in ("u_dot_v", "u_dot_q", "margin", "hat_s", "lower_s")
                    )
                    and block["margin"] >= 0
                    and close(block["u_dot_v"], float(round_u_dot_v[slice_name])),
                    f"{block_label} has an invalid score schema or candidate-varying u_dot_v",
                )
                expected_hat = -float(block["u_dot_v"]) + float(eta) * float(block["u_dot_q"])
                expected_lower = expected_hat - float(block["margin"])
                require(
                    close(block["hat_s"], expected_hat)
                    and close(block["lower_s"], expected_lower),
                    f"{block_label} hat_s or lower_s does not reproduce from -u^T v + eta u^T q - margin",
                )
                return expected_lower

            s0_lower = check_score_block(
                cand["target_score"], f"{c_label} target_score", target_slice
            )
            prot_scores = cand["protected_scores"]
            require(
                isinstance(prot_scores, dict) and set(prot_scores) == set(protected),
                f"{c_label} protected_scores must cover exactly the protected slices",
            )
            prot_feasible = True
            for s_name in protected:
                sc_lower = check_score_block(
                    prot_scores[s_name],
                    f"{c_label} protected_scores[{s_name}]",
                    s_name,
                )
                threshold = 0.5 * float(beta_upper[s_name]) * q_bar - round_tau
                if sc_lower < threshold - TOLERANCE:
                    prot_feasible = False
            is_feasible = (
                s0_lower >= -TOLERANCE
                and prot_feasible
                and q_bar <= float(rho_sq) + TOLERANCE
                and c_reserve <= remaining_budget + TOLERANCE
            )
            require(
                cand["feasible"] is is_feasible,
                f"{c_label} feasible flag does not reproduce from Eq. (selection)",
            )
            if is_feasible:
                feasible_candidates.append((s0_lower, -c_reserve, c_id))
            if c_id == plan_id:
                selected_entry = cand

        require(
            selected_entry is not None
            and selected_entry["pack_loads"] == pack_loads
            and close(selected_entry["reserve_accelerator_seconds"], expected_reserve)
            and close(selected_entry["v_dot_q"], recomputed_v_dot_q)
            and close(
                selected_entry["target_score"]["u_dot_q"],
                recomputed_u_dot_q[target_slice],
            )
            and all(
                close(
                    selected_entry["protected_scores"][s_name]["u_dot_q"],
                    recomputed_u_dot_q[s_name],
                )
                for s_name in protected
            ),
            f"{label} selected_plan does not match its shortlist entry or global-N_B per-example sketch aggregation",
        )
        require(
            feasible_candidates,
            f"{label} selected a non-null plan when no shortlist candidate was feasible",
        )
        best_s0 = max(item[0] for item in feasible_candidates)
        best_ids = {
            item[2]
            for item in feasible_candidates
            if abs(item[0] - best_s0) <= TOLERANCE * max(1.0, abs(best_s0))
        }
        require(
            plan_id in best_ids,
            f"{label} did not select the feasible shortlist plan maximizing target lower_s",
        )
    return candidate_ids, candidate_slices_by_id


def replay_ledger(
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    section: dict[str, Any],
    slices: list[str],
) -> dict[str, Any]:
    gate = section["actual_update_gate"]
    window = section["optimizer_window"]
    scales = list(gate["scale_grid"])
    require(scales == sorted(scales, reverse=True), "frozen scale grid must be descending")
    budget = manifest["budget_accelerator_seconds"]
    require(rows, "train ledger must contain at least one round")
    require(len(rows) <= gate["maximum_rounds"], "train ledger exceeds maximum rounds")
    state = manifest["initial_state_sha256"]
    step = 0
    charged_total = 0.0
    tolerance_used = {name: 0.0 for name in slices}
    block_ids: set[str] = set()
    counts = {decision: 0 for decision in DECISIONS}
    overrun_round = None
    for index, row in enumerate(rows):
        label = f"round {index}"
        require(row.get("round") == index, f"{label} is out of order")
        require(overrun_round is None, "no round may follow a reserve overrun")
        require(row.get("state_sha256_before") == state, f"{label} breaks the state chain")
        require(row.get("optimizer_step_before") == step, f"{label} breaks the step chain")
        remaining = budget - charged_total
        require(
            close(row.get("remaining_budget_before"), remaining),
            f"{label} remaining budget does not reproduce",
        )
        reserve = row.get("reserve_accelerator_seconds")
        require(
            isinstance(reserve, (int, float)) and 0 < reserve <= remaining + TOLERANCE,
            f"{label} was admitted without a fitting reserve",
        )
        charged = row.get("charged")
        require(
            isinstance(charged, dict) and set(charged) == set(CHARGE_COMPONENTS),
            f"{label} must charge every cost component",
        )
        require(
            all(
                isinstance(charged[k], (int, float)) and charged[k] >= 0
                for k in CHARGE_COMPONENTS
            ),
            f"{label} has a negative or non-numeric charge",
        )
        round_cost = sum(charged[k] for k in CHARGE_COMPONENTS)
        decision = row.get("decision")
        require(decision in DECISIONS, f"{label} has an unknown decision")
        counts[decision] += 1
        if decision == "null_action":
            require(
                row.get("gate") is None
                and row.get("committed_scale") is None
                and row.get("response_tokens") is None
                and all(charged[k] == 0 for k in ("gate_forwards", "training", "rejected_update")),
                f"{label} null action cannot carry an update, gate, or update charge",
            )
            passing: list[float] = []
        else:
            tokens = row.get("response_tokens")
            packs = row.get("physical_packs")
            require(
                isinstance(tokens, int)
                and window["response_tokens_minimum"]
                <= tokens
                <= window["response_tokens_maximum"],
                f"{label} response tokens lie outside the frozen window",
            )
            require(
                isinstance(packs, int)
                and 1 <= packs <= window["physical_packs_per_update"],
                f"{label} exceeds the physical pack capacity",
            )
            evidence = row.get("gate")
            require(
                isinstance(evidence, dict) and set(evidence) == set(slices),
                f"{label} gate must cover exactly the protected slices",
            )
            tolerance = row.get("tolerance")
            require(
                isinstance(tolerance, dict)
                and set(tolerance) == set(slices)
                and all(close(tolerance[s], gate["per_round_tolerance"]) for s in slices),
                f"{label} tolerance differs from the frozen per-round tolerance",
            )
            passing = list(scales)
            for name in slices:
                block = evidence[name]
                require(
                    isinstance(block.get("block_id"), str)
                    and block["block_id"] not in block_ids,
                    f"{label} reuses a gate block for {name}",
                )
                block_ids.add(block["block_id"])
                require(
                    block.get("samples") == gate["samples_per_slice_round"],
                    f"{label} gate block for {name} has the wrong sample count",
                )
                means = block.get("mean_change")
                require(
                    isinstance(means, list)
                    and len(means) == len(scales)
                    and all(isinstance(v, (int, float)) and math.isfinite(v) for v in means),
                    f"{label} gate for {name} must report one mean per frozen scale",
                )
                uppers = [v + gate["empirical_margin"] for v in means]
                require(
                    block.get("upper") is not None
                    and len(block["upper"]) == len(scales)
                    and all(close(a, b) for a, b in zip(block["upper"], uppers)),
                    f"{label} gate upper statistic for {name} does not reproduce",
                )
                passing = [
                    scale
                    for scale, upper in zip(scales, uppers)
                    if scale in passing and upper <= tolerance[name] + TOLERANCE
                ]
                tolerance_used[name] += tolerance[name]
        committed = row.get("committed_scale")
        if decision == "accepted":
            require(passing, f"{label} accepted an update with no passing scale")
            require(
                committed == max(passing),
                f"{label} did not commit the largest passing scale",
            )
            require(charged["rejected_update"] == 0, f"{label} charges an accepted update as rejected")
            require(row.get("optimizer_step_after") == step + 1, f"{label} step count is wrong")
            require(
                is_sha256(row.get("state_sha256_after"))
                and row["state_sha256_after"] != state,
                f"{label} accepted update must change the parameter state",
            )
        else:
            if decision == "rejected":
                require(not passing, f"{label} rejected an update although a scale passed")
            require(committed is None, f"{label} commits a scale without an accepted update")
            require(
                row.get("optimizer_step_after") == step
                and row.get("state_sha256_after") == state,
                f"{label} must restore parameter and optimizer state",
            )
        state = row["state_sha256_after"]
        step = row["optimizer_step_after"]
        charged_total += round_cost
        if round_cost > reserve + TOLERANCE:
            overrun_round = index
    for name, used in tolerance_used.items():
        require(
            used <= gate["cumulative_tolerance_budget"] + TOLERANCE,
            f"cumulative tolerance for {name} exceeds its budget",
        )
    reason = manifest.get("terminal_reason")
    require(reason in TERMINAL_REASONS, "manifest has an unknown terminal reason")
    if overrun_round is not None:
        require(
            reason == "protocol_failure" and manifest.get("protocol_failure") is True,
            "a reserve overrun must be retained as a protocol failure",
        )
    else:
        require(
            reason != "protocol_failure" and manifest.get("protocol_failure") is False,
            "protocol failure is declared without a retained reserve overrun",
        )
        if reason == "maximum_rounds":
            require(len(rows) == gate["maximum_rounds"], "stopped before maximum rounds")
        else:
            next_reserve = manifest.get("next_round_reserve_accelerator_seconds")
            require(
                len(rows) < gate["maximum_rounds"]
                and isinstance(next_reserve, (int, float))
                and next_reserve > budget - charged_total + TOLERANCE,
                "reserve-exhausted stop does not reproduce",
            )
    return {
        "rounds": len(rows),
        "charged_accelerator_seconds": charged_total,
        "accepted_updates": counts["accepted"],
        "rejected_updates": counts["rejected"],
        "null_actions": counts["null_action"],
        "final_state_sha256": state,
        "final_optimizer_step": step,
        "tolerance_used": tolerance_used,
        "protocol_failure": overrun_round is not None,
    }


def recompute_gate_blocks(
    manifest: dict[str, Any],
    manifest_root: Path,
    rows: list[dict[str, Any]],
    section: dict[str, Any],
    slices: list[str],
) -> tuple[dict[str, str], dict[str, set[str]]]:
    """Recompute every round's gate mean changes from per-draw loss records."""
    gate = section["actual_update_gate"]
    scales = list(gate["scale_grid"])
    n_samples = gate["samples_per_slice_round"]
    clip_bound = float(gate["clipped_example_nll_bound"])
    require(clip_bound > 0, "clipped_example_nll_bound must be positive")
    declared_partitions = manifest.get("gate_partitions")
    require(
        isinstance(declared_partitions, dict) and set(declared_partitions) == set(slices),
        "run manifest must declare gate_partitions for every protected slice",
    )
    gate_partition_hashes: dict[str, str] = {}
    gate_partition_ids: dict[str, set[str]] = {}
    for name in slices:
        entry = declared_partitions[name]
        require(
            isinstance(entry, dict)
            and set(entry) == {"gate_partition_sha256", "example_ids"}
            and is_sha256(entry.get("gate_partition_sha256")),
            f"gate partition declaration for {name} is invalid",
        )
        ids = entry.get("example_ids")
        require(
            isinstance(ids, list)
            and ids
            and all(isinstance(item, str) and item for item in ids)
            and len(set(ids)) == len(ids)
            and canonical_digest(sorted(ids)) == entry["gate_partition_sha256"],
            f"gate partition example IDs for {name} do not reproduce its hash",
        )
        gate_partition_hashes[name] = entry["gate_partition_sha256"]
        gate_partition_ids[name] = set(ids)

    require(
        is_sha256(manifest.get("gate_loss_records_sha256")),
        "run manifest needs gate_loss_records_sha256",
    )
    records = load_ledger(
        safe_child(manifest_root, manifest.get("gate_loss_records_path")),
        manifest["gate_loss_records_sha256"],
        "gate-loss records",
    )
    expected_keys = {
        (row["round"], name)
        for row in rows
        if row.get("decision") != "null_action"
        for name in slices
    }
    by_key: dict[tuple[Any, Any], dict[str, Any]] = {}
    for index, record in enumerate(records):
        label = f"gate-loss record {index}"
        require(
            set(record)
            == {
                "round",
                "slice",
                "block_id",
                "gate_partition_sha256",
                "state_sha256_before",
                "candidate_state_sha256_by_scale",
                "example_ids",
                "response_tokens",
                "base_nll_sum",
                "candidate_nll_sum_by_scale",
            },
            f"{label} has an unexpected schema",
        )
        key = (record["round"], record["slice"])
        require(key not in by_key, f"duplicate gate-loss record for {key}")
        by_key[key] = record
    require(
        set(by_key) == expected_keys,
        "gate-loss records must cover exactly the non-null round/slice blocks",
    )

    for row in rows:
        if row["decision"] == "null_action":
            continue
        round_index = row["round"]
        round_candidate_states: list[str] | None = None
        for name in slices:
            label = f"round {round_index} {name} gate block"
            record = by_key[(round_index, name)]
            block = row["gate"][name]
            require(
                record["block_id"] == block["block_id"],
                f"{label} block_id differs from the train ledger",
            )
            require(
                record["gate_partition_sha256"] == gate_partition_hashes[name],
                f"{label} gate_partition_sha256 differs from the run manifest",
            )
            require(
                record["state_sha256_before"] == row["state_sha256_before"],
                f"{label} was not evaluated at state_sha256_before",
            )
            candidate_states = record["candidate_state_sha256_by_scale"]
            require(
                isinstance(candidate_states, list)
                and len(candidate_states) == len(scales)
                and all(is_sha256(state) for state in candidate_states)
                and len(set(candidate_states)) == len(scales)
                and row["state_sha256_before"] not in candidate_states,
                f"{label} needs distinct candidate state hashes per scale",
            )
            if round_candidate_states is None:
                round_candidate_states = candidate_states
            else:
                require(
                    candidate_states == round_candidate_states,
                    f"round {round_index} slices do not evaluate the same scale candidate states",
                )
            if row["decision"] == "accepted":
                committed_index = scales.index(row["committed_scale"])
                require(
                    candidate_states[committed_index] == row["state_sha256_after"],
                    f"round {round_index} accepted state_sha256_after differs from the committed scale candidate state",
                )
            draw_ids = record["example_ids"]
            allowed_ids = gate_partition_ids[name]
            require(
                isinstance(draw_ids, list)
                and len(draw_ids) == n_samples
                and set(map(type, draw_ids)) <= {str}
                and set(draw_ids) <= allowed_ids,
                f"{label} draws an example outside the {name} gate partition",
            )
            tokens = record["response_tokens"]
            require(
                isinstance(tokens, list)
                and len(tokens) == n_samples
                and set(map(type, tokens)) <= {int}
                and min(tokens) > 0,
                f"{label} needs positive integer response-token counts",
            )
            base_nll = record["base_nll_sum"]
            require(
                isinstance(base_nll, list)
                and len(base_nll) == n_samples
                and set(map(type, base_nll)) <= {int, float}
                and min(base_nll) >= 0
                and math.isfinite(sum(base_nll)),
                f"{label} needs finite nonnegative base NLL sums",
            )
            cand_by_scale = record["candidate_nll_sum_by_scale"]
            require(
                isinstance(cand_by_scale, list) and len(cand_by_scale) == len(scales),
                f"{label} must provide candidate NLL sums for every frozen scale",
            )
            base_clipped_sum = math.fsum(
                min(float(nll) / tok, clip_bound) for nll, tok in zip(base_nll, tokens)
            )
            for scale_index, cand_nll in enumerate(cand_by_scale):
                require(
                    isinstance(cand_nll, list)
                    and len(cand_nll) == n_samples
                    and set(map(type, cand_nll)) <= {int, float}
                    and min(cand_nll) >= 0
                    and math.isfinite(sum(cand_nll)),
                    f"{label} scale {scale_index} needs finite nonnegative candidate NLL sums",
                )
                recomputed_mean = (
                    math.fsum(
                        min(float(nll) / tok, clip_bound)
                        for nll, tok in zip(cand_nll, tokens)
                    )
                    - base_clipped_sum
                ) / n_samples
                require(
                    close(block["mean_change"][scale_index], recomputed_mean),
                    f"{label} scale {scales[scale_index]} mean_change does not reproduce from clipped per-draw records",
                )
    return gate_partition_hashes, gate_partition_ids


def recompute_audit_slice(
    entry: Any,
    name: str,
    manifest_root: Path,
    states: dict[str, str],
) -> tuple[dict[str, float], str, set[str]]:
    """Return recomputed initial/final audit NLLs, partition hash, and IDs."""
    require(isinstance(entry, dict), f"audit endpoint for {name} must be an object")
    partition = entry.get("audit_partition_sha256")
    require(is_sha256(partition), f"audit endpoint for {name} needs a partition hash")
    bindings = entry.get("loss_records")
    require(
        isinstance(bindings, dict) and set(bindings) == set(states),
        f"audit endpoint for {name} must bind initial and final loss records",
    )
    recomputed: dict[str, float] = {}
    example_sets: dict[str, list[str]] = {}
    for phase, state in states.items():
        binding = bindings[phase]
        require(
            isinstance(binding, dict) and is_sha256(binding.get("sha256")),
            f"{name} {phase} loss records need a SHA-256 binding",
        )
        records = load_ledger(
            safe_child(manifest_root, binding.get("path")),
            binding["sha256"],
            f"{name} {phase} audit-loss records",
        )
        require(records, f"{name} {phase} audit-loss records are empty")
        identifiers: list[str] = []
        per_example: list[float] = []
        for index, record in enumerate(records):
            label = f"{name} {phase} audit record {index}"
            require(
                set(record)
                == {
                    "example_id",
                    "slice",
                    "evaluated_state_sha256",
                    "response_tokens",
                    "nll_sum",
                },
                f"{label} has an unexpected schema",
            )
            require(
                isinstance(record["example_id"], str) and record["example_id"],
                f"{label} needs an example identifier",
            )
            require(record["slice"] == name, f"{label} belongs to another slice")
            require(
                record["evaluated_state_sha256"] == state,
                f"{label} was not evaluated on the {phase} state",
            )
            tokens = record["response_tokens"]
            require(
                isinstance(tokens, int) and not isinstance(tokens, bool) and tokens > 0,
                f"{label} needs a positive response-token count",
            )
            nll = record["nll_sum"]
            require(
                isinstance(nll, (int, float))
                and not isinstance(nll, bool)
                and math.isfinite(nll)
                and nll >= 0,
                f"{label} needs a finite nonnegative summed NLL",
            )
            identifiers.append(record["example_id"])
            per_example.append(nll / tokens)
        require(
            len(set(identifiers)) == len(identifiers),
            f"{name} {phase} audit-loss records repeat an example",
        )
        example_sets[phase] = sorted(identifiers)
        recomputed[phase] = math.fsum(per_example) / len(per_example)
        declared = entry.get(phase)
        require(
            close(declared, recomputed[phase]),
            f"{name} {phase} audit NLL does not reproduce from per-example records",
        )
    require(
        example_sets["initial"] == example_sets["final"],
        f"{name} initial and final audits cover different examples",
    )
    require(
        canonical_digest(example_sets["initial"]) == partition,
        f"{name} audit records do not cover exactly the declared partition",
    )
    return recomputed, partition, set(example_sets["initial"])


def validate_train_manifest(
    manifest: dict[str, Any],
    manifest_root: Path,
    row: dict[str, Any],
    section: dict[str, Any],
    slices_by_setting: dict[str, list[str]],
    targets_by_setting: dict[str, str] | None = None,
) -> dict[str, Any]:
    if targets_by_setting is None:
        targets_by_setting = target_slices(ROOT, section)
    require(manifest.get("schema_version") == 1, "unsupported run-manifest schema")
    identity = manifest.get("run_identity")
    expected_identity = {
        key: row[key]
        for key in ("seed", "target_setting", "budget_fraction", "cell", "protected_tolerance")
    }
    require(identity == expected_identity, "run manifest identity differs from its endpoint row")
    for key in (
        "split_manifest_sha256",
        "initial_state_sha256",
        "train_ledger_sha256",
        "cost_calibration_bundle_sha256",
    ):
        require(is_sha256(manifest.get(key)), f"run manifest needs {key}")
    reference = manifest.get("reference_cost_accelerator_seconds")
    require(
        isinstance(reference, (int, float)) and reference > 0,
        "run manifest needs a positive reference cost",
    )
    require(
        close(manifest.get("budget_accelerator_seconds"), row["budget_fraction"] * reference),
        "run budget is not the frozen fraction of the reference cost",
    )
    slices = slices_by_setting[row["target_setting"]]
    target_slice = targets_by_setting[row["target_setting"]]
    rows = load_ledger(
        safe_child(manifest_root, manifest.get("train_ledger_path")),
        manifest["train_ledger_sha256"],
    )
    cost_bundle = load_json_verified(
        safe_child(manifest_root, manifest.get("cost_calibration_bundle_path")),
        manifest["cost_calibration_bundle_sha256"],
        "cost-calibration bundle",
    )
    calibration = validate_cost_calibration_bundle(
        cost_bundle,
        section,
        expected_seed=row["seed"],
        expected_target_setting=row["target_setting"],
    )
    candidate_example_ids, candidate_slices_by_id = recompute_round_plans_and_reserves(
        manifest, rows, section, calibration, target_slice, slices
    )
    closure = replay_ledger(manifest, rows, section, slices)
    gate_partitions, gate_partition_ids = recompute_gate_blocks(
        manifest, manifest_root, rows, section, slices
    )
    for gate_name, gate_ids in gate_partition_ids.items():
        require(
            not (gate_ids & candidate_example_ids),
            f"candidate plan examples overlap {gate_name} gate partition",
        )
        require(
            not (gate_ids & calibration["calibration_example_ids"]),
            f"cost-calibration examples overlap {gate_name} gate partition",
        )
    totals = manifest.get("totals")
    require(isinstance(totals, dict), "run manifest needs ledger totals")
    for key in (
        "accepted_updates",
        "rejected_updates",
        "null_actions",
        "final_state_sha256",
        "final_optimizer_step",
    ):
        require(totals.get(key) == closure[key], f"manifest total {key} does not reproduce")
    require(
        close(totals.get("charged_accelerator_seconds"), closure["charged_accelerator_seconds"]),
        "manifest charged accelerator-seconds do not reproduce",
    )
    return {
        **closure,
        "split_manifest_sha256": manifest["split_manifest_sha256"],
        "initial_state_sha256": manifest["initial_state_sha256"],
        "cost_calibration_bundle_sha256": manifest["cost_calibration_bundle_sha256"],
        "reference_cost_accelerator_seconds": reference,
        "gate_partitions": gate_partitions,
        "gate_partition_ids": gate_partition_ids,
        "candidate_example_ids": candidate_example_ids,
        "candidate_slices_by_id": candidate_slices_by_id,
        "calibration_example_ids": calibration["calibration_example_ids"],
    }


def validate_run(
    manifest: dict[str, Any],
    manifest_root: Path,
    row: dict[str, Any],
    section: dict[str, Any],
    slices_by_setting: dict[str, list[str]],
    targets_by_setting: dict[str, str] | None = None,
) -> dict[str, Any]:
    if targets_by_setting is None:
        targets_by_setting = target_slices(ROOT, section)
    closure = validate_train_manifest(
        manifest, manifest_root, row, section, slices_by_setting, targets_by_setting
    )
    slices = slices_by_setting[row["target_setting"]]
    gate_partition_ids = closure.pop("gate_partition_ids")
    candidate_example_ids = closure.pop("candidate_example_ids")
    closure.pop("candidate_slices_by_id", None)
    calibration_example_ids = closure.pop("calibration_example_ids")
    endpoint = manifest.get("audit_endpoint")
    audit_nll: dict[str, dict[str, float]] = {}
    partitions: dict[str, str] = {}
    if closure["protocol_failure"]:
        require(row["status"] == "failed", "a protocol-failure run must enter as failed")
        require(
            endpoint is None,
            "sealed audits must remain closed for a protocol-failure run",
        )
    else:
        require(row["status"] == "completed", "a closed run must enter as completed")
        require(isinstance(endpoint, dict), "run manifest needs an audit endpoint")
        require(
            endpoint.get("evaluated_state_sha256") == closure["final_state_sha256"],
            "audit endpoint was not evaluated on the final ledger state",
        )
        protected = endpoint.get("protected_audit_nll")
        require(
            isinstance(protected, dict) and set(protected) == set(slices),
            "audit endpoint must cover exactly the protected slices",
        )
        states = {
            "initial": manifest["initial_state_sha256"],
            "final": closure["final_state_sha256"],
        }
        target_name = targets_by_setting[row["target_setting"]]
        require(target_name not in slices, "target slice cannot also be protected")
        entries = {target_name: endpoint.get("target_audit_nll"), **protected}
        audit_ids_by_slice: dict[str, set[str]] = {}
        for name, entry in entries.items():
            audit_nll[name], partitions[name], audit_ids = recompute_audit_slice(
                entry, name, manifest_root, states
            )
            audit_ids_by_slice[name] = audit_ids
            require(
                not (candidate_example_ids & audit_ids),
                f"candidate plan examples overlap {name} audit partition",
            )
            require(
                not (calibration_example_ids & audit_ids),
                f"cost-calibration examples overlap {name} audit partition",
            )
        for gate_name, gate_ids in gate_partition_ids.items():
            for audit_name, audit_ids in audit_ids_by_slice.items():
                require(
                    not (gate_ids & audit_ids),
                    f"gate partition for {gate_name} overlaps audit partition for {audit_name}",
                )
        target = audit_nll[target_name]
        require(
            close(row["target_delta_nll"], target["final"] - target["initial"]),
            "endpoint target NLL change does not reproduce",
        )
        worst = max(audit_nll[s]["final"] - audit_nll[s]["initial"] for s in slices)
        require(
            close(row["protected_delta_nll"], worst),
            "endpoint worst protected NLL change does not reproduce",
        )
    return {
        **closure,
        "initial_audit_nll": {name: value["initial"] for name, value in audit_nll.items()},
        "audit_partitions": partitions,
    }


def check_pairing(closures: list[tuple[dict[str, Any], dict[str, Any]]]) -> None:
    splits = {closure["split_manifest_sha256"] for _, closure in closures}
    require(len(splits) == 1, "factorial rows do not share one split-manifest hash")
    reference: dict[str, float] = {}
    initial: dict[tuple[Any, str], str] = {}
    calibration_bundles: dict[tuple[Any, str], str] = {}
    gate_partitions: dict[str, str] = {}
    partitions: dict[str, str] = {}
    initial_nll: dict[tuple[tuple[Any, str], str], float] = {}
    for row, closure in closures:
        setting = row["target_setting"]
        cost = reference.setdefault(setting, closure["reference_cost_accelerator_seconds"])
        require(close(cost, closure["reference_cost_accelerator_seconds"]), (
            f"reference cost differs within target setting {setting}"
        ))
        key = (row["seed"], setting)
        state = initial.setdefault(key, closure["initial_state_sha256"])
        require(state == closure["initial_state_sha256"], (
            f"paired runs for {key} do not share an initial state"
        ))
        calib_sha = calibration_bundles.setdefault(
            key, closure["cost_calibration_bundle_sha256"]
        )
        require(
            calib_sha == closure["cost_calibration_bundle_sha256"],
            f"paired runs for {key} do not share a cost-calibration bundle",
        )
        for name, partition in closure.get("gate_partitions", {}).items():
            shared_gate = gate_partitions.setdefault(name, partition)
            require(shared_gate == partition, f"gate partition for {name} differs across rows")
        for name, partition in closure.get("audit_partitions", {}).items():
            shared = partitions.setdefault(name, partition)
            require(shared == partition, f"audit partition for {name} differs across rows")
            require(
                gate_partitions.get(name) != partition,
                f"gate and audit partition hashes for {name} collide",
            )
        for name, value in closure.get("initial_audit_nll", {}).items():
            shared_nll = initial_nll.setdefault((key, name), value)
            require(close(value, shared_nll), (
                f"paired runs for {key} report different initial {name} audit NLLs"
            ))


def fake_hash(*parts: Any) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


FIXTURE_AUDIT_TOKENS = (17, 40, 64, 9, 128, 33, 72, 21)
FIXTURE_AUDIT_OFFSETS = (0.3, -0.3, 0.15, -0.15, 0.05, -0.05, 0.2, -0.2)
FIXTURE_GATE_POOL_SIZE = 16


def fixture_audit_ids(name: str) -> list[str]:
    return [f"{name}-audit-{index}" for index in range(len(FIXTURE_AUDIT_TOKENS))]


def fixture_gate_ids(name: str) -> list[str]:
    return [f"{name}-gate-{index}" for index in range(FIXTURE_GATE_POOL_SIZE)]


@functools.lru_cache(maxsize=64)
def cached_gate_draw_arrays(
    name: str,
    n_samples: int,
    clip_bound: float,
    means: tuple[float, ...],
) -> tuple[list[str], list[int], list[float], list[list[float]], dict[str, str]]:
    """Construct paired 512-draw gate arrays with clipped tail and token variation.

    The first 1/8 of draws exceed ``clip_bound`` at both base and candidate
    states (1.35 vs 1.20 per token), so clipping each example-average NLL to
    ``[0, clip_bound]`` before differencing yields zero change on those draws,
    whereas unclipped or post-difference clipping shifts the mean by
    ``-0.15 / 8 = -0.01875``.  The remaining 7/8 of draws alternate token
    counts (16 vs 48) and opposite zero-sum offsets (+0.02 vs -0.02), so a
    token-weighted mean differs from the draw-uniform mean by ``-0.01``.
    """
    pool = fixture_gate_ids(name)
    example_ids = [pool[(j * 7 + 3) % len(pool)] for j in range(n_samples)]
    clipped_prefix = n_samples // 8
    active_count = n_samples - clipped_prefix
    require(active_count > 0 and active_count % 2 == 0, "invalid gate sample count for fixture")
    tokens = [16 if j % 2 == 0 else 48 for j in range(n_samples)]
    base_nll = [
        1.35 * clip_bound * tokens[j]
        if j < clipped_prefix
        else (0.50 + (-0.01 if j % 2 == 0 else 0.01)) * tokens[j]
        for j in range(n_samples)
    ]
    cand_by_scale: list[list[float]] = []
    scale_factor = n_samples / active_count
    for target_mean in means:
        shifted_delta = target_mean * scale_factor
        cand_nll = []
        for j in range(n_samples):
            if j < clipped_prefix:
                cand_nll.append(1.20 * clip_bound * tokens[j])
            else:
                base_rate = 0.50 + (-0.01 if j % 2 == 0 else 0.01)
                offset = 0.02 if j % 2 == 0 else -0.02
                cand_rate = base_rate + shifted_delta + offset
                cand_nll.append(cand_rate * tokens[j])
        cand_by_scale.append(cand_nll)
    serialized = {
        "base_nll_sum": json.dumps(base_nll, allow_nan=False),
        "candidate_nll_sum_by_scale": json.dumps(cand_by_scale, allow_nan=False),
        "example_ids": json.dumps(example_ids, allow_nan=False),
        "response_tokens": json.dumps(tokens, allow_nan=False),
    }
    return example_ids, tokens, base_nll, cand_by_scale, serialized


def write_audit_records(
    directory: Path, run_key: str, name: str, phase: str, state: str, value: float
) -> dict[str, str]:
    """Write generated per-example records whose row-uniform mean is ``value``."""
    records = [
        {
            "example_id": example_id,
            "slice": name,
            "evaluated_state_sha256": state,
            "response_tokens": tokens,
            "nll_sum": (value + offset) * tokens,
        }
        for example_id, tokens, offset in zip(
            fixture_audit_ids(name), FIXTURE_AUDIT_TOKENS, FIXTURE_AUDIT_OFFSETS
        )
    ]
    relative = f"{run_key}-audit-{name}-{phase}.jsonl"
    payload = "".join(
        json.dumps(record, sort_keys=True, allow_nan=False) + "\n" for record in records
    ).encode()
    (directory / relative).write_bytes(payload)
    return {"path": relative, "sha256": hashlib.sha256(payload).hexdigest()}


def fixture_audit_entry(
    directory: Path,
    run_key: str,
    name: str,
    states: dict[str, str],
    values: dict[str, float],
) -> dict[str, Any]:
    return {
        **values,
        "audit_partition_sha256": canonical_digest(sorted(fixture_audit_ids(name))),
        "loss_records": {
            phase: write_audit_records(directory, run_key, name, phase, states[phase], values[phase])
            for phase in ("initial", "final")
        },
    }


@functools.lru_cache(maxsize=64)
def cached_cost_calibration_bundle_bytes(
    seed: int,
    target_setting: str,
    model_key: str,
    max_seq_len: int,
    n_plans: int,
    quantile_prob: float,
    n_scales: int,
    n_slices: int,
    n_samples: int,
) -> tuple[bytes, str]:
    cost_model = {
        "launch_overhead_seconds": 0.5,
        "linear_token_coeff": 0.00025,
        "quadratic_load_coeff": 0.0003125,
        "max_sequence_length": max_seq_len,
    }
    c_fwd = 0.00125
    gate_reserve = c_fwd * (n_scales + 1) * n_slices * n_samples
    q_index = max(0, min(math.ceil(quantile_prob * n_plans) - 1, n_plans - 1))
    plans = []
    for idx in range(n_plans):
        pack_loads = [2048 + 256 * ((idx + p) % 5) for p in range(4 + (idx % 5))]
        pred = predicted_pack_cost(pack_loads, cost_model)
        if idx < q_index:
            residual = round(0.02 * (idx + 1), 6)
        elif idx == q_index:
            residual = 4.432
        else:
            residual = 4.5 + 0.01 * (idx - q_index)
        plans.append(
            {
                "plan_id": f"calib-plan-{idx:03d}",
                "example_ids": [f"calib-{idx:03d}-ex-{k:02d}" for k in range(len(pack_loads))],
                "pack_loads": pack_loads,
                "predicted_cost_seconds": pred,
                "measured_cost_seconds": pred + residual,
                "residual_seconds": residual,
            }
        )
    bundle = {
        "schema_version": 1,
        "synthetic_audit": True,
        "empirical_evidence": False,
        "seed": seed,
        "target_setting": target_setting,
        "model_key": model_key,
        "cost_model": cost_model,
        "gate_forward_cost_per_example_seconds": c_fwd,
        "worst_case_gate_reserve_seconds": gate_reserve,
        "reserve_quantile": quantile_prob,
        "reserve_margin_seconds": 4.432,
        "calibration_plans": plans,
    }
    payload = (json.dumps(bundle, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    return payload, hashlib.sha256(payload).hexdigest()


def build_fixture_run(
    directory: Path,
    section: dict[str, Any],
    slices_by_setting: dict[str, list[str]],
    identity: dict[str, Any],
    target_delta: float,
    protected_delta: float,
    overrun: bool = False,
    path_prefix: str = "",
) -> tuple[str, str, dict[str, Any]]:
    """Write a generated, closed run and return (path, sha256, endpoint row).

    Every written path is ``path_prefix`` plus a run-specific file name, so a
    caller can place a run under a stage directory such as ``train/``.
    """
    gate = section["actual_update_gate"]
    scales = gate["scale_grid"]
    slices = slices_by_setting[identity["target_setting"]]
    gate_partitions = {
        name: {
            "gate_partition_sha256": canonical_digest(sorted(fixture_gate_ids(name))),
            "example_ids": sorted(fixture_gate_ids(name)),
        }
        for name in slices
    }
    reference = 1000.0
    budget = identity["budget_fraction"] * reference
    run_key = path_prefix + canonical_digest(identity)[:16]
    calib_bytes, calib_sha256 = cached_cost_calibration_bundle_bytes(
        identity["seed"],
        identity["target_setting"],
        section["primary_factorial_model_key"],
        section["optimizer_window"]["maximum_sequence_length"],
        section["cost_design"]["independent_cost_calibration_plans"],
        float(section["cost_design"]["reserve_quantile"]),
        len(scales),
        gate["protected_slices"],
        gate["samples_per_slice_round"],
    )
    calib_name = f"{run_key}-cost-calibration.json"
    (directory / calib_name).write_bytes(calib_bytes)
    target_name = target_slices(ROOT, section)[identity["target_setting"]]
    ordered_all_slices = [target_name, *slices]
    eta = float(section["adamw"]["learning_rate"])
    shortlist_size = section["selector"]["shortlist_size"]
    round_tau = float(gate["per_round_tolerance"])
    initial = fake_hash("initial", identity["seed"], identity["target_setting"])
    state = initial
    step = 0
    charged_total = 0.0
    rows = []
    gate_record_lines: list[str] = []
    round_index = 0
    while round_index < gate["maximum_rounds"]:
        reserve = 40.0
        if reserve > budget - charged_total:
            break
        pattern = round_index % 4
        decision = "null_action" if pattern == 3 else ("rejected" if pattern == 2 else "accepted")
        charged = dict.fromkeys(CHARGE_COMPONENTS, 0.0)
        charged.update(packing=1.0, reference_gradients=4.0, candidate_scoring=3.0, solving=2.0)
        row: dict[str, Any] = {
            "round": round_index,
            "state_sha256_before": state,
            "optimizer_step_before": step,
            "remaining_budget_before": budget - charged_total,
            "reserve_accelerator_seconds": reserve,
            "decision": decision,
            "gate": None,
            "tolerance": None,
            "committed_scale": None,
            "response_tokens": None,
            "physical_packs": None,
            "selected_plan": None,
            "shortlist": None,
        }
        candidate_states = [
            fake_hash("candidate-state", run_key, round_index, scale)
            for scale in scales
        ]
        candidate_states_json = json.dumps(candidate_states, allow_nan=False)
        first_pass = 1 if decision == "accepted" else len(scales)
        if decision != "null_action":
            charged.update(gate_forwards=6.0, training=12.0)
            if decision == "rejected":
                charged["rejected_update"] = 3.0
            row["gate"] = {}
            for slice_index, name in enumerate(slices):
                means = [
                    0.02 - 0.004 * k if k < first_pass else -0.001 * (k + slice_index)
                    for k in range(len(scales))
                ]
                block_id = f"{run_key[len(path_prefix):]}-{round_index}-{name}"
                row["gate"][name] = {
                    "block_id": block_id,
                    "samples": gate["samples_per_slice_round"],
                    "mean_change": means,
                    "upper": [m + gate["empirical_margin"] for m in means],
                }
                _, _, _, _, serialized = cached_gate_draw_arrays(
                    name,
                    gate["samples_per_slice_round"],
                    float(gate["clipped_example_nll_bound"]),
                    tuple(means),
                )
                gate_record_lines.append(
                    f'{{"base_nll_sum": {serialized["base_nll_sum"]}, '
                    f'"block_id": "{block_id}", '
                    f'"candidate_nll_sum_by_scale": {serialized["candidate_nll_sum_by_scale"]}, '
                    f'"candidate_state_sha256_by_scale": {candidate_states_json}, '
                    f'"example_ids": {serialized["example_ids"]}, '
                    f'"gate_partition_sha256": "{gate_partitions[name]["gate_partition_sha256"]}", '
                    f'"response_tokens": {serialized["response_tokens"]}, '
                    f'"round": {round_index}, '
                    f'"slice": "{name}", '
                    f'"state_sha256_before": "{state}"}}\n'
                )
            row["tolerance"] = dict.fromkeys(slices, gate["per_round_tolerance"])
            row["response_tokens"] = 6144
            row["physical_packs"] = section["optimizer_window"]["physical_packs_per_update"]
            selected_plan_id = f"plan-{round_index:02d}-00"
            selected_pack_loads = [4096 if p < 4 else 3072 for p in range(8)]
            resp_by_ex = [256 if k % 2 == 0 else 512 for k in range(16)]
            u_dot_q_by_slice = {
                target_name: [
                    200.0 * 256.0 - 64.0 if k % 2 == 0 else 200.0 * 512.0 + 64.0
                    for k in range(16)
                ],
                **{
                    s_name: [
                        (50.0 - 2.0 * s_idx) * 256.0 - 32.0
                        if k % 2 == 0
                        else (50.0 - 2.0 * s_idx) * 512.0 + 32.0
                        for k in range(16)
                    ]
                    for s_idx, s_name in enumerate(slices)
                },
            }
            row["selected_plan"] = {
                "plan_id": selected_plan_id,
                "example_ids": [f"cand-{round_index:02d}-{k:02d}" for k in range(16)],
                "source_slices": [ordered_all_slices[k % 3] for k in range(16)],
                "packed_lengths": [2048 if k < 8 else 1536 for k in range(16)],
                "response_tokens_by_example": resp_by_ex,
                "pack_assignments": [k // 2 for k in range(16)],
                "u_dot_q_by_slice": u_dot_q_by_slice,
                "v_dot_q_by_example": [-16.0 if k % 2 == 0 else 16.0 for k in range(16)],
            }
            v_norm_sq = 0.08
            round_u_dot_v = {
                target_name: -0.02,
                **{s_name: -0.01 - 0.002 * s_idx for s_idx, s_name in enumerate(slices)},
            }
            shortlist_candidates = []
            for m in range(shortlist_size):
                u_dot_v_0 = round_u_dot_v[target_name]
                u_dot_q_0 = 200.0 - 5.0 * m if m < 14 else -500.0
                margin_0 = 0.01
                hat_s_0 = -u_dot_v_0 + eta * u_dot_q_0
                lower_s_0 = hat_s_0 - margin_0
                v_dot_q = -5.0 * m
                q_norm_sq = 25000.0 * (m + 1)
                q_hat = v_norm_sq - 2.0 * eta * v_dot_q + (eta**2) * q_norm_sq
                q_margin = 0.02 if m < 15 else 0.40
                q_upper = q_hat + q_margin
                prot_scores = {}
                prot_ok = True
                for s_idx, s_name in enumerate(slices):
                    u_dot_v_c = round_u_dot_v[s_name]
                    u_dot_q_c = 50.0 - 2.0 * s_idx if m != 13 else -500.0
                    margin_c = 0.005
                    hat_s_c = -u_dot_v_c + eta * u_dot_q_c
                    lower_s_c = hat_s_c - margin_c
                    prot_scores[s_name] = {
                        "u_dot_v": u_dot_v_c,
                        "u_dot_q": u_dot_q_c,
                        "margin": margin_c,
                        "hat_s": hat_s_c,
                        "lower_s": lower_s_c,
                    }
                    if lower_s_c < 0.5 * 0.1 * q_upper - round_tau:
                        prot_ok = False
                cand_feasible = (
                    lower_s_0 >= 0.0
                    and prot_ok
                    and q_upper <= 0.25
                    and reserve <= budget - charged_total
                )
                shortlist_candidates.append(
                    {
                        "plan_id": f"plan-{round_index:02d}-{m:02d}",
                        "pack_loads": list(selected_pack_loads),
                        "reserve_accelerator_seconds": reserve,
                        "v_dot_q": v_dot_q,
                        "q_norm_sq": q_norm_sq,
                        "step_norm_sq_hat": q_hat,
                        "step_norm_sq_margin": q_margin,
                        "step_norm_sq_upper": q_upper,
                        "target_score": {
                            "u_dot_v": u_dot_v_0,
                            "u_dot_q": u_dot_q_0,
                            "margin": margin_0,
                            "hat_s": hat_s_0,
                            "lower_s": lower_s_0,
                        },
                        "protected_scores": prot_scores,
                        "feasible": cand_feasible,
                    }
                )
            row["shortlist"] = {
                "learning_rate": eta,
                "v_norm_sq": v_norm_sq,
                "u_dot_v": round_u_dot_v,
                "trust_radius_sq": 0.25,
                "curvature_upper": dict.fromkeys(slices, 0.1),
                "candidates": shortlist_candidates,
            }
            if decision == "accepted":
                row["committed_scale"] = scales[first_pass]
        if overrun and round_index == 1:
            charged["training"] += 30.0
        after_state = (
            candidate_states[first_pass] if decision == "accepted" else state
        )
        after_step = step + 1 if decision == "accepted" else step
        row["charged"] = charged
        row["state_sha256_after"] = after_state
        row["optimizer_step_after"] = after_step
        rows.append(row)
        state, step = after_state, after_step
        charged_total += sum(charged.values())
        round_index += 1
        if overrun and round_index == 2:
            break
    if overrun:
        reason = "protocol_failure"
    elif round_index == gate["maximum_rounds"]:
        reason = "maximum_rounds"
    else:
        reason = "reserve_exhausted"
    ledger_name = f"{run_key}-ledger.jsonl"
    ledger_bytes = "".join(
        json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows
    ).encode()
    (directory / ledger_name).write_bytes(ledger_bytes)
    gate_records_name = f"{run_key}-gate-records.jsonl"
    gate_records_bytes = "".join(gate_record_lines).encode()
    (directory / gate_records_name).write_bytes(gate_records_bytes)
    target_initial = 2.0
    protected_initial = {name: 1.5 + 0.1 * i for i, name in enumerate(slices)}
    target_name = target_slices(ROOT, section)[identity["target_setting"]]
    phase_states = {"initial": initial, "final": state}
    audit_endpoint = None
    if not overrun:
        audit_endpoint = {
            "evaluated_state_sha256": state,
            "target_audit_nll": fixture_audit_entry(
                directory,
                run_key,
                target_name,
                phase_states,
                {"initial": target_initial, "final": target_initial + target_delta},
            ),
            "protected_audit_nll": {
                name: fixture_audit_entry(
                    directory,
                    run_key,
                    name,
                    phase_states,
                    {
                        "initial": protected_initial[name],
                        "final": protected_initial[name] + protected_delta - 0.01 * i,
                    },
                )
                for i, name in enumerate(slices)
            },
        }
    manifest = {
        "schema_version": 1,
        "preregistration_sha256": canonical_digest(section),
        "synthetic_audit": True,
        "run_identity": identity,
        "split_manifest_sha256": fake_hash("split-manifest"),
        "initial_state_sha256": initial,
        "reference_cost_accelerator_seconds": reference,
        "budget_accelerator_seconds": budget,
        "train_ledger_path": ledger_name,
        "train_ledger_sha256": hashlib.sha256(ledger_bytes).hexdigest(),
        "cost_calibration_bundle_path": calib_name,
        "cost_calibration_bundle_sha256": calib_sha256,
        "gate_partitions": gate_partitions,
        "gate_loss_records_path": gate_records_name,
        "gate_loss_records_sha256": hashlib.sha256(gate_records_bytes).hexdigest(),
        "terminal_reason": reason,
        "next_round_reserve_accelerator_seconds": 40.0 if reason == "reserve_exhausted" else None,
        "protocol_failure": overrun,
        "totals": {
            "charged_accelerator_seconds": charged_total,
            "accepted_updates": sum(r["decision"] == "accepted" for r in rows),
            "rejected_updates": sum(r["decision"] == "rejected" for r in rows),
            "null_actions": sum(r["decision"] == "null_action" for r in rows),
            "final_state_sha256": state,
            "final_optimizer_step": step,
        },
        "audit_endpoint": audit_endpoint,
    }
    manifest_name = f"{run_key}-manifest.json"
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    (directory / manifest_name).write_bytes(manifest_bytes)
    endpoint = manifest["audit_endpoint"]
    row = {
        **identity,
        "status": "failed" if overrun else "completed",
        "target_delta_nll": None if overrun else (
            endpoint["target_audit_nll"]["final"] - endpoint["target_audit_nll"]["initial"]
        ),
        "protected_delta_nll": None if overrun else max(
            v["final"] - v["initial"] for v in endpoint["protected_audit_nll"].values()
        ),
        "run_manifest_path": manifest_name,
        "run_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "empirical_evidence": False,
    }
    return manifest_name, row["run_manifest_sha256"], row


def rewrite_manifest(directory: Path, row: dict[str, Any], manifest: dict[str, Any]) -> None:
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    (directory / row["run_manifest_path"]).write_bytes(payload)
    row["run_manifest_sha256"] = hashlib.sha256(payload).hexdigest()


def rewrite_ledger(directory: Path, manifest: dict[str, Any], rows: list[dict]) -> None:
    payload = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode()
    (directory / manifest["train_ledger_path"]).write_bytes(payload)
    manifest["train_ledger_sha256"] = hashlib.sha256(payload).hexdigest()


def self_test(preregistration: dict[str, Any]) -> dict[str, Any]:
    section = preregistration["sft"]
    slices_by_setting = protected_slices(ROOT, section)
    identity = {
        "seed": section["seeds"][0],
        "target_setting": "math",
        "budget_fraction": section["cost_design"]["budget_fractions"][0],
        "cell": next(iter(section["primary_factorial"]["cells"])),
        "protected_tolerance": section["primary_factorial"]["protected_tolerance_grid"][1],
    }
    mutations: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="sft-ledger-") as name:
        directory = Path(name)

        def load(row: dict[str, Any]) -> tuple[dict, list[dict]]:
            manifest = json.loads((directory / row["run_manifest_path"]).read_text())
            ledger = load_ledger(
                directory / manifest["train_ledger_path"], manifest["train_ledger_sha256"]
            )
            return manifest, ledger

        closures = {}
        for label, fraction, overrun in (
            ("reserve_exhausted", 0.5, False),
            ("maximum_rounds", 1.0, False),
            ("protocol_failure", 1.0, True),
        ):
            run_identity = {**identity, "budget_fraction": fraction}
            if overrun:
                run_identity["protected_tolerance"] = 0.0
            _, _, row = build_fixture_run(
                directory, section, slices_by_setting, run_identity, -0.2, 0.03, overrun
            )
            manifest, _ = load(row)
            closure = validate_run(manifest, directory, row, section, slices_by_setting)
            require(closure["protocol_failure"] == overrun, f"{label} fixture misclassified")
            require(manifest["terminal_reason"] == label, f"{label} fixture has wrong stop")
            closures[label] = (row, closure)
        check_pairing(list(closures.values()))
        base_row = closures["reserve_exhausted"][0]

        def mutate(label: str, change) -> None:
            _, _, row = build_fixture_run(
                directory, section, slices_by_setting, dict(identity), -0.2, 0.03
            )
            manifest, ledger = load(row)
            change(row, manifest, ledger)
            try:
                validate_run(manifest, directory, row, section, slices_by_setting)
            except LedgerError:
                mutations[label] = "rejected"
                return
            raise LedgerError(f"ledger mutation {label!r} was accepted")

        def relog(manifest: dict, ledger: list[dict]) -> None:
            rewrite_ledger(directory, manifest, ledger)

        mutate("ledger_bytes_changed", lambda r, m, l: (directory / m["train_ledger_path"]).write_text("{}\n"))

        def hidden_charge(r, m, l):
            l[1]["charged"]["solving"] += 5.0
            relog(m, l)

        mutate("uncharged_manifest_total", hidden_charge)

        def extra_step(r, m, l):
            l[2]["optimizer_step_after"] += 1
            relog(m, l)

        mutate("rejected_update_advances_optimizer", extra_step)

        def smaller_scale(r, m, l):
            l[0]["committed_scale"] = section["actual_update_gate"]["scale_grid"][-1]
            relog(m, l)

        mutate("commits_smaller_than_largest_passing_scale", smaller_scale)

        def reused_block(r, m, l):
            first = next(iter(l[0]["gate"]))
            l[1]["gate"][first]["block_id"] = l[0]["gate"][first]["block_id"]
            relog(m, l)

        mutate("reused_gate_block", reused_block)

        def dropped_round(r, m, l):
            del l[2]
            relog(m, l)

        mutate("dropped_rejected_round", dropped_round)

        def stale_endpoint(r, m, l):
            m["audit_endpoint"]["evaluated_state_sha256"] = m["initial_state_sha256"]

        mutate("endpoint_on_non_final_state", stale_endpoint)

        def wrong_delta(r, m, l):
            r["protected_delta_nll"] -= 0.02

        mutate("endpoint_delta_not_reproduced", wrong_delta)

        def hidden_overrun(r, m, l):
            del l[1:]
            l[0]["charged"]["training"] += 30.0
            relog(m, l)

        mutate("reserve_overrun_not_marked_failure", hidden_overrun)

        def early_stop(r, m, l):
            m["next_round_reserve_accelerator_seconds"] = 1.0

        mutate("stop_with_fitting_reserve", early_stop)

        def wrong_identity(r, m, l):
            r["cell"] = list(section["primary_factorial"]["cells"])[1]

        mutate("manifest_identity_mismatch", wrong_identity)

        def tokens_outside(r, m, l):
            l[0]["response_tokens"] = section["optimizer_window"]["response_tokens_maximum"] + 1
            relog(m, l)

        mutate("response_tokens_outside_window", tokens_outside)

        target_name = target_slices(ROOT, section)[identity["target_setting"]]
        protected_name = slices_by_setting[identity["target_setting"]][1]

        def entry_for(m: dict, name: str) -> dict:
            endpoint = m["audit_endpoint"]
            if name == target_name:
                return endpoint["target_audit_nll"]
            return endpoint["protected_audit_nll"][name]

        def edit_records(m: dict, name: str, phase: str, change) -> list[dict]:
            binding = entry_for(m, name)["loss_records"][phase]
            records = load_ledger(directory / binding["path"], binding["sha256"])
            change(records)
            payload = "".join(json.dumps(x, sort_keys=True) + "\n" for x in records).encode()
            (directory / binding["path"]).write_bytes(payload)
            binding["sha256"] = hashlib.sha256(payload).hexdigest()
            return records

        def row_mean(records: list[dict]) -> float:
            return math.fsum(x["nll_sum"] / x["response_tokens"] for x in records) / len(records)

        def tampered_records(r, m, l):
            binding = entry_for(m, target_name)["loss_records"]["final"]
            path = directory / binding["path"]
            path.write_bytes(path.read_bytes().replace(b'"nll_sum": ', b'"nll_sum": 1'))

        mutate("audit_record_bytes_changed", tampered_records)

        def declared_not_reproduced(r, m, l):
            entry_for(m, target_name)["final"] += 0.01
            r["target_delta_nll"] += 0.01

        mutate("declared_audit_nll_not_reproduced", declared_not_reproduced)

        def wrong_state(r, m, l):
            def change(records):
                records[0]["evaluated_state_sha256"] = m["initial_state_sha256"]
            edit_records(m, protected_name, "final", change)

        mutate("audit_record_on_non_final_state", wrong_state)

        def dropped_example(r, m, l):
            entry = entry_for(m, protected_name)
            for phase in ("initial", "final"):
                records = edit_records(m, protected_name, phase, lambda x: x.pop(3))
                entry[phase] = row_mean(records)

        mutate("audit_example_dropped_from_partition", dropped_example)

        def duplicated_example(r, m, l):
            def change(records):
                records[1]["example_id"] = records[0]["example_id"]
            for phase in ("initial", "final"):
                edit_records(m, protected_name, phase, change)

        mutate("duplicate_audit_example", duplicated_example)

        def token_weighted(r, m, l):
            entry = entry_for(m, target_name)
            for phase in ("initial", "final"):
                binding = entry["loss_records"][phase]
                records = load_ledger(directory / binding["path"], binding["sha256"])
                entry[phase] = math.fsum(x["nll_sum"] for x in records) / sum(
                    x["response_tokens"] for x in records
                )
            r["target_delta_nll"] = entry["final"] - entry["initial"]

        mutate("token_weighted_instead_of_row_uniform_mean", token_weighted)

        def edit_gate_records(m: dict, change) -> list[dict]:
            path = directory / m["gate_loss_records_path"]
            records = load_ledger(path, m["gate_loss_records_sha256"], "gate-loss records")
            change(records)
            payload = "".join(
                json.dumps(x, sort_keys=True, allow_nan=False) + "\n" for x in records
            ).encode()
            path.write_bytes(payload)
            m["gate_loss_records_sha256"] = hashlib.sha256(payload).hexdigest()
            return records

        def tampered_gate_records(r, m, l):
            path = directory / m["gate_loss_records_path"]
            path.write_bytes(path.read_bytes().replace(b'"round": 0', b'"round": 9'))

        mutate("gate_record_bytes_changed", tampered_gate_records)

        def gate_mean_not_reproduced(r, m, l):
            block = l[0]["gate"][protected_name]
            block["mean_change"][1] -= 0.002
            block["upper"][1] -= 0.002
            relog(m, l)

        mutate("gate_mean_not_reproduced_from_draws", gate_mean_not_reproduced)

        def post_difference_clipped_gate(r, m, l):
            records = load_ledger(
                directory / m["gate_loss_records_path"],
                m["gate_loss_records_sha256"],
                "gate-loss records",
            )
            record = next(
                x for x in records if x["round"] == 0 and x["slice"] == protected_name
            )
            tokens = record["response_tokens"]
            base = record["base_nll_sum"]
            margin = section["actual_update_gate"]["empirical_margin"]
            unclipped_means = [
                math.fsum((c - b) / t for c, b, t in zip(cand, base, tokens)) / len(tokens)
                for cand in record["candidate_nll_sum_by_scale"]
            ]
            l[0]["gate"][protected_name]["mean_change"] = unclipped_means
            l[0]["gate"][protected_name]["upper"] = [v + margin for v in unclipped_means]
            relog(m, l)

        mutate(
            "gate_clipped_after_differencing_instead_of_per_example",
            post_difference_clipped_gate,
        )

        def token_weighted_gate(r, m, l):
            records = load_ledger(
                directory / m["gate_loss_records_path"],
                m["gate_loss_records_sha256"],
                "gate-loss records",
            )
            record = next(
                x for x in records if x["round"] == 0 and x["slice"] == protected_name
            )
            tokens = record["response_tokens"]
            total_tokens = sum(tokens)
            bound = float(section["actual_update_gate"]["clipped_example_nll_bound"])
            margin = section["actual_update_gate"]["empirical_margin"]
            base_c = [min(b / t, bound) for b, t in zip(record["base_nll_sum"], tokens)]
            weighted_means = [
                math.fsum(
                    (min(c / t, bound) - b_c) * t
                    for c, b_c, t in zip(cand, base_c, tokens)
                )
                / total_tokens
                for cand in record["candidate_nll_sum_by_scale"]
            ]
            l[0]["gate"][protected_name]["mean_change"] = weighted_means
            l[0]["gate"][protected_name]["upper"] = [v + margin for v in weighted_means]
            relog(m, l)

        mutate("gate_token_weighted_instead_of_draw_uniform_mean", token_weighted_gate)

        def foreign_gate_draw(r, m, l):
            def change(records):
                records[0]["example_ids"][0] = f"{records[0]['slice']}-foreign-0"

            edit_gate_records(m, change)

        mutate("gate_example_outside_partition", foreign_gate_draw)

        def overlapping_gate_audit(r, m, l):
            overlap_id = fixture_audit_ids(protected_name)[0]
            ids = sorted({*m["gate_partitions"][protected_name]["example_ids"], overlap_id})
            digest = canonical_digest(ids)
            m["gate_partitions"][protected_name] = {
                "gate_partition_sha256": digest,
                "example_ids": ids,
            }

            def change(records):
                for rec in records:
                    if rec["slice"] == protected_name:
                        rec["gate_partition_sha256"] = digest

            edit_gate_records(m, change)

        mutate("gate_and_audit_partitions_overlap", overlapping_gate_audit)

        def wrong_committed_candidate_state(r, m, l):
            def change(records):
                for rec in records:
                    if rec["round"] == 0:
                        rec["candidate_state_sha256_by_scale"][1] = fake_hash(
                            "wrong-committed-candidate-state"
                        )

            edit_gate_records(m, change)

        mutate(
            "committed_state_differs_from_accepted_scale_candidate",
            wrong_committed_candidate_state,
        )

        def edit_calibration_bundle(m: dict, change) -> dict:
            path = directory / m["cost_calibration_bundle_path"]
            bundle = load_json_verified(
                path, m["cost_calibration_bundle_sha256"], "cost-calibration bundle"
            )
            change(bundle)
            payload = (json.dumps(bundle, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
            path.write_bytes(payload)
            m["cost_calibration_bundle_sha256"] = hashlib.sha256(payload).hexdigest()
            return bundle

        mutate(
            "cost_calibration_plan_count_wrong",
            lambda r, m, l: edit_calibration_bundle(
                m, lambda b: b["calibration_plans"].pop()
            ),
        )
        mutate(
            "cost_calibration_margin_quantile_wrong",
            lambda r, m, l: edit_calibration_bundle(
                m, lambda b: b.__setitem__("reserve_margin_seconds", 4.0)
            ),
        )
        mutate(
            "cost_calibration_overlaps_gate_or_audit",
            lambda r, m, l: edit_calibration_bundle(
                m,
                lambda b: b["calibration_plans"][0]["example_ids"].__setitem__(
                    0, fixture_gate_ids(protected_name)[0]
                ),
            ),
        )

        def pack_load_exceeds_max(r, m, l):
            l[0]["selected_plan"]["packed_lengths"][0] = 2049
            relog(m, l)

        mutate("pack_load_exceeds_maximum_sequence_length", pack_load_exceeds_max)

        def non_best_fit_in_best_fit_cell(r, m, l):
            l[0]["selected_plan"]["pack_assignments"] = [k % 8 for k in range(16)]
            l[0]["reserve_accelerator_seconds"] = 39.84
            l[0]["shortlist"]["candidates"][0]["pack_loads"] = [3584] * 8
            l[0]["shortlist"]["candidates"][0]["reserve_accelerator_seconds"] = 39.84
            relog(m, l)

        mutate(
            "deterministic_best_fit_cell_uses_non_best_fit_packing",
            non_best_fit_in_best_fit_cell,
        )

        # Positive control: a fixed_budget_beam cell may use the balanced 3584-load
        # packing when its reserve matches that packing's calibrated cost (39.84s).
        _, _, beam_row = build_fixture_run(
            directory,
            section,
            slices_by_setting,
            {**identity, "cell": "measured_nonlinear__fixed_budget_beam"},
            -0.2,
            0.03,
        )
        beam_manifest, beam_ledger = load(beam_row)
        beam_ledger[0]["selected_plan"]["pack_assignments"] = [k % 8 for k in range(16)]
        beam_ledger[0]["reserve_accelerator_seconds"] = 39.84
        beam_ledger[0]["shortlist"]["candidates"][0]["pack_loads"] = [3584] * 8
        beam_ledger[0]["shortlist"]["candidates"][0]["reserve_accelerator_seconds"] = 39.84
        rewrite_ledger(directory, beam_manifest, beam_ledger)
        validate_run(beam_manifest, directory, beam_row, section, slices_by_setting)

        def reserve_disagrees_with_calibrated_cost(r, m, l):
            l[0]["reserve_accelerator_seconds"] -= 0.1
            relog(m, l)

        mutate(
            "round_reserve_disagrees_with_calibrated_pack_cost",
            reserve_disagrees_with_calibrated_cost,
        )

        def candidate_overlaps_gate_or_audit(r, m, l):
            l[0]["selected_plan"]["example_ids"][0] = fixture_audit_ids(target_name)[0]
            relog(m, l)

        mutate("candidate_plan_overlaps_gate_or_audit", candidate_overlaps_gate_or_audit)

        def source_coverage_floor_violated(r, m, l):
            l[0]["selected_plan"]["source_slices"] = [target_name] * 16
            relog(m, l)

        mutate(
            "source_coverage_floor_violated",
            source_coverage_floor_violated,
        )

        def shortlist_size_wrong(r, m, l):
            l[0]["shortlist"]["candidates"].pop()
            relog(m, l)

        mutate(
            "shortlist_size_wrong",
            shortlist_size_wrong,
        )

        def shortlist_hat_s_mismatch(r, m, l):
            l[0]["shortlist"]["candidates"][0]["target_score"]["hat_s"] += 0.01
            l[0]["shortlist"]["candidates"][0]["target_score"]["lower_s"] += 0.01
            relog(m, l)

        mutate(
            "shortlist_hat_s_disagrees_with_sketch_equation",
            shortlist_hat_s_mismatch,
        )

        def shortlist_protected_constraint_violated(r, m, l):
            l[0]["shortlist"]["candidates"][13]["feasible"] = True
            relog(m, l)

        mutate(
            "shortlist_protected_constraint_violated",
            shortlist_protected_constraint_violated,
        )

        def shortlist_trust_radius_violated(r, m, l):
            l[0]["shortlist"]["candidates"][15]["feasible"] = True
            relog(m, l)

        mutate(
            "shortlist_trust_radius_violated",
            shortlist_trust_radius_violated,
        )

        def shortlist_non_argmax_selected(r, m, l):
            l[0]["selected_plan"]["plan_id"] = "plan-00-01"
            relog(m, l)

        mutate(
            "shortlist_suboptimal_feasible_plan_selected",
            shortlist_non_argmax_selected,
        )

        def shortlist_u_dot_v_varies(r, m, l):
            cand = l[0]["shortlist"]["candidates"][1]["target_score"]
            cand["u_dot_v"] -= 0.005
            cand["hat_s"] += 0.005
            cand["lower_s"] += 0.005
            relog(m, l)

        mutate(
            "shortlist_u_dot_v_varies_across_candidates",
            shortlist_u_dot_v_varies,
        )

        def selected_u_dot_q_example_weighted(r, m, l):
            plan = l[0]["selected_plan"]
            resp = plan["response_tokens_by_example"]
            per_ex_rates = [
                val / tok
                for val, tok in zip(plan["u_dot_q_by_slice"][target_name], resp)
            ]
            wrong_u_dot_q = math.fsum(per_ex_rates) / len(per_ex_rates)
            cand0 = l[0]["shortlist"]["candidates"][0]["target_score"]
            eta_val = l[0]["shortlist"]["learning_rate"]
            cand0["u_dot_q"] = wrong_u_dot_q
            cand0["hat_s"] = -cand0["u_dot_v"] + eta_val * wrong_u_dot_q
            cand0["lower_s"] = cand0["hat_s"] - cand0["margin"]
            relog(m, l)

        mutate(
            "selected_plan_u_dot_q_example_weighted_instead_of_global_nb",
            selected_u_dot_q_example_weighted,
        )

        def selected_v_dot_q_mismatch(r, m, l):
            l[0]["selected_plan"]["v_dot_q_by_example"][0] += 10.0
            relog(m, l)

        mutate(
            "selected_plan_v_dot_q_mismatch",
            selected_v_dot_q_mismatch,
        )

        def shortlist_step_norm_quadratic_mismatch(r, m, l):
            cand = l[0]["shortlist"]["candidates"][0]
            cand["step_norm_sq_hat"] += 0.005
            cand["step_norm_sq_upper"] += 0.005
            relog(m, l)

        mutate(
            "shortlist_step_norm_sq_hat_disagrees_with_sketch_quadratic",
            shortlist_step_norm_quadratic_mismatch,
        )

        def shortlist_cauchy_schwarz_violated(r, m, l):
            cand = l[0]["shortlist"]["candidates"][1]
            eta_val = l[0]["shortlist"]["learning_rate"]
            v_norm_sq = l[0]["shortlist"]["v_norm_sq"]
            cand["v_dot_q"] = -100.0
            cand["q_norm_sq"] = 100.0
            cand["step_norm_sq_hat"] = (
                v_norm_sq - 2.0 * eta_val * cand["v_dot_q"] + (eta_val**2) * cand["q_norm_sq"]
            )
            cand["step_norm_sq_upper"] = cand["step_norm_sq_hat"] + cand["step_norm_sq_margin"]
            relog(m, l)

        mutate(
            "shortlist_step_norm_cauchy_schwarz_violated",
            shortlist_cauchy_schwarz_violated,
        )

        _, _, base_row = build_fixture_run(
            directory, section, slices_by_setting, dict(identity), -0.2, 0.03
        )
        _, _, failed_row = build_fixture_run(
            directory,
            section,
            slices_by_setting,
            {**identity, "protected_tolerance": 0.0},
            -0.2,
            0.03,
            overrun=True,
        )
        failed_manifest, _ = load(failed_row)
        failed_manifest["audit_endpoint"] = load(base_row)[0]["audit_endpoint"]
        try:
            validate_run(failed_manifest, directory, failed_row, section, slices_by_setting)
        except LedgerError:
            mutations["protocol_failure_opens_sealed_audit"] = "rejected"
        else:
            raise LedgerError("a protocol-failure run opened its sealed audit")

        pairing_failure = dict(closures["maximum_rounds"][1])
        pairing_failure["initial_state_sha256"] = fake_hash("other-initial")
        try:
            check_pairing([closures["reserve_exhausted"], (base_row, pairing_failure)])
        except LedgerError:
            mutations["unpaired_initial_state"] = "rejected"
        else:
            raise LedgerError("unpaired initial state was accepted")
        unpaired_calib = dict(closures["maximum_rounds"][1])
        unpaired_calib["cost_calibration_bundle_sha256"] = fake_hash("other-calib")
        try:
            check_pairing([closures["reserve_exhausted"], (base_row, unpaired_calib)])
        except LedgerError:
            mutations["unpaired_cost_calibration_bundle"] = "rejected"
        else:
            raise LedgerError("unpaired cost_calibration_bundle_sha256 was accepted")
        for label, field, value in (
            ("unpaired_gate_partition", "gate_partitions", fake_hash("other-gate-partition")),
            ("unpaired_audit_partition", "audit_partitions", fake_hash("other-partition")),
            ("unpaired_initial_audit_nll", "initial_audit_nll", 9.0),
        ):
            altered = dict(closures["maximum_rounds"][1])
            altered[field] = {key: value for key in altered[field]}
            try:
                check_pairing([closures["reserve_exhausted"], (base_row, altered)])
            except LedgerError:
                mutations[label] = "rejected"
            else:
                raise LedgerError(f"{label} was accepted")
    return {
        "schema_version": 1,
        "audit_type": "sft_train_ledger_endpoint_closure",
        "empirical_evidence": False,
        "fixtures": {
            label: {
                "rounds": closure["rounds"],
                "accepted_updates": closure["accepted_updates"],
                "rejected_updates": closure["rejected_updates"],
                "null_actions": closure["null_actions"],
                "charged_accelerator_seconds": closure["charged_accelerator_seconds"],
                "protocol_failure": closure["protocol_failure"],
            }
            for label, (_, closure) in closures.items()
        },
        "mutations_rejected": len(mutations),
        "mutations": mutations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.self_test:
        parser.error("use --self-test; real closure runs inside aggregate_sft_factorial.py")
    preregistration = json.loads(args.preregistration.read_text(encoding="utf-8"))
    try:
        summary = self_test(preregistration)
    except LedgerError as exc:
        raise SystemExit(f"error: {exc}") from exc
    if args.output is not None:
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        "validated SFT train-ledger-to-endpoint closure; "
        f"{summary['mutations_rejected']} mutations rejected"
    )


if __name__ == "__main__":
    main()
