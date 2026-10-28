#!/usr/bin/env python3
"""Reconstruct RL estimator terms from raw trajectory fields.

The estimator aggregator consumes projected per-trajectory terms.  This
validator rebuilds every such term from hash-bound raw fields instead of
trusting the producer:

* prompt masses come only from the frozen header laws, so
  ``log P = log d(x) + log pi(y|x)`` and ``log G = log q(x) + log mu(y|x)``;
* ``w = P / (alpha P + (1 - alpha) G)`` is recomputed in log space;
* ``U = R - lambda C`` and ``h = h^R - lambda h^C`` use the frozen lambda;
* current rows give ``A=(h-b)s+alpha*w*(U-h)s``, stale rows ``B=w*(U-h)s``,
  and reference rows ``U*s``;
* every attempted row, including infrastructure-failed/aborted attempts,
  carries raw ``response_tokens``, ``verifier_called``, and
  ``added_score_passes``, from which per-attempt and stage-level
  ``K_gen``, ``K_ver``, ``K_score``, and ``allocated_accelerator_seconds``
  counters are reconstructed so no cost disappears when an attempt fails.

Raw rows must also carry the frozen decoder temperature for both scored
policies, the frozen current or behavior tensor hash as generator, zero prior
uses, the frozen nuisance model and freeze record, and one prompt-only baseline
per prompt.  Nuisance training prompts must be disjoint from every audited
prompt.  A declared projected ledger passes only if it matches the
reconstruction row for row.

The self-test builds a golden raw ledger on the exact four-cell Gate 0 law and
checks unbiasedness by enumeration.  It then replays five faults through the
reconstruction: omitted prompt ratio, weight clipping, base-versus-decoder
scoring, focal nuisance fitting, and consumed-trajectory reuse.  Each fault
must be rejected and its null control must pass.  All outputs are software
checks with empirical_evidence=false.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import rl_gate0_fault_injections as gate0

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREREGISTRATION = ROOT / "experiments" / "pilot_preregistration.json"
MASS_TOLERANCE = 1e-12
TERM_RELATIVE_TOLERANCE = 1e-9
IDENTITY_TOLERANCE = gate0.IDENTITY_TOLERANCE
DETECTION_THRESHOLD = gate0.DETECTION_THRESHOLD

COMMON_FIELDS = (
    "attempt_id",
    "target_law",
    "estimator_role",
    "source_component",
    "infrastructure_failure",
    "allocated_accelerator_seconds",
    "response_tokens",
    "verifier_called",
    "added_score_passes",
)
CANDIDATE_FIELDS = ("alpha", "allocation_id", "replication_id")
TRAJECTORY_FIELDS = (
    "trajectory_id",
    "prompt_id",
    "generating_policy_sha256",
    "scoring",
    "current_sequence_log_prob",
    "behavior_sequence_log_prob",
    "reward",
    "termination",
    "nuisance",
    "prior_use_count",
    "freeze_record_sha256",
    "projected_score",
)
NUISANCE_FIELDS = ("model_sha256", "reward_prediction", "length_prediction", "prompt_baseline")
HEADER_FROZEN_FIELDS = (
    "current_policy_sha256",
    "behavior_policy_sha256",
    "decoder_temperature",
    "maximum_response_tokens",
    "lambda",
    "support_floor_epsilon",
    "nuisance_training_partition",
    "nuisance_model_sha256",
    "prompt_baseline_sha256",
    "freeze_record_sha256",
    "projection_dimension",
)


class ReconstructionError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReconstructionError(message)


def canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def fake_hash(*parts: Any) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def frozen_constants(section: dict[str, Any]) -> dict[str, Any]:
    """Return the preregistered constants a real raw ledger must reproduce."""
    pair = section["stale_actor"]["frozen_policy_pair"]
    require(
        is_sha256(pair.get("current_tensor_sha256"))
        and is_sha256(pair.get("behavior_tensor_sha256")),
        "raw reconstruction is blocked until both policy tensor hashes are frozen",
    )
    audit = section["estimator_audit"]
    return {
        "current_policy_sha256": pair["current_tensor_sha256"],
        "behavior_policy_sha256": pair["behavior_tensor_sha256"],
        "decoder_temperature": section["decoder"]["temperature"],
        "maximum_response_tokens": section["decoder"]["maximum_response_tokens"],
        "lambda": audit["audit_lambda"],
        "support_floor_epsilon": section["proposal"]["support_floor_epsilon"],
        "nuisance_training_partition": section["nuisance_models"]["training_partition"],
        "projection_dimension": audit["projection_dimension"],
        "cells": sorted(
            (row["alpha"], row["allocation_id"]) for row in audit["stratum_counts_by_alpha"]
        ),
    }


def validate_header(header: Any, frozen: dict[str, Any]) -> dict[str, Any]:
    require(isinstance(header, dict), "raw trajectory header must be an object")
    require(header.get("schema_version") == 1, "unsupported raw trajectory header schema")
    declared = header.get("frozen")
    require(
        isinstance(declared, dict) and set(declared) == set(HEADER_FROZEN_FIELDS),
        "raw trajectory header must declare exactly the frozen fields",
    )
    for key in (
        "current_policy_sha256",
        "behavior_policy_sha256",
        "decoder_temperature",
        "maximum_response_tokens",
        "lambda",
        "support_floor_epsilon",
        "nuisance_training_partition",
        "projection_dimension",
    ):
        require(declared[key] == frozen[key], f"raw trajectory header {key} differs from the frozen value")
    require(
        declared["current_policy_sha256"] != declared["behavior_policy_sha256"],
        "current and behavior policy tensors must differ",
    )
    for key in ("nuisance_model_sha256", "prompt_baseline_sha256", "freeze_record_sha256"):
        require(is_sha256(declared[key]), f"raw trajectory header needs {key}")
    reward_range = header.get("reward_range")
    require(
        isinstance(reward_range, list)
        and len(reward_range) == 2
        and all(is_number(value) for value in reward_range)
        and reward_range[0] < reward_range[1],
        "raw trajectory header needs a bounded reward range",
    )
    laws = header.get("prompt_laws")
    require(
        isinstance(laws, dict) and set(laws) == {"select", "confirm"},
        "raw trajectory header must define the select and confirm prompt laws",
    )
    epsilon = declared["support_floor_epsilon"]
    prompt_sets: dict[str, set[str]] = {}
    for law, masses in laws.items():
        require(isinstance(masses, dict) and masses, f"{law} prompt law is empty")
        target_total = []
        proposal_total = []
        for prompt_id, mass in masses.items():
            require(
                isinstance(mass, dict) and set(mass) == {"target_mass", "proposal_mass"},
                f"{law} prompt {prompt_id} needs target and proposal masses",
            )
            d, q = mass["target_mass"], mass["proposal_mass"]
            require(
                is_number(d) and is_number(q) and d > 0 and q > 0,
                f"{law} prompt {prompt_id} needs positive finite masses",
            )
            require(
                q >= epsilon * d * (1.0 - MASS_TOLERANCE),
                f"{law} prompt {prompt_id} violates the proposal support floor",
            )
            target_total.append(d)
            proposal_total.append(q)
        for name, values in (("target", target_total), ("proposal", proposal_total)):
            require(
                abs(math.fsum(values) - 1.0) <= MASS_TOLERANCE,
                f"{law} {name} prompt masses do not sum to one",
            )
        prompt_sets[law] = set(masses)
    require(
        not prompt_sets["select"] & prompt_sets["confirm"],
        "select and confirm prompt laws overlap",
    )
    training = header.get("nuisance_training_prompt_ids")
    require(
        isinstance(training, list)
        and training
        and all(isinstance(value, str) and value for value in training)
        and len(set(training)) == len(training),
        "raw trajectory header needs unique nuisance training prompt identifiers",
    )
    require(
        not set(training) & (prompt_sets["select"] | prompt_sets["confirm"]),
        "nuisance models were fitted on audited prompts (focal or partition leak)",
    )
    return {"declared": declared, "reward_range": reward_range, "laws": laws}


def finite_vector(value: Any, dimension: int, label: str) -> list[float]:
    require(
        isinstance(value, list)
        and len(value) == dimension
        and all(is_number(component) for component in value),
        f"{label} must be a finite projected vector of dimension {dimension}",
    )
    return [float(component) for component in value]


def mixture_weight(log_p: float, log_g: float, alpha: float) -> float:
    if alpha == 1:
        return 1.0
    return 1.0 / (alpha + (1.0 - alpha) * math.exp(log_g - log_p))


def empty_cost_bucket() -> dict[str, Any]:
    return {
        "attempts": 0,
        "aborted_attempts": 0,
        "k_gen": 0,
        "k_ver": 0,
        "k_score": 0,
        "accelerator_seconds": 0.0,
    }


def add_cost_bucket(
    bucket: dict[str, Any],
    *,
    failed: bool,
    k_gen: int,
    k_ver: int,
    k_score: int,
    acc_sec: float,
) -> None:
    bucket["attempts"] += 1
    if failed:
        bucket["aborted_attempts"] += 1
    bucket["k_gen"] += k_gen
    bucket["k_ver"] += k_ver
    bucket["k_score"] += k_score
    bucket["accelerator_seconds"] = math.fsum([bucket["accelerator_seconds"], acc_sec])


def reconstruct_cost_ledger(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Reconstruct K_gen, K_ver, K_score, and accelerator-seconds across all attempts."""
    candidate = empty_cost_bucket()
    reference = empty_cost_bucket()
    aborted = empty_cost_bucket()
    total = empty_cost_bucket()
    by_cell: dict[str, dict[str, Any]] = {}
    for row in sorted(rows, key=lambda item: item["attempt_id"]):
        failed = bool(row["infrastructure_failure"])
        k_gen = int(row["generated_tokens"])
        k_ver = int(row["verifier_calls"])
        k_score = int(row["added_score_tokens"])
        acc = float(row["allocated_accelerator_seconds"])
        add_cost_bucket(
            total, failed=failed, k_gen=k_gen, k_ver=k_ver, k_score=k_score, acc_sec=acc
        )
        if failed:
            add_cost_bucket(
                aborted, failed=True, k_gen=k_gen, k_ver=k_ver, k_score=k_score, acc_sec=acc
            )
        if row["estimator_role"] == "candidate":
            add_cost_bucket(
                candidate, failed=failed, k_gen=k_gen, k_ver=k_ver, k_score=k_score, acc_sec=acc
            )
            cell_key = f"{row['target_law']}|{float(row['alpha'])}|{row['allocation_id']}"
            cell_bucket = by_cell.setdefault(cell_key, empty_cost_bucket())
            add_cost_bucket(
                cell_bucket, failed=failed, k_gen=k_gen, k_ver=k_ver, k_score=k_score, acc_sec=acc
            )
        else:
            add_cost_bucket(
                reference, failed=failed, k_gen=k_gen, k_ver=k_ver, k_score=k_score, acc_sec=acc
            )
    return {
        "candidate": candidate,
        "reference": reference,
        "aborted": aborted,
        "total": total,
        "by_cell": dict(sorted(by_cell.items())),
    }


def compare_cost_bucket(observed: Any, expected: dict[str, Any], label: str) -> None:
    require(
        isinstance(observed, dict) and set(observed) == set(expected),
        f"{label} cost bucket has unexpected keys",
    )
    for key in ("attempts", "aborted_attempts", "k_gen", "k_ver", "k_score"):
        require(
            isinstance(observed[key], int)
            and not isinstance(observed[key], bool)
            and observed[key] == expected[key],
            f"{label} {key} does not reproduce from raw attempts",
        )
    require(
        is_number(observed["accelerator_seconds"])
        and abs(float(observed["accelerator_seconds"]) - expected["accelerator_seconds"])
        <= MASS_TOLERANCE * max(1.0, expected["accelerator_seconds"]),
        f"{label} accelerator_seconds does not reproduce from raw attempts",
    )


def compare_cost_ledger(observed: Any, expected: dict[str, Any]) -> None:
    require(
        isinstance(observed, dict)
        and set(observed) == {"candidate", "reference", "aborted", "total", "by_cell"},
        "raw trajectory header must declare a complete cost_ledger",
    )
    for section_name in ("candidate", "reference", "aborted", "total"):
        compare_cost_bucket(
            observed[section_name], expected[section_name], f"cost_ledger.{section_name}"
        )
    by_cell = observed["by_cell"]
    require(
        isinstance(by_cell, dict) and set(by_cell) == set(expected["by_cell"]),
        "cost_ledger.by_cell keys do not reproduce",
    )
    for cell_key, expected_bucket in expected["by_cell"].items():
        compare_cost_bucket(by_cell[cell_key], expected_bucket, f"cost_ledger.by_cell[{cell_key}]")


def reconstruct(
    header: Any, raw_rows: list[dict[str, Any]], frozen: dict[str, Any]
) -> dict[str, Any]:
    checked = validate_header(header, frozen)
    declared = checked["declared"]
    laws = checked["laws"]
    low, high = checked["reward_range"]
    dimension = declared["projection_dimension"]
    lam = declared["lambda"]
    maximum = declared["maximum_response_tokens"]
    cells = {(float(alpha), allocation) for alpha, allocation in frozen["cells"]}
    attempt_ids: set[str] = set()
    trajectory_ids: set[str] = set()
    baselines: dict[tuple[str, str], float] = {}
    rows: list[dict[str, Any]] = []
    max_alpha_weight = 0.0
    for index, raw in enumerate(raw_rows):
        label = f"raw row {index}"
        require(isinstance(raw, dict), f"{label} must be an object")
        attempt_id = raw.get("attempt_id")
        law = raw.get("target_law")
        role = raw.get("estimator_role")
        source = raw.get("source_component")
        failed = raw.get("infrastructure_failure")
        cost = raw.get("allocated_accelerator_seconds")
        require(
            isinstance(attempt_id, str)
            and attempt_id
            and attempt_id not in attempt_ids
            and law in laws
            and role in {"candidate", "r1", "r2"}
            and source in {"current", "stale"}
            and isinstance(failed, bool)
            and is_number(cost)
            and cost >= 0,
            f"{label} has invalid common fields",
        )
        attempt_ids.add(attempt_id)
        expected_keys = set(COMMON_FIELDS)
        if role == "candidate":
            expected_keys |= set(CANDIDATE_FIELDS)
            alpha = raw.get("alpha")
            require(
                is_number(alpha)
                and (float(alpha), raw.get("allocation_id")) in cells
                and isinstance(raw.get("replication_id"), int)
                and not isinstance(raw.get("replication_id"), bool)
                and raw["replication_id"] >= 0,
                f"{label} has an invalid allocation or replication",
            )
            require(
                not (alpha == 1 and source == "stale"),
                f"{label} places a stale trajectory in the alpha=1 cell",
            )
            expected_added_passes = 0 if float(alpha) == 1.0 else 1
        else:
            require(source == "current", f"{label} reference rows must be current-policy draws")
            expected_added_passes = 0

        tokens = raw.get("response_tokens")
        verifier_called = raw.get("verifier_called")
        added_score_passes = raw.get("added_score_passes")
        require(
            isinstance(tokens, int)
            and not isinstance(tokens, bool)
            and isinstance(verifier_called, bool)
            and isinstance(added_score_passes, int)
            and not isinstance(added_score_passes, bool),
            f"{label} needs integer response_tokens/added_score_passes and boolean verifier_called",
        )
        row = {
            "attempt_id": attempt_id,
            "target_law": law,
            "estimator_role": role,
            "source_component": source,
            "infrastructure_failure": failed,
            "allocated_accelerator_seconds": float(cost),
            "generated_tokens": tokens,
            "verifier_calls": int(verifier_called),
            "added_score_tokens": tokens * added_score_passes,
        }
        if role == "candidate":
            row.update({key: raw[key] for key in CANDIDATE_FIELDS})
        if failed:
            require(set(raw) == expected_keys, f"{label} failed attempt carries trajectory fields")
            require(
                0 <= tokens <= maximum,
                f"{label} failed attempt response-token count lies outside [0, maximum]",
            )
            require(
                0 <= added_score_passes <= expected_added_passes,
                f"{label} failed attempt has invalid added_score_passes",
            )
            if tokens == 0:
                require(
                    not verifier_called and added_score_passes == 0,
                    f"{label} zero-token failed attempt cannot call verifier or score tokens",
                )
            rows.append(row)
            continue
        require(
            set(raw) == expected_keys | set(TRAJECTORY_FIELDS),
            f"{label} has an unexpected raw trajectory schema",
        )
        require(
            1 <= tokens <= maximum,
            f"{label} response-token count lies outside [1, maximum]",
        )
        require(
            verifier_called is True,
            f"{label} successful trajectory must record verifier_called=True",
        )
        require(
            added_score_passes == expected_added_passes,
            f"{label} added_score_passes does not match its cell or role",
        )
        trajectory_id = raw["trajectory_id"]
        require(
            isinstance(trajectory_id, str) and trajectory_id and trajectory_id not in trajectory_ids,
            f"{label} repeats or omits its trajectory identifier",
        )
        trajectory_ids.add(trajectory_id)
        require(
            raw["prior_use_count"] == 0,
            f"{label} reuses a consumed trajectory",
        )
        require(
            raw["freeze_record_sha256"] == declared["freeze_record_sha256"],
            f"{label} was not generated under the frozen iteration record",
        )
        generator = (
            declared["current_policy_sha256"]
            if source == "current"
            else declared["behavior_policy_sha256"]
        )
        require(
            raw["generating_policy_sha256"] == generator,
            f"{label} was generated by a policy other than its frozen source",
        )
        require(
            raw["scoring"]
            == {
                "current_policy_sha256": declared["current_policy_sha256"],
                "behavior_policy_sha256": declared["behavior_policy_sha256"],
                "temperature": declared["decoder_temperature"],
            },
            f"{label} was not scored by both frozen policies under the rollout decoder",
        )
        prompt_id = raw["prompt_id"]
        require(prompt_id in laws[law], f"{label} prompt lies outside its target law")
        log_current = raw["current_sequence_log_prob"]
        log_behavior = raw["behavior_sequence_log_prob"]
        require(
            is_number(log_current)
            and is_number(log_behavior)
            and log_current <= 0
            and log_behavior <= 0,
            f"{label} needs finite nonpositive sequence log-probabilities",
        )
        reward = raw["reward"]
        require(is_number(reward) and low <= reward <= high, f"{label} reward lies outside its range")
        require(
            raw["termination"] == "eos"
            or (raw["termination"] == "truncated" and tokens == maximum),
            f"{label} termination does not match its token count",
        )
        nuisance = raw["nuisance"]
        require(
            isinstance(nuisance, dict)
            and set(nuisance) == set(NUISANCE_FIELDS)
            and nuisance["model_sha256"] == declared["nuisance_model_sha256"]
            and all(is_number(nuisance[key]) for key in NUISANCE_FIELDS[1:]),
            f"{label} nuisance predictions do not come from the frozen model",
        )
        baseline = float(nuisance["prompt_baseline"])
        known = baselines.setdefault((law, prompt_id), baseline)
        require(known == baseline, f"{label} prompt baseline depends on more than the prompt")
        score = finite_vector(raw["projected_score"], dimension, f"{label} projected score")
        utility = reward - lam * tokens
        h = nuisance["reward_prediction"] - lam * nuisance["length_prediction"]
        if role in {"r1", "r2"}:
            term = [utility * value for value in score]
            semantics = "target=U*s"
        else:
            alpha = float(raw["alpha"])
            masses = laws[law][prompt_id]
            log_p = math.log(masses["target_mass"]) + log_current
            log_g = math.log(masses["proposal_mass"]) + log_behavior
            weight = mixture_weight(log_p, log_g, alpha)
            max_alpha_weight = max(max_alpha_weight, alpha * weight)
            if source == "current":
                coefficient = (h - baseline) + alpha * weight * (utility - h)
                semantics = "A=(h-b)s+alpha*w*(U-h)s"
            else:
                coefficient = weight * (utility - h)
                semantics = "B=w*(U-h)s"
            term = [coefficient * value for value in score]
        row.update(
            {
                "trajectory_id": trajectory_id,
                "use_count": 1,
                "scored_outcome": True,
                "projected_gradient": term,
                "projected_term_semantics": semantics,
            }
        )
        rows.append(row)
    require(max_alpha_weight <= 1.0 + MASS_TOLERANCE, "a mixture weight exceeds 1/alpha")
    cost_ledger = reconstruct_cost_ledger(rows)
    compare_cost_ledger(header.get("cost_ledger"), cost_ledger)
    return {
        "rows": rows,
        "attempts": len(rows),
        "successful_trajectories": len(trajectory_ids),
        "maximum_alpha_times_weight": max_alpha_weight,
        "cost_ledger": cost_ledger,
    }


def compare_ledgers(
    reconstructed: list[dict[str, Any]], declared: list[dict[str, Any]]
) -> None:
    """Require a declared projected ledger to equal the reconstruction."""
    by_attempt = {row["attempt_id"]: row for row in reconstructed}
    seen: set[str] = set()
    for index, row in enumerate(declared):
        attempt_id = row.get("attempt_id") if isinstance(row, dict) else None
        require(
            attempt_id in by_attempt and attempt_id not in seen,
            f"declared row {index} has no unique raw counterpart",
        )
        seen.add(attempt_id)
        expected = by_attempt[attempt_id]
        require(
            set(row) == set(expected),
            f"declared row {attempt_id} has different fields from its reconstruction",
        )
        for key, value in expected.items():
            if key == "projected_gradient":
                observed = row[key]
                require(
                    isinstance(observed, list)
                    and len(observed) == len(value)
                    and all(
                        is_number(a)
                        and abs(a - b) <= TERM_RELATIVE_TOLERANCE * max(1.0, abs(a), abs(b))
                        for a, b in zip(observed, value)
                    ),
                    f"declared projected term for {attempt_id} does not reproduce",
                )
            else:
                require(row[key] == value, f"declared {key} for {attempt_id} does not reproduce")
    require(seen == set(by_attempt), "declared ledger omits raw attempts")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file() and not path.is_symlink(), f"raw ledger not found: {path}")
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReconstructionError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        require(isinstance(value, dict), f"{path}:{line_number}: expected an object")
        rows.append(value)
    return rows


def verify_work_dir(
    work_dir: Path, section: dict[str, Any], declared_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Runner entry point for the real train stage."""
    frozen = frozen_constants(section)
    header_path = work_dir / "train" / "raw_trajectory_header.json"
    ledger_path = work_dir / "train" / "raw_trajectory_ledger.jsonl"
    require(
        header_path.is_file() and not header_path.is_symlink(),
        "real RL train stage needs train/raw_trajectory_header.json",
    )
    header = json.loads(header_path.read_text(encoding="utf-8"))
    require(header.get("execution_class") == "real", "raw trajectory header is not real")
    result = reconstruct(header, load_jsonl(ledger_path), frozen)
    compare_ledgers(result["rows"], declared_rows)
    return {
        "raw_reconstruction_status": "pass",
        "raw_header_sha256": hashlib.sha256(header_path.read_bytes()).hexdigest(),
        "raw_ledger_sha256": hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
        "raw_reconstructed_attempts": result["attempts"],
        "cost_ledger": result["cost_ledger"],
    }


# ---------------------------------------------------------------------------
# Golden four-cell fixture and fault replays.

GOLDEN_CURRENT = fake_hash("golden-current-policy")
GOLDEN_BEHAVIOR = fake_hash("golden-behavior-policy")
GOLDEN_PREVIOUS = fake_hash("golden-previous-policy")
GOLDEN_NUISANCE = fake_hash("golden-nuisance")
GOLDEN_FREEZE = fake_hash("golden-freeze")
HALF_CELL = (gate0.ALPHA, "half")
ON_POLICY_CELL = (1.0, "on-policy")


def golden_frozen(temperature: float = gate0.DECODER_TEMPERATURE) -> dict[str, Any]:
    return {
        "current_policy_sha256": GOLDEN_CURRENT,
        "behavior_policy_sha256": GOLDEN_BEHAVIOR,
        "decoder_temperature": temperature,
        "maximum_response_tokens": 3,
        "lambda": gate0.DUAL,
        "support_floor_epsilon": 0.2,
        "nuisance_training_partition": "nuisance_fit",
        "projection_dimension": 1,
        "cells": sorted([HALF_CELL, ON_POLICY_CELL]),
    }


def golden_cost_ledger() -> dict[str, Any]:
    return {
        "candidate": {
            "attempts": 12,
            "aborted_attempts": 0,
            "k_gen": 24,
            "k_ver": 12,
            "k_score": 16,
            "accelerator_seconds": 12.0,
        },
        "reference": {
            "attempts": 4,
            "aborted_attempts": 0,
            "k_gen": 8,
            "k_ver": 4,
            "k_score": 0,
            "accelerator_seconds": 4.0,
        },
        "aborted": {
            "attempts": 0,
            "aborted_attempts": 0,
            "k_gen": 0,
            "k_ver": 0,
            "k_score": 0,
            "accelerator_seconds": 0.0,
        },
        "total": {
            "attempts": 16,
            "aborted_attempts": 0,
            "k_gen": 32,
            "k_ver": 16,
            "k_score": 16,
            "accelerator_seconds": 16.0,
        },
        "by_cell": {
            "select|0.5|half": {
                "attempts": 8,
                "aborted_attempts": 0,
                "k_gen": 16,
                "k_ver": 8,
                "k_score": 16,
                "accelerator_seconds": 8.0,
            },
            "select|1.0|on-policy": {
                "attempts": 4,
                "aborted_attempts": 0,
                "k_gen": 8,
                "k_ver": 4,
                "k_score": 0,
                "accelerator_seconds": 4.0,
            },
        },
    }


def golden_header(
    frozen: dict[str, Any],
    proposal: tuple[float, float] = gate0.STALE_PROMPT_MASS,
    training_ids: tuple[str, ...] = ("fit-0", "fit-1"),
    cost_ledger: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prompts = {
        f"p{x}": {"target_mass": gate0.PROMPT_MASS[x], "proposal_mass": proposal[x]}
        for x in range(2)
    }
    return {
        "schema_version": 1,
        "execution_class": "golden",
        "frozen": {
            **{key: frozen[key] for key in HEADER_FROZEN_FIELDS if key in frozen},
            "nuisance_model_sha256": GOLDEN_NUISANCE,
            "prompt_baseline_sha256": fake_hash("golden-baseline"),
            "freeze_record_sha256": GOLDEN_FREEZE,
        },
        "reward_range": [0.0, 1.0],
        "prompt_laws": {
            "select": prompts,
            "confirm": {"c0": {"target_mass": 1.0, "proposal_mass": 1.0}},
        },
        "nuisance_training_prompt_ids": list(training_ids),
        "cost_ledger": copy.deepcopy(cost_ledger) if cost_ledger is not None else golden_cost_ledger(),
    }


def decoder_success(temperature_ratio: float) -> tuple[float, float]:
    """Success probabilities when decoder logits are rescaled by a ratio."""
    return tuple(
        gate0.sigmoid(temperature_ratio * gate0.logit(p)) for p in gate0.DECODER_SUCCESS
    )


def log_mass(success: tuple[float, float], x: int, y: int) -> float:
    return math.log(success[x] if y else 1.0 - success[x])


def golden_raw_row(
    attempt: str,
    role: str,
    source: str,
    x: int,
    y: int,
    cell: tuple[float, str] | None,
    frozen: dict[str, Any],
    current_success: tuple[float, float] = gate0.DECODER_SUCCESS,
) -> dict[str, Any]:
    tokens = 1 + y + x
    h = gate0.frozen_h(x, y)
    added_passes = 1 if (role == "candidate" and cell is not None and cell[0] < 1.0) else 0
    row: dict[str, Any] = {
        "attempt_id": attempt,
        "target_law": "select",
        "estimator_role": role,
        "source_component": source,
        "infrastructure_failure": False,
        "allocated_accelerator_seconds": 1.0,
        "response_tokens": tokens,
        "verifier_called": True,
        "added_score_passes": added_passes,
        "trajectory_id": f"trajectory-{attempt}",
        "prompt_id": f"p{x}",
        "generating_policy_sha256": GOLDEN_CURRENT if source == "current" else GOLDEN_BEHAVIOR,
        "scoring": {
            "current_policy_sha256": GOLDEN_CURRENT,
            "behavior_policy_sha256": GOLDEN_BEHAVIOR,
            "temperature": frozen["decoder_temperature"],
        },
        "current_sequence_log_prob": log_mass(current_success, x, y),
        "behavior_sequence_log_prob": log_mass(gate0.STALE_SUCCESS, x, y),
        "reward": float(y if x == 0 else 1 - y),
        "termination": "truncated" if tokens == frozen["maximum_response_tokens"] else "eos",
        "nuisance": {
            "model_sha256": GOLDEN_NUISANCE,
            "reward_prediction": h + gate0.DUAL * tokens,
            "length_prediction": float(tokens),
            "prompt_baseline": gate0.prompt_baseline(0.0, x),
        },
        "prior_use_count": 0,
        "freeze_record_sha256": GOLDEN_FREEZE,
        "projected_score": [gate0.score(0.0, x, y)],
    }
    if role == "candidate":
        row.update({"alpha": cell[0], "allocation_id": cell[1], "replication_id": 0})
    return row


def golden_raw_rows(
    frozen: dict[str, Any], current_success: tuple[float, float] = gate0.DECODER_SUCCESS
) -> list[dict[str, Any]]:
    rows = []
    for x, y in gate0.CELLS:
        z = f"{x}{y}"
        rows.append(golden_raw_row(f"half-current-{z}", "candidate", "current", x, y, HALF_CELL, frozen, current_success))
        rows.append(golden_raw_row(f"half-stale-{z}", "candidate", "stale", x, y, HALF_CELL, frozen, current_success))
        rows.append(golden_raw_row(f"on-policy-{z}", "candidate", "current", x, y, ON_POLICY_CELL, frozen, current_success))
        rows.append(golden_raw_row(f"reference-{z}", "r1", "current", x, y, None, frozen, current_success))
    return rows


def enumerated_expectations(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Exact expectations of reconstructed terms under the true Gate 0 laws."""
    p_law = gate0.current_law()
    g_law = gate0.stale_law()
    totals: dict[str, float] = defaultdict(float)
    for row in rows:
        z = (int(row["attempt_id"][-2]), int(row["attempt_id"][-1]))
        value = row["projected_gradient"][0]
        if row["attempt_id"].startswith("half-current"):
            totals["half"] += p_law[z] * value
        elif row["attempt_id"].startswith("half-stale"):
            totals["half"] += (1.0 - gate0.ALPHA) * g_law[z] * value
        elif row["attempt_id"].startswith("on-policy"):
            totals["on_policy"] += p_law[z] * value
        else:
            totals["reference"] += p_law[z] * value
    return dict(totals)


def independent_projected_ledger(
    weight_rule: str = "exact", clip: float | None = None
) -> list[dict[str, Any]]:
    """Projected rows written by an independent producer from Gate 0 functions.

    ``exact`` uses the joint prompt-and-response mixture weight; ``omit_prompt``
    drops the prompt ratio; ``clip`` truncates the exact weight at ``clip``.
    """
    p_law = gate0.current_law()
    g_law = gate0.stale_law()
    rows = []

    def weight(x: int, y: int) -> float:
        z = (x, y)
        if weight_rule == "omit_prompt":
            p = p_law[z] / gate0.PROMPT_MASS[x]
            g = g_law[z] / gate0.STALE_PROMPT_MASS[x]
            return p / (gate0.ALPHA * p + (1.0 - gate0.ALPHA) * g)
        exact = p_law[z] / (gate0.ALPHA * p_law[z] + (1.0 - gate0.ALPHA) * g_law[z])
        return min(exact, clip) if clip is not None else exact

    for x, y in gate0.CELLS:
        z = f"{x}{y}"
        tokens = 1 + y + x
        u = gate0.utility(x, y)
        h = gate0.frozen_h(x, y)
        b = gate0.prompt_baseline(0.0, x)
        s = gate0.score(0.0, x, y)
        w = weight(x, y)
        terms = (
            (f"half-current-{z}", "candidate", "current", HALF_CELL, (h - b) * s + gate0.ALPHA * w * (u - h) * s, "A=(h-b)s+alpha*w*(U-h)s", tokens),
            (f"half-stale-{z}", "candidate", "stale", HALF_CELL, w * (u - h) * s, "B=w*(U-h)s", tokens),
            (f"on-policy-{z}", "candidate", "current", ON_POLICY_CELL, (u - b) * s, "A=(h-b)s+alpha*w*(U-h)s", 0),
            (f"reference-{z}", "r1", "current", None, u * s, "target=U*s", 0),
        )
        for attempt, role, source, cell, value, semantics, added_score_tokens in terms:
            row = {
                "attempt_id": attempt,
                "target_law": "select",
                "estimator_role": role,
                "source_component": source,
                "infrastructure_failure": False,
                "allocated_accelerator_seconds": 1.0,
                "generated_tokens": tokens,
                "verifier_calls": 1,
                "added_score_tokens": added_score_tokens,
                "trajectory_id": f"trajectory-{attempt}",
                "use_count": 1,
                "scored_outcome": True,
                "projected_gradient": [value],
                "projected_term_semantics": semantics,
            }
            if cell is not None:
                row.update({"alpha": cell[0], "allocation_id": cell[1], "replication_id": 0})
            rows.append(row)
    return rows


def rejected(action) -> str | None:
    try:
        action()
    except ReconstructionError as exc:
        return str(exc)
    return None


def weight_bias(rows: list[dict[str, Any]]) -> float:
    return enumerated_expectations(rows)["half"] - gate0.target_gradient()


def self_test() -> dict[str, Any]:
    target = gate0.target_gradient()
    frozen = golden_frozen()
    header = golden_header(frozen)
    golden = reconstruct(header, golden_raw_rows(frozen), frozen)
    expectations = enumerated_expectations(golden["rows"])
    p_law, g_law = gate0.current_law(), gate0.stale_law()
    mixture = {z: gate0.ALPHA * p_law[z] + (1 - gate0.ALPHA) * g_law[z] for z in gate0.CELLS}
    select_law = header["prompt_laws"]["select"]
    weight_mass = math.fsum(
        mixture[(x, y)]
        * mixture_weight(
            math.log(select_law[f"p{x}"]["target_mass"])
            + log_mass(gate0.DECODER_SUCCESS, x, y),
            math.log(select_law[f"p{x}"]["proposal_mass"])
            + log_mass(gate0.STALE_SUCCESS, x, y),
            gate0.ALPHA,
        )
        for x, y in gate0.CELLS
    )
    golden_checks = {
        "half_cell_unbiased": abs(expectations["half"] - target) <= IDENTITY_TOLERANCE,
        "on_policy_cell_unbiased": abs(expectations["on_policy"] - target) <= IDENTITY_TOLERANCE,
        "reference_unbiased": abs(expectations["reference"] - target) <= IDENTITY_TOLERANCE,
        "mixture_weight_normalizes": abs(weight_mass - 1.0) <= IDENTITY_TOLERANCE,
        "weight_bound_holds": golden["maximum_alpha_times_weight"] <= 1.0 + MASS_TOLERANCE,
        "cost_ledger_reconstructs": golden["cost_ledger"] == golden_cost_ledger(),
        "aborted_attempt_costs_retained": aborted_attempt_fixture_passes(frozen),
        "independent_producer_matches": rejected(
            lambda: compare_ledgers(golden["rows"], independent_projected_ledger())
        )
        is None,
    }
    require(all(golden_checks.values()), f"golden reconstruction failed: {golden_checks}")

    faults: dict[str, dict[str, Any]] = {}

    # 1. Omitted prompt ratio in the producer's weight.
    omitted = independent_projected_ledger("omit_prompt")
    equal_q_header = golden_header(frozen, proposal=gate0.PROMPT_MASS)
    faults["omitted_prompt_ratio"] = {
        "producer_bias": weight_bias(omitted),
        "rejection": rejected(lambda: compare_ledgers(golden["rows"], omitted)),
        "null_control_passes": rejected(
            lambda: compare_ledgers(
                reconstruct(equal_q_header, golden_raw_rows(frozen), frozen)["rows"],
                omitted_with_equal_prompt_law(),
            )
        )
        is None,
    }

    # 2. Weight clipping below the defensive bound 1/alpha.
    max_weight = max(p_law[z] / mixture[z] for z in gate0.CELLS)
    clipped = independent_projected_ledger(clip=0.9 * max_weight)
    faults["weight_clipping"] = {
        "clip": 0.9 * max_weight,
        "producer_bias": weight_bias(clipped),
        "rejection": rejected(lambda: compare_ledgers(golden["rows"], clipped)),
        "null_control_passes": rejected(
            lambda: compare_ledgers(
                golden["rows"], independent_projected_ledger(clip=1.0 / gate0.ALPHA)
            )
        )
        is None,
    }

    # 3. Base logits (temperature one) instead of the rollout decoder.
    base_success = decoder_success(gate0.DECODER_TEMPERATURE)
    base_rows = golden_raw_rows(frozen, base_success)
    for row in base_rows:
        row["scoring"]["temperature"] = 1.0
    relaxed = golden_frozen(temperature=1.0)
    base_terms = reconstruct(golden_header(relaxed), relabel(base_rows, 1.0), relaxed)["rows"]
    gate0_base = gate0.run_base_decoder_fault()
    unit_frozen = golden_frozen(temperature=1.0)
    faults["base_versus_decoder_scoring"] = {
        "reconstructed_bias": weight_bias(base_terms),
        "gate0_bias": gate0_base["base_scored_bias"],
        "bias_matches_gate0": abs(weight_bias(base_terms) - gate0_base["base_scored_bias"])
        <= IDENTITY_TOLERANCE,
        "rejection": rejected(lambda: reconstruct(header, base_rows, frozen)),
        "null_control_passes": rejected(
            lambda: reconstruct(golden_header(unit_frozen), golden_raw_rows(unit_frozen), unit_frozen)
        )
        is None,
    }

    # 4. Nuisance model fitted on the focal audited prompts.
    gate0_focal = gate0.run_focal_nuisance_fault()
    focal_rows = golden_raw_rows(frozen)
    refit = fake_hash("focal-refit-nuisance")
    for row in focal_rows:
        row["nuisance"]["model_sha256"] = refit
    faults["focal_nuisance_fitting"] = {
        "gate0_bias": gate0_focal["focal_fit_bias"],
        "rejection_training_overlap": rejected(
            lambda: reconstruct(golden_header(frozen, training_ids=("fit-0", "p0")), golden_raw_rows(frozen), frozen)
        ),
        "rejection_unfrozen_model": rejected(lambda: reconstruct(header, focal_rows, frozen)),
        "null_control_passes": golden_checks["independent_producer_matches"],
    }

    # 5. Reuse of trajectories that already formed the current policy.
    gate0_reuse = gate0.run_reuse_fault()
    reused = golden_raw_rows(frozen)
    reused[0]["prior_use_count"] = 1
    rescored = golden_raw_rows(frozen)
    rescored[0]["generating_policy_sha256"] = GOLDEN_PREVIOUS
    faults["consumed_trajectory_reuse"] = {
        "gate0_bias_without_ratio": gate0_reuse["reuse_without_ratio_bias"],
        "gate0_bias_with_ratio": gate0_reuse["reuse_with_recomputed_ratio_bias"],
        "rejection_prior_use": rejected(lambda: reconstruct(header, reused, frozen)),
        "rejection_previous_generator": rejected(lambda: reconstruct(header, rescored, frozen)),
        "null_control_passes": golden_checks["half_cell_unbiased"],
    }

    for name, fault in faults.items():
        rejections = [value for key, value in fault.items() if key.startswith("rejection")]
        require(
            rejections and all(isinstance(value, str) for value in rejections),
            f"fault {name} was not rejected by the reconstruction",
        )
        require(fault["null_control_passes"] is True, f"null control for {name} failed")
    for name in ("omitted_prompt_ratio", "weight_clipping"):
        require(
            abs(faults[name]["producer_bias"]) > DETECTION_THRESHOLD,
            f"fault {name} has no detectable bias on the golden law",
        )
    require(
        faults["base_versus_decoder_scoring"]["bias_matches_gate0"],
        "base-scored reconstruction does not reproduce the Gate 0 bias",
    )

    mutations = {
        "declared_term_tampered": rejected(
            lambda: compare_ledgers(golden["rows"], tamper_term(golden["rows"]))
        ),
        "declared_row_omitted": rejected(
            lambda: compare_ledgers(golden["rows"], golden["rows"][1:])
        ),
        "prompt_mass_not_normalized": rejected(
            lambda: reconstruct(perturb_mass(header), golden_raw_rows(frozen), frozen)
        ),
        "support_floor_violated": rejected(
            lambda: reconstruct(
                golden_header(frozen, proposal=(0.1, 0.9)), golden_raw_rows(frozen), frozen
            )
        ),
        "action_dependent_baseline": rejected(
            lambda: reconstruct(header, shift_baseline(golden_raw_rows(frozen)), frozen)
        ),
        "truncation_mislabeled": rejected(
            lambda: reconstruct(header, mislabel_truncation(golden_raw_rows(frozen)), frozen)
        ),
        "unfrozen_lambda": rejected(
            lambda: reconstruct(header, golden_raw_rows(frozen), {**frozen, "lambda": 0.0})
        ),
        "aborted_attempt_cost_dropped": rejected(
            lambda: reconstruct(header, golden_raw_rows(frozen) + [aborted_raw_row()], frozen)
        ),
        "failed_attempt_carries_scored_fields": rejected(
            lambda: reconstruct(*failed_with_scored_fields(frozen), frozen)
        ),
        "mixture_row_omits_added_score_pass": rejected(
            lambda: reconstruct(*miscount_added_score_passes(frozen, "half-current-00", 0), frozen)
        ),
        "on_policy_row_claims_added_score_pass": rejected(
            lambda: reconstruct(*miscount_added_score_passes(frozen, "on-policy-00", 1), frozen)
        ),
        "header_cost_ledger_tampered": rejected(
            lambda: reconstruct(tamper_header_cost(header), golden_raw_rows(frozen), frozen)
        ),
        "declared_cost_counter_tampered": rejected(
            lambda: compare_ledgers(golden["rows"], tamper_declared_cost(golden["rows"]))
        ),
    }
    require(
        all(isinstance(value, str) for value in mutations.values()),
        f"a raw-ledger mutation was accepted: {mutations}",
    )
    return {
        "schema_version": 1,
        "audit_type": "rl_raw_trajectory_reconstruction",
        "empirical_evidence": False,
        "status": "pass",
        "target_gradient": target,
        "golden_expectations": expectations,
        "golden_cost_ledger": golden["cost_ledger"],
        "golden_checks": golden_checks,
        "faults": faults,
        "mutations_rejected": len(mutations),
        "mutations": mutations,
    }


def aborted_raw_row() -> dict[str, Any]:
    return {
        "attempt_id": "half-stale-aborted-00",
        "target_law": "select",
        "estimator_role": "candidate",
        "source_component": "stale",
        "infrastructure_failure": True,
        "allocated_accelerator_seconds": 0.75,
        "response_tokens": 2,
        "verifier_called": True,
        "added_score_passes": 1,
        "alpha": gate0.ALPHA,
        "allocation_id": "half",
        "replication_id": 0,
    }


def aborted_attempt_fixture_passes(frozen: dict[str, Any]) -> bool:
    rows = golden_raw_rows(frozen) + [aborted_raw_row()]
    ledger = golden_cost_ledger()
    for section_name in ("candidate", "aborted", "total"):
        bucket = ledger[section_name]
        bucket["attempts"] += 1
        bucket["aborted_attempts"] += 1
        bucket["k_gen"] += 2
        bucket["k_ver"] += 1
        bucket["k_score"] += 2
        bucket["accelerator_seconds"] += 0.75
    cell_bucket = ledger["by_cell"]["select|0.5|half"]
    cell_bucket["attempts"] += 1
    cell_bucket["aborted_attempts"] += 1
    cell_bucket["k_gen"] += 2
    cell_bucket["k_ver"] += 1
    cell_bucket["k_score"] += 2
    cell_bucket["accelerator_seconds"] += 0.75
    result = reconstruct(golden_header(frozen, cost_ledger=ledger), rows, frozen)
    return (
        result["attempts"] == 17
        and result["successful_trajectories"] == 16
        and result["cost_ledger"]["aborted"]["k_gen"] == 2
        and result["cost_ledger"]["aborted"]["k_ver"] == 1
        and result["cost_ledger"]["aborted"]["k_score"] == 2
    )


def failed_with_scored_fields(frozen: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = golden_raw_rows(frozen)
    rows[0]["infrastructure_failure"] = True
    return golden_header(frozen), rows


def miscount_added_score_passes(
    frozen: dict[str, Any], attempt_id: str, new_passes: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = golden_raw_rows(frozen)
    for row in rows:
        if row["attempt_id"] == attempt_id:
            delta = (new_passes - row["added_score_passes"]) * row["response_tokens"]
            row["added_score_passes"] = new_passes
            break
    ledger = golden_cost_ledger()
    ledger["candidate"]["k_score"] += delta
    ledger["total"]["k_score"] += delta
    cell_key = "select|0.5|half" if attempt_id.startswith("half") else "select|1.0|on-policy"
    ledger["by_cell"][cell_key]["k_score"] += delta
    return golden_header(frozen, cost_ledger=ledger), rows


def tamper_header_cost(header: dict[str, Any]) -> dict[str, Any]:
    changed = copy.deepcopy(header)
    changed["cost_ledger"]["candidate"]["k_score"] -= 1
    changed["cost_ledger"]["total"]["k_score"] -= 1
    return changed


def tamper_declared_cost(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    changed = copy.deepcopy(rows)
    changed[0]["added_score_tokens"] -= 1
    return changed


def omitted_with_equal_prompt_law() -> list[dict[str, Any]]:
    """Omitting the prompt ratio is harmless when the proposal equals D."""
    saved = gate0.STALE_PROMPT_MASS
    gate0.STALE_PROMPT_MASS = gate0.PROMPT_MASS
    try:
        return independent_projected_ledger("omit_prompt")
    finally:
        gate0.STALE_PROMPT_MASS = saved


def relabel(rows: list[dict[str, Any]], temperature: float) -> list[dict[str, Any]]:
    for row in rows:
        row["scoring"]["temperature"] = temperature
    return rows


def tamper_term(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    changed = copy.deepcopy(rows)
    changed[0]["projected_gradient"][0] += 1e-6
    return changed


def perturb_mass(header: dict[str, Any]) -> dict[str, Any]:
    changed = copy.deepcopy(header)
    changed["prompt_laws"]["select"]["p0"]["target_mass"] += 1e-6
    return changed


def shift_baseline(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        if row["attempt_id"] == "half-stale-01":
            row["nuisance"]["prompt_baseline"] += 0.01
    return rows


def mislabel_truncation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        if row["response_tokens"] < 3:
            row["termination"] = "truncated"
            break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.self_test:
        parser.error("use --self-test; real reconstruction runs inside the pipeline runner")
    try:
        summary = self_test()
    except ReconstructionError as exc:
        raise SystemExit(f"error: {exc}") from exc
    if args.output is not None:
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        "validated RL raw trajectory reconstruction on the exact Gate 0 law; "
        f"{len(summary['faults'])} faults rejected with passing null controls, "
        f"{summary['mutations_rejected']} mutations rejected"
    )


if __name__ == "__main__":
    main()
