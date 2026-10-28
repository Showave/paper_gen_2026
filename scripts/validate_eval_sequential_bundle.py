#!/usr/bin/env python3
"""Recompute the evaluation paper's sequential decision bundle and frame hierarchy.

A bundle contains the root/cell/rater frame hierarchy, the weighted target
frame, one frozen commitment per batch, and every fixed assignment slot.  The
validator independently recomputes

* each source's SHA-256(revision || question_id) root ranks, selected roots,
  prompt-variant weights, root--prompt--order cells, model-jury call count,
  and human target-unit product masses pi(i,c,h,o) = w_i * a_(c|i) * rho_h / 2;
* each batch's candidate-mixture law p_b = eps*pi + (1-eps)*r_b from the
  committed candidate multiset, EVI values, annotation costs, and temperature;
* predictability: the batch commitment hash and fit slots strictly before the
  batch;
* the draw-level augmented Hansen--Hurwitz contributions, their worst-case
  nonresponse endpoints, the tie-rate analogue, ESS, and maximum weight;
* the complete-outcome studentized interval or the martingale--Hoeffding
  interval when any slot is missing, the response-drift widenings, and the
  terminal four-way action.

Every reported number must reproduce.  The self-test builds generated golden
bundles whose probabilities and unit masses are exact rationals and checks the
float recomputation against an independent Fraction implementation, validates
the 528-root / 2,112-cell / 8,448-jury-call / 168,960-unit pilot hierarchy,
and applies mutation tests that must fail.  Real bundles are rejected while the
preregistration's estimand-defining artifact hashes remain null.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from fractions import Fraction
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREREGISTRATION = ROOT / "experiments" / "pilot_preregistration.json"
TOLERANCE = 1e-12
ACTIONS = ("A", "B", "equivalent", "not-established")
POLICIES = ("target_random", "evi_candidate_mixture")


class BundleError(ValueError):
    pass


def canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def close(a: float, b: float) -> bool:
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= TOLERANCE * max(
        1.0, abs(a), abs(b)
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BundleError(message)


def commitment_payload(batch: dict[str, Any]) -> dict[str, Any]:
    return {
        key: batch[key]
        for key in (
            "batch",
            "policy",
            "slot_start",
            "slot_end",
            "candidate_multiset",
            "evi",
            "annotation_cost",
            "predictions",
            "tie_predictions",
            "fit_slots",
        )
    }


def config_from_preregistration(section: dict[str, Any]) -> dict[str, Any]:
    acquisition = section["acquisition"]
    decision = section["decision"]
    return {
        "exploration_epsilon": acquisition["exploration_epsilon"],
        "evi_temperature": acquisition["evi_temperature"],
        "batch_size": acquisition["batch_size"],
        "fixed_budget_attempts": acquisition["adaptive_fixed_budget_attempts"],
        "evi_candidate_multiset_size": acquisition["evi_candidate_multiset_size"],
        "alpha": round(1.0 - decision["confidence_level"], 12),
        "normal_quantile": decision["normal_quantile"],
        "practical_margin": decision["practical_margin"],
        "response_drift_gamma_grid": section["nonresponse"]["response_drift_gamma_grid"],
    }


def batch_law(
    batch: dict[str, Any], mass: dict[str, float], config: dict[str, Any]
) -> dict[str, float]:
    """Return p_b restricted to the units it can differ from eps*pi on."""
    eps = config["exploration_epsilon"]
    if batch["policy"] == "target_random":
        return {}
    multiset = batch["candidate_multiset"]
    logits = {}
    for unit, count in multiset.items():
        cost = batch["annotation_cost"][unit]
        value = batch["evi"][unit]
        require(
            isinstance(value, (int, float)) and math.isfinite(value),
            f"batch {batch['batch']} has non-finite EVI for {unit}",
        )
        require(cost > 0, f"batch {batch['batch']} has nonpositive cost for {unit}")
        logits[unit] = math.log(count) + config["evi_temperature"] * value / cost
    top = max(logits.values())
    normalizer = sum(math.exp(value - top) for value in logits.values())
    return {
        unit: eps * mass[unit] + (1.0 - eps) * math.exp(value - top) / normalizer
        for unit, value in logits.items()
    }


def draw_probability(
    unit: str, law: dict[str, float], mass: dict[str, float], policy: str, eps: float
) -> float:
    if policy == "target_random":
        return mass[unit]
    return law.get(unit, eps * mass[unit])


def root_selection_sha256(revision: str, question_id: str) -> str:
    return hashlib.sha256(f"{revision}||{question_id}".encode("utf-8")).hexdigest()


def recompute_frame_hierarchy(
    hierarchy: Any,
    *,
    expected_sources: dict[str, int] | None = None,
    expected_variants: int | None = None,
    expected_orders: int | None = None,
    expected_panel_size: int | None = None,
    expected_jury_keys: list[str] | None = None,
) -> dict[str, Any]:
    """Recompute root SHA-256 ranks, selected roots, cells, jury calls, and unit masses."""
    require(isinstance(hierarchy, dict), "frame_hierarchy must be an object")
    sources = hierarchy.get("sources")
    require(isinstance(sources, dict) and sources, "frame_hierarchy needs sources")
    if expected_sources is not None:
        require(
            set(sources) == set(expected_sources),
            "frame_hierarchy sources differ from the preregistration",
        )
    variants_per_root = hierarchy.get("prompt_variants_per_root")
    require(
        isinstance(variants_per_root, int)
        and not isinstance(variants_per_root, bool)
        and variants_per_root > 0,
        "frame_hierarchy needs positive prompt_variants_per_root",
    )
    if expected_variants is not None:
        require(
            variants_per_root == expected_variants,
            "prompt_variants_per_root differs from the preregistration",
        )
    display_orders = hierarchy.get("display_orders")
    require(
        display_orders == [-1, 1],
        "frame_hierarchy display_orders must be [-1, 1]",
    )
    if expected_orders is not None:
        require(len(display_orders) == expected_orders, "display_orders count mismatch")
    jury_keys = hierarchy.get("jury_model_keys")
    require(
        isinstance(jury_keys, list)
        and jury_keys
        and all(isinstance(k, str) and k for k in jury_keys)
        and len(set(jury_keys)) == len(jury_keys),
        "frame_hierarchy needs unique jury_model_keys",
    )
    if expected_jury_keys is not None:
        require(
            jury_keys == expected_jury_keys,
            "frame_hierarchy jury_model_keys differ from the preregistration",
        )
    raters = hierarchy.get("raters")
    require(isinstance(raters, list) and raters, "frame_hierarchy needs raters")
    if expected_panel_size is not None:
        require(len(raters) == expected_panel_size, "rater panel size mismatch")
    rater_weights: dict[str, float] = {}
    for rater in raters:
        require(isinstance(rater, dict), "rater entry must be an object")
        rater_id = rater.get("rater_id")
        weight = rater.get("target_weight")
        require(
            isinstance(rater_id, str)
            and rater_id
            and rater_id not in rater_weights
            and isinstance(weight, (int, float))
            and not isinstance(weight, bool)
            and math.isfinite(weight)
            and weight > 0,
            "rater entry has invalid or duplicate rater_id or target_weight",
        )
        rater_weights[rater_id] = float(weight)
    require(
        close(math.fsum(rater_weights.values()), 1.0),
        "rater target weights do not sum to one",
    )

    n_sources = len(sources)
    row_uids: set[str] = set()
    selected_roots: dict[str, dict[str, Any]] = {}
    expected_cells: dict[tuple[str, str, int], float] = {}
    for source_key, source_entry in sorted(sources.items()):
        require(isinstance(source_entry, dict), f"source {source_key} must be an object")
        revision = source_entry.get("revision")
        selected_count = source_entry.get("selected_count")
        candidates = source_entry.get("candidate_roots")
        require(
            isinstance(revision, str)
            and revision
            and isinstance(selected_count, int)
            and not isinstance(selected_count, bool)
            and selected_count > 0
            and isinstance(candidates, list)
            and len(candidates) >= selected_count,
            f"source {source_key} has invalid revision, selected_count, or candidate_roots",
        )
        if expected_sources is not None:
            require(
                selected_count == expected_sources[source_key],
                f"source {source_key} selected_count differs from the preregistration",
            )
        seen_questions: set[str] = set()
        recomputed_candidates = []
        for candidate in candidates:
            require(isinstance(candidate, dict), f"{source_key} candidate root must be an object")
            qid = candidate.get("question_id")
            uid = candidate.get("row_uid")
            sel_hash = candidate.get("selection_sha256")
            require(
                isinstance(qid, str)
                and qid
                and qid not in seen_questions
                and isinstance(uid, str)
                and uid
                and uid not in row_uids
                and sel_hash == root_selection_sha256(revision, qid),
                f"{source_key} candidate root has invalid question_id, row_uid, or selection_sha256",
            )
            seen_questions.add(qid)
            row_uids.add(uid)
            recomputed_candidates.append((sel_hash, qid, candidate))
        recomputed_candidates.sort(key=lambda item: (item[0], item[1]))
        expected_root_mass = 1.0 / (n_sources * selected_count)
        for expected_rank, (_, _, candidate) in enumerate(recomputed_candidates):
            uid = candidate["row_uid"]
            require(
                candidate.get("hash_rank") == expected_rank,
                f"{source_key} root {uid} hash_rank does not reproduce from sorted SHA-256(revision || question_id)",
            )
            should_select = expected_rank < selected_count
            require(
                candidate.get("selected") is should_select,
                f"{source_key} root {uid} selection flag violates the hash-rank cutoff",
            )
            if not should_select:
                require(
                    candidate.get("target_mass") in (None, 0.0),
                    f"unselected root {uid} cannot carry positive target_mass",
                )
                continue
            require(
                isinstance(candidate.get("target_mass"), (int, float))
                and close(float(candidate["target_mass"]), expected_root_mass),
                f"selected root {uid} target_mass does not equal 1/(n_sources * n_source)",
            )
            variants = candidate.get("prompt_variants")
            require(
                isinstance(variants, list) and len(variants) == variants_per_root,
                f"selected root {uid} must define prompt_variants_per_root variants",
            )
            variant_weights: dict[str, float] = {}
            for variant in variants:
                require(isinstance(variant, dict), f"root {uid} variant must be an object")
                vid = variant.get("variant_id")
                v_weight = variant.get("weight")
                require(
                    isinstance(vid, str)
                    and vid
                    and vid not in variant_weights
                    and isinstance(v_weight, (int, float))
                    and not isinstance(v_weight, bool)
                    and math.isfinite(v_weight)
                    and v_weight > 0,
                    f"root {uid} variant has invalid or duplicate variant_id or weight",
                )
                variant_weights[vid] = float(v_weight)
            require(
                close(math.fsum(variant_weights.values()), 1.0),
                f"root {uid} prompt-variant weights do not sum to one",
            )
            selected_roots[uid] = {
                "source_key": source_key,
                "target_mass": expected_root_mass,
                "variants": variant_weights,
            }
            for vid, v_weight in variant_weights.items():
                for order in display_orders:
                    expected_cells[(uid, vid, order)] = (
                        expected_root_mass * v_weight / 2.0
                    )

    require(
        close(math.fsum(expected_cells.values()), 1.0),
        "root--prompt--order cell masses do not sum to one",
    )
    declared_cells = hierarchy.get("cells")
    require(
        isinstance(declared_cells, list) and len(declared_cells) == len(expected_cells),
        "frame_hierarchy must list every root--prompt--order cell",
    )
    seen_cells: set[tuple[str, str, int]] = set()
    for cell in declared_cells:
        require(isinstance(cell, dict), "cell entry must be an object")
        key = (cell.get("row_uid"), cell.get("variant_id"), cell.get("display_order"))
        require(
            key in expected_cells
            and key not in seen_cells
            and isinstance(cell.get("cell_mass"), (int, float))
            and close(float(cell["cell_mass"]), expected_cells[key]),
            f"root--prompt--order cell {key} is invalid, duplicated, or misweighted",
        )
        seen_cells.add(key)

    return {
        "selected_roots": len(selected_roots),
        "root_prompt_order_cells": len(expected_cells),
        "model_jury_calls": len(expected_cells) * len(jury_keys),
        "human_target_units": len(expected_cells) * len(rater_weights),
        "cell_masses": expected_cells,
        "rater_weights": rater_weights,
    }


def recompute(bundle: dict[str, Any], preregistration: dict[str, Any]) -> dict[str, Any]:
    require(bundle.get("schema_version") == 1, "unsupported bundle schema")
    execution_class = bundle.get("execution_class")
    require(execution_class in {"fixture", "real"}, "unknown execution class")
    section = preregistration["eval"]
    if execution_class == "real":
        require(bundle.get("empirical_evidence") is True, "real bundle must be empirical")
        require(
            bundle.get("preregistration_section_sha256") == canonical_digest(section),
            "real bundle has a stale evaluation preregistration",
        )
        missing = sorted(
            key for key, value in section["frozen_artifact_hashes"].items() if value is None
        )
        require(
            not missing,
            "real bundle is blocked by unfrozen estimand artifacts: " + ", ".join(missing),
        )
        require(
            bundle.get("config") == config_from_preregistration(section),
            "real bundle config differs from the preregistration",
        )
        frame_spec = section["frame"]
        hierarchy = recompute_frame_hierarchy(
            bundle.get("frame_hierarchy"),
            expected_sources=frame_spec["root_items_by_source"],
            expected_variants=frame_spec["prompt_variants_per_root"],
            expected_orders=frame_spec["display_orders"],
            expected_panel_size=section["human_protocol"]["panel_size"],
            expected_jury_keys=section["jury_model_keys"],
        )
    else:
        require(
            bundle.get("empirical_evidence") is False,
            "fixture bundle must be marked non-empirical",
        )
        hierarchy = recompute_frame_hierarchy(bundle.get("frame_hierarchy"))
    config = bundle["config"]
    eps = config["exploration_epsilon"]
    total_slots = config["fixed_budget_attempts"]
    require(0 < eps <= 1 and total_slots > 1, "invalid epsilon or fixed budget")

    frame = bundle["frame"]
    require(
        len(frame) == hierarchy["human_target_units"],
        "bundle frame count does not match the hierarchy human_target_units",
    )
    mass = {}
    seen_tuples: set[tuple[str, str, str, int]] = set()
    for row in frame:
        require(isinstance(row, dict), "frame unit must be an object")
        unit_id = row.get("unit_id")
        cell_key = (row.get("row_uid"), row.get("variant_id"), row.get("display_order"))
        rater_id = row.get("rater_id")
        tuple_key = (cell_key[0], cell_key[1], rater_id, cell_key[2])
        require(
            isinstance(unit_id, str)
            and unit_id
            and unit_id not in mass
            and cell_key in hierarchy["cell_masses"]
            and rater_id in hierarchy["rater_weights"]
            and tuple_key not in seen_tuples,
            f"frame unit {unit_id!r} does not map uniquely to a (root, variant, rater, order) cell",
        )
        expected_unit_mass = (
            hierarchy["cell_masses"][cell_key] * hierarchy["rater_weights"][rater_id]
        )
        require(
            isinstance(row.get("target_mass"), (int, float))
            and close(float(row["target_mass"]), expected_unit_mass),
            f"frame unit {unit_id} target_mass disagrees with w_i * a_(c|i) * rho_h / 2",
        )
        seen_tuples.add(tuple_key)
        mass[unit_id] = float(row["target_mass"])
    require(abs(sum(mass.values()) - 1.0) <= TOLERANCE, "frame masses must sum to one")

    batches = sorted(bundle["batches"], key=lambda row: row["batch"])
    require(
        [row["batch"] for row in batches] == list(range(len(batches))),
        "batch indices must be contiguous from zero",
    )
    laws: dict[int, dict[str, float]] = {}
    expected_start = 0
    for batch in batches:
        index = batch["batch"]
        require(batch["policy"] in POLICIES, f"batch {index} has an unknown policy")
        require(
            batch["slot_start"] == expected_start
            and batch["slot_end"] - batch["slot_start"] == config["batch_size"],
            f"batch {index} does not occupy the next fixed slot block",
        )
        expected_start = batch["slot_end"]
        require(
            batch.get("commitment_sha256") == canonical_digest(commitment_payload(batch)),
            f"batch {index} commitment hash does not match its frozen objects",
        )
        require(
            all(0 <= slot < batch["slot_start"] for slot in batch["fit_slots"]),
            f"batch {index} predictor uses a slot that is not strictly earlier",
        )
        require(
            set(batch["predictions"]) == set(mass)
            and set(batch["tie_predictions"]) == set(mass),
            f"batch {index} predictions must cover the complete frame",
        )
        require(
            all(-1.0 <= value <= 1.0 for value in batch["predictions"].values())
            and all(0.0 <= value <= 1.0 for value in batch["tie_predictions"].values()),
            f"batch {index} predictions are outside their bounded ranges",
        )
        if batch["policy"] == "evi_candidate_mixture":
            multiset = batch["candidate_multiset"]
            require(
                set(multiset) <= set(mass)
                and all(isinstance(n, int) and n > 0 for n in multiset.values())
                and sum(multiset.values()) == config["evi_candidate_multiset_size"],
                f"batch {index} candidate multiset is invalid",
            )
            require(
                set(batch["evi"]) == set(multiset)
                and set(batch["annotation_cost"]) == set(multiset),
                f"batch {index} EVI and costs must cover exactly its candidates",
            )
        else:
            require(
                not batch["candidate_multiset"] and not batch["evi"],
                f"target-random batch {index} cannot carry EVI candidates",
            )
        law = batch_law(batch, mass, config)
        rest = eps * (1.0 - sum(mass[unit] for unit in law))
        if batch["policy"] == "evi_candidate_mixture":
            require(
                abs(sum(law.values()) + rest - 1.0) <= TOLERANCE,
                f"batch {index} draw law does not normalize",
            )
        laws[index] = law
    require(expected_start == total_slots, "batches must cover exactly the fixed budget")

    slots = sorted(bundle["slots"], key=lambda row: row["slot"])
    require(
        [row["slot"] for row in slots] == list(range(total_slots)),
        "every fixed assignment slot must appear exactly once",
    )
    by_batch = {batch["batch"]: batch for batch in batches}
    psi: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    tie_psi: list[float] = []
    tie_lower: list[float] = []
    tie_upper: list[float] = []
    weights: list[float] = []
    missing = 0
    for row in slots:
        batch = by_batch.get(row["batch"])
        require(batch is not None, f"slot {row['slot']} references an unknown batch")
        require(
            batch["slot_start"] <= row["slot"] < batch["slot_end"],
            f"slot {row['slot']} lies outside its batch",
        )
        require(
            row.get("batch_commitment_sha256") == batch["commitment_sha256"],
            f"slot {row['slot']} is not bound to its batch commitment",
        )
        unit = row["unit_id"]
        require(unit in mass, f"slot {row['slot']} draws a unit outside the frame")
        p = draw_probability(unit, laws[batch["batch"]], mass, batch["policy"], eps)
        require(
            close(row["logged_probability"], p),
            f"slot {row['slot']} logged probability does not reproduce",
        )
        require(p >= eps * mass[unit] - TOLERANCE, f"slot {row['slot']} violates positivity")
        weight = mass[unit] / p
        weights.append(weight)
        m_bar = sum(mass[u] * batch["predictions"][u] for u in mass)
        tie_bar = sum(mass[u] * batch["tie_predictions"][u] for u in mass)
        m = batch["predictions"][unit]
        m0 = batch["tie_predictions"][unit]
        if row["observed"]:
            require(row["outcome"] in (-1, 0, 1), f"slot {row['slot']} outcome invalid")
            value = m_bar + weight * (row["outcome"] - m)
            tie = tie_bar + weight * (float(row["outcome"] == 0) - m0)
            psi.append(value)
            lower.append(value)
            upper.append(value)
            tie_psi.append(tie)
            tie_lower.append(tie)
            tie_upper.append(tie)
        else:
            require(row["outcome"] is None, f"missing slot {row['slot']} carries an outcome")
            missing += 1
            lower.append(m_bar + weight * (-1.0 - m))
            upper.append(m_bar + weight * (1.0 - m))
            tie_lower.append(tie_bar + weight * (0.0 - m0))
            tie_upper.append(tie_bar + weight * (1.0 - m0))

    alpha = config["alpha"]
    delta = config["practical_margin"]
    count = total_slots
    if missing == 0:
        estimate = sum(psi) / count
        variance = sum((value - estimate) ** 2 for value in psi) / (count - 1)
        radius = config["normal_quantile"] * math.sqrt(variance / count)
        interval = [max(-1.0, estimate - radius), min(1.0, estimate + radius)]
        tie_estimate = sum(tie_psi) / count
        tie_variance = sum((v - tie_estimate) ** 2 for v in tie_psi) / (count - 1)
        tie_radius = config["normal_quantile"] * math.sqrt(tie_variance / count)
        tie_interval = [
            max(0.0, tie_estimate - tie_radius),
            min(1.0, tie_estimate + tie_radius),
        ]
        interval_type = "studentized_complete"
        point: dict[str, Any] = {
            "estimate": estimate,
            "variance": variance,
            "tie_estimate": tie_estimate,
        }
    else:
        bound = 1.0 + 2.0 / eps
        radius = bound * math.sqrt(2.0 * math.log(2.0 / alpha) / count)
        interval = [
            max(-1.0, sum(lower) / count - radius),
            min(1.0, sum(upper) / count + radius),
        ]
        tie_radius = bound * math.sqrt(math.log(2.0 / alpha) / (2.0 * count))
        tie_interval = [
            max(0.0, sum(tie_lower) / count - tie_radius),
            min(1.0, sum(tie_upper) / count + tie_radius),
        ]
        interval_type = "martingale_hoeffding_worst_case"
        point = {
            "estimate_lower": sum(lower) / count,
            "estimate_upper": sum(upper) / count,
        }

    def action(bounds: list[float]) -> str:
        if bounds[0] > delta:
            return "A"
        if bounds[1] < -delta:
            return "B"
        if -delta <= bounds[0] and bounds[1] <= delta:
            return "equivalent"
        return "not-established"

    drift = {}
    for gamma in config["response_drift_gamma_grid"]:
        widened = [max(-1.0, interval[0] - gamma), min(1.0, interval[1] + gamma)]
        drift[f"{gamma:g}"] = {"interval": widened, "action": action(widened)}
    recomputed = {
        **point,
        "selected_roots": hierarchy["selected_roots"],
        "root_prompt_order_cells": hierarchy["root_prompt_order_cells"],
        "model_jury_calls": hierarchy["model_jury_calls"],
        "human_target_units": hierarchy["human_target_units"],
        "interval_type": interval_type,
        "missing_slots": missing,
        "interval": interval,
        "tie_interval": tie_interval,
        "action": action(interval),
        "response_drift": drift,
        "maximum_weight": max(weights),
        "effective_sample_size": sum(weights) ** 2 / sum(w * w for w in weights),
    }
    return recomputed


def validate_bundle(
    bundle: dict[str, Any], preregistration: dict[str, Any]
) -> dict[str, Any]:
    recomputed = recompute(bundle, preregistration)
    compare_reported(bundle.get("reported"), recomputed)
    return {
        "schema_version": 1,
        "execution_class": bundle["execution_class"],
        "empirical_evidence": bundle["execution_class"] == "real",
        "bundle_sha256": canonical_digest(bundle),
        "slots": len(bundle["slots"]),
        "batches": len(bundle["batches"]),
        "recomputed": recomputed,
    }


def compare_reported(reported: dict[str, Any], recomputed: dict[str, Any]) -> None:
    require(set(reported) == set(recomputed), "reported fields differ from the contract")

    def walk(a: Any, b: Any, path: str) -> None:
        if isinstance(b, dict):
            require(isinstance(a, dict) and set(a) == set(b), f"{path} keys differ")
            for key in b:
                walk(a[key], b[key], f"{path}.{key}")
        elif isinstance(b, list):
            require(isinstance(a, list) and len(a) == len(b), f"{path} length differs")
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{path}[{i}]")
        elif isinstance(b, float):
            require(
                isinstance(a, (int, float)) and close(float(a), b),
                f"{path} does not reproduce",
            )
        else:
            require(a == b, f"{path} does not reproduce")

    walk(reported, recomputed, "reported")


def fraction_draw_probabilities(
    batch: dict[str, Any],
    mass: dict[str, Fraction],
    eps: Fraction,
    integer_scores: dict[str, int],
) -> dict[str, Fraction]:
    """Independent exact law for fixtures whose exp(g) values are integers."""
    if batch["policy"] == "target_random":
        return dict(mass)
    weights = {
        unit: count * integer_scores[unit]
        for unit, count in batch["candidate_multiset"].items()
    }
    total = sum(weights.values())
    return {
        unit: eps * mass[unit] + (1 - eps) * Fraction(weights.get(unit, 0), total)
        for unit in mass
    }


def build_fixture_hierarchy() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build an 8-unit frame hierarchy whose product masses equal raw/16."""
    source_specs = (
        ("source-a", "rev-a", (("v0", 0.25), ("v1", 0.75))),
        ("source-b", "rev-b", (("v0", 0.50), ("v1", 0.50))),
    )
    sources: dict[str, Any] = {}
    selected_root_uids: list[str] = []
    cells: list[dict[str, Any]] = []
    for source_key, revision, variants in source_specs:
        raw_candidates = [
            {
                "question_id": f"{source_key}-q{idx}",
                "row_uid": f"{source_key}-r{idx}",
                "selection_sha256": root_selection_sha256(revision, f"{source_key}-q{idx}"),
            }
            for idx in range(2)
        ]
        raw_candidates.sort(key=lambda item: (item["selection_sha256"], item["question_id"]))
        candidate_roots = []
        for rank, cand in enumerate(raw_candidates):
            selected = rank == 0
            if selected:
                selected_root_uids.append(cand["row_uid"])
                for vid, v_weight in variants:
                    for order in (-1, 1):
                        cells.append(
                            {
                                "row_uid": cand["row_uid"],
                                "variant_id": vid,
                                "display_order": order,
                                "cell_mass": 0.5 * v_weight * 0.5,
                            }
                        )
            candidate_roots.append(
                {
                    **cand,
                    "hash_rank": rank,
                    "selected": selected,
                    "target_mass": 0.5 if selected else 0.0,
                    "prompt_variants": [
                        {"variant_id": vid, "weight": v_weight}
                        for vid, v_weight in variants
                    ]
                    if selected
                    else [],
                }
            )
        sources[source_key] = {
            "revision": revision,
            "selected_count": 1,
            "candidate_roots": candidate_roots,
        }
    r0, r1 = selected_root_uids
    unit_map = [
        ("u0", r0, "v0", "h0", -1, 1 / 16),
        ("u1", r1, "v0", "h0", -1, 2 / 16),
        ("u2", r0, "v1", "h0", -1, 3 / 16),
        ("u3", r1, "v0", "h0", 1, 2 / 16),
        ("u4", r0, "v0", "h0", 1, 1 / 16),
        ("u5", r0, "v1", "h0", 1, 3 / 16),
        ("u6", r1, "v1", "h0", -1, 2 / 16),
        ("u7", r1, "v1", "h0", 1, 2 / 16),
    ]
    frame = [
        {
            "unit_id": uid,
            "row_uid": root_uid,
            "variant_id": vid,
            "rater_id": rater_id,
            "display_order": order,
            "target_mass": target_mass,
        }
        for uid, root_uid, vid, rater_id, order, target_mass in unit_map
    ]
    hierarchy = {
        "sources": sources,
        "prompt_variants_per_root": 2,
        "display_orders": [-1, 1],
        "jury_model_keys": [
            "qwen25-instruct-1p5b",
            "smollm2-instruct-1p7b",
            "mistral-instruct-7b",
            "phi3-mini-instruct",
        ],
        "raters": [{"rater_id": "h0", "target_weight": 1.0}],
        "cells": cells,
    }
    return hierarchy, frame


def build_pilot_frame_hierarchy(section: dict[str, Any]) -> dict[str, Any]:
    """Build the full 528-root / 2,112-cell / 8,448-jury-call / 168,960-unit hierarchy."""
    frame_spec = section["frame"]
    expected_sources = frame_spec["root_items_by_source"]
    variants_per_root = frame_spec["prompt_variants_per_root"]
    panel_size = section["human_protocol"]["panel_size"]
    n_sources = len(expected_sources)
    sources: dict[str, Any] = {}
    cells: list[dict[str, Any]] = []
    for source_key, selected_count in sorted(expected_sources.items()):
        revision = f"pinned-rev-{source_key}"
        pool_count = selected_count + (50 if source_key == "livebench-math-eval" else 0)
        raw_candidates = [
            {
                "question_id": f"{source_key}-q{idx:04d}",
                "row_uid": f"{source_key}-root-{idx:04d}",
                "selection_sha256": root_selection_sha256(
                    revision, f"{source_key}-q{idx:04d}"
                ),
            }
            for idx in range(pool_count)
        ]
        raw_candidates.sort(key=lambda item: (item["selection_sha256"], item["question_id"]))
        root_mass = 1.0 / (n_sources * selected_count)
        candidate_roots = []
        for rank, cand in enumerate(raw_candidates):
            selected = rank < selected_count
            variants = (
                [
                    {"variant_id": f"v{v}", "weight": 1.0 / variants_per_root}
                    for v in range(variants_per_root)
                ]
                if selected
                else []
            )
            if selected:
                for variant in variants:
                    for order in (-1, 1):
                        cells.append(
                            {
                                "row_uid": cand["row_uid"],
                                "variant_id": variant["variant_id"],
                                "display_order": order,
                                "cell_mass": root_mass * variant["weight"] / 2.0,
                            }
                        )
            candidate_roots.append(
                {
                    **cand,
                    "hash_rank": rank,
                    "selected": selected,
                    "target_mass": root_mass if selected else 0.0,
                    "prompt_variants": variants,
                }
            )
        sources[source_key] = {
            "revision": revision,
            "selected_count": selected_count,
            "candidate_roots": candidate_roots,
        }
    return {
        "sources": sources,
        "prompt_variants_per_root": variants_per_root,
        "display_orders": [-1, 1],
        "jury_model_keys": list(section["jury_model_keys"]),
        "raters": [
            {"rater_id": f"rater-{h:03d}", "target_weight": 1.0 / panel_size}
            for h in range(panel_size)
        ],
        "cells": cells,
    }


def build_fixture(missing_slots: tuple[int, ...] = ()) -> tuple[dict[str, Any], dict]:
    """Generated golden bundle whose exp(g) values are integers."""
    hierarchy, frame = build_fixture_hierarchy()
    units = [row["unit_id"] for row in frame]
    raw = [1, 2, 3, 2, 1, 3, 2, 2]
    mass_fraction = {u: Fraction(r, sum(raw)) for u, r in zip(units, raw)}
    config = {
        "exploration_epsilon": 0.25,
        "evi_temperature": 2.0,
        "batch_size": 10,
        "fixed_budget_attempts": 40,
        "evi_candidate_multiset_size": 6,
        "alpha": 0.05,
        "normal_quantile": 1.959963984540054,
        "practical_margin": 0.05,
        "response_drift_gamma_grid": [0.0, 0.02, 0.05],
    }
    eps_fraction = Fraction(1, 4)
    response = {
        u: (0.45 + 0.05 * (i % 3), 0.2 + 0.03 * (i % 2)) for i, u in enumerate(units)
    }
    rng = random.Random(2026092901)
    batches: list[dict[str, Any]] = []
    slots: list[dict[str, Any]] = []
    exact_rows: list[tuple[Fraction, Fraction, Fraction, Fraction, int]] = []
    for index in range(4):
        start = index * config["batch_size"]
        policy = "target_random" if index == 0 else "evi_candidate_mixture"
        multiset: dict[str, int] = {}
        integer_scores: dict[str, int] = {}
        if policy == "evi_candidate_mixture":
            for _ in range(config["evi_candidate_multiset_size"]):
                unit = rng.choices(units, weights=raw)[0]
                multiset[unit] = multiset.get(unit, 0) + 1
            integer_scores = {u: 1 + (int(u[1:]) + index) % 4 for u in multiset}
        batch = {
            "batch": index,
            "policy": policy,
            "slot_start": start,
            "slot_end": start + config["batch_size"],
            "candidate_multiset": multiset,
            "evi": {
                u: math.log(score) / config["evi_temperature"]
                for u, score in integer_scores.items()
            },
            "annotation_cost": {u: 1.0 for u in multiset},
            "predictions": {
                u: 0.0 if index == 0 else round(0.1 * ((i + index) % 5) - 0.2, 3)
                for i, u in enumerate(units)
            },
            "tie_predictions": {
                u: 0.0 if index == 0 else 0.2 for u in units
            },
            "fit_slots": list(range(start)),
        }
        batch["commitment_sha256"] = canonical_digest(commitment_payload(batch))
        batches.append(batch)
        law = fraction_draw_probabilities(batch, mass_fraction, eps_fraction, integer_scores)
        require(sum(law.values()) == 1, "fixture Fraction law does not normalize")
        pred = {u: Fraction(str(v)) for u, v in batch["predictions"].items()}
        tie_pred = {u: Fraction(str(v)) for u, v in batch["tie_predictions"].items()}
        m_bar = sum(mass_fraction[u] * pred[u] for u in units)
        tie_bar = sum(mass_fraction[u] * tie_pred[u] for u in units)
        for slot in range(start, start + config["batch_size"]):
            unit = rng.choices(units, weights=[float(law[u]) for u in units])[0]
            win, tie = response[unit]
            draw = rng.random()
            outcome = 0 if draw < tie else (1 if draw < tie + win * (1 - tie) else -1)
            observed = slot not in missing_slots
            slots.append(
                {
                    "slot": slot,
                    "batch": index,
                    "batch_commitment_sha256": batch["commitment_sha256"],
                    "unit_id": unit,
                    "logged_probability": float(law[unit]),
                    "observed": observed,
                    "outcome": outcome if observed else None,
                }
            )
            weight = mass_fraction[unit] / law[unit]
            psi = m_bar + weight * (outcome - pred[unit])
            exact_rows.append(
                {
                    "psi": psi,
                    "tie": tie_bar + weight * (int(outcome == 0) - tie_pred[unit]),
                    "weight": weight,
                    "lower": psi if observed else m_bar + weight * (-1 - pred[unit]),
                    "upper": psi if observed else m_bar + weight * (1 - pred[unit]),
                }
            )
    bundle = {
        "schema_version": 1,
        "execution_class": "fixture",
        "empirical_evidence": False,
        "preregistration_section_sha256": None,
        "config": config,
        "frame_hierarchy": hierarchy,
        "frame": frame,
        "batches": batches,
        "slots": slots,
        "reported": {},
    }
    return bundle, {"rows": exact_rows}


def fill_reported(bundle: dict[str, Any], preregistration: dict[str, Any]) -> None:
    bundle["reported"] = recompute(bundle, preregistration)


def expect_failure(
    name: str, bundle: dict[str, Any], preregistration: dict[str, Any]
) -> None:
    try:
        validate_bundle(bundle, preregistration)
    except BundleError:
        return
    raise BundleError(f"mutation {name!r} was accepted")


def self_test(preregistration: dict[str, Any]) -> dict[str, Any]:
    results = {}
    for label, missing in (("complete", ()), ("one_missing_slot", (17,))):
        bundle, exact = build_fixture(missing)
        fill_reported(bundle, preregistration)
        summary = validate_bundle(bundle, preregistration)
        recomputed = summary["recomputed"]
        rows = exact["rows"]
        count = len(rows)
        weights = [row["weight"] for row in rows]
        ess = float(sum(weights) ** 2 / sum(w * w for w in weights))
        require(close(recomputed["effective_sample_size"], ess), "ESS differs from Fraction")
        require(
            close(recomputed["maximum_weight"], float(max(weights))),
            "maximum weight differs from Fraction",
        )
        if not missing:
            estimate = sum(row["psi"] for row in rows) / count
            variance = sum((row["psi"] - estimate) ** 2 for row in rows) / (count - 1)
            radius = 1.959963984540054 * math.sqrt(float(variance) / count)
            require(
                close(recomputed["estimate"], float(estimate))
                and close(recomputed["variance"], float(variance))
                and close(recomputed["interval"][0], max(-1.0, float(estimate) - radius))
                and close(recomputed["interval"][1], min(1.0, float(estimate) + radius)),
                "complete-case estimate or interval differs from Fraction",
            )
            require(
                close(
                    recomputed["tie_estimate"],
                    float(sum(row["tie"] for row in rows) / count),
                ),
                "tie estimate differs from the independent Fraction computation",
            )
            require(recomputed["interval_type"] == "studentized_complete", "wrong interval")
        else:
            require(
                close(
                    recomputed["estimate_lower"],
                    float(sum(row["lower"] for row in rows) / count),
                )
                and close(
                    recomputed["estimate_upper"],
                    float(sum(row["upper"] for row in rows) / count),
                ),
                "worst-case endpoint means differ from Fraction",
            )
            require(
                recomputed["interval_type"] == "martingale_hoeffding_worst_case"
                and recomputed["action"] == "not-established",
                "a missing slot must force the worst-case interval and abstention here",
            )
        results[label] = summary

    base, _ = build_fixture()
    fill_reported(base, preregistration)
    mutations = {}

    def mutate(name: str, change) -> None:
        candidate = copy.deepcopy(base)
        change(candidate)
        expect_failure(name, candidate, preregistration)
        mutations[name] = "rejected"

    mutate("tampered_logged_probability", lambda b: b["slots"][15].__setitem__(
        "logged_probability", b["slots"][15]["logged_probability"] * 1.01))
    mutate("evi_changed_after_commitment", lambda b: b["batches"][2]["evi"].__setitem__(
        next(iter(b["batches"][2]["evi"])), 0.9))
    def fit_on_current_batch(candidate: dict[str, Any]) -> None:
        batch = candidate["batches"][1]
        batch["fit_slots"].append(12)
        batch["commitment_sha256"] = canonical_digest(commitment_payload(batch))
        for row in candidate["slots"]:
            if row["batch"] == 1:
                row["batch_commitment_sha256"] = batch["commitment_sha256"]

    mutate("predictor_fit_on_current_batch", fit_on_current_batch)
    mutate("dropped_slot", lambda b: b["slots"].pop(23))
    mutate("replacement_slot_added", lambda b: b["slots"].append(
        {**b["slots"][23], "outcome": 1}))
    mutate("outcome_on_missing_slot", lambda b: b["slots"][5].__setitem__("observed", False))
    mutate("reported_action_altered", lambda b: b["reported"].__setitem__(
        "action", "A" if b["reported"]["action"] != "A" else "B"))
    mutate("reported_interval_altered", lambda b: b["reported"]["interval"].__setitem__(
        0, b["reported"]["interval"][0] + 1e-6))
    mutate("unit_outside_frame", lambda b: b["slots"][3].__setitem__("unit_id", "u99"))
    mutate("fixture_marked_empirical", lambda b: b.__setitem__("empirical_evidence", True))
    mutate("real_class_with_null_artifacts", lambda b: b.update(
        {"execution_class": "real", "empirical_evidence": True,
         "preregistration_section_sha256": canonical_digest(preregistration["eval"])}))

    missing_bundle, _ = build_fixture((17,))
    fill_reported(missing_bundle, preregistration)
    complete_claim = copy.deepcopy(missing_bundle)
    complete_claim["reported"] = copy.deepcopy(base["reported"])
    expect_failure("complete_case_interval_despite_missing", complete_claim, preregistration)
    mutations["complete_case_interval_despite_missing"] = "rejected"

    mutate(
        "tampered_root_selection_hash",
        lambda b: b["frame_hierarchy"]["sources"]["source-a"]["candidate_roots"][0].__setitem__(
            "selection_sha256", "0" * 64
        ),
    )

    def swap_selected_rank(b: dict[str, Any]) -> None:
        roots = b["frame_hierarchy"]["sources"]["source-a"]["candidate_roots"]
        roots[0]["selected"] = False
        roots[0]["target_mass"] = 0.0
        roots[1]["selected"] = True
        roots[1]["target_mass"] = 0.5
        roots[1]["prompt_variants"] = roots[0]["prompt_variants"]

    mutate("wrong_root_hash_rank_selected", swap_selected_rank)

    mutate(
        "variant_weights_not_normalized",
        lambda b: b["frame_hierarchy"]["sources"]["source-a"]["candidate_roots"][0][
            "prompt_variants"
        ][0].__setitem__("weight", 0.30),
    )
    mutate(
        "incomplete_root_prompt_order_cells",
        lambda b: b["frame_hierarchy"]["cells"].pop(),
    )
    mutate(
        "unit_mass_disagrees_with_cell_rater_product",
        lambda b: b["frame"][0].__setitem__("target_mass", b["frame"][0]["target_mass"] + 0.01),
    )

    section = preregistration["eval"]
    frame_spec = section["frame"]
    pilot_hierarchy = recompute_frame_hierarchy(
        build_pilot_frame_hierarchy(section),
        expected_sources=frame_spec["root_items_by_source"],
        expected_variants=frame_spec["prompt_variants_per_root"],
        expected_orders=frame_spec["display_orders"],
        expected_panel_size=section["human_protocol"]["panel_size"],
        expected_jury_keys=section["jury_model_keys"],
    )
    require(
        pilot_hierarchy["selected_roots"] == frame_spec["root_items"]
        and pilot_hierarchy["root_prompt_order_cells"] == frame_spec["root_prompt_order_cells"]
        and pilot_hierarchy["model_jury_calls"] == frame_spec["model_jury_calls"]
        and pilot_hierarchy["human_target_units"] == frame_spec["human_target_units"],
        "528-root pilot hierarchy did not reproduce 2,112 cells, 8,448 jury calls, and 168,960 units",
    )

    return {
        "schema_version": 1,
        "audit_type": "eval_sequential_bundle_golden_fixture",
        "empirical_evidence": False,
        "pilot_frame_hierarchy": {
            "selected_roots": pilot_hierarchy["selected_roots"],
            "root_prompt_order_cells": pilot_hierarchy["root_prompt_order_cells"],
            "model_jury_calls": pilot_hierarchy["model_jury_calls"],
            "human_target_units": pilot_hierarchy["human_target_units"],
        },
        "fixtures": {
            label: {
                "bundle_sha256": summary["bundle_sha256"],
                "interval_type": summary["recomputed"]["interval_type"],
                "action": summary["recomputed"]["action"],
            }
            for label, summary in results.items()
        },
        "mutations_rejected": len(mutations),
        "mutations": mutations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    preregistration = json.loads(args.preregistration.read_text(encoding="utf-8"))
    try:
        if args.self_test:
            summary = self_test(preregistration)
        elif args.bundle is not None:
            summary = validate_bundle(
                json.loads(args.bundle.read_text(encoding="utf-8")), preregistration
            )
        else:
            parser.error("--bundle or --self-test is required")
    except BundleError as exc:
        raise SystemExit(f"error: {exc}") from exc
    if args.output is not None:
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if args.self_test:
        print(
            "validated evaluation sequential bundle recomputation; "
            f"{summary['mutations_rejected']} mutations rejected"
        )
    else:
        print(f"validated bundle with action {summary['recomputed']['action']}")


if __name__ == "__main__":
    main()
