#!/usr/bin/env python3
"""Run deterministic, CPU-only design audits for the three paper protocols.

These audits exercise the estimators and experimental ledgers; they are not
substitutes for the model-training or human-subject experiments in the papers.
Each invocation implements one stage of the shared five-stage protocol and
writes the artifact expected by ``run_experiment_pipeline.py``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Iterable


PAPERS = ("sft", "rl", "eval")
STAGES = ("acquire", "process", "build", "train", "evaluate")
ARTIFACTS = {
    "sft": {
        "acquire": "data_manifest.json",
        "process": "splits_manifest.json",
        "build": "model_manifest.json",
        "train": "run_ledger.jsonl",
        "evaluate": "summary.json",
    },
    "rl": {
        "acquire": "data_manifest.json",
        "process": "trajectory_schema.json",
        "build": "model_manifest.json",
        "train": "trajectory_ledger.jsonl",
        "evaluate": "summary.json",
    },
    "eval": {
        "acquire": "frame_manifest.json",
        "process": "cell_manifest.json",
        "build": "evaluation_model_manifest.json",
        "train": "acquisition_ledger.jsonl",
        "evaluate": "summary.json",
    },
}
COST_KEYS = {
    "sft": {
        "acquire": ("cpu_seconds", "wall_seconds", "bytes_downloaded"),
        "process": ("cpu_seconds", "wall_seconds", "accelerator_seconds"),
        "build": ("cpu_seconds", "wall_seconds", "accelerator_seconds"),
        "train": (
            "cpu_seconds",
            "wall_seconds",
            "accelerator_seconds",
            "scored_examples",
            "accepted_updates",
            "rejected_updates",
        ),
        "evaluate": (
            "cpu_seconds",
            "wall_seconds",
            "accelerator_seconds",
            "evaluated_checkpoints",
        ),
    },
    "rl": {
        "acquire": ("cpu_seconds", "wall_seconds", "bytes_downloaded"),
        "process": (
            "cpu_seconds",
            "wall_seconds",
            "accelerator_seconds",
            "verifier_calls",
        ),
        "build": ("cpu_seconds", "wall_seconds", "accelerator_seconds"),
        "train": (
            "cpu_seconds",
            "wall_seconds",
            "accelerator_seconds",
            "generated_tokens",
            "verifier_calls",
            "likelihood_tokens",
        ),
        "evaluate": (
            "cpu_seconds",
            "wall_seconds",
            "accelerator_seconds",
            "evaluated_checkpoints",
            "verifier_calls",
        ),
    },
    "eval": {
        "acquire": ("cpu_seconds", "wall_seconds", "annotation_currency"),
        "process": (
            "cpu_seconds",
            "wall_seconds",
            "accelerator_seconds",
            "jury_calls",
        ),
        "build": (
            "cpu_seconds",
            "wall_seconds",
            "accelerator_seconds",
            "jury_calls",
        ),
        "train": (
            "cpu_seconds",
            "wall_seconds",
            "accelerator_seconds",
            "assignment_attempts",
            "completed_ratings",
            "nonresponses",
            "jury_calls",
            "annotation_currency",
        ),
        "evaluate": (
            "cpu_seconds",
            "wall_seconds",
            "assignment_attempts",
            "completed_ratings",
            "nonresponses",
            "jury_calls",
            "annotation_currency",
        ),
    },
}


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        fail(f"required prior-stage artifact is missing: {path}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    except FileNotFoundError:
        fail(f"required prior-stage artifact is missing: {path}")


def categorical_index(rng: random.Random, probabilities: list[float]) -> int:
    draw = rng.random()
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if draw < cumulative:
            return index
    return len(probabilities) - 1


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def mse(values: list[float], target: float) -> float:
    return mean([(value - target) ** 2 for value in values])


def monte_carlo_interval(values: list[float]) -> list[float]:
    center = mean(values)
    half_width = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return [center - half_width, center + half_width]


def stage_artifact(work_dir: Path, paper: str, stage: str) -> Path:
    return work_dir / stage / ARTIFACTS[paper][stage]


def prior_artifact(work_dir: Path, paper: str, stage: str) -> Path:
    index = STAGES.index(stage)
    if index == 0:
        fail("acquire has no prior artifact")
    prior = STAGES[index - 1]
    return stage_artifact(work_dir, paper, prior)


def pack_bins(lengths: list[int], capacity: int, max_bins: int) -> list[list[int]] | None:
    """Return an exact feasible packing, using symmetry-pruned backtracking."""
    order = sorted(range(len(lengths)), key=lambda index: lengths[index], reverse=True)
    bins: list[list[int]] = []
    loads: list[int] = []

    def search(position: int) -> bool:
        if position == len(order):
            return True
        item = order[position]
        length = lengths[item]
        seen_loads: set[int] = set()
        for bin_index, load in enumerate(loads):
            if load in seen_loads or load + length > capacity:
                continue
            seen_loads.add(load)
            loads[bin_index] += length
            bins[bin_index].append(item)
            if search(position + 1):
                return True
            bins[bin_index].pop()
            loads[bin_index] -= length
        if len(bins) < max_bins:
            bins.append([item])
            loads.append(length)
            if search(position + 1):
                return True
            bins.pop()
            loads.pop()
        return False

    return bins if search(0) else None


def sft_best_subset(
    items: list[dict[str, Any]],
    feasible: Callable[[list[dict[str, Any]]], bool],
) -> list[dict[str, Any]]:
    best: list[dict[str, Any]] = []
    best_key = (-math.inf, -math.inf, ())
    for size in range(1, len(items) + 1):
        for indices in itertools.combinations(range(len(items)), size):
            subset = [items[index] for index in indices]
            if not feasible(subset):
                continue
            key = (
                sum(item["utility"] for item in subset),
                -sum(item["length"] for item in subset),
                tuple(item["id"] for item in subset),
            )
            if key > best_key:
                best_key = key
                best = subset
    return best


def sft_pack(subset: list[dict[str, Any]], config: dict[str, Any]) -> list[list[int]] | None:
    return pack_bins(
        [item["length"] for item in subset],
        config["pack_capacity"],
        config["max_packs"],
    )


def sft_select(items: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    risk_limit = config["planning_risk_limit"]
    total_capacity = config["pack_capacity"] * config["max_packs"]

    def common(subset: list[dict[str, Any]]) -> bool:
        return sum(item["protected_risk"] for item in subset) <= risk_limit

    additive = sft_best_subset(
        items,
        lambda subset: common(subset)
        and sum(item["length"] for item in subset) <= total_capacity,
    )
    proposed_additive = list(additive)
    while additive and sft_pack(additive, config) is None:
        removed = min(
            additive,
            key=lambda item: (item["utility"] / item["length"], item["utility"]),
        )
        additive = [item for item in additive if item["id"] != removed["id"]]

    joint = sft_best_subset(
        items,
        lambda subset: common(subset) and sft_pack(subset, config) is not None,
    )
    profiled = sft_best_subset(
        items,
        lambda subset: common(subset)
        and (
            sft_pack(subset, config) is not None
        ),
    )

    def result(subset: list[dict[str, Any]]) -> dict[str, Any]:
        risk = sum(item["protected_risk"] for item in subset)
        count = len(subset)
        scales = config["gate_scales"]
        accepted_scale = 0.0
        for scale in scales:
            actual_change = scale * risk + config["curvature"] * scale**2 * count**2
            if actual_change <= config["gate_tolerance"]:
                accepted_scale = scale
                break
        return {
            "ids": [item["id"] for item in subset],
            "utility": sum(item["utility"] for item in subset),
            "protected_proxy": risk,
            "pack_count": len(sft_pack(subset, config) or []),
            "accepted_scale": accepted_scale,
        }

    return {
        "additive_proposal_ids": [item["id"] for item in proposed_additive],
        "additive_proposal_packable": sft_pack(proposed_additive, config) is not None,
        "score_then_pack": result(additive),
        "joint": result(joint),
        "profiled": result(profiled),
    }


def run_sft(stage: str, work_dir: Path) -> dict[str, float]:
    artifact = stage_artifact(work_dir, "sft", stage)
    if stage == "acquire":
        items = [
            {"id": "long-a", "length": 6, "utility": 10.0, "protected_risk": 1.0},
            {"id": "long-b", "length": 6, "utility": 9.0, "protected_risk": 1.0},
            {"id": "long-c", "length": 6, "utility": 8.0, "protected_risk": 1.0},
            {"id": "short-a", "length": 4, "utility": 3.5, "protected_risk": 0.0},
            {"id": "short-b", "length": 4, "utility": 3.0, "protected_risk": 0.0},
            {"id": "medium", "length": 5, "utility": 4.0, "protected_risk": 2.0},
        ]
        write_json(
            artifact,
            {
                "audit_type": "synthetic_design_audit",
                "empirical_evidence": False,
                "seed": 20260902,
                "license": "generated mathematical fixture; no external data",
                "items": items,
            },
        )
        return {"bytes_downloaded": 0.0}

    previous = (
        None
        if stage == "evaluate"
        else read_json(prior_artifact(work_dir, "sft", stage))
    )
    if stage == "process":
        items = previous["items"]
        unique_ids = len({item["id"] for item in items}) == len(items)
        positive_lengths = all(item["length"] > 0 for item in items)
        if not unique_ids or not positive_lengths:
            fail("invalid synthetic SFT candidate fixture")
        write_json(
            artifact,
            {
                "audit_type": previous["audit_type"],
                "checks": {
                    "unique_ids": unique_ids,
                    "positive_lengths": positive_lengths,
                    "isolated_gradient_invariance": "exact by fixture construction",
                },
                "items": items,
            },
        )
        return {"accelerator_seconds": 0.0}

    if stage == "build":
        config = {
            "pack_capacity": 10,
            "max_packs": 2,
            "planning_risk_limit": 3.0,
            "gate_tolerance": 3.2,
            "curvature": 0.2,
            "gate_scales": [1.0, 0.5, 0.25, 0.125],
            "trials": 512,
            "seed": 20260902,
            "utility_noise_sd": 0.35,
        }
        write_json(
            artifact,
            {
                "audit_type": previous["audit_type"],
                "empirical_evidence": False,
                "config": config,
                "items": previous["items"],
                "assertions": [
                    "joint selection equals exact selection under profiled pack cost",
                    "all committed synthetic steps pass the actual-change gate",
                ],
            },
        )
        return {"accelerator_seconds": 0.0}

    if stage == "train":
        config = previous["config"]
        base_items = previous["items"]
        rng = random.Random(config["seed"])
        rows = []
        accepted = 0
        rejected = 0
        for trial in range(config["trials"]):
            items = [
                {
                    **item,
                    "utility": item["utility"]
                    + rng.gauss(0.0, config["utility_noise_sd"]),
                }
                for item in base_items
            ]
            selection = sft_select(items, config)
            if selection["joint"]["accepted_scale"] > 0:
                accepted += 1
            else:
                rejected += 1
            rows.append(
                {
                    "trial": trial,
                    "synthetic": True,
                    **selection,
                }
            )
        write_jsonl(artifact, rows)
        return {
            "accelerator_seconds": 0.0,
            "scored_examples": float(config["trials"] * len(base_items)),
            "accepted_updates": float(accepted),
            "rejected_updates": float(rejected),
        }

    rows = read_jsonl(prior_artifact(work_dir, "sft", stage))
    equivalence = all(row["joint"] == row["profiled"] for row in rows)
    if not equivalence:
        fail("joint and exact profiled-cost selections diverged")
    summary = {
        "audit_type": "synthetic_design_audit",
        "empirical_evidence": False,
        "trials": len(rows),
        "checks": {
            "profiled_cost_equivalence": equivalence,
            "all_joint_plans_packable": all(row["joint"]["pack_count"] <= 2 for row in rows),
            "gate_always_found_safe_scale": all(
                row["joint"]["accepted_scale"] > 0 for row in rows
            ),
            "constructed_nonadditive_case_exercised": any(
                not row["additive_proposal_packable"] for row in rows
            )
            and any(
                row["joint"]["utility"] > row["score_then_pack"]["utility"]
                for row in rows
            ),
        },
        "diagnostics": {
            "infeasible_additive_proposal_rate": mean(
                [not row["additive_proposal_packable"] for row in rows]
            ),
            "mean_joint_minus_repaired_utility": mean(
                [
                    row["joint"]["utility"] - row["score_then_pack"]["utility"]
                    for row in rows
                ]
            ),
            "joint_full_step_acceptance_rate": mean(
                [row["joint"]["accepted_scale"] == 1.0 for row in rows]
            ),
        },
        "interpretation": (
            "Software/design check only; these values must not populate the paper's "
            "model-training result tables."
        ),
    }
    write_json(artifact, summary)
    return {"accelerator_seconds": 0.0, "evaluated_checkpoints": float(len(rows))}


def rl_cells() -> list[dict[str, float | int]]:
    d = [0.6, 0.4]
    q = [0.2, 0.8]
    pi = [0.72, 0.31]
    mu = [0.38, 0.79]
    dual = 0.12
    cells: list[dict[str, float | int]] = []
    for x in range(2):
        for y in range(2):
            p_action = pi[x] if y == 1 else 1.0 - pi[x]
            g_action = mu[x] if y == 1 else 1.0 - mu[x]
            reward = float(y if x == 0 else 1 - y)
            length = float(1 + y + x)
            utility = reward - dual * length
            score = float(y) - pi[x]
            cells.append(
                {
                    "x": x,
                    "y": y,
                    "p": d[x] * p_action,
                    "g": q[x] * g_action,
                    "pi_action": p_action,
                    "mu_action": g_action,
                    "utility": utility,
                    "length": length,
                    "score": score,
                }
            )
    return cells


def rl_enrich(cells: list[dict[str, Any]], alpha: float) -> list[dict[str, Any]]:
    enriched = []
    for cell in cells:
        mixture = alpha * cell["p"] + (1.0 - alpha) * cell["g"]
        exact_weight = cell["p"] / mixture
        conditional_denominator = (
            alpha * cell["pi_action"] + (1.0 - alpha) * cell["mu_action"]
        )
        wrong_weight = cell["pi_action"] / conditional_denominator
        h = 0.72 * cell["utility"] + 0.08 * (1.0 - 2.0 * cell["x"])
        enriched.append(
            {
                **cell,
                "mixture": mixture,
                "weight": exact_weight,
                "wrong_weight": wrong_weight,
                "h": h,
            }
        )
    for x in range(2):
        conditional = [cell for cell in enriched if cell["x"] == x]
        mass = sum(cell["p"] for cell in conditional)
        baseline = sum(cell["p"] * cell["utility"] for cell in conditional) / mass
        for cell in conditional:
            cell["baseline"] = baseline
    return enriched


def rl_target(cells: list[dict[str, Any]]) -> float:
    return sum(cell["p"] * cell["utility"] * cell["score"] for cell in cells)


def rl_expectation(
    cells: list[dict[str, Any]],
    alpha: float,
    weight_key: str,
    clip: float | None = None,
) -> float:
    def weight(cell: dict[str, Any]) -> float:
        value = cell[weight_key]
        return min(value, clip) if clip is not None else value

    direct = sum(
        cell["p"] * (cell["h"] - cell["baseline"]) * cell["score"]
        for cell in cells
    )
    residual_p = alpha * sum(
        cell["p"]
        * weight(cell)
        * (cell["utility"] - cell["h"])
        * cell["score"]
        for cell in cells
    )
    residual_g = (1.0 - alpha) * sum(
        cell["g"]
        * weight(cell)
        * (cell["utility"] - cell["h"])
        * cell["score"]
        for cell in cells
    )
    return direct + residual_p + residual_g


def rl_draw(rng: random.Random, cells: list[dict[str, Any]], law: str) -> dict[str, Any]:
    index = categorical_index(rng, [cell[law] for cell in cells])
    return cells[index]


def rl_estimate(
    rng: random.Random,
    cells: list[dict[str, Any]],
    n_p: int,
    n_g: int,
    alpha: float,
    weight_key: str,
    clip: float | None = None,
) -> tuple[float, float]:
    p_terms = []
    g_terms = []
    tokens = 0.0
    for _ in range(n_p):
        cell = rl_draw(rng, cells, "p")
        weight = cell[weight_key]
        if clip is not None:
            weight = min(weight, clip)
        p_terms.append(
            (cell["h"] - cell["baseline"]) * cell["score"]
            + alpha * weight * (cell["utility"] - cell["h"]) * cell["score"]
        )
        tokens += cell["length"]
    for _ in range(n_g):
        cell = rl_draw(rng, cells, "g")
        weight = cell[weight_key]
        if clip is not None:
            weight = min(weight, clip)
        g_terms.append(weight * (cell["utility"] - cell["h"]) * cell["score"])
        tokens += cell["length"]
    estimate = mean(p_terms) + (1.0 - alpha) * mean(g_terms)
    return estimate, tokens


def run_rl(stage: str, work_dir: Path) -> dict[str, float]:
    artifact = stage_artifact(work_dir, "rl", stage)
    if stage == "acquire":
        cells = rl_cells()
        write_json(
            artifact,
            {
                "audit_type": "finite_support_estimator_audit",
                "empirical_evidence": False,
                "seed": 20260902,
                "license": "generated finite probability model; no external data",
                "cells": cells,
            },
        )
        return {"bytes_downloaded": 0.0}

    previous = (
        None
        if stage == "evaluate"
        else read_json(prior_artifact(work_dir, "rl", stage))
    )
    if stage == "process":
        cells = previous["cells"]
        checks = {
            "target_mass": sum(cell["p"] for cell in cells),
            "stale_mass": sum(cell["g"] for cell in cells),
            "all_outcomes_retained": len(cells) == 4,
        }
        if abs(checks["target_mass"] - 1.0) > 1e-12:
            fail("target trajectory law does not normalize")
        if abs(checks["stale_mass"] - 1.0) > 1e-12:
            fail("stale trajectory law does not normalize")
        write_json(
            artifact,
            {
                "audit_type": previous["audit_type"],
                "decoder": "finite Bernoulli sampler with exactly scored masses",
                "checks": checks,
                "cells": cells,
            },
        )
        return {"accelerator_seconds": 0.0, "verifier_calls": 4.0}

    if stage == "build":
        config = {
            "alpha": 0.5,
            "n_current": 32,
            "n_stale": 32,
            "replications": 2000,
            "seed": 20260902,
            "fault_clip": 1.2,
        }
        cells = rl_enrich(previous["cells"], config["alpha"])
        target = rl_target(cells)
        exact_expectation = rl_expectation(cells, config["alpha"], "weight")
        wrong_expectation = rl_expectation(cells, config["alpha"], "wrong_weight")
        clipped_expectation = rl_expectation(
            cells, config["alpha"], "weight", config["fault_clip"]
        )
        if abs(exact_expectation - target) > 1e-12:
            fail("augmented-mixture expectation identity failed")
        write_json(
            artifact,
            {
                "audit_type": previous["audit_type"],
                "empirical_evidence": False,
                "config": config,
                "cells": cells,
                "exact": {
                    "target_gradient": target,
                    "augmented_expectation": exact_expectation,
                    "wrong_prompt_ratio_expectation": wrong_expectation,
                    "clipped_expectation": clipped_expectation,
                    "max_weight": max(cell["weight"] for cell in cells),
                    "weight_bound": 1.0 / config["alpha"],
                },
            },
        )
        return {"accelerator_seconds": 0.0}

    if stage == "train":
        config = previous["config"]
        cells = previous["cells"]
        rng = random.Random(config["seed"])
        rows = []
        generated_tokens = 0.0
        for replication in range(config["replications"]):
            exact, tokens = rl_estimate(
                rng,
                cells,
                config["n_current"],
                config["n_stale"],
                config["alpha"],
                "weight",
            )
            wrong, wrong_tokens = rl_estimate(
                rng,
                cells,
                config["n_current"],
                config["n_stale"],
                config["alpha"],
                "wrong_weight",
            )
            clipped, clipped_tokens = rl_estimate(
                rng,
                cells,
                config["n_current"],
                config["n_stale"],
                config["alpha"],
                "weight",
                config["fault_clip"],
            )
            generated_tokens += tokens + wrong_tokens + clipped_tokens
            rows.append(
                {
                    "replication": replication,
                    "synthetic": True,
                    "exact_augmented": exact,
                    "omitted_prompt_ratio": wrong,
                    "clipped_weight": clipped,
                }
            )
        write_jsonl(artifact, rows)
        trajectories = config["replications"] * (
            config["n_current"] + config["n_stale"]
        ) * 3
        return {
            "accelerator_seconds": 0.0,
            "generated_tokens": generated_tokens,
            "verifier_calls": float(trajectories),
            "likelihood_tokens": generated_tokens * 2.0,
        }

    model = read_json(work_dir / "build" / ARTIFACTS["rl"]["build"])
    rows = read_jsonl(prior_artifact(work_dir, "rl", stage))
    target = model["exact"]["target_gradient"]
    summaries = {}
    for key in ("exact_augmented", "omitted_prompt_ratio", "clipped_weight"):
        values = [row[key] for row in rows]
        summaries[key] = {
            "mean": mean(values),
            "bias": mean(values) - target,
            "mse": mse(values, target),
            "monte_carlo_mean_95_interval": monte_carlo_interval(values),
        }
    summary = {
        "audit_type": model["audit_type"],
        "empirical_evidence": False,
        "replications": len(rows),
        "target_gradient": target,
        "exact_enumeration": model["exact"],
        "monte_carlo": summaries,
        "checks": {
            "exact_identity_error_below_1e-12": abs(
                model["exact"]["augmented_expectation"] - target
            )
            < 1e-12,
            "weight_bound_holds": model["exact"]["max_weight"]
            <= model["exact"]["weight_bound"] + 1e-12,
            "omitting_prompt_ratio_changes_expectation": abs(
                model["exact"]["wrong_prompt_ratio_expectation"] - target
            )
            > 1e-6,
            "clipping_changes_expectation": abs(
                model["exact"]["clipped_expectation"] - target
            )
            > 1e-6,
            "target_in_exact_monte_carlo_interval": (
                summaries["exact_augmented"]["monte_carlo_mean_95_interval"][0]
                <= target
                <= summaries["exact_augmented"]["monte_carlo_mean_95_interval"][1]
            ),
        },
        "interpretation": (
            "Finite-support estimator check only; these values do not establish "
            "language-model reward or compute improvements."
        ),
    }
    write_json(artifact, summary)
    return {
        "accelerator_seconds": 0.0,
        "evaluated_checkpoints": float(len(rows)),
        "verifier_calls": 0.0,
    }


def eval_units() -> list[dict[str, Any]]:
    raw_weights = [1 + (index % 4) for index in range(16)]
    total = sum(raw_weights)
    units = []
    for index, raw_weight in enumerate(raw_weights):
        tie = 0.12 + 0.04 * (index % 3)
        signal = 0.34 * math.sin(0.7 * index) + 0.08
        plus = (1.0 - tie + signal) / 2.0
        minus = (1.0 - tie - signal) / 2.0
        if min(plus, minus) <= 0:
            fail("invalid synthetic preference probabilities")
        mu = plus - minus
        units.append(
            {
                "id": f"unit-{index:02d}",
                "pi": raw_weight / total,
                "p_plus": plus,
                "p_tie": tie,
                "p_minus": minus,
                "mu": mu,
                "initial_prediction": max(
                    -1.0, min(1.0, -0.55 * mu + 0.18 * math.cos(index))
                ),
                "initial_tie_prediction": 0.25,
            }
        )
    return units


def eval_draw_outcome(rng: random.Random, unit: dict[str, Any]) -> int:
    index = categorical_index(
        rng, [unit["p_minus"], unit["p_tie"], unit["p_plus"]]
    )
    return (-1, 0, 1)[index]


def eval_probabilities(
    units: list[dict[str, Any]],
    predictions: list[float],
    epsilon: float,
    temperature: float,
) -> list[float]:
    adaptive_raw = [
        unit["pi"] * math.exp(temperature * (1.0 - abs(prediction)))
        for unit, prediction in zip(units, predictions)
    ]
    normalizer = sum(adaptive_raw)
    adaptive = [value / normalizer for value in adaptive_raw]
    return [
        epsilon * unit["pi"] + (1.0 - epsilon) * probability
        for unit, probability in zip(units, adaptive)
    ]


def run_eval_trial(
    rng: random.Random,
    units: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    predictions = [unit["initial_prediction"] for unit in units]
    tie_predictions = [unit["initial_tie_prediction"] for unit in units]
    counts = [0] * len(units)
    sums = [0.0] * len(units)
    tie_sums = [0.0] * len(units)
    psi_values = []
    tie_psi_values = []
    observed = []
    max_weight = 0.0
    min_floor_ratio = math.inf
    prior_count = config["prediction_prior_count"]
    for _ in range(config["budget"]):
        probabilities = eval_probabilities(
            units, predictions, config["epsilon"], config["temperature"]
        )
        bar_prediction = sum(
            unit["pi"] * prediction
            for unit, prediction in zip(units, predictions)
        )
        bar_tie_prediction = sum(
            unit["pi"] * prediction
            for unit, prediction in zip(units, tie_predictions)
        )
        index = categorical_index(rng, probabilities)
        unit = units[index]
        outcome = eval_draw_outcome(rng, unit)
        weight = unit["pi"] / probabilities[index]
        max_weight = max(max_weight, weight)
        min_floor_ratio = min(min_floor_ratio, probabilities[index] / unit["pi"])
        psi_values.append(
            bar_prediction + weight * (outcome - predictions[index])
        )
        tie_value = float(outcome == 0)
        tie_psi_values.append(
            bar_tie_prediction
            + weight * (tie_value - tie_predictions[index])
        )
        observed.append(float(outcome))
        counts[index] += 1
        sums[index] += outcome
        tie_sums[index] += tie_value
        predictions[index] = (
            prior_count * unit["initial_prediction"] + sums[index]
        ) / (prior_count + counts[index])
        tie_predictions[index] = (
            prior_count * unit["initial_tie_prediction"] + tie_sums[index]
        ) / (prior_count + counts[index])

    estimate = mean(psi_values)
    standard_error = statistics.stdev(psi_values) / math.sqrt(len(psi_values))
    tie_estimate = mean(tie_psi_values)
    tie_standard_error = statistics.stdev(tie_psi_values) / math.sqrt(
        len(tie_psi_values)
    )
    target = sum(unit["pi"] * unit["mu"] for unit in units)
    tie_target = sum(unit["pi"] * unit["p_tie"] for unit in units)
    return {
        "estimate": estimate,
        "standard_error": standard_error,
        "covered": abs(estimate - target) <= 1.96 * standard_error,
        "tie_estimate": tie_estimate,
        "tie_covered": abs(tie_estimate - tie_target) <= 1.96
        * tie_standard_error,
        "naive_adaptive_mean": mean(observed),
        "max_weight": max_weight,
        "min_observed_p_over_pi": min_floor_ratio,
    }


def run_eval(stage: str, work_dir: Path) -> dict[str, float]:
    artifact = stage_artifact(work_dir, "eval", stage)
    if stage == "acquire":
        units = eval_units()
        write_json(
            artifact,
            {
                "audit_type": "synthetic_adaptive_sampling_audit",
                "empirical_evidence": False,
                "seed": 20260902,
                "license": "generated finite preference frame; no human data",
                "complete_response": True,
                "units": units,
            },
        )
        return {"annotation_currency": 0.0}

    previous = (
        None
        if stage == "evaluate"
        else read_json(prior_artifact(work_dir, "eval", stage))
    )
    if stage == "process":
        units = previous["units"]
        target_mass = sum(unit["pi"] for unit in units)
        outcome_mass_error = max(
            abs(unit["p_plus"] + unit["p_tie"] + unit["p_minus"] - 1.0)
            for unit in units
        )
        if abs(target_mass - 1.0) > 1e-12 or outcome_mass_error > 1e-12:
            fail("synthetic evaluation frame does not normalize")
        write_json(
            artifact,
            {
                "audit_type": previous["audit_type"],
                "complete_response": previous["complete_response"],
                "checks": {
                    "target_mass": target_mass,
                    "maximum_outcome_mass_error": outcome_mass_error,
                    "ties_retained": all(unit["p_tie"] > 0 for unit in units),
                },
                "units": units,
            },
        )
        return {"accelerator_seconds": 0.0, "jury_calls": 0.0}

    if stage == "build":
        config = {
            "budget": 300,
            "trials": 2000,
            "epsilon": 0.2,
            "temperature": 2.0,
            "prediction_prior_count": 5,
            "seed": 20260902,
            "confidence_level": 0.95,
        }
        units = previous["units"]
        initial_predictions = [unit["initial_prediction"] for unit in units]
        initial_probabilities = eval_probabilities(
            units,
            initial_predictions,
            config["epsilon"],
            config["temperature"],
        )
        bar_prediction = sum(
            unit["pi"] * prediction
            for unit, prediction in zip(units, initial_predictions)
        )
        exact_one_step = sum(
            probability
            * (
                bar_prediction
                + unit["pi"]
                / probability
                * (unit["mu"] - prediction)
            )
            for unit, prediction, probability in zip(
                units, initial_predictions, initial_probabilities
            )
        )
        target = sum(unit["pi"] * unit["mu"] for unit in units)
        if abs(exact_one_step - target) > 1e-12:
            fail("sequential augmented HH one-step identity failed")
        write_json(
            artifact,
            {
                "audit_type": previous["audit_type"],
                "empirical_evidence": False,
                "complete_response": previous["complete_response"],
                "config": config,
                "units": units,
                "exact": {
                    "target": target,
                    "tie_target": sum(
                        unit["pi"] * unit["p_tie"] for unit in units
                    ),
                    "one_step_expectation": exact_one_step,
                    "minimum_initial_p_over_pi": min(
                        probability / unit["pi"]
                        for unit, probability in zip(units, initial_probabilities)
                    ),
                    "maximum_initial_weight": max(
                        unit["pi"] / probability
                        for unit, probability in zip(units, initial_probabilities)
                    ),
                    "weight_bound": 1.0 / config["epsilon"],
                },
            },
        )
        return {"accelerator_seconds": 0.0, "jury_calls": 0.0}

    if stage == "train":
        config = previous["config"]
        units = previous["units"]
        rng = random.Random(config["seed"])
        rows = []
        for trial in range(config["trials"]):
            rows.append(
                {
                    "trial": trial,
                    "synthetic": True,
                    **run_eval_trial(rng, units, config),
                }
            )
        write_jsonl(artifact, rows)
        assignments = config["trials"] * config["budget"]
        return {
            "accelerator_seconds": 0.0,
            "assignment_attempts": float(assignments),
            "completed_ratings": float(assignments),
            "nonresponses": 0.0,
            "jury_calls": 0.0,
            "annotation_currency": 0.0,
        }

    model = read_json(work_dir / "build" / ARTIFACTS["eval"]["build"])
    rows = read_jsonl(prior_artifact(work_dir, "eval", stage))
    target = model["exact"]["target"]
    tie_target = model["exact"]["tie_target"]
    estimates = [row["estimate"] for row in rows]
    tie_estimates = [row["tie_estimate"] for row in rows]
    naive = [row["naive_adaptive_mean"] for row in rows]
    coverage = mean([row["covered"] for row in rows])
    tie_coverage = mean([row["tie_covered"] for row in rows])
    nominal_coverage = model["config"]["confidence_level"]
    coverage_mc_se = math.sqrt(
        nominal_coverage * (1.0 - nominal_coverage) / len(rows)
    )
    summary = {
        "audit_type": model["audit_type"],
        "empirical_evidence": False,
        "complete_response": model["complete_response"],
        "trials": len(rows),
        "budget_per_trial": model["config"]["budget"],
        "target": target,
        "tie_target": tie_target,
        "checks": {
            "one_step_identity_error_below_1e-12": abs(
                model["exact"]["one_step_expectation"] - target
            )
            < 1e-12,
            "positivity_floor_observed": min(
                row["min_observed_p_over_pi"] for row in rows
            )
            >= model["config"]["epsilon"] - 1e-12,
            "weight_bound_observed": max(row["max_weight"] for row in rows)
            <= model["exact"]["weight_bound"] + 1e-12,
            "coverage_within_three_monte_carlo_se": abs(
                coverage - nominal_coverage
            )
            <= 3.0 * coverage_mc_se,
            "tie_coverage_within_three_monte_carlo_se": abs(
                tie_coverage - nominal_coverage
            )
            <= 3.0 * coverage_mc_se,
        },
        "sequential_augmented_hh": {
            "mean": mean(estimates),
            "bias": mean(estimates) - target,
            "rmse": math.sqrt(mse(estimates, target)),
            "studentized_95_coverage": coverage,
            "coverage_monte_carlo_se": coverage_mc_se,
            "monte_carlo_mean_95_interval": monte_carlo_interval(estimates),
        },
        "tie_estimator": {
            "mean": mean(tie_estimates),
            "bias": mean(tie_estimates) - tie_target,
            "studentized_95_coverage": tie_coverage,
            "coverage_monte_carlo_se": coverage_mc_se,
        },
        "naive_adaptive_mean": {
            "mean": mean(naive),
            "bias": mean(naive) - target,
        },
        "interpretation": (
            "Synthetic complete-response design check only; it neither replaces "
            "the preregistered condition grid nor provides human-evaluation evidence."
        ),
    }
    write_json(artifact, summary)
    assignments = len(rows) * model["config"]["budget"]
    return {
        "assignment_attempts": float(assignments),
        "completed_ratings": float(assignments),
        "nonresponses": 0.0,
        "jury_calls": 0.0,
        "annotation_currency": 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", required=True, choices=PAPERS)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--work-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    if args.paper == "sft":
        measured = run_sft(args.stage, args.work_dir)
    elif args.paper == "rl":
        measured = run_rl(args.stage, args.work_dir)
    else:
        measured = run_eval(args.stage, args.work_dir)
    elapsed_cpu = max(0.0, time.process_time() - start_cpu)
    elapsed_wall = max(0.0, time.perf_counter() - start_wall)
    costs = {key: 0.0 for key in COST_KEYS[args.paper][args.stage]}
    costs.update(measured)
    if "cpu_seconds" in costs:
        costs["cpu_seconds"] = elapsed_cpu
    if "wall_seconds" in costs:
        costs["wall_seconds"] = elapsed_wall
    write_json(args.work_dir / args.stage / "costs.json", costs)


if __name__ == "__main__":
    main()
