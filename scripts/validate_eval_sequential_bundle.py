#!/usr/bin/env python3
"""Recompute the evaluation paper's sequential decision bundle.

A bundle contains the weighted target frame, one frozen commitment per batch,
and every fixed assignment slot.  The validator independently recomputes

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
bundles whose probabilities are exact rationals and checks the float
recomputation against an independent Fraction implementation, then applies
mutation tests that must fail.  Real bundles are rejected while the
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
    else:
        require(
            bundle.get("empirical_evidence") is False,
            "fixture bundle must be marked non-empirical",
        )
    config = bundle["config"]
    eps = config["exploration_epsilon"]
    total_slots = config["fixed_budget_attempts"]
    require(0 < eps <= 1 and total_slots > 1, "invalid epsilon or fixed budget")

    frame = bundle["frame"]
    mass = {row["unit_id"]: row["target_mass"] for row in frame}
    require(len(mass) == len(frame), "frame unit identifiers must be unique")
    require(all(value > 0 for value in mass.values()), "frame masses must be positive")
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


def build_fixture(missing_slots: tuple[int, ...] = ()) -> tuple[dict[str, Any], dict]:
    """Generated golden bundle whose exp(g) values are integers."""
    units = [f"u{index}" for index in range(8)]
    raw = [1, 2, 3, 2, 1, 3, 2, 2]
    mass_fraction = {u: Fraction(r, sum(raw)) for u, r in zip(units, raw)}
    mass = {u: float(value) for u, value in mass_fraction.items()}
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
        "frame": [{"unit_id": u, "target_mass": mass[u]} for u in units],
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
    return {
        "schema_version": 1,
        "audit_type": "eval_sequential_bundle_golden_fixture",
        "empirical_evidence": False,
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
