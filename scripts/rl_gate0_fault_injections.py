#!/usr/bin/env python3
"""Exact finite-law Gate 0 fault injections for the RL estimator.

The four-cell law matches the committed Gate 0 audit: two prompts, a binary
response, a scalar logit shift theta applied in decoder-logit space, and the
score s=y-pi_x(theta).  Every expectation below is computed by exhaustive
enumeration of the joint sample space, so each check is an identity to 1e-12
rather than a Monte Carlo tolerance.

Three faults are injected and must each be detected:

* base-versus-decoder scoring: the current-policy mass entering w and M is
  scored with temperature-one base logits instead of the temperature-T_d
  rollout decoder; its bias must equal the common implemented-weight identity;
* focal nuisance fitting: h is refit from the same focal batch as a pooled
  per-prompt outcome mean; a frozen h and an h fit on an independent prior
  batch are exact positive controls;
* consumed-trajectory reuse: the batch that formed theta_t is reused at
  theta_t, with and without ratio recomputation; fresh draws at theta_t are the
  positive control for the random post-update gradient.

All outputs are generated software checks with empirical_evidence=false.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Callable

IDENTITY_TOLERANCE = 1e-12
DETECTION_THRESHOLD = 1e-6

PROMPT_MASS = (0.6, 0.4)
STALE_PROMPT_MASS = (0.2, 0.8)
DECODER_SUCCESS = (0.72, 0.31)
STALE_SUCCESS = (0.38, 0.79)
DUAL = 0.12
ALPHA = 0.5
DECODER_TEMPERATURE = 0.7
FROZEN_H_SLOPE = 0.72
FROZEN_H_OFFSET = 0.08
FOCAL_BATCH = {"current": 2, "stale": 2}
PRIOR_BATCH = 2
REUSE_BATCH = 2
REUSE_STEP_SIZE = 0.5


class FaultInjectionError(ValueError):
    pass


def logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def utility(x: int, y: int) -> float:
    reward = float(y if x == 0 else 1 - y)
    return reward - DUAL * float(1 + y + x)


def frozen_h(x: int, y: int) -> float:
    return FROZEN_H_SLOPE * utility(x, y) + FROZEN_H_OFFSET * (1.0 - 2.0 * x)


CELLS = tuple((x, y) for x in range(2) for y in range(2))


def success_at(theta: float) -> tuple[float, float]:
    return tuple(sigmoid(logit(p) + theta) for p in DECODER_SUCCESS)


def current_law(theta: float = 0.0) -> dict[tuple[int, int], float]:
    success = success_at(theta)
    return {
        (x, y): PROMPT_MASS[x] * (success[x] if y else 1.0 - success[x])
        for x, y in CELLS
    }


def base_current_law(
    temperature: float = DECODER_TEMPERATURE,
) -> dict[tuple[int, int], float]:
    base = [sigmoid(temperature * logit(p)) for p in DECODER_SUCCESS]
    return {
        (x, y): PROMPT_MASS[x] * (base[x] if y else 1.0 - base[x]) for x, y in CELLS
    }


def stale_law() -> dict[tuple[int, int], float]:
    return {
        (x, y): STALE_PROMPT_MASS[x]
        * (STALE_SUCCESS[x] if y else 1.0 - STALE_SUCCESS[x])
        for x, y in CELLS
    }


def score(theta: float, x: int, y: int) -> float:
    return float(y) - success_at(theta)[x]


def prompt_baseline(theta: float, x: int) -> float:
    success = success_at(theta)[x]
    return success * utility(x, 1) + (1.0 - success) * utility(x, 0)


def target_gradient(theta: float = 0.0) -> float:
    law = current_law(theta)
    return sum(law[z] * utility(*z) * score(theta, *z) for z in CELLS)


def check_law(law: dict[tuple[int, int], float], name: str) -> None:
    if abs(sum(law.values()) - 1.0) > IDENTITY_TOLERANCE or min(law.values()) <= 0:
        raise FaultInjectionError(f"{name} law is not a positive normalized law")


def estimator(
    current: list[tuple[int, int]],
    stale: list[tuple[int, int]],
    weight: Callable[[tuple[int, int]], float],
    h: Callable[[tuple[int, int]], float],
    theta: float = 0.0,
) -> float:
    a_terms = [
        (h(z) - prompt_baseline(theta, z[0])) * score(theta, *z)
        + ALPHA * weight(z) * (utility(*z) - h(z)) * score(theta, *z)
        for z in current
    ]
    b_terms = [weight(z) * (utility(*z) - h(z)) * score(theta, *z) for z in stale]
    return sum(a_terms) / len(a_terms) + (1.0 - ALPHA) * sum(b_terms) / len(b_terms)


def enumerate_batches(
    law: dict[tuple[int, int], float], size: int
) -> list[tuple[float, list[tuple[int, int]]]]:
    return [
        (math.prod(law[z] for z in draws), list(draws))
        for draws in itertools.product(CELLS, repeat=size)
    ]


def focal_expectation(
    make_h: Callable[[list[tuple[int, int]], list[tuple[int, int]]], Callable],
    weight: Callable[[tuple[int, int]], float],
) -> tuple[float, float]:
    """Return the exact expectation and the enumerated probability mass."""
    p_law = current_law()
    g_law = stale_law()
    total = 0.0
    mass = 0.0
    for p_mass, current in enumerate_batches(p_law, FOCAL_BATCH["current"]):
        for g_mass, stale in enumerate_batches(g_law, FOCAL_BATCH["stale"]):
            probability = p_mass * g_mass
            total += probability * estimator(
                current, stale, weight, make_h(current, stale)
            )
            mass += probability
    return total, mass


def pooled_prompt_mean_h(
    batch: list[tuple[int, int]],
) -> Callable[[tuple[int, int]], float]:
    """Action-independent per-prompt outcome mean; frozen h if x is unseen."""
    means = {}
    for x in range(2):
        values = [utility(*z) for z in batch if z[0] == x]
        if values:
            means[x] = sum(values) / len(values)

    def h(z: tuple[int, int]) -> float:
        return means[z[0]] if z[0] in means else frozen_h(*z)

    return h


def run_base_decoder_fault(
    temperature: float = DECODER_TEMPERATURE,
) -> dict[str, Any]:
    p_law = current_law()
    b_law = base_current_law(temperature)
    g_law = stale_law()
    for law, name in ((p_law, "decoder"), (b_law, "base"), (g_law, "stale")):
        check_law(law, name)
    mixture = {z: ALPHA * p_law[z] + (1.0 - ALPHA) * g_law[z] for z in CELLS}
    exact_weight = {z: p_law[z] / mixture[z] for z in CELLS}
    base_weight = {
        z: b_law[z] / (ALPHA * b_law[z] + (1.0 - ALPHA) * g_law[z]) for z in CELLS
    }
    target = target_gradient()

    def expectation(weight: dict[tuple[int, int], float]) -> float:
        direct = sum(
            p_law[z] * (frozen_h(*z) - prompt_baseline(0.0, z[0])) * score(0.0, *z)
            for z in CELLS
        )
        residual = sum(
            mixture[z] * weight[z] * (utility(*z) - frozen_h(*z)) * score(0.0, *z)
            for z in CELLS
        )
        return direct + residual

    exact = expectation(exact_weight)
    faulty = expectation(base_weight)
    identity = sum(
        mixture[z]
        * (base_weight[z] - exact_weight[z])
        * (utility(*z) - frozen_h(*z))
        * score(0.0, *z)
        for z in CELLS
    )
    total_variation = 0.5 * sum(abs(p_law[z] - b_law[z]) for z in CELLS)
    return {
        "decoder_temperature": temperature,
        "decoder_base_total_variation": total_variation,
        "target_gradient": target,
        "exact_decoder_expectation": exact,
        "base_scored_expectation": faulty,
        "base_scored_bias": faulty - target,
        "common_weight_bias_identity": identity,
        "checks": {
            "decoder_route_is_exact": abs(exact - target) <= IDENTITY_TOLERANCE,
            "base_scoring_is_detected": abs(faulty - target) > DETECTION_THRESHOLD,
            "bias_matches_identity": abs((faulty - target) - identity)
            <= IDENTITY_TOLERANCE,
            "mixture_weight_normalizes": abs(
                sum(mixture[z] * exact_weight[z] for z in CELLS) - 1.0
            )
            <= IDENTITY_TOLERANCE,
            "weight_bound_holds": max(exact_weight.values())
            <= 1.0 / ALPHA + IDENTITY_TOLERANCE,
        },
    }


def run_focal_nuisance_fault(
    focal_fit: Callable[[list[tuple[int, int]]], Callable] = pooled_prompt_mean_h,
) -> dict[str, Any]:
    p_law = current_law()
    g_law = stale_law()
    mixture = {z: ALPHA * p_law[z] + (1.0 - ALPHA) * g_law[z] for z in CELLS}

    def weight(z: tuple[int, int]) -> float:
        return p_law[z] / mixture[z]

    target = target_gradient()
    frozen, frozen_mass = focal_expectation(
        lambda c, s: (lambda z: frozen_h(*z)), weight
    )
    focal, focal_mass = focal_expectation(
        lambda c, s: focal_fit(c + s), weight
    )
    independent_total = 0.0
    independent_mass = 0.0
    for prior_mass, prior in enumerate_batches(p_law, PRIOR_BATCH):
        fitted = pooled_prompt_mean_h(prior)
        conditional, conditional_mass = focal_expectation(
            lambda c, s, fitted=fitted: fitted, weight
        )
        independent_total += prior_mass * conditional
        independent_mass += prior_mass * conditional_mass
    return {
        "focal_batch": dict(FOCAL_BATCH),
        "independent_prior_batch": PRIOR_BATCH,
        "focal_fit_rule": (
            "pooled per-prompt mean of U over both focal strata; frozen h for a "
            "prompt absent from the batch"
        ),
        "target_gradient": target,
        "frozen_h_expectation": frozen,
        "independent_prior_fit_expectation": independent_total,
        "focal_fit_expectation": focal,
        "focal_fit_bias": focal - target,
        "checks": {
            "enumeration_mass_is_one": all(
                abs(value - 1.0) <= IDENTITY_TOLERANCE
                for value in (frozen_mass, focal_mass, independent_mass)
            ),
            "frozen_h_is_exact": abs(frozen - target) <= IDENTITY_TOLERANCE,
            "independent_prior_fit_is_exact": abs(independent_total - target)
            <= IDENTITY_TOLERANCE,
            "focal_fit_is_detected": abs(focal - target) > DETECTION_THRESHOLD,
        },
    }


def on_policy_estimate(theta: float, batch: list[tuple[int, int]]) -> float:
    return sum(
        (utility(*z) - prompt_baseline(theta, z[0])) * score(theta, *z) for z in batch
    ) / len(batch)


def run_reuse_fault(step_size: float = REUSE_STEP_SIZE) -> dict[str, Any]:
    start_law = current_law(0.0)
    fresh_target = 0.0
    fresh_estimate = 0.0
    reuse_ratio = 0.0
    reuse_plain = 0.0
    mass = 0.0
    max_kl = 0.0
    for batch_mass, batch in enumerate_batches(start_law, REUSE_BATCH):
        theta_next = step_size * on_policy_estimate(0.0, batch)
        next_law = current_law(theta_next)
        max_kl = max(
            max_kl,
            sum(
                next_law[z] * math.log(next_law[z] / start_law[z]) for z in CELLS
            ),
        )
        fresh_target += batch_mass * target_gradient(theta_next)
        fresh_estimate += batch_mass * sum(
            fresh_mass * on_policy_estimate(theta_next, fresh)
            for fresh_mass, fresh in enumerate_batches(next_law, REUSE_BATCH)
        )
        reuse_plain += batch_mass * on_policy_estimate(theta_next, batch)
        reuse_ratio += batch_mass * sum(
            next_law[z]
            / start_law[z]
            * (utility(*z) - prompt_baseline(theta_next, z[0]))
            * score(theta_next, *z)
            for z in batch
        ) / len(batch)
        mass += batch_mass
    return {
        "forming_batch": REUSE_BATCH,
        "step_size": step_size,
        "maximum_post_update_kl": max_kl,
        "expected_post_update_target_gradient": fresh_target,
        "fresh_draw_expectation": fresh_estimate,
        "reused_without_ratio_expectation": reuse_plain,
        "reused_with_recomputed_ratio_expectation": reuse_ratio,
        "reuse_without_ratio_bias": reuse_plain - fresh_target,
        "reuse_with_recomputed_ratio_bias": reuse_ratio - fresh_target,
        "checks": {
            "enumeration_mass_is_one": abs(mass - 1.0) <= IDENTITY_TOLERANCE,
            "fresh_draws_are_exact": abs(fresh_estimate - fresh_target)
            <= IDENTITY_TOLERANCE,
            "reuse_without_ratio_is_detected": abs(reuse_plain - fresh_target)
            > DETECTION_THRESHOLD,
            "reuse_with_recomputed_ratio_is_detected": abs(reuse_ratio - fresh_target)
            > DETECTION_THRESHOLD,
        },
    }


def canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def run_all() -> dict[str, Any]:
    results = {
        "base_versus_decoder_scoring": run_base_decoder_fault(),
        "focal_nuisance_fitting": run_focal_nuisance_fault(),
        "consumed_trajectory_reuse": run_reuse_fault(),
    }
    failed = [
        f"{name}.{check}"
        for name, result in results.items()
        for check, passed in result["checks"].items()
        if passed is not True
    ]
    summary = {
        "schema_version": 1,
        "audit_type": "rl_gate0_exact_fault_injections",
        "empirical_evidence": False,
        "identity_tolerance": IDENTITY_TOLERANCE,
        "detection_threshold": DETECTION_THRESHOLD,
        "alpha": ALPHA,
        "fault_injections": results,
        "failed_checks": failed,
        "status": "pass" if not failed else "fail",
        "interpretation": (
            "Exact enumeration on a generated four-cell law. It shows that the "
            "Gate 0 checks detect each injected fault; it is not language-model "
            "evidence and cannot populate any result cell."
        ),
    }
    summary["summary_sha256"] = canonical_digest(summary)
    return summary


def self_test() -> dict[str, Any]:
    summary = run_all()
    if summary["status"] != "pass":
        raise FaultInjectionError(
            "Gate 0 fault injections failed: " + ", ".join(summary["failed_checks"])
        )
    # Null-fault controls: removing each fault must silence its detector, so a
    # detector that always fires cannot pass.
    null_checks = {
        "base_scoring_is_detected": run_base_decoder_fault(temperature=1.0)[
            "checks"
        ]["base_scoring_is_detected"],
        "focal_fit_is_detected": run_focal_nuisance_fault(
            focal_fit=lambda batch: (lambda z: frozen_h(*z))
        )["checks"]["focal_fit_is_detected"],
        "reuse_with_recomputed_ratio_is_detected": run_reuse_fault(step_size=0.0)[
            "checks"
        ]["reuse_with_recomputed_ratio_is_detected"],
    }
    fired = [name for name, detected in null_checks.items() if detected]
    if fired:
        raise FaultInjectionError(
            "detectors fired without an injected fault: " + ", ".join(fired)
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        summary = self_test() if args.self_test else run_all()
    except FaultInjectionError as exc:
        raise SystemExit(f"error: {exc}") from exc
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    results = summary["fault_injections"]
    print(
        "validated RL Gate 0 fault injections: "
        f"base-scoring bias {results['base_versus_decoder_scoring']['base_scored_bias']:.6g}, "
        f"focal-fit bias {results['focal_nuisance_fitting']['focal_fit_bias']:.6g}, "
        "reuse bias "
        f"{results['consumed_trajectory_reuse']['reuse_with_recomputed_ratio_bias']:.6g}"
    )
    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
