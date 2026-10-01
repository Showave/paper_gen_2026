#!/usr/bin/env python3
"""Recompute committed EVI values from the frozen Laplace working posterior.

For every adaptive batch of a sequential decision bundle, a posterior artifact
records the working-model mode and precision.  This validator

* checks that the artifact is bound to the batch commitment and that the frame
  design covers exactly the bundle frame;
* closes the posterior against the observed outcomes in the batch's strictly
  earlier fit slots: the mode must be stationary for the Gaussian-prior
  Davidson log posterior, and the precision must equal prior precision plus
  the observed information at that mode;
* recomputes Q_b, the posterior predictive, the preregistered hypothetical
  damped Newton update for each outcome, and therefore EVI_b(x) for every
  committed candidate, with the frozen common standard-normal draws, ridge,
  damping, Cholesky jitter ladder, margin, and loss matrix.

The working model scored here has Davidson natural parameters that are linear
in the coefficients (eta = F_x theta, kappa = G_x theta), i.e. fixed
discrimination.  Jointly estimated log-discrimination makes eta bilinear and
needs its own Jacobian; that case is rejected rather than approximated.

The self-test builds a generated golden bundle, recomputes EVI with numpy and
with an independent pure-Python implementation, and applies mutations that
must fail.  Outputs are software checks with empirical_evidence=false.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np

from validate_eval_sequential_bundle import canonical_digest, commitment_payload

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREREGISTRATION = ROOT / "experiments" / "pilot_preregistration.json"
WORKING_MODEL = "davidson_linear_natural_parameters"
STATIONARITY_TOLERANCE = 1e-8
PRECISION_TOLERANCE = 1e-9
EVI_TOLERANCE = 1e-9
BOUNDARY_BAND = 1e-9
OUTCOMES = (1, 0, -1)
STATES = ("A", "B", "E")


class EVIError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EVIError(message)


def config_from_section(section: dict[str, Any]) -> dict[str, Any]:
    posterior = section["working_posterior"]
    decision = section["decision"]
    return {
        "seed": posterior["standard_normal_common_random_seed"],
        "draws": posterior["common_monte_carlo_draws"],
        "ridge": posterior["newton_ridge"],
        "damping": posterior["hessian_damping"],
        "jitter": [0.0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6],
        "margin": decision["practical_margin"],
        "loss": decision["loss_matrix"],
        "prior_standard_deviations": sorted(
            {
                posterior["fixed_effect_prior_standard_deviation"],
                posterior["random_effect_prior_standard_deviation"],
                posterior["log_discrimination_prior_standard_deviation"],
            }
        ),
    }


def davidson(eta: np.ndarray, kappa: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return P(Y=1), P(Y=0), P(Y=-1) with a stable log-sum-exp."""
    stacked = np.stack([eta / 2.0, kappa, -eta / 2.0])
    top = stacked.max(axis=0)
    exp = np.exp(stacked - top)
    total = exp.sum(axis=0)
    return exp[0] / total, exp[1] / total, exp[2] / total


def natural_gradient_and_information(
    eta: float, kappa: float, outcome: int | None
) -> tuple[np.ndarray, np.ndarray]:
    """Gradient of log p(outcome) and the information in (eta, kappa)."""
    p_plus, p_tie, p_minus = (float(v) for v in davidson(np.array(eta), np.array(kappa)))
    directions = np.array([[0.5, 0.0], [0.0, 1.0], [-0.5, 0.0]])
    probabilities = np.array([p_plus, p_tie, p_minus])
    mean = probabilities @ directions
    centered = directions - mean
    information = (centered.T * probabilities) @ centered
    if outcome is None:
        return np.zeros(2), information
    index = OUTCOMES.index(outcome)
    return directions[index] - mean, information


def cholesky_with_jitter(matrix: np.ndarray, ladder: list[float]) -> np.ndarray | None:
    identity = np.eye(matrix.shape[0])
    for jitter in ladder:
        try:
            return np.linalg.cholesky(matrix + jitter * identity)
        except np.linalg.LinAlgError:
            continue
    return None


def common_normals(config: dict[str, Any], dimension: int) -> np.ndarray:
    generator = np.random.Generator(np.random.PCG64(config["seed"]))
    return generator.standard_normal((config["draws"], dimension))


def posterior_draws(mode: np.ndarray, factor: np.ndarray, normals: np.ndarray) -> np.ndarray:
    return mode + np.linalg.solve(factor.T, normals.T).T


def state_probabilities(
    draws: np.ndarray, design: dict[str, np.ndarray], margin: float
) -> tuple[np.ndarray, int]:
    p_plus, _, p_minus = davidson(draws @ design["eta"].T, draws @ design["kappa"].T)
    delta = (p_plus - p_minus) @ design["mass"]
    counts = np.array(
        [np.sum(delta > margin), np.sum(delta < -margin), 0], dtype=np.float64
    )
    counts[2] = len(delta) - counts[0] - counts[1]
    boundary = int(
        np.sum(np.abs(delta - margin) <= BOUNDARY_BAND)
        + np.sum(np.abs(delta + margin) <= BOUNDARY_BAND)
    )
    return counts / len(delta), boundary


def bayes_risk(state: np.ndarray, loss: list[list[float]]) -> float:
    return float(min(np.asarray(row, dtype=np.float64) @ state for row in loss))


def frame_design(
    artifact: dict[str, Any], mass: dict[str, float], dimension: int
) -> dict[str, Any]:
    units = sorted(mass)
    design = artifact.get("frame_design")
    require(
        isinstance(design, dict) and set(design) == set(units),
        "posterior frame design must cover exactly the bundle frame",
    )
    rows: dict[str, list[list[float]]] = {"eta": [], "kappa": []}
    for unit in units:
        entry = design[unit]
        require(
            isinstance(entry, dict) and set(entry) == {"eta", "kappa"},
            f"frame design for {unit} needs eta and kappa rows",
        )
        for key in ("eta", "kappa"):
            row = entry[key]
            require(
                isinstance(row, list)
                and len(row) == dimension
                and all(isinstance(v, (int, float)) and math.isfinite(v) for v in row),
                f"frame design {key} row for {unit} is invalid",
            )
            rows[key].append([float(v) for v in row])
    return {
        "units": units,
        "index": {unit: i for i, unit in enumerate(units)},
        "eta": np.array(rows["eta"]),
        "kappa": np.array(rows["kappa"]),
        "mass": np.array([mass[unit] for unit in units]),
    }


def close_laplace_fit(
    mode: np.ndarray,
    precision: np.ndarray,
    prior_precision: np.ndarray,
    design: dict[str, Any],
    observations: list[tuple[str, int]],
) -> None:
    gradient = -prior_precision @ mode
    expected = prior_precision.copy()
    for unit, outcome in observations:
        i = design["index"][unit]
        jacobian = np.stack([design["eta"][i], design["kappa"][i]])
        local_gradient, information = natural_gradient_and_information(
            float(design["eta"][i] @ mode), float(design["kappa"][i] @ mode), outcome
        )
        gradient += jacobian.T @ local_gradient
        expected += jacobian.T @ information @ jacobian
    scale = max(1.0, float(len(observations)))
    require(
        float(np.max(np.abs(gradient))) <= STATIONARITY_TOLERANCE * scale,
        "posterior mode is not stationary for the committed fit slots",
    )
    require(
        np.allclose(precision, expected, rtol=PRECISION_TOLERANCE, atol=PRECISION_TOLERANCE),
        "posterior precision is not prior precision plus fit-slot information",
    )


def recompute_evi(
    mode: np.ndarray,
    precision: np.ndarray,
    design: dict[str, Any],
    candidates: list[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    dimension = len(mode)
    normals = common_normals(config, dimension)
    factor = cholesky_with_jitter(precision, config["jitter"])
    require(factor is not None, "batch posterior precision cannot be factorized")
    base = posterior_draws(mode, factor, normals)
    prior_state, boundary = state_probabilities(base, design, config["margin"])
    prior_risk = bayes_risk(prior_state, config["loss"])
    values: dict[str, float] = {}
    failures: list[str] = []
    identity = np.eye(dimension)
    for unit in sorted(set(candidates)):
        i = design["index"][unit]
        eta_row, kappa_row = design["eta"][i], design["kappa"][i]
        predictive = np.array(
            [float(np.mean(p)) for p in davidson(base @ eta_row, base @ kappa_row)]
        )
        jacobian = np.stack([eta_row, kappa_row])
        expected_risk = 0.0
        failed = False
        for index, outcome in enumerate(OUTCOMES):
            local_gradient, information = natural_gradient_and_information(
                float(eta_row @ mode), float(kappa_row @ mode), outcome
            )
            updated = precision + jacobian.T @ information @ jacobian + config["damping"] * identity
            regularized = updated + config["ridge"] * identity
            new_mode = mode + np.linalg.solve(regularized, jacobian.T @ local_gradient)
            new_factor = cholesky_with_jitter(regularized, config["jitter"])
            if new_factor is None:
                failed = True
                break
            state, extra = state_probabilities(
                posterior_draws(new_mode, new_factor, normals), design, config["margin"]
            )
            boundary += extra
            expected_risk += float(predictive[index]) * bayes_risk(state, config["loss"])
        if failed:
            failures.append(unit)
            values[unit] = 0.0
        else:
            values[unit] = float(prior_risk - expected_risk)
    return {
        "evi": values,
        "prior_state_probabilities": dict(zip(STATES, prior_state.tolist())),
        "prior_bayes_risk": prior_risk,
        "factorization_failures": failures,
        "boundary_draws": boundary,
    }


def validate_bundle_evi(
    bundle: dict[str, Any], posteriors: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    require(posteriors.get("schema_version") == 1, "unsupported posterior artifact schema")
    require(
        posteriors.get("working_model") == WORKING_MODEL,
        "only the fixed-discrimination linear Davidson working model is recomputable",
    )
    names = posteriors.get("parameter_names")
    prior_sd = posteriors.get("prior_standard_deviations")
    require(
        isinstance(names, list)
        and names
        and len(set(names)) == len(names)
        and isinstance(prior_sd, list)
        and len(prior_sd) == len(names),
        "posterior artifact needs unique parameter names and one prior SD each",
    )
    require(
        all(value in config["prior_standard_deviations"] for value in prior_sd),
        "posterior prior standard deviations are not preregistered values",
    )
    dimension = len(names)
    prior_precision = np.diag([1.0 / value**2 for value in prior_sd])
    mass = {row["unit_id"]: row["target_mass"] for row in bundle["frame"]}
    design = frame_design(posteriors, mass, dimension)
    slots = {row["slot"]: row for row in bundle["slots"]}
    by_batch = {entry.get("batch"): entry for entry in posteriors.get("batches", [])}
    adaptive = [b for b in bundle["batches"] if b["policy"] == "evi_candidate_mixture"]
    require(
        set(by_batch) == {b["batch"] for b in adaptive},
        "posterior artifact must cover exactly the adaptive batches",
    )
    reports = []
    for batch in adaptive:
        entry = by_batch[batch["batch"]]
        label = f"batch {batch['batch']}"
        require(
            entry.get("commitment_sha256") == batch["commitment_sha256"]
            == canonical_digest(commitment_payload(batch)),
            f"{label} posterior is not bound to the batch commitment",
        )
        mode = np.array(entry.get("theta_hat"), dtype=np.float64)
        precision = np.array(entry.get("precision"), dtype=np.float64)
        require(
            mode.shape == (dimension,)
            and precision.shape == (dimension, dimension)
            and np.all(np.isfinite(mode))
            and np.all(np.isfinite(precision))
            and np.allclose(precision, precision.T, rtol=0, atol=0),
            f"{label} posterior mode or precision has the wrong shape or symmetry",
        )
        require(
            all(slot < batch["slot_start"] for slot in batch["fit_slots"]),
            f"{label} posterior uses a slot that is not strictly earlier",
        )
        observations = [
            (slots[slot]["unit_id"], slots[slot]["outcome"])
            for slot in batch["fit_slots"]
            if slots[slot]["observed"]
        ]
        close_laplace_fit(mode, precision, prior_precision, design, observations)
        result = recompute_evi(mode, precision, design, list(batch["candidate_multiset"]), config)
        require(
            result["boundary_draws"] == 0,
            f"{label} has draws within {BOUNDARY_BAND} of the decision margin",
        )
        require(set(batch["evi"]) == set(result["evi"]), f"{label} EVI units differ")
        for unit, value in result["evi"].items():
            require(
                abs(batch["evi"][unit] - value) <= EVI_TOLERANCE,
                f"{label} committed EVI for {unit} does not reproduce",
            )
        reports.append(
            {
                "batch": batch["batch"],
                "fit_observations": len(observations),
                "candidates": len(result["evi"]),
                "prior_bayes_risk": result["prior_bayes_risk"],
                "factorization_failures": result["factorization_failures"],
            }
        )
    return {"evi_batches_recomputed": len(reports), "batches": reports}


# ---------------------------------------------------------------------------
# Independent pure-Python implementation used only by the self-test.


def py_davidson(eta: float, kappa: float) -> tuple[float, float, float]:
    terms = (eta / 2.0, kappa, -eta / 2.0)
    top = max(terms)
    exp = [math.exp(t - top) for t in terms]
    total = sum(exp)
    return exp[0] / total, exp[1] / total, exp[2] / total


def py_cholesky(matrix: list[list[float]]) -> list[list[float]]:
    n = len(matrix)
    lower = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            total = matrix[i][j] - sum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                if total <= 0:
                    raise EVIError("pure-Python Cholesky failed")
                lower[i][j] = math.sqrt(total)
            else:
                lower[i][j] = total / lower[j][j]
    return lower


def py_solve_upper_transpose(lower: list[list[float]], z: list[float]) -> list[float]:
    n = len(lower)
    out = [0.0] * n
    for i in reversed(range(n)):
        out[i] = (z[i] - sum(lower[k][i] * out[k] for k in range(i + 1, n))) / lower[i][i]
    return out


def py_solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    lower = py_cholesky(matrix)
    n = len(lower)
    forward = [0.0] * n
    for i in range(n):
        forward[i] = (vector[i] - sum(lower[i][k] * forward[k] for k in range(i))) / lower[i][i]
    return py_solve_upper_transpose(lower, forward)


def py_evi(
    mode: list[float],
    precision: list[list[float]],
    frame: list[tuple[list[float], list[float], float]],
    candidate: tuple[list[float], list[float]],
    normals: list[list[float]],
    config: dict[str, Any],
) -> float:
    n = len(mode)

    def risk(center: list[float], matrix: list[list[float]]) -> tuple[float, list[list[float]]]:
        lower = py_cholesky(matrix)
        counts = [0, 0, 0]
        draws = []
        for z in normals:
            offset = py_solve_upper_transpose(lower, z)
            theta = [c + o for c, o in zip(center, offset)]
            draws.append(theta)
            delta = 0.0
            for eta_row, kappa_row, mass in frame:
                p, _, m = py_davidson(
                    sum(a * b for a, b in zip(eta_row, theta)),
                    sum(a * b for a, b in zip(kappa_row, theta)),
                )
                delta += mass * (p - m)
            if delta > config["margin"]:
                counts[0] += 1
            elif delta < -config["margin"]:
                counts[1] += 1
            else:
                counts[2] += 1
        state = [c / len(normals) for c in counts]
        return min(sum(l * s for l, s in zip(row, state)) for row in config["loss"]), draws

    prior_risk, base = risk(mode, precision)
    eta_row, kappa_row = candidate
    predictive = [0.0, 0.0, 0.0]
    for theta in base:
        probabilities = py_davidson(
            sum(a * b for a, b in zip(eta_row, theta)),
            sum(a * b for a, b in zip(kappa_row, theta)),
        )
        for k in range(3):
            predictive[k] += probabilities[k] / len(base)
    eta = sum(a * b for a, b in zip(eta_row, mode))
    kappa = sum(a * b for a, b in zip(kappa_row, mode))
    probabilities = py_davidson(eta, kappa)
    directions = ((0.5, 0.0), (0.0, 1.0), (-0.5, 0.0))
    mean = [sum(p * d[k] for p, d in zip(probabilities, directions)) for k in range(2)]
    covariance = [
        [
            sum(p * (d[a] - mean[a]) * (d[b] - mean[b]) for p, d in zip(probabilities, directions))
            for b in range(2)
        ]
        for a in range(2)
    ]
    rows = (eta_row, kappa_row)
    expected = 0.0
    for index in range(3):
        local = [directions[index][k] - mean[k] for k in range(2)]
        gradient = [sum(rows[k][i] * local[k] for k in range(2)) for i in range(n)]
        matrix = [
            [
                precision[i][j]
                + sum(rows[a][i] * covariance[a][b] * rows[b][j] for a in range(2) for b in range(2))
                + (config["damping"] + config["ridge"]) * (i == j)
                for j in range(n)
            ]
            for i in range(n)
        ]
        step = py_solve(matrix, gradient)
        new_risk, _ = risk([m + s for m, s in zip(mode, step)], matrix)
        expected += predictive[index] * new_risk
    return prior_risk - expected


# ---------------------------------------------------------------------------
# Golden fixture.

FIXTURE_LENGTHS = (-1.0, -0.5, 0.0, 0.5, 1.0, 1.5)
FIXTURE_MASS_WEIGHTS = (1, 2, 3, 2, 1, 3)
FIXTURE_TRUE_THETA = (0.25, -0.6, 0.3)
FIXTURE_PRIOR_SD = (2.0, 2.0, 1.0)
FIXTURE_BATCH_SIZE = 80
DIAGNOSTIC_FIT_SIZES = (20, 40, 80, 120, 200)


def fit_map(
    prior_precision: np.ndarray, design: dict[str, Any], observations: list[tuple[str, int]]
) -> tuple[np.ndarray, np.ndarray]:
    """Producer-side Newton MAP fit; the validator only checks its closure."""
    mode = np.zeros(prior_precision.shape[0])
    for _ in range(100):
        gradient = -prior_precision @ mode
        precision = prior_precision.copy()
        for unit, outcome in observations:
            i = design["index"][unit]
            jacobian = np.stack([design["eta"][i], design["kappa"][i]])
            local, information = natural_gradient_and_information(
                float(design["eta"][i] @ mode), float(design["kappa"][i] @ mode), outcome
            )
            gradient += jacobian.T @ local
            precision += jacobian.T @ information @ jacobian
        step = np.linalg.solve(precision, gradient)
        mode = mode + step
        if float(np.max(np.abs(step))) < 1e-14:
            break
    _, precision = laplace_terms(mode, prior_precision, design, observations)
    return mode, precision


def laplace_terms(
    mode: np.ndarray,
    prior_precision: np.ndarray,
    design: dict[str, Any],
    observations: list[tuple[str, int]],
) -> tuple[np.ndarray, np.ndarray]:
    gradient = -prior_precision @ mode
    precision = prior_precision.copy()
    for unit, outcome in observations:
        i = design["index"][unit]
        jacobian = np.stack([design["eta"][i], design["kappa"][i]])
        local, information = natural_gradient_and_information(
            float(design["eta"][i] @ mode), float(design["kappa"][i] @ mode), outcome
        )
        gradient += jacobian.T @ local
        precision += jacobian.T @ information @ jacobian
    return gradient, (precision + precision.T) / 2.0


def build_fixture(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    units = [f"u{i}" for i in range(len(FIXTURE_LENGTHS))]
    total = sum(FIXTURE_MASS_WEIGHTS)
    mass = {u: w / total for u, w in zip(units, FIXTURE_MASS_WEIGHTS)}
    posteriors = {
        "schema_version": 1,
        "working_model": WORKING_MODEL,
        "parameter_names": ["preference", "tie", "length"],
        "prior_standard_deviations": list(FIXTURE_PRIOR_SD),
        "frame_design": {
            u: {"eta": [1.0, 0.0, length], "kappa": [0.0, 1.0, 0.0]}
            for u, length in zip(units, FIXTURE_LENGTHS)
        },
        "batches": [],
    }
    design = frame_design(posteriors, mass, 3)
    rng = random.Random(2026100101)
    slots = []
    batch_size = FIXTURE_BATCH_SIZE
    for slot in range(2 * batch_size):
        unit = rng.choices(units, weights=FIXTURE_MASS_WEIGHTS)[0]
        i = design["index"][unit]
        theta = np.array(FIXTURE_TRUE_THETA)
        p_plus, p_tie, _ = (
            float(v) for v in davidson(design["eta"][i] @ theta, design["kappa"][i] @ theta)
        )
        draw = rng.random()
        outcome = 1 if draw < p_plus else (0 if draw < p_plus + p_tie else -1)
        observed = slot != 7
        slots.append(
            {
                "slot": slot,
                "batch": slot // batch_size,
                "unit_id": unit,
                "observed": observed,
                "outcome": outcome if observed else None,
            }
        )
    multiset: dict[str, int] = {}
    for _ in range(6):
        unit = rng.choices(units, weights=FIXTURE_MASS_WEIGHTS)[0]
        multiset[unit] = multiset.get(unit, 0) + 1
    prior_precision = np.diag([1.0 / v**2 for v in FIXTURE_PRIOR_SD])
    fit_slots = list(range(batch_size))
    observations = [
        (slots[s]["unit_id"], slots[s]["outcome"]) for s in fit_slots if slots[s]["observed"]
    ]
    mode, precision = fit_map(prior_precision, design, observations)
    evi = recompute_evi(mode, precision, design, list(multiset), config)
    batches = [
        {
            "batch": 0,
            "policy": "target_random",
            "slot_start": 0,
            "slot_end": batch_size,
            "candidate_multiset": {},
            "evi": {},
            "annotation_cost": {},
            "predictions": {u: 0.0 for u in units},
            "tie_predictions": {u: 0.0 for u in units},
            "fit_slots": [],
        },
        {
            "batch": 1,
            "policy": "evi_candidate_mixture",
            "slot_start": batch_size,
            "slot_end": 2 * batch_size,
            "candidate_multiset": multiset,
            "evi": evi["evi"],
            "annotation_cost": {u: 1.0 for u in multiset},
            "predictions": {u: 0.1 for u in units},
            "tie_predictions": {u: 0.2 for u in units},
            "fit_slots": fit_slots,
        },
    ]
    for batch in batches:
        batch["commitment_sha256"] = canonical_digest(commitment_payload(batch))
    posteriors["batches"] = [
        {
            "batch": 1,
            "commitment_sha256": batches[1]["commitment_sha256"],
            "theta_hat": mode.tolist(),
            "precision": precision.tolist(),
        }
    ]
    bundle = {
        "frame": [{"unit_id": u, "target_mass": mass[u]} for u in units],
        "batches": batches,
        "slots": slots,
    }
    return bundle, {"posteriors": posteriors, "evi": evi, "design": design}


def fit_size_diagnostic(config: dict[str, Any], temperature: float) -> list[dict[str, Any]]:
    """EVI range on the golden law as the number of fit outcomes grows."""
    units = [f"u{i}" for i in range(len(FIXTURE_LENGTHS))]
    total = sum(FIXTURE_MASS_WEIGHTS)
    mass = {u: w / total for u, w in zip(units, FIXTURE_MASS_WEIGHTS)}
    design = frame_design(
        {
            "frame_design": {
                u: {"eta": [1.0, 0.0, length], "kappa": [0.0, 1.0, 0.0]}
                for u, length in zip(units, FIXTURE_LENGTHS)
            }
        },
        mass,
        3,
    )
    prior_precision = np.diag([1.0 / v**2 for v in FIXTURE_PRIOR_SD])
    theta = np.array(FIXTURE_TRUE_THETA)
    rows = []
    for size in DIAGNOSTIC_FIT_SIZES:
        rng = random.Random(2026100102)
        observations = []
        for _ in range(size):
            unit = rng.choices(units, weights=FIXTURE_MASS_WEIGHTS)[0]
            i = design["index"][unit]
            p_plus, p_tie, _ = (
                float(v) for v in davidson(design["eta"][i] @ theta, design["kappa"][i] @ theta)
            )
            draw = rng.random()
            observations.append((unit, 1 if draw < p_plus else (0 if draw < p_plus + p_tie else -1)))
        mode, precision = fit_map(prior_precision, design, observations)
        result = recompute_evi(mode, precision, design, units, config)
        values = list(result["evi"].values())
        rows.append(
            {
                "fit_outcomes": size,
                "prior_bayes_risk": result["prior_bayes_risk"],
                "minimum_evi": min(values),
                "maximum_evi": max(values),
                "negative_evi_units": int(sum(v < -EVI_TOLERANCE for v in values)),
                "zero_evi_units": int(sum(abs(v) <= EVI_TOLERANCE for v in values)),
                "maximum_softmax_tilt": math.exp(temperature * (max(values) - min(values))),
            }
        )
    return rows


def recommit(bundle: dict[str, Any], posteriors: dict[str, Any]) -> None:
    batch = bundle["batches"][1]
    batch["commitment_sha256"] = canonical_digest(commitment_payload(batch))
    posteriors["batches"][0]["commitment_sha256"] = batch["commitment_sha256"]


def self_test(section: dict[str, Any]) -> dict[str, Any]:
    config = config_from_section(section)
    bundle, extra = build_fixture(config)
    posteriors = extra["posteriors"]
    summary = validate_bundle_evi(bundle, posteriors, config)
    evi = extra["evi"]["evi"]
    require(
        any(abs(v) > 1e-6 for v in evi.values()) and len(set(evi.values())) > 1,
        "golden EVI values must be nonzero and distinct",
    )
    design = extra["design"]
    frame = [
        (design["eta"][i].tolist(), design["kappa"][i].tolist(), float(design["mass"][i]))
        for i in range(len(design["units"]))
    ]
    normals = common_normals(config, 3).tolist()
    entry = posteriors["batches"][0]
    independent = {}
    for unit in evi:
        i = design["index"][unit]
        independent[unit] = py_evi(
            entry["theta_hat"],
            entry["precision"],
            frame,
            (design["eta"][i].tolist(), design["kappa"][i].tolist()),
            normals,
            config,
        )
        require(
            abs(independent[unit] - evi[unit]) <= EVI_TOLERANCE,
            f"independent EVI for {unit} disagrees with the numpy recomputation",
        )
    require(
        cholesky_with_jitter(np.array([[-1.0]]), config["jitter"]) is None,
        "jitter ladder accepted a negative-definite precision",
    )

    mutations: dict[str, str] = {}

    def mutate(name: str, change) -> None:
        candidate_bundle = copy.deepcopy(bundle)
        candidate_posteriors = copy.deepcopy(posteriors)
        change(candidate_bundle, candidate_posteriors)
        try:
            validate_bundle_evi(candidate_bundle, candidate_posteriors, config)
        except EVIError as exc:
            mutations[name] = str(exc)
            return
        raise EVIError(f"EVI mutation {name!r} was accepted")

    def zero_evi(b, p):
        b["batches"][1]["evi"] = {u: 0.0 for u in b["batches"][1]["evi"]}
        recommit(b, p)

    mutate("evi_without_hypothetical_update", zero_evi)

    def swapped(b, p):
        values = b["batches"][1]["evi"]
        units = sorted(values, key=values.get)
        values[units[0]], values[units[-1]] = values[units[-1]], values[units[0]]
        recommit(b, p)

    mutate("evi_assigned_to_wrong_unit", swapped)

    def other_seed(b, p):
        mode = np.array(p["batches"][0]["theta_hat"])
        precision = np.array(p["batches"][0]["precision"])
        changed = {**config, "seed": config["seed"] + 1}
        b["batches"][1]["evi"] = recompute_evi(
            mode, precision, design, list(b["batches"][1]["candidate_multiset"]), changed
        )["evi"]
        recommit(b, p)

    mutate("evi_from_unregistered_draws", other_seed)

    def own_batch_fit(b, p):
        prior_precision = np.diag([1.0 / v**2 for v in FIXTURE_PRIOR_SD])
        observations = [
            (row["unit_id"], row["outcome"]) for row in b["slots"] if row["observed"]
        ]
        mode, precision = fit_map(prior_precision, design, observations)
        p["batches"][0]["theta_hat"] = mode.tolist()
        p["batches"][0]["precision"] = precision.tolist()

    mutate("posterior_fit_on_its_own_batch", own_batch_fit)

    def no_prior(b, p):
        precision = np.array(p["batches"][0]["precision"])
        prior = np.diag([1.0 / v**2 for v in FIXTURE_PRIOR_SD])
        p["batches"][0]["precision"] = (precision - prior).tolist()

    mutate("precision_omits_prior", no_prior)

    mutate(
        "stale_commitment",
        lambda b, p: p["batches"][0].__setitem__("commitment_sha256", "0" * 64),
    )
    mutate(
        "frame_design_omits_unit",
        lambda b, p: p["frame_design"].pop("u0"),
    )
    mutate(
        "unregistered_prior_sd",
        lambda b, p: p["prior_standard_deviations"].__setitem__(0, 10.0),
    )
    mutate(
        "bilinear_working_model",
        lambda b, p: p.__setitem__("working_model", "davidson_estimated_discrimination"),
    )
    diagnostic = fit_size_diagnostic(config, section["acquisition"]["evi_temperature"])
    return {
        "schema_version": 1,
        "audit_type": "eval_evi_laplace_recomputation",
        "fit_size_diagnostic": diagnostic,
        "empirical_evidence": False,
        "status": "pass",
        "common_monte_carlo_draws": config["draws"],
        "golden_evi": evi,
        "independent_python_evi": independent,
        "prior_state_probabilities": extra["evi"]["prior_state_probabilities"],
        "validation": summary,
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
        parser.error("use --self-test; real recomputation runs inside the pipeline runner")
    section = json.loads(args.preregistration.read_text(encoding="utf-8"))["eval"]
    try:
        summary = self_test(section)
    except EVIError as exc:
        raise SystemExit(f"error: {exc}") from exc
    if args.output is not None:
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        "validated EVI recomputation from the frozen Laplace posterior; "
        f"numpy and pure-Python agree on {len(summary['golden_evi'])} candidates, "
        f"{summary['mutations_rejected']} mutations rejected"
    )


if __name__ == "__main__":
    main()
