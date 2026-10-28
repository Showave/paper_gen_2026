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

Both the fixed-discrimination linear Davidson model
(``eta = f_x^T theta``, ``kappa = g_x^T theta``) and the jointly estimated
log-discrimination bilinear Davidson model
(``eta = f_x^T theta + exp(u_x^T theta) * (s_x^T theta)``,
``kappa = g_x^T theta``) are recomputed.  In the bilinear model, the validator
enforces the human-panel identifiability constraint ``sum_x pi(x) u_x = 0``
and applies the exact parameter Jacobian
``d eta / d theta = f_x + d_x s_x + (d_x q_x) u_x`` and observed negative
Hessian ``J_x^T I_(eta,kappa) J_x - g_eta(y) d^2 eta / d theta^2`` in both
Laplace closure and every hypothetical Newton update; linear or Gauss-Newton
approximations are rejected.

The self-test builds generated golden bundles for both working models,
recomputes EVI with numpy and with an independent pure-Python implementation,
and applies mutations that must fail.  Outputs are software checks with
empirical_evidence=false.
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
WORKING_MODEL_BILINEAR = "davidson_estimated_discrimination"
SUPPORTED_WORKING_MODELS = {WORKING_MODEL, WORKING_MODEL_BILINEAR}
IDENTIFIABILITY_TOLERANCE = 1e-12
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


def evaluate_natural_parameters(
    draws: np.ndarray, design: dict[str, Any], unit_index: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate (eta, kappa) for 1D mode or 2D posterior draws."""
    if unit_index is not None:
        f_row = design["eta_additive"][unit_index]
        s_row = design["eta_scaled"][unit_index]
        u_row = design["log_discrimination"][unit_index]
        g_row = design["kappa"][unit_index]
        eta = draws @ f_row + np.exp(draws @ u_row) * (draws @ s_row)
        kappa = draws @ g_row
        return eta, kappa
    eta = draws @ design["eta_additive"].T + np.exp(
        draws @ design["log_discrimination"].T
    ) * (draws @ design["eta_scaled"].T)
    kappa = draws @ design["kappa"].T
    return eta, kappa


def unit_gradient_and_observed_information(
    mode: np.ndarray,
    design: dict[str, Any],
    unit_index: int,
    outcome: int,
    *,
    include_log_disc_chain: bool = True,
    include_second_derivative: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact parameter gradient and observed information -d^2 log p(y|x)/d theta^2."""
    f_row = design["eta_additive"][unit_index]
    s_row = design["eta_scaled"][unit_index]
    u_row = design["log_discrimination"][unit_index]
    g_row = design["kappa"][unit_index]
    phi = float(u_row @ mode)
    disc = math.exp(phi)
    scaled_quality = float(s_row @ mode)
    eta = float(f_row @ mode) + disc * scaled_quality
    kappa = float(g_row @ mode)
    grad_eta = f_row + disc * s_row
    if include_log_disc_chain:
        grad_eta = grad_eta + (disc * scaled_quality) * u_row
    jacobian = np.stack([grad_eta, g_row])
    local_gradient, information = natural_gradient_and_information(eta, kappa, outcome)
    param_gradient = jacobian.T @ local_gradient
    observed_info = jacobian.T @ information @ jacobian
    if include_second_derivative and design["working_model"] == WORKING_MODEL_BILINEAR:
        hessian_eta = disc * (
            np.outer(u_row, s_row) + np.outer(s_row, u_row)
        ) + (disc * scaled_quality) * np.outer(u_row, u_row)
        observed_info = observed_info - float(local_gradient[0]) * hessian_eta
    return param_gradient, observed_info


def state_probabilities(
    draws: np.ndarray, design: dict[str, Any], margin: float
) -> tuple[np.ndarray, int]:
    eta_mat, kappa_mat = evaluate_natural_parameters(draws, design)
    p_plus, _, p_minus = davidson(eta_mat, kappa_mat)
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
    working_model = artifact.get("working_model", WORKING_MODEL)
    require(
        working_model in SUPPORTED_WORKING_MODELS,
        f"unsupported working_model: {working_model!r}",
    )
    design = artifact.get("frame_design")
    require(
        isinstance(design, dict) and set(design) == set(units),
        "posterior frame design must cover exactly the bundle frame",
    )
    mass_vec = np.array([mass[unit] for unit in units], dtype=np.float64)
    if working_model == WORKING_MODEL:
        rows: dict[str, list[list[float]]] = {"eta": [], "kappa": []}
        for unit in units:
            entry = design[unit]
            require(
                isinstance(entry, dict) and set(entry) == {"eta", "kappa"},
                f"linear frame design for {unit} needs eta and kappa rows",
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
        eta_additive = np.array(rows["eta"], dtype=np.float64)
        eta_scaled = np.zeros_like(eta_additive)
        log_discrimination = np.zeros_like(eta_additive)
        kappa = np.array(rows["kappa"], dtype=np.float64)
    else:
        bilinear_keys = {"eta_additive", "eta_scaled", "log_discrimination", "kappa"}
        rows = {key: [] for key in bilinear_keys}
        for unit in units:
            entry = design[unit]
            require(
                isinstance(entry, dict) and set(entry) == bilinear_keys,
                f"bilinear frame design for {unit} needs eta_additive, eta_scaled, log_discrimination, and kappa",
            )
            for key in bilinear_keys:
                row = entry[key]
                require(
                    isinstance(row, list)
                    and len(row) == dimension
                    and all(isinstance(v, (int, float)) and math.isfinite(v) for v in row),
                    f"frame design {key} row for {unit} is invalid",
                )
                rows[key].append([float(v) for v in row])
        eta_additive = np.array(rows["eta_additive"], dtype=np.float64)
        eta_scaled = np.array(rows["eta_scaled"], dtype=np.float64)
        log_discrimination = np.array(rows["log_discrimination"], dtype=np.float64)
        kappa = np.array(rows["kappa"], dtype=np.float64)
        require(
            bool(np.any(np.abs(eta_scaled) > 0))
            and bool(np.any(np.abs(log_discrimination) > 0)),
            "estimated-discrimination frame design must have nonzero eta_scaled and log_discrimination rows",
        )
        require(
            float(np.max(np.abs(mass_vec @ log_discrimination))) <= IDENTIFIABILITY_TOLERANCE,
            "estimated-discrimination frame design violates the weighted human-panel log-discrimination zero constraint",
        )
    return {
        "working_model": working_model,
        "units": units,
        "index": {unit: i for i, unit in enumerate(units)},
        "eta": eta_additive,
        "eta_additive": eta_additive,
        "eta_scaled": eta_scaled,
        "log_discrimination": log_discrimination,
        "kappa": kappa,
        "mass": mass_vec,
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
        param_gradient, observed_info = unit_gradient_and_observed_information(
            mode, design, i, outcome
        )
        gradient += param_gradient
        expected += observed_info
    expected = (expected + expected.T) / 2.0
    scale = max(1.0, float(len(observations)))
    require(
        float(np.max(np.abs(gradient))) <= STATIONARITY_TOLERANCE * scale,
        "posterior mode is not stationary for the committed fit slots",
    )
    require(
        np.allclose(precision, expected, rtol=PRECISION_TOLERANCE, atol=PRECISION_TOLERANCE),
        "posterior precision is not prior precision plus fit-slot observed information",
    )


def recompute_evi(
    mode: np.ndarray,
    precision: np.ndarray,
    design: dict[str, Any],
    candidates: list[str],
    config: dict[str, Any],
    *,
    include_log_disc_chain: bool = True,
    include_second_derivative: bool = True,
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
        eta_draws, kappa_draws = evaluate_natural_parameters(base, design, i)
        predictive = np.array(
            [float(np.mean(p)) for p in davidson(eta_draws, kappa_draws)]
        )
        expected_risk = 0.0
        failed = False
        for index, outcome in enumerate(OUTCOMES):
            param_gradient, observed_info = unit_gradient_and_observed_information(
                mode,
                design,
                i,
                outcome,
                include_log_disc_chain=include_log_disc_chain,
                include_second_derivative=include_second_derivative,
            )
            updated = precision + observed_info + config["damping"] * identity
            regularized = (updated + updated.T) / 2.0 + config["ridge"] * identity
            new_mode = mode + np.linalg.solve(regularized, param_gradient)
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
        posteriors.get("working_model") in SUPPORTED_WORKING_MODELS,
        "unsupported Davidson working model in posterior artifact",
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


def py_eta_kappa(
    f_row: list[float],
    s_row: list[float],
    u_row: list[float],
    g_row: list[float],
    theta: list[float],
) -> tuple[float, float, float, float]:
    phi = sum(a * b for a, b in zip(u_row, theta))
    disc = math.exp(phi)
    scaled = sum(a * b for a, b in zip(s_row, theta))
    eta = sum(a * b for a, b in zip(f_row, theta)) + disc * scaled
    kappa = sum(a * b for a, b in zip(g_row, theta))
    return eta, kappa, disc, scaled


def py_evi(
    mode: list[float],
    precision: list[list[float]],
    frame: list[tuple[list[float], list[float], list[float], list[float], float]],
    candidate: tuple[list[float], list[float], list[float], list[float]],
    normals: list[list[float]],
    config: dict[str, Any],
    bilinear: bool = False,
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
            for f_row, s_row, u_row, g_row, mass in frame:
                eta_val, kappa_val, _, _ = py_eta_kappa(f_row, s_row, u_row, g_row, theta)
                p, _, m = py_davidson(eta_val, kappa_val)
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
    f_row, s_row, u_row, g_row = candidate
    predictive = [0.0, 0.0, 0.0]
    for theta in base:
        eta_val, kappa_val, _, _ = py_eta_kappa(f_row, s_row, u_row, g_row, theta)
        probabilities = py_davidson(eta_val, kappa_val)
        for k in range(3):
            predictive[k] += probabilities[k] / len(base)
    eta, kappa, disc, scaled = py_eta_kappa(f_row, s_row, u_row, g_row, mode)
    grad_eta = [
        f_row[i] + disc * s_row[i] + (disc * scaled) * u_row[i] for i in range(n)
    ]
    hessian_eta = [
        [
            disc * (u_row[i] * s_row[j] + s_row[i] * u_row[j])
            + (disc * scaled) * u_row[i] * u_row[j]
            if bilinear
            else 0.0
            for j in range(n)
        ]
        for i in range(n)
    ]
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
    rows = (grad_eta, g_row)
    expected = 0.0
    for index in range(3):
        local = [directions[index][k] - mean[k] for k in range(2)]
        gradient = [sum(rows[k][i] * local[k] for k in range(2)) for i in range(n)]
        matrix = [
            [
                precision[i][j]
                + sum(rows[a][i] * covariance[a][b] * rows[b][j] for a in range(2) for b in range(2))
                - local[0] * hessian_eta[i][j]
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
# Golden fixtures (linear and bilinear estimated-discrimination).

FIXTURE_LENGTHS = (-1.0, -0.5, 0.0, 0.5, 1.0, 1.5)
FIXTURE_MASS_WEIGHTS = (1, 2, 3, 2, 1, 3)
FIXTURE_TRUE_THETA = (0.25, -0.6, 0.3)
FIXTURE_PRIOR_SD = (2.0, 2.0, 1.0)
BILINEAR_LOG_DISC_COEFFS = (0.6, 0.3, -0.4, -0.3, 0.6, 0.0)
BILINEAR_TRUE_THETA = (0.22, 0.18, -0.55, 0.35)
BILINEAR_PRIOR_SD = (2.0, 1.0, 2.0, 0.5)
FIXTURE_BATCH_SIZE = 80
DIAGNOSTIC_FIT_SIZES = (20, 40, 80, 120, 200)


def fit_map(
    prior_precision: np.ndarray,
    design: dict[str, Any],
    observations: list[tuple[str, int]],
    *,
    include_log_disc_chain: bool = True,
    include_second_derivative: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Producer-side Newton MAP fit; the validator only checks its closure."""
    mode = np.zeros(prior_precision.shape[0])
    for _ in range(100):
        gradient, precision = laplace_terms(
            mode,
            prior_precision,
            design,
            observations,
            include_log_disc_chain=include_log_disc_chain,
            include_second_derivative=include_second_derivative,
        )
        step = np.linalg.solve(precision, gradient)
        mode = mode + step
        if float(np.max(np.abs(step))) < 1e-14:
            break
    _, precision = laplace_terms(
        mode,
        prior_precision,
        design,
        observations,
        include_log_disc_chain=include_log_disc_chain,
        include_second_derivative=include_second_derivative,
    )
    return mode, precision


def laplace_terms(
    mode: np.ndarray,
    prior_precision: np.ndarray,
    design: dict[str, Any],
    observations: list[tuple[str, int]],
    *,
    include_log_disc_chain: bool = True,
    include_second_derivative: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    gradient = -prior_precision @ mode
    precision = prior_precision.copy()
    for unit, outcome in observations:
        i = design["index"][unit]
        param_gradient, observed_info = unit_gradient_and_observed_information(
            mode,
            design,
            i,
            outcome,
            include_log_disc_chain=include_log_disc_chain,
            include_second_derivative=include_second_derivative,
        )
        gradient += param_gradient
        precision += observed_info
    return gradient, (precision + precision.T) / 2.0


def build_bilinear_fixture(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    units = [f"u{i}" for i in range(len(FIXTURE_LENGTHS))]
    total = sum(FIXTURE_MASS_WEIGHTS)
    mass = {u: w / total for u, w in zip(units, FIXTURE_MASS_WEIGHTS)}
    posteriors = {
        "schema_version": 1,
        "working_model": WORKING_MODEL_BILINEAR,
        "parameter_names": ["preference", "item_shift", "tie", "rater_log_disc"],
        "prior_standard_deviations": list(BILINEAR_PRIOR_SD),
        "frame_design": {
            u: {
                "eta_additive": [0.0, 0.2 * length, 0.0, 0.0],
                "eta_scaled": [1.0, 0.5 * length, 0.0, 0.0],
                "log_discrimination": [0.0, 0.0, 0.0, disc_coeff],
                "kappa": [0.0, 0.0, 1.0, 0.1 * length],
            }
            for u, length, disc_coeff in zip(
                units, FIXTURE_LENGTHS, BILINEAR_LOG_DISC_COEFFS
            )
        },
        "batches": [],
    }
    design = frame_design(posteriors, mass, 4)
    rng = random.Random(2026102801)
    slots = []
    batch_size = FIXTURE_BATCH_SIZE
    theta = np.array(BILINEAR_TRUE_THETA, dtype=np.float64)
    for slot in range(2 * batch_size):
        unit = rng.choices(units, weights=FIXTURE_MASS_WEIGHTS)[0]
        i = design["index"][unit]
        eta_val, kappa_val = evaluate_natural_parameters(theta, design, i)
        p_plus, p_tie, _ = (float(v) for v in davidson(eta_val, kappa_val))
        draw = rng.random()
        outcome = 1 if draw < p_plus else (0 if draw < p_plus + p_tie else -1)
        observed = slot != 9
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
    prior_precision = np.diag([1.0 / v**2 for v in BILINEAR_PRIOR_SD])
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
    return bundle, {"posteriors": posteriors, "evi": evi, "design": design, "observations": observations}


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
        (
            design["eta_additive"][i].tolist(),
            design["eta_scaled"][i].tolist(),
            design["log_discrimination"][i].tolist(),
            design["kappa"][i].tolist(),
            float(design["mass"][i]),
        )
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
            (
                design["eta_additive"][i].tolist(),
                design["eta_scaled"][i].tolist(),
                design["log_discrimination"][i].tolist(),
                design["kappa"][i].tolist(),
            ),
            normals,
            config,
            bilinear=False,
        )
        require(
            abs(independent[unit] - evi[unit]) <= EVI_TOLERANCE,
            f"independent EVI for {unit} disagrees with the numpy recomputation",
        )

    bilinear_bundle, bilinear_extra = build_bilinear_fixture(config)
    bilinear_posteriors = bilinear_extra["posteriors"]
    bilinear_summary = validate_bundle_evi(bilinear_bundle, bilinear_posteriors, config)
    bilinear_evi = bilinear_extra["evi"]["evi"]
    require(
        any(abs(v) > 1e-6 for v in bilinear_evi.values())
        and len(set(bilinear_evi.values())) > 1,
        "bilinear golden EVI values must be nonzero and distinct",
    )
    bilinear_design = bilinear_extra["design"]
    bilinear_frame = [
        (
            bilinear_design["eta_additive"][i].tolist(),
            bilinear_design["eta_scaled"][i].tolist(),
            bilinear_design["log_discrimination"][i].tolist(),
            bilinear_design["kappa"][i].tolist(),
            float(bilinear_design["mass"][i]),
        )
        for i in range(len(bilinear_design["units"]))
    ]
    bilinear_normals = common_normals(config, 4).tolist()
    bilinear_entry = bilinear_posteriors["batches"][0]
    bilinear_independent = {}
    for unit in bilinear_evi:
        i = bilinear_design["index"][unit]
        bilinear_independent[unit] = py_evi(
            bilinear_entry["theta_hat"],
            bilinear_entry["precision"],
            bilinear_frame,
            (
                bilinear_design["eta_additive"][i].tolist(),
                bilinear_design["eta_scaled"][i].tolist(),
                bilinear_design["log_discrimination"][i].tolist(),
                bilinear_design["kappa"][i].tolist(),
            ),
            bilinear_normals,
            config,
            bilinear=True,
        )
        require(
            abs(bilinear_independent[unit] - bilinear_evi[unit]) <= EVI_TOLERANCE,
            f"independent bilinear EVI for {unit} disagrees with the numpy recomputation",
        )

    require(
        cholesky_with_jitter(np.array([[-1.0]]), config["jitter"]) is None,
        "jitter ladder accepted a negative-definite precision",
    )

    mutations: dict[str, str] = {}

    def mutate(name: str, change, base_b=bundle, base_p=posteriors) -> None:
        candidate_bundle = copy.deepcopy(base_b)
        candidate_posteriors = copy.deepcopy(base_p)
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
        "linear_design_mislabeled_as_bilinear",
        lambda b, p: p.__setitem__("working_model", "davidson_estimated_discrimination"),
    )
    mutate(
        "unknown_working_model",
        lambda b, p: p.__setitem__("working_model", "davidson_unregistered_variant"),
    )

    def omit_log_disc_chain_rule(b, p):
        prior_precision = np.diag([1.0 / v**2 for v in BILINEAR_PRIOR_SD])
        mode, precision = fit_map(
            prior_precision,
            bilinear_design,
            bilinear_extra["observations"],
            include_log_disc_chain=False,
        )
        p["batches"][0]["theta_hat"] = mode.tolist()
        p["batches"][0]["precision"] = precision.tolist()
        b["batches"][1]["evi"] = recompute_evi(
            mode,
            precision,
            bilinear_design,
            list(b["batches"][1]["candidate_multiset"]),
            config,
            include_log_disc_chain=False,
        )["evi"]
        recommit(b, p)

    mutate(
        "bilinear_jacobian_omits_log_disc_chain_rule",
        omit_log_disc_chain_rule,
        bilinear_bundle,
        bilinear_posteriors,
    )

    def omit_second_derivative_term(b, p):
        prior_precision = np.diag([1.0 / v**2 for v in BILINEAR_PRIOR_SD])
        _, gauss_newton_precision = laplace_terms(
            np.array(p["batches"][0]["theta_hat"]),
            prior_precision,
            bilinear_design,
            bilinear_extra["observations"],
            include_second_derivative=False,
        )
        p["batches"][0]["precision"] = gauss_newton_precision.tolist()

    mutate(
        "bilinear_hessian_omits_second_derivative_term",
        omit_second_derivative_term,
        bilinear_bundle,
        bilinear_posteriors,
    )

    def uncentered_log_disc(b, p):
        p["frame_design"]["u0"]["log_discrimination"][3] += 0.1

    mutate(
        "bilinear_uncentered_panel_log_discrimination",
        uncentered_log_disc,
        bilinear_bundle,
        bilinear_posteriors,
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
        "bilinear_golden_evi": bilinear_evi,
        "bilinear_independent_python_evi": bilinear_independent,
        "prior_state_probabilities": extra["evi"]["prior_state_probabilities"],
        "bilinear_prior_state_probabilities": bilinear_extra["evi"][
            "prior_state_probabilities"
        ],
        "validation": summary,
        "bilinear_validation": bilinear_summary,
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
