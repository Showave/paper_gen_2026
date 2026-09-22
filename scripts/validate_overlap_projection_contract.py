#!/usr/bin/env python3
"""Validate source-complete, multi-field lexical-overlap projections.

The committed contract deliberately leaves source-specific field inventories
unreviewed.  This validator checks exact source/role coverage and validates the
shape and binding of generated or reviewed projection attestations.  Its
self-test uses generated field names only and is not data or model evidence.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "experiments" / "overlap_projection_contract.json"
DEFAULT_REGISTRY = ROOT / "experiments" / "public_source_registry.json"
DEFAULT_PREREGISTRATION = ROOT / "experiments" / "pilot_preregistration.json"
HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")
SLOT_RE = re.compile(r"^[a-z][a-z0-9-]*$")
ROLE_REQUIREMENTS = {
    ("sft", "candidate-general"): {"model-input", "training-target"},
    ("sft", "candidate-math"): {"model-input", "training-target"},
    ("sft", "candidate-code"): {"model-input", "training-target"},
    ("sft", "sealed-code-benchmark"): {"protected-input", "protected-target"},
    ("rl", "prompt-train"): {"model-input", "protected-target"},
    ("eval", "sealed-root-frame"): {"protected-input", "protected-target"},
}


class ProjectionContractError(ValueError):
    """Raised when an overlap-projection contract or attestation is invalid."""


def fail(message: str) -> None:
    raise ProjectionContractError(message)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        fail(f"file not found: {path}")
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON in {path}: {exc}")


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        fail(f"value is not canonical JSON: {exc}")


def digest_value(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and HEX_64_RE.fullmatch(value) is not None


def registry_entries(registry: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(registry, dict) or registry.get("schema_version") != 1:
        fail("source registry must be a schema-version 1 object")
    sources = registry.get("sources")
    if not isinstance(sources, list) or not sources:
        fail("source registry has no sources")
    entries: dict[str, dict[str, Any]] = {}
    for entry in sources:
        key = entry.get("key") if isinstance(entry, dict) else None
        if not isinstance(key, str) or not key or key in entries:
            fail("source registry keys must be unique nonempty strings")
        entries[key] = entry
    return entries


def expected_source_roles(
    preregistration: Any, registry: Any
) -> dict[tuple[str, str, str], dict[str, Any]]:
    if not isinstance(preregistration, dict) or preregistration.get(
        "schema_version"
    ) != 1:
        fail("pilot preregistration must be a schema-version 1 object")
    entries = registry_entries(registry)
    expected: dict[tuple[str, str, str], dict[str, Any]] = {}
    for paper in ("sft", "rl", "eval"):
        section = preregistration.get(paper)
        keys = section.get("source_keys") if isinstance(section, dict) else None
        if not isinstance(keys, list) or not keys:
            fail(f"{paper} preregistration has no source_keys")
        for source_key in keys:
            entry = entries.get(source_key)
            if not isinstance(entry, dict) or entry.get("kind") != "dataset":
                fail(f"{paper} source {source_key!r} is not a registered dataset")
            roles = [
                usage.split(":", 1)[1]
                for usage in entry.get("usage", [])
                if isinstance(usage, str) and usage.startswith(paper + ":")
            ]
            if len(roles) != 1 or (paper, roles[0]) not in ROLE_REQUIREMENTS:
                fail(
                    f"{paper} source {source_key!r} must have one supported role"
                )
            expected[(paper, source_key, roles[0])] = entry
    return expected


def validate_contract(
    contract: Any, registry: Any, preregistration: Any
) -> dict[str, Any]:
    required_top_level = {
        "schema_version",
        "artifact_type",
        "execution_status",
        "evidence_status",
        "normalization",
        "comparison_unit",
        "field_classes",
        "source_projections",
        "review_requirements",
        "failure_policy",
        "limitations",
    }
    if (
        not isinstance(contract, dict)
        or set(contract) != required_top_level
        or contract.get("schema_version") != 1
        or contract.get("artifact_type")
        != "multi_field_overlap_projection_contract"
        or contract.get("normalization")
        != "unicode-nfkc-casefold-whitespace-v1"
        or contract.get("execution_status")
        not in {"unfrozen_review_blocker", "reviewed_pass"}
    ):
        fail("overlap-projection contract has an unsupported schema")
    field_classes = contract.get("field_classes")
    if (
        not isinstance(field_classes, dict)
        or set(field_classes)
        != {
            "model-input",
            "training-target",
            "protected-input",
            "protected-target",
        }
        or not all(
            isinstance(description, str) and description.strip()
            for description in field_classes.values()
        )
    ):
        fail("overlap-projection field classes are incomplete")
    comparison = contract.get("comparison_unit")
    if (
        not isinstance(comparison, dict)
        or set(comparison)
        != {
            "field_segment",
            "record_aggregate",
            "cross_field_rule",
            "pair_identity",
            "empty_or_missing_rule",
        }
        or not all(
            isinstance(description, str) and description.strip()
            for description in comparison.values()
        )
    ):
        fail("overlap-projection comparison unit is incomplete")

    expected = expected_source_roles(preregistration, registry)
    declarations = contract.get("source_projections")
    if not isinstance(declarations, list) or not declarations:
        fail("overlap-projection contract has no source declarations")
    observed: dict[tuple[str, str, str], dict[str, Any]] = {}
    pending = 0
    for declaration in declarations:
        if not isinstance(declaration, dict):
            fail("source-projection declarations must be objects")
        paper = declaration.get("paper")
        source_key = declaration.get("source_key")
        role = declaration.get("role")
        key = (paper, source_key, role)
        entry = expected.get(key)
        requirements = ROLE_REQUIREMENTS.get((paper, role))
        status = declaration.get("projection_status")
        if (
            entry is None
            or key in observed
            or declaration.get("source_revision") != entry.get("revision")
            or not isinstance(declaration.get("required_field_classes"), list)
            or set(declaration["required_field_classes"]) != requirements
            or len(declaration["required_field_classes"]) != len(requirements)
            or status not in {"unfrozen_site_schema", "reviewed_pass"}
        ):
            fail("source projection is missing, duplicated, stale, or misclassified")
        if status == "reviewed_pass":
            if (
                not is_sha256(declaration.get("field_inventory_sha256"))
                or not isinstance(declaration.get("review_record"), str)
                or not declaration["review_record"].strip()
            ):
                fail("reviewed source projection needs an inventory hash and record")
        else:
            pending += 1
            if (
                declaration.get("field_inventory_sha256") is not None
                or declaration.get("review_record") is not None
            ):
                fail("unreviewed source projection cannot carry approval evidence")
        observed[key] = declaration
    if set(observed) != set(expected):
        fail("overlap-projection contract does not cover every pilot source and role")
    if contract["execution_status"] == "reviewed_pass":
        if pending:
            fail("reviewed overlap-projection contract contains pending sources")
    elif pending == 0:
        fail("unfrozen overlap-projection contract must retain a pending blocker")
    return {
        "execution_status": contract["execution_status"],
        "empirical_evidence": False,
        "source_projections": len(observed),
        "pending_source_projections": pending,
        "contract_sha256": digest_value(contract),
    }


def field_inventory_digest(attestation: dict[str, Any]) -> str:
    return digest_value(
        {
            "paper": attestation.get("paper"),
            "source_key": attestation.get("source_key"),
            "source_revision": attestation.get("source_revision"),
            "role": attestation.get("role"),
            "exporter": attestation.get("exporter"),
            "fields": attestation.get("fields"),
        }
    )


def validate_attestation(
    attestation: Any,
    content_projection: Any,
    contract: dict[str, Any],
    registry: dict[str, Any],
    preregistration: dict[str, Any],
) -> dict[str, Any]:
    validate_contract(contract, registry, preregistration)
    if (
        not isinstance(attestation, dict)
        or attestation.get("schema_version") != 1
        or attestation.get("artifact_type")
        != "multi_field_overlap_projection_attestation"
        or attestation.get("contract_sha256") != digest_value(contract)
        or attestation.get("execution_class") not in {"real", "synthetic-audit"}
        or attestation.get("empirical_evidence") is not False
    ):
        fail("overlap-projection attestation has an invalid schema or binding")
    key = (
        attestation.get("paper"),
        attestation.get("source_key"),
        attestation.get("role"),
    )
    declarations = {
        (
            declaration["paper"],
            declaration["source_key"],
            declaration["role"],
        ): declaration
        for declaration in contract["source_projections"]
    }
    declaration = declarations.get(key)
    exporter = attestation.get("exporter")
    fields = attestation.get("fields")
    if (
        declaration is None
        or attestation.get("source_revision")
        != declaration.get("source_revision")
        or not isinstance(exporter, dict)
        or set(exporter) != {"name", "version", "implementation_sha256"}
        or not isinstance(exporter["name"], str)
        or not exporter["name"].strip()
        or not isinstance(exporter["version"], str)
        or not exporter["version"].strip()
        or not is_sha256(exporter["implementation_sha256"])
        or not isinstance(fields, list)
        or not fields
    ):
        fail("overlap-projection attestation source or exporter is invalid")
    slots: set[str] = set()
    paths: set[str] = set()
    observed_classes: set[str] = set()
    projected: list[dict[str, str]] = []
    for field in fields:
        slot = field.get("slot") if isinstance(field, dict) else None
        field_class = field.get("field_class") if isinstance(field, dict) else None
        path = field.get("field_path") if isinstance(field, dict) else None
        flattening = (
            field.get("structured_flattening") if isinstance(field, dict) else None
        )
        if (
            not isinstance(slot, str)
            or SLOT_RE.fullmatch(slot) is None
            or slot in slots
            or field_class not in contract["field_classes"]
            or not isinstance(path, str)
            or not path
            or path in paths
            or any(not component for component in path.split("."))
            or not isinstance(flattening, str)
            or not flattening.strip()
        ):
            fail("overlap-projection field is invalid or duplicated")
        slots.add(slot)
        paths.add(path)
        observed_classes.add(field_class)
        projected.append({"slot": slot, "field": path})
    required_classes = set(declaration["required_field_classes"])
    if not required_classes.issubset(observed_classes):
        fail("overlap-projection attestation omits a required field class")
    if content_projection != projected:
        fail("materialization projection differs from the reviewed field inventory")
    if attestation.get("field_inventory_sha256") != field_inventory_digest(
        attestation
    ):
        fail("overlap-projection field-inventory hash does not reproduce")

    if attestation["execution_class"] == "real":
        if (
            contract["execution_status"] != "reviewed_pass"
            or declaration["projection_status"] != "reviewed_pass"
            or attestation.get("projection_status") != "reviewed_pass"
            or attestation.get("field_inventory_sha256")
            != declaration["field_inventory_sha256"]
            or attestation.get("review_record") != declaration["review_record"]
        ):
            fail("real projection attestation is not contract-reviewed")
    elif (
        attestation.get("projection_status") != "generated_pass"
        or attestation.get("review_record") is not None
    ):
        fail("synthetic projection attestation cannot claim a real review")
    return {
        "execution_class": attestation["execution_class"],
        "empirical_evidence": False,
        "fields": len(fields),
        "field_classes": sorted(observed_classes),
        "field_inventory_sha256": attestation["field_inventory_sha256"],
    }


def generated_attestation(
    contract: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    declaration = contract["source_projections"][0]
    fields = [
        {
            "slot": "model-input",
            "field_class": "model-input",
            "field_path": "generated_prompt",
            "structured_flattening": "generated scalar UTF-8 fixture",
        },
        {
            "slot": "training-target",
            "field_class": "training-target",
            "field_path": "generated_response",
            "structured_flattening": "generated scalar UTF-8 fixture",
        },
    ]
    attestation = {
        "schema_version": 1,
        "artifact_type": "multi_field_overlap_projection_attestation",
        "execution_class": "synthetic-audit",
        "empirical_evidence": False,
        "contract_sha256": digest_value(contract),
        "paper": declaration["paper"],
        "source_key": declaration["source_key"],
        "source_revision": declaration["source_revision"],
        "role": declaration["role"],
        "projection_status": "generated_pass",
        "review_record": None,
        "exporter": {
            "name": "generated-overlap-projection-fixture",
            "version": "1",
            "implementation_sha256": hashlib.sha256(
                b"generated-overlap-projection-exporter"
            ).hexdigest(),
        },
        "fields": fields,
        "field_inventory_sha256": None,
    }
    attestation["field_inventory_sha256"] = field_inventory_digest(attestation)
    projection = [
        {"slot": field["slot"], "field": field["field_path"]} for field in fields
    ]
    return attestation, projection


def expect_attestation_rejection(
    attestation: dict[str, Any],
    projection: list[dict[str, str]],
    contract: dict[str, Any],
    registry: dict[str, Any],
    preregistration: dict[str, Any],
    description: str,
) -> None:
    try:
        validate_attestation(
            attestation, projection, contract, registry, preregistration
        )
    except ProjectionContractError:
        return
    fail(f"self-test accepted {description}")


def self_test(
    contract: dict[str, Any],
    registry: dict[str, Any],
    preregistration: dict[str, Any],
) -> dict[str, Any]:
    summary = validate_contract(contract, registry, preregistration)
    attestation, projection = generated_attestation(contract)
    validated = validate_attestation(
        attestation, projection, contract, registry, preregistration
    )

    missing_class = copy.deepcopy(attestation)
    missing_class["fields"].pop()
    missing_class["field_inventory_sha256"] = field_inventory_digest(missing_class)
    expect_attestation_rejection(
        missing_class,
        projection[:1],
        contract,
        registry,
        preregistration,
        "a missing required field class",
    )

    duplicate_slot = copy.deepcopy(attestation)
    duplicate_slot["fields"][1]["slot"] = duplicate_slot["fields"][0]["slot"]
    duplicate_slot["field_inventory_sha256"] = field_inventory_digest(duplicate_slot)
    expect_attestation_rejection(
        duplicate_slot,
        projection,
        contract,
        registry,
        preregistration,
        "a duplicated canonical slot",
    )

    stale_revision = copy.deepcopy(attestation)
    stale_revision["source_revision"] = "0" * 40
    stale_revision["field_inventory_sha256"] = field_inventory_digest(stale_revision)
    expect_attestation_rejection(
        stale_revision,
        projection,
        contract,
        registry,
        preregistration,
        "a stale source revision",
    )

    mismatched_projection = copy.deepcopy(projection)
    mismatched_projection[0]["field"] = "substituted_prompt"
    expect_attestation_rejection(
        attestation,
        mismatched_projection,
        contract,
        registry,
        preregistration,
        "a substituted materialization projection",
    )

    relabelled = copy.deepcopy(attestation)
    relabelled["execution_class"] = "real"
    relabelled["projection_status"] = "reviewed_pass"
    relabelled["review_record"] = "invented-review"
    expect_attestation_rejection(
        relabelled,
        projection,
        contract,
        registry,
        preregistration,
        "a generated attestation relabelled real",
    )
    return {
        **summary,
        "generated_fields_validated": validated["fields"],
        "fault_injections_rejected": 5,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument(
        "--preregistration", type=Path, default=DEFAULT_PREREGISTRATION
    )
    parser.add_argument("--attestation", type=Path)
    parser.add_argument("--content-projection", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract = read_json(args.contract)
    registry = read_json(args.registry)
    preregistration = read_json(args.preregistration)
    if args.self_test:
        result = self_test(contract, registry, preregistration)
    elif args.attestation or args.content_projection:
        if args.attestation is None or args.content_projection is None:
            fail("--attestation and --content-projection are required together")
        result = validate_attestation(
            read_json(args.attestation),
            read_json(args.content_projection),
            contract,
            registry,
            preregistration,
        )
    else:
        result = validate_contract(contract, registry, preregistration)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except ProjectionContractError as exc:
        raise SystemExit(f"error: {exc}") from exc
