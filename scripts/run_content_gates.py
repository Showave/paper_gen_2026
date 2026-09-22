#!/usr/bin/env python3
"""Audit materialized text for sensitive content and lexical overlap.

The gate consumes the hash-verified output of ``materialize_public_data.py``.
It never prints matched text.  Findings contain only detector names, roles,
record identifiers, canonical slots, and similarity scores.  Every reviewed
content slot remains a separate lexical comparison segment, so unlike field
classes are checked without dilution from record-level concatenation.  The
lexical screen is deliberately described as heuristic; it cannot certify
semantic independence.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import stat
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from validate_external_overlap_audit import (
    ExternalAuditError,
    expected_binding as external_audit_binding,
    validate_contract as validate_external_audit_contract,
    validate_external_overlap_bundle,
)
from validate_overlap_projection_contract import (
    DEFAULT_CONTRACT as DEFAULT_OVERLAP_PROJECTION_CONTRACT,
    DEFAULT_PREREGISTRATION as DEFAULT_PILOT_PREREGISTRATION,
    DEFAULT_REGISTRY as DEFAULT_PUBLIC_SOURCE_REGISTRY,
    ProjectionContractError,
    validate_attestation as validate_overlap_projection_attestation,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPECIFICATION = ROOT / "experiments" / "content_gates.json"
DEFAULT_EXTERNAL_AUDIT_BUNDLE = Path("process") / "external_overlap_audit"
SPACE_RE = re.compile(r"\s+")
HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_U64 = (1 << 64) - 1

PII_PATTERNS = {
    "email": re.compile(
        r"(?<![\w.+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,63}(?![\w.-])",
        re.IGNORECASE,
    ),
    "e164_phone": re.compile(r"(?<![\w+])\+[1-9][0-9]{7,14}(?![0-9])"),
    "us_ssn": re.compile(r"(?<![0-9])[0-9]{3}-[0-9]{2}-[0-9]{4}(?![0-9])"),
}
SECRET_PATTERNS = {
    "aws_access_key_id": re.compile(r"(?<![A-Z0-9])AKIA[A-Z0-9]{16}(?![A-Z0-9])"),
    "github_token": re.compile(r"(?<![A-Za-z0-9])gh[pousr]_[A-Za-z0-9]{36,255}"),
    "huggingface_token": re.compile(r"(?<![A-Za-z0-9])hf_[A-Za-z0-9]{20,}"),
    "private_key_header": re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
    ),
    "sk_style_api_key": re.compile(
        r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}(?![A-Za-z0-9_-])"
    ),
}


class GateError(ValueError):
    """Raised when the content-readiness contract is invalid or fails."""


def fail(message: str) -> None:
    raise GateError(message)


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        fail(f"value is not canonical JSON: {exc}")


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_value(value: Any) -> str:
    return digest_bytes(canonical_bytes(value))


def digest_file(path: Path) -> str:
    hasher = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
    except FileNotFoundError:
        fail(f"file not found: {path}")
    return hasher.hexdigest()


def read_json(path: Path) -> Any:
    value, _ = read_json_with_sha256(path)
    return value


def read_json_with_sha256(path: Path) -> tuple[Any, str]:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
    except FileNotFoundError:
        fail(f"file not found: {path}")
    except OSError as exc:
        fail(f"could not open regular JSON file {path}: {exc}")
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            fail(f"JSON input must be a regular file: {path}")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            payload = handle.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    observed = hashlib.sha256(payload).hexdigest()
    try:
        return json.loads(payload.decode("utf-8")), observed
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail(f"invalid UTF-8 JSON in {path}: {exc}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        fail(f"file not found: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            fail(f"{path}:{line_number}: blank JSONL rows are forbidden")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            fail(f"{path}:{line_number}: invalid JSON: {exc}")
        if not isinstance(value, dict):
            fail(f"{path}:{line_number}: row must be an object")
        rows.append(value)
    if not rows:
        fail(f"{path}: artifact is empty")
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def normalize_text(value: str) -> str:
    return SPACE_RE.sub(" ", unicodedata.normalize("NFKC", value).casefold()).strip()


def nested(value: Any, dotted_path: str) -> Any:
    current = value
    for part in dotted_path.split("."):
        if not part or not isinstance(current, dict) or part not in current:
            fail(f"row is missing projected field {dotted_path!r}")
        current = current[part]
    return current


def confined_path(root: Path, declared: Any) -> Path:
    if not isinstance(declared, str) or not declared:
        fail("artifact path must be a nonempty string")
    relative = Path(declared)
    if relative.is_absolute() or ".." in relative.parts:
        fail(f"artifact path escapes the materialization root: {declared!r}")
    candidate = root / relative
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            fail(f"artifact path traverses a symlink: {declared!r}")
    resolved = candidate.resolve()
    if resolved != root and root not in resolved.parents:
        fail(f"artifact path escapes the materialization root: {declared!r}")
    return resolved


def require_number(value: Any, name: str, lower: float, upper: float) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or not lower <= float(value) <= upper
    ):
        fail(f"{name} must be finite and in [{lower}, {upper}]")
    return float(value)


def validate_specification(value: Any, path: Path) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 1
        or value.get("artifact_type") != "content_readiness_gate_specification"
        or value.get("normalization") != "unicode-nfkc-casefold-whitespace-v1"
    ):
        fail(f"{path}: invalid content gate specification")
    similarity = value.get("similarity")
    if not isinstance(similarity, dict) or similarity.get("method") != (
        "character_5gram_jaccard"
    ):
        fail(f"{path}: unsupported similarity method")
    if (
        similarity.get("comparison_unit") != "canonical_field_segment"
        or similarity.get("record_candidate_signature")
        != (
            "union of per-segment character 5-grams; exhaustive mode evaluates "
            "every cross-record segment pair"
        )
        or similarity.get("pair_identity")
        != (
            "sha256(canonical-json({left_record_id,left_slot,right_record_id,"
            "right_slot}))"
        )
    ):
        fail(f"{path}: multi-field similarity unit or identity is not frozen")
    if similarity.get("minimum_normalized_characters", 0) < 5:
        fail(f"{path}: minimum text length must be at least five")
    for field in (
        "within_role_threshold",
        "cross_role_threshold",
        "training_protected_threshold",
    ):
        require_number(similarity.get(field), field, 0.5, 1.0)
    candidate = similarity.get("candidate_generation")
    if not isinstance(candidate, dict) or candidate.get("method") != (
        "exhaustive_then_deterministic_one_permutation_minhash_lsh"
    ):
        fail(f"{path}: unsupported candidate-generation method")
    integer_fields = (
        "exhaustive_max_rows",
        "minhash_permutations",
        "lsh_bands",
        "rows_per_band",
        "maximum_candidate_pairs",
    )
    if any(
        not isinstance(candidate.get(field), int) or candidate[field] <= 0
        for field in integer_fields
    ):
        fail(f"{path}: candidate-generation sizes must be positive integers")
    if candidate["minhash_permutations"] != (
        candidate["lsh_bands"] * candidate["rows_per_band"]
    ):
        fail(f"{path}: LSH bands and rows do not cover the signature")
    if not isinstance(candidate.get("seed"), str) or not candidate["seed"]:
        fail(f"{path}: candidate-generation seed is required")

    sensitive = value.get("sensitive_content")
    pii_detectors = sensitive.get("pii_detectors") if isinstance(sensitive, dict) else None
    secret_detectors = (
        sensitive.get("secret_detectors") if isinstance(sensitive, dict) else None
    )
    if (
        not isinstance(sensitive, dict)
        or sensitive.get("implementation") != "named-regex-detectors-v1"
        or not isinstance(pii_detectors, list)
        or len(pii_detectors) != len(set(pii_detectors))
        or set(pii_detectors) != set(PII_PATTERNS)
        or not isinstance(secret_detectors, list)
        or len(secret_detectors) != len(set(secret_detectors))
        or set(secret_detectors) != set(SECRET_PATTERNS)
    ):
        fail(f"{path}: invalid sensitive-content detector list")
    classes = value.get("role_classes")
    if not isinstance(classes, dict):
        fail(f"{path}: role classes are required")
    role_sets: list[set[str]] = []
    for name in ("training", "protected"):
        roles = classes.get(name)
        if (
            not isinstance(roles, list)
            or not roles
            or not all(isinstance(role, str) and role for role in roles)
            or len(roles) != len(set(roles))
        ):
            fail(f"{path}: role class {name!r} must be a unique string list")
        role_sets.append(set(roles))
    if role_sets[0].intersection(role_sets[1]):
        fail(f"{path}: training and protected roles must be disjoint")
    policies = value.get("failure_policy")
    required_policies = {
        "pii_in_retained_content",
        "secret_in_retained_content",
        "text_too_short_for_similarity",
        "within_role_near_duplicate",
        "cross_role_near_duplicate",
        "training_protected_near_duplicate",
        "approximate_candidate_generation",
    }
    if (
        not isinstance(policies, dict)
        or set(policies) != required_policies
        or any(policy != "block" for policy in policies.values())
    ):
        fail(f"{path}: every v1 finding must use the fail-closed policy")

    external = value.get("external_overlap_audit")
    expected_paths = {
        "contract": "experiments/external_overlap_audit.json",
        "validator": "scripts/validate_external_overlap_audit.py",
    }
    if (
        not isinstance(external, dict)
        or set(external)
        != {
            "bundle_path",
            "contract_path",
            "contract_sha256",
            "validator_path",
            "validator_sha256",
        }
    ):
        fail(f"{path}: external overlap audit binding is required")
    for name, expected_path in expected_paths.items():
        declared = external.get(f"{name}_path")
        declared_sha256 = external.get(f"{name}_sha256")
        if (
            declared != expected_path
            or not isinstance(declared_sha256, str)
            or not HEX_64_RE.fullmatch(declared_sha256)
            or digest_file(ROOT / expected_path) != declared_sha256
        ):
            fail(f"{path}: external overlap {name} binding is stale")
    if external.get("bundle_path") != str(DEFAULT_EXTERNAL_AUDIT_BUNDLE):
        fail(f"{path}: external overlap bundle path is not frozen")
    return value


@dataclass(frozen=True)
class ScanRow:
    record_id: str
    role: str
    similarity_segments: tuple[tuple[str, str, str], ...]
    sensitive_text: str


def textual_leaves(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from textual_leaves(item)
    elif isinstance(value, dict):
        for key in sorted(value):
            yield str(key)
            yield from textual_leaves(value[key])


def _projection_key(provenance: dict[str, Any]) -> tuple[str, str, str, str]:
    fields = ("source_key", "config", "upstream_split", "requested_role")
    if not all(isinstance(provenance.get(field), str) for field in fields):
        fail("retained row has incomplete provenance for content projection")
    return tuple(provenance[field] for field in fields)  # type: ignore[return-value]


def load_materialized_rows(
    materialization: Path, specification: dict[str, Any]
) -> tuple[list[ScanRow], dict[str, Any]]:
    acquisition_path = confined_path(
        materialization, "acquire/data_manifest.json"
    )
    content_path = confined_path(
        materialization, "process/content_manifest.json"
    )
    acquisition = read_json(acquisition_path)
    content = read_json(content_path)
    if (
        not isinstance(acquisition, dict)
        or acquisition.get("schema_version") != 1
        or acquisition.get("artifact_type") != "row_level_source_materialization"
        or not isinstance(content, dict)
        or content.get("schema_version") != 1
        or content.get("artifact_type") != "normalized_deduplication_manifest"
        or acquisition.get("paper") != content.get("paper")
        or not isinstance(acquisition.get("synthetic_fixture"), bool)
        or acquisition.get("synthetic_fixture")
        is not content.get("synthetic_fixture")
    ):
        fail("materialization manifests are missing or inconsistent")
    provenance_path = confined_path(
        materialization, acquisition.get("row_provenance_path")
    )
    if (
        digest_file(provenance_path) != acquisition.get("row_provenance_sha256")
        or len(read_jsonl(provenance_path)) != acquisition.get("row_count")
    ):
        fail("row provenance does not match the acquisition manifest")
    materialized_rows = 0
    for input_file in acquisition.get("inputs", []):
        if not isinstance(input_file, dict) or not isinstance(
            input_file.get("materialized_path"), str
        ):
            fail("acquisition manifest has an invalid materialized input")
        path = confined_path(materialization, input_file["materialized_path"])
        values = read_jsonl(path)
        if (
            digest_file(path) != input_file.get("materialized_sha256")
            or len(values) != input_file.get("input_rows")
        ):
            fail(f"materialized input does not match acquisition manifest: {path}")
        materialized_rows += len(values)
    if materialized_rows != acquisition.get("row_count"):
        fail("materialized input counts do not reproduce acquisition row count")
    for path_field, hash_field in (
        ("duplicate_suppressed_path", "duplicate_suppressed_sha256"),
        ("quarantine_path", "quarantine_sha256"),
    ):
        artifact_path = confined_path(materialization, content.get(path_field))
        if digest_file(artifact_path) != content.get(hash_field):
            fail(f"processing artifact hash mismatch: {artifact_path}")

    overlap_contract = read_json(DEFAULT_OVERLAP_PROJECTION_CONTRACT)
    source_registry = read_json(DEFAULT_PUBLIC_SOURCE_REGISTRY)
    pilot_preregistration = read_json(DEFAULT_PILOT_PREREGISTRATION)
    projections: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for input_file in acquisition.get("inputs", []):
        if not isinstance(input_file, dict):
            fail("acquisition input declaration must be an object")
        key_fields = ("source_key", "config", "split", "role")
        if not all(isinstance(input_file.get(field), str) for field in key_fields):
            fail("acquisition input declaration has incomplete role identity")
        projection = input_file.get("content_projection")
        if (
            not isinstance(projection, list)
            or not projection
            or not all(
                isinstance(item, dict)
                and isinstance(item.get("slot"), str)
                and isinstance(item.get("field"), str)
                for item in projection
            )
        ):
            fail("acquisition manifest does not retain the content projection")
        try:
            attestation = input_file.get("overlap_projection_attestation")
            validate_overlap_projection_attestation(
                attestation,
                projection,
                overlap_contract,
                source_registry,
                pilot_preregistration,
            )
            expected_execution_class = (
                "synthetic-audit"
                if acquisition.get("synthetic_fixture") is True
                else "real"
            )
            if attestation.get("execution_class") != expected_execution_class:
                fail(
                    "acquisition projection attestation execution class "
                    "does not match the materialization"
                )
            if attestation.get("exporter") != input_file.get("exporter"):
                fail("acquisition projection attestation binds another exporter")
        except ProjectionContractError as exc:
            fail(f"acquisition overlap projection does not verify: {exc}")
        field_class_by_slot = {
            field["slot"]: field["field_class"]
            for field in attestation["fields"]
        }
        key = tuple(input_file[field] for field in key_fields)
        if key in projections:
            fail(f"duplicate acquisition projection for {key}")
        projections[key] = [
            {**item, "field_class": field_class_by_slot[item["slot"]]}
            for item in projection
        ]

    rows: list[ScanRow] = []
    record_ids: set[str] = set()
    role_file_hashes: list[dict[str, Any]] = []
    for role_file in content.get("retained_role_files", []):
        if (
            not isinstance(role_file, dict)
            or not isinstance(role_file.get("path"), str)
            or not isinstance(role_file.get("role"), str)
            or not HEX_64_RE.fullmatch(str(role_file.get("sha256", "")))
        ):
            fail("content manifest has an invalid retained-role declaration")
        path = confined_path(materialization, role_file["path"])
        observed_hash = digest_file(path)
        if observed_hash != role_file["sha256"]:
            fail(f"retained role file hash mismatch: {path}")
        values = read_jsonl(path)
        if len(values) != role_file.get("rows"):
            fail(f"retained role file row count mismatch: {path}")
        role_file_hashes.append(
            {
                "path": role_file["path"],
                "role": role_file["role"],
                "rows": len(values),
                "sha256": observed_hash,
            }
        )
        for value in values:
            provenance = value.get("provenance")
            raw_row = value.get("row")
            if not isinstance(provenance, dict) or not isinstance(raw_row, dict):
                fail(f"{path}: retained row needs provenance and raw row objects")
            if digest_value(raw_row) != provenance.get("raw_row_sha256"):
                fail(f"{path}: retained raw row hash differs from provenance")
            if (
                provenance.get("requested_role") != role_file["role"]
                or provenance.get("normalization") != specification["normalization"]
            ):
                fail(f"{path}: retained row role or normalization mismatch")
            projection = projections.get(_projection_key(provenance))
            if projection is None:
                fail(f"{path}: no content projection matches retained row")
            selected = {
                item["slot"]: nested(raw_row, item["field"]) for item in projection
            }
            if not all(
                isinstance(projected, str) and projected.strip()
                for projected in selected.values()
            ):
                fail(f"{path}: projected content must be nonempty text")
            if digest_value(selected) != provenance.get("exact_content_sha256"):
                fail(f"{path}: projected content hash differs from provenance")
            normalized_with_slots = "\u241d".join(
                f"{slot}\u241e{normalize_text(projected)}"
                for slot, projected in sorted(selected.items())
            )
            if digest_bytes(normalized_with_slots.encode("utf-8")) != provenance.get(
                "normalized_content_sha256"
            ):
                fail(f"{path}: normalized content hash differs from provenance")
            record_id = provenance.get("record_id")
            if (
                not isinstance(record_id, str)
                or not HEX_64_RE.fullmatch(record_id)
                or record_id in record_ids
            ):
                fail(
                    f"{path}: retained record identifiers must be unique SHA-256 "
                    "digests"
                )
            record_ids.add(record_id)
            rows.append(
                ScanRow(
                    record_id=record_id,
                    role=role_file["role"],
                    similarity_segments=tuple(
                        (
                            item["slot"],
                            item["field_class"],
                            selected[item["slot"]],
                        )
                        for item in sorted(projection, key=lambda item: item["slot"])
                    ),
                    sensitive_text="\n".join(textual_leaves(raw_row)),
                )
            )
    if not rows or len(rows) != content.get("retained_row_count"):
        fail("retained rows do not reproduce the content manifest count")
    return rows, {
        "paper": content["paper"],
        "synthetic_fixture": bool(content.get("synthetic_fixture")),
        "acquisition_manifest_sha256": digest_file(acquisition_path),
        "content_manifest_sha256": digest_file(content_path),
        "role_files": sorted(role_file_hashes, key=lambda row: row["path"]),
        "retained_record_ids_sha256": digest_value(sorted(record_ids)),
    }


def character_ngrams(text: str, size: int = 5) -> frozenset[str]:
    normalized = normalize_text(text)
    if len(normalized) < size:
        return frozenset()
    return frozenset(
        normalized[index : index + size]
        for index in range(len(normalized) - size + 1)
    )


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left.intersection(right)) / len(left.union(right))


def densified_one_permutation_signature(
    ngrams: frozenset[str], width: int, seed: str
) -> tuple[int, ...]:
    bins = [MAX_U64] * width
    seed_bytes = seed.encode("utf-8")
    for ngram in ngrams:
        digest = hashlib.blake2b(
            seed_bytes + b"\0" + ngram.encode("utf-8"), digest_size=16
        ).digest()
        bucket = int.from_bytes(digest[:8], "big") % width
        value = int.from_bytes(digest[8:], "big")
        bins[bucket] = min(bins[bucket], value)
    populated = [index for index, value in enumerate(bins) if value != MAX_U64]
    if not populated:
        return tuple(bins)
    for index, value in enumerate(bins):
        if value != MAX_U64:
            continue
        distance, donor = min(
            ((candidate - index) % width, candidate) for candidate in populated
        )
        payload = (
            seed_bytes
            + b"\0densify\0"
            + index.to_bytes(4, "big")
            + distance.to_bytes(4, "big")
            + bins[donor].to_bytes(8, "big")
        )
        bins[index] = int.from_bytes(
            hashlib.blake2b(payload, digest_size=8).digest(), "big"
        )
    return tuple(bins)


def candidate_pairs(
    ngrams: list[frozenset[str]], configuration: dict[str, Any]
) -> tuple[Iterable[tuple[int, int]], str, int]:
    count = len(ngrams)
    if count <= configuration["exhaustive_max_rows"]:
        pair_count = count * (count - 1) // 2
        if pair_count > configuration["maximum_candidate_pairs"]:
            fail("exhaustive near-duplicate comparison exceeds the frozen maximum")
        return itertools.combinations(range(count), 2), "exhaustive", pair_count
    width = configuration["minhash_permutations"]
    rows_per_band = configuration["rows_per_band"]
    bands = configuration["lsh_bands"]
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = {}
    pairs: set[tuple[int, int]] = set()
    for index, tokens in enumerate(ngrams):
        if not tokens:
            continue
        signature = densified_one_permutation_signature(
            tokens, width, configuration["seed"]
        )
        for band in range(bands):
            start = band * rows_per_band
            key = (band, signature[start : start + rows_per_band])
            previous = buckets.setdefault(key, [])
            pairs.update((other, index) for other in previous)
            if len(pairs) > configuration["maximum_candidate_pairs"]:
                fail(
                    "near-duplicate candidate generation exceeded the frozen "
                    "maximum; use a reviewed scalable site-local audit"
                )
            previous.append(index)
    ordered = sorted(pairs)
    return ordered, "deterministic_one_permutation_minhash_lsh", len(ordered)


def sensitive_findings(
    rows: Iterable[ScanRow], detector_names: list[str], patterns: dict[str, re.Pattern[str]]
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for row in rows:
        normalized = unicodedata.normalize("NFKC", row.sensitive_text)
        for detector in detector_names:
            if patterns[detector].search(normalized):
                findings.append(
                    {
                        "detector": detector,
                        "record_id": row.record_id,
                        "role": row.role,
                    }
                )
    return sorted(
        findings,
        key=lambda finding: (
            finding["detector"],
            finding["role"],
            finding["record_id"],
        ),
    )


def near_duplicate_findings(
    rows: list[ScanRow], specification: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, str]], str, int]:
    similarity = specification["similarity"]
    minimum = similarity["minimum_normalized_characters"]
    too_short = [
        {
            "record_id": row.record_id,
            "role": row.role,
            "slot": slot,
            "field_class": field_class,
        }
        for row in rows
        for slot, field_class, text in row.similarity_segments
        if len(normalize_text(text)) < minimum
    ]
    segment_ngrams = [
        {
            slot: (
                field_class,
                (
                    character_ngrams(text)
                    if len(normalize_text(text)) >= minimum
                    else frozenset()
                ),
            )
            for slot, field_class, text in row.similarity_segments
        }
        for row in rows
    ]
    ngrams = [
        (
            frozenset().union(
                *(field_ngrams for _, field_ngrams in by_slot.values())
            )
            if by_slot
            else frozenset()
        )
        for by_slot in segment_ngrams
    ]
    pairs, method, pair_count = candidate_pairs(
        ngrams, similarity["candidate_generation"]
    )
    training = set(specification["role_classes"]["training"])
    protected = set(specification["role_classes"]["protected"])
    findings: list[dict[str, Any]] = []
    for left_index, right_index in pairs:
        left = rows[left_index]
        right = rows[right_index]
        if not ngrams[left_index] or not ngrams[right_index]:
            continue
        for left_slot, (
            left_field_class,
            left_ngrams,
        ) in segment_ngrams[left_index].items():
            for right_slot, (
                right_field_class,
                right_ngrams,
            ) in segment_ngrams[right_index].items():
                if not left_ngrams or not right_ngrams:
                    continue
                field_classes = {left_field_class, right_field_class}
                if (
                    any(name.startswith("protected-") for name in field_classes)
                    and any(
                        not name.startswith("protected-")
                        for name in field_classes
                    )
                ):
                    finding_type = "training_protected_near_duplicate"
                    threshold = similarity["training_protected_threshold"]
                elif left.role == right.role:
                    finding_type = "within_role_near_duplicate"
                    threshold = similarity["within_role_threshold"]
                elif (
                    left.role in training
                    and right.role in protected
                    or right.role in training
                    and left.role in protected
                ):
                    finding_type = "training_protected_near_duplicate"
                    threshold = similarity["training_protected_threshold"]
                else:
                    finding_type = "cross_role_near_duplicate"
                    threshold = similarity["cross_role_threshold"]
                length_upper_bound = min(
                    len(left_ngrams), len(right_ngrams)
                ) / max(len(left_ngrams), len(right_ngrams))
                if length_upper_bound < threshold:
                    continue
                similarity_value = jaccard(left_ngrams, right_ngrams)
                if similarity_value < threshold:
                    continue
                ordered = sorted(
                    (
                        (left.record_id, left.role, left_slot),
                        (right.record_id, right.role, right_slot),
                    )
                )
                findings.append(
                    {
                        "finding_type": finding_type,
                        "pair_id": digest_value(
                            {
                                "left_record_id": ordered[0][0],
                                "left_slot": ordered[0][2],
                                "right_record_id": ordered[1][0],
                                "right_slot": ordered[1][2],
                            }
                        ),
                        "left_record_id": ordered[0][0],
                        "left_role": ordered[0][1],
                        "left_slot": ordered[0][2],
                        "left_field_class": (
                            left_field_class
                            if ordered[0][0] == left.record_id
                            else right_field_class
                        ),
                        "right_record_id": ordered[1][0],
                        "right_role": ordered[1][1],
                        "right_slot": ordered[1][2],
                        "right_field_class": (
                            right_field_class
                            if ordered[1][0] == right.record_id
                            else left_field_class
                        ),
                        "character_5gram_jaccard": round(similarity_value, 12),
                        "threshold": threshold,
                    }
                )
    return (
        sorted(
            findings,
            key=lambda finding: (
                finding["left_record_id"],
                finding["left_slot"],
                finding["right_record_id"],
                finding["right_slot"],
            ),
        ),
        too_short,
        method,
        pair_count,
    )


def ledger_status(blocking_failures: list[dict[str, Any]]) -> str:
    if not blocking_failures:
        return "pass"
    if {failure["gate"] for failure in blocking_failures} == {
        "approximate_candidate_generation"
    }:
        return "inconclusive"
    return "blocked"


def build_ledger(
    materialization: Path, specification_path: Path
) -> dict[str, Any]:
    specification_value, specification_sha256 = read_json_with_sha256(
        specification_path
    )
    specification = validate_specification(specification_value, specification_path)
    rows, inputs = load_materialized_rows(materialization, specification)
    sensitive = specification["sensitive_content"]
    pii = sensitive_findings(rows, sensitive["pii_detectors"], PII_PATTERNS)
    secrets = sensitive_findings(
        rows, sensitive["secret_detectors"], SECRET_PATTERNS
    )
    external_result: dict[str, Any] | None = None
    external_bundle = materialization / DEFAULT_EXTERNAL_AUDIT_BUNDLE
    if external_bundle.exists():
        resolved_bundle = external_bundle.resolve()
        if (
            external_bundle.is_symlink()
            or materialization.resolve() not in resolved_bundle.parents
        ):
            fail("external overlap bundle must stay inside the materialization")
        if len(rows) <= specification["similarity"]["candidate_generation"][
            "exhaustive_max_rows"
        ]:
            fail("external overlap bundle is forbidden for an exhaustive artifact")
        if any(len(row.similarity_segments) != 1 for row in rows):
            fail(
                "external overlap contract v1 is record-level and cannot clear "
                "a multi-field artifact; freeze a segment-pair engine first"
            )
        minimum = specification["similarity"]["minimum_normalized_characters"]
        too_short = [
            {
                "record_id": row.record_id,
                "role": row.role,
                "slot": row.similarity_segments[0][0],
                "field_class": row.similarity_segments[0][1],
            }
            for row in rows
            if len(normalize_text(row.similarity_segments[0][2])) < minimum
        ]
        ngrams_by_record_id = {
            row.record_id: (
                character_ngrams(row.similarity_segments[0][2])
                if len(normalize_text(row.similarity_segments[0][2])) >= minimum
                else frozenset()
            )
            for row in rows
        }
        external_configuration = specification["external_overlap_audit"]
        contract_path = ROOT / external_configuration["contract_path"]
        try:
            contract_value, contract_sha256 = read_json_with_sha256(contract_path)
            if contract_sha256 != external_configuration["contract_sha256"]:
                fail("external overlap contract changed after gate validation")
            contract = validate_external_audit_contract(
                contract_value, contract_path
            )
            binding = external_audit_binding(
                paper=inputs["paper"],
                retained_rows=len(rows),
                retained_record_ids_sha256=inputs[
                    "retained_record_ids_sha256"
                ],
                gate_specification_sha256=specification_sha256,
                acquisition_manifest_sha256=inputs[
                    "acquisition_manifest_sha256"
                ],
                content_manifest_sha256=inputs["content_manifest_sha256"],
                role_files=inputs["role_files"],
            )
            external_result = validate_external_overlap_bundle(
                resolved_bundle,
                contract=contract,
                binding=binding,
                roles_by_record_id={row.record_id: row.role for row in rows},
                ngrams_by_record_id=ngrams_by_record_id,
                gate_specification=specification,
            )
        except ExternalAuditError as exc:
            fail(f"external overlap audit failed validation: {exc}")
        near_duplicates = external_result["findings"]
        candidate_method = "external_exact_all_pairs_sharded"
        pairs_scored = external_result["pairs_compared"]
    else:
        (
            near_duplicates,
            too_short,
            candidate_method,
            pairs_scored,
        ) = near_duplicate_findings(rows, specification)
    counts = {
        "pii_in_retained_content": len(pii),
        "secret_in_retained_content": len(secrets),
        "text_too_short_for_similarity": len(too_short),
        "within_role_near_duplicate": sum(
            finding["finding_type"] == "within_role_near_duplicate"
            for finding in near_duplicates
        ),
        "cross_role_near_duplicate": sum(
            finding["finding_type"] == "cross_role_near_duplicate"
            for finding in near_duplicates
        ),
        "training_protected_near_duplicate": sum(
            finding["finding_type"] == "training_protected_near_duplicate"
            for finding in near_duplicates
        ),
        "approximate_candidate_generation": int(
            candidate_method
            not in {"exhaustive", "external_exact_all_pairs_sharded"}
        ),
    }
    blocking = [
        {"gate": gate, "count": count}
        for gate, count in sorted(counts.items())
        if count > 0
    ]
    return {
        "schema_version": 1,
        "artifact_type": "content_readiness_ledger",
        "paper": inputs["paper"],
        "synthetic_fixture": inputs["synthetic_fixture"],
        "empirical_evidence": False,
        "evidence_scope": (
            "Software and data-contract audit only; no model-training or "
            "human-study result."
        ),
        "status": ledger_status(blocking),
        "gate_specification_path": str(
            specification_path.resolve().relative_to(ROOT.resolve())
            if ROOT.resolve() in specification_path.resolve().parents
            else specification_path.resolve()
        ),
        "gate_specification_sha256": specification_sha256,
        "gate_implementation_sha256": digest_file(Path(__file__).resolve()),
        "acquisition_manifest_sha256": inputs["acquisition_manifest_sha256"],
        "content_manifest_sha256": inputs["content_manifest_sha256"],
        "retained_record_ids_sha256": inputs["retained_record_ids_sha256"],
        "role_files": inputs["role_files"],
        "summary": {
            "retained_rows_scanned": len(rows),
            "candidate_generation": candidate_method,
            "candidate_pairs_scored": pairs_scored,
            **counts,
            "blocking_failure_classes": len(blocking),
        },
        "findings": {
            "pii": pii,
            "secrets": secrets,
            "too_short_for_similarity": too_short,
            "near_duplicates": near_duplicates,
        },
        "external_overlap_audit": (
            None
            if external_result is None
            else {
                key: value
                for key, value in external_result.items()
                if key != "findings"
            }
        ),
        "blocking_failures": blocking,
        "checks": {
            "materialization_hash_chain_verified": True,
            "no_pii_detector_matches": not pii,
            "no_secret_detector_matches": not secrets,
            "all_rows_similarity_eligible": not too_short,
            "no_lexical_near_duplicates": not near_duplicates,
            "similarity_candidate_generation_recall_complete": (
                candidate_method
                in {"exhaustive", "external_exact_all_pairs_sharded"}
            ),
            "manual_clearance_inferred_from_pass": False,
            "semantic_independence_claimed": False,
        },
        "limitations": specification["limitations"],
    }


def audit_materialization(
    materialization: Path,
    specification_path: Path = DEFAULT_SPECIFICATION,
    *,
    write_ledger: bool = True,
    require_pass: bool = True,
) -> dict[str, Any]:
    materialization = materialization.resolve()
    specification_path = specification_path.resolve()
    ledger = build_ledger(materialization, specification_path)
    if write_ledger:
        write_json(
            materialization / "process" / "content_readiness_ledger.json", ledger
        )
    if require_pass and ledger["status"] != "pass":
        classes = ", ".join(
            f"{finding['gate']}={finding['count']}"
            for finding in ledger["blocking_failures"]
        )
        fail(f"retained content failed readiness gates: {classes}")
    return ledger


def verify_readiness_ledger(
    materialization: Path,
    specification_path: Path = DEFAULT_SPECIFICATION,
    *,
    allow_inconclusive: bool = False,
) -> None:
    materialization = materialization.resolve()
    specification_path = specification_path.resolve()
    ledger = read_json(
        confined_path(materialization, "process/content_readiness_ledger.json")
    )
    expected = build_ledger(materialization, specification_path)
    if not isinstance(ledger, dict) or ledger != expected:
        fail("content-readiness ledger is missing, stale, or blocked")
    if allow_inconclusive and ledger["status"] == "inconclusive":
        if ledger["blocking_failures"] != [
            {"gate": "approximate_candidate_generation", "count": 1}
        ]:
            fail("inconclusive ledger contains a non-overlap blocking failure")
        return
    if ledger["status"] != "pass" or ledger["blocking_failures"]:
        fail("content-readiness ledger is not passing")


def run_self_test() -> None:
    specification = validate_specification(
        read_json(DEFAULT_SPECIFICATION), DEFAULT_SPECIFICATION
    )
    def fixture(
        record_id: str,
        role: str,
        text: str,
        sensitive_text: str | None = None,
        slot: str = "dedup-text",
    ) -> ScanRow:
        return ScanRow(
            record_id,
            role,
            ((slot, "model-input", text),),
            sensitive_text or text,
        )

    prefix = "This deliberately generated fixture sentence checks the scanner"
    rows = [
        fixture("clean-a", "candidate-general", prefix + " with stable wording."),
        fixture(
            "clean-b",
            "candidate-general",
            prefix + " with stable wording!",
        ),
        fixture(
            "pii-email",
            "candidate-math",
            "Generated contact: fixture.user@example.invalid",
        ),
        fixture(
            "pii-phone",
            "candidate-math",
            "Generated phone: " + "+12025550123",
        ),
        fixture(
            "pii-ssn",
            "candidate-math",
            "Generated identifier: " + "000-00-0000",
        ),
        fixture(
            "secret-aws",
            "candidate-code",
            "A generated safe projection for raw-field scanning.",
            "Generated unprojected token: " + "AKIA" + ("X" * 16),
        ),
        fixture(
            "secret-github",
            "candidate-code",
            "Generated token: " + "ghp_" + ("x" * 36),
        ),
        fixture(
            "secret-huggingface",
            "candidate-code",
            "Generated token: " + "hf_" + ("x" * 20),
        ),
        fixture(
            "secret-private-key",
            "candidate-code",
            "Generated marker: " + "-----BEGIN " + "PRIVATE KEY-----",
        ),
        fixture(
            "secret-sk",
            "candidate-code",
            "Generated token: " + "sk-" + ("x" * 20),
        ),
    ]
    pii = sensitive_findings(
        rows,
        specification["sensitive_content"]["pii_detectors"],
        PII_PATTERNS,
    )
    secrets = sensitive_findings(
        rows,
        specification["sensitive_content"]["secret_detectors"],
        SECRET_PATTERNS,
    )
    duplicates, too_short, method, _ = near_duplicate_findings(
        rows[:2], specification
    )
    if {finding["detector"] for finding in pii} != set(PII_PATTERNS):
        fail("self-test did not detect every generated PII fixture")
    if {finding["detector"] for finding in secrets} != set(SECRET_PATTERNS):
        fail("self-test did not detect every generated secret fixture")
    if (
        method != "exhaustive"
        or too_short
        or len(duplicates) != 1
        or duplicates[0]["finding_type"] != "within_role_near_duplicate"
    ):
        fail("self-test did not detect the generated lexical near-duplicate")
    cross_field_text = (
        "A generated protected answer explains a deliberately copied derivation."
    )
    cross_field_rows = [
        ScanRow(
            "multi-training",
            "candidate-math",
            (
                (
                    "model-input",
                    "model-input",
                    "A distinct generated training prompt.",
                ),
                ("training-target", "training-target", cross_field_text),
            ),
            "generated training row",
        ),
        ScanRow(
            "multi-protected",
            "sealed-code-benchmark",
            (
                ("protected-input", "protected-input", cross_field_text),
                (
                    "protected-target",
                    "protected-target",
                    "A distinct generated protected target.",
                ),
            ),
            "generated protected row",
        ),
    ]
    cross_field_findings = near_duplicate_findings(
        cross_field_rows, specification
    )[0]
    if (
        len(cross_field_findings) != 1
        or cross_field_findings[0]["finding_type"]
        != "training_protected_near_duplicate"
        or {
            cross_field_findings[0]["left_slot"],
            cross_field_findings[0]["right_slot"],
        }
        != {"training-target", "protected-input"}
    ):
        fail("self-test missed a cross-class multi-field overlap")
    same_role_base = "generated protected overlap alpha beta gamma delta epsilon"
    same_role_findings = near_duplicate_findings(
        [
            ScanRow(
                "same-role-input",
                "prompt-train",
                (("model-input", "model-input", same_role_base),),
                same_role_base,
            ),
            ScanRow(
                "same-role-target",
                "prompt-train",
                (
                    (
                        "protected-target",
                        "protected-target",
                        same_role_base + " extended suffix",
                    ),
                ),
                same_role_base + " extended suffix",
            ),
        ],
        specification,
    )[0]
    if (
        len(same_role_findings) != 1
        or same_role_findings[0]["finding_type"]
        != "training_protected_near_duplicate"
        or not (
            specification["similarity"]["training_protected_threshold"]
            <= same_role_findings[0]["character_5gram_jaccard"]
            < specification["similarity"]["within_role_threshold"]
        )
    ):
        fail("self-test did not apply field classes before the role threshold")
    _, short_rows, _, _ = near_duplicate_findings(
        [fixture("short", "candidate-general", "tiny")], specification
    )
    if short_rows != [
        {
            "record_id": "short",
            "role": "candidate-general",
            "slot": "dedup-text",
            "field_class": "model-input",
        }
    ]:
        fail("self-test did not block text below the similarity minimum")
    approximate_rows = [
        fixture(
            f"approx-{index}",
            "candidate-general",
            hashlib.sha256(f"generated-{index}".encode("utf-8")).hexdigest(),
        )
        for index in range(
            specification["similarity"]["candidate_generation"][
                "exhaustive_max_rows"
            ]
            + 1
        )
    ]
    _, _, approximate_method, _ = near_duplicate_findings(
        approximate_rows, specification
    )
    if (
        approximate_method != "deterministic_one_permutation_minhash_lsh"
        or ledger_status(
            [{"gate": "approximate_candidate_generation", "count": 1}]
        )
        != "inconclusive"
    ):
        fail("self-test allowed approximate candidate generation to pass")
    clean = [
        fixture(
            "clean-1",
            "candidate-general",
            "A generated note about photosynthesis and chlorophyll.",
        ),
        fixture(
            "clean-2",
            "sealed-code-benchmark",
            "A generated program computes a prefix sum over integers.",
        ),
    ]
    if (
        sensitive_findings(clean, list(PII_PATTERNS), PII_PATTERNS)
        or sensitive_findings(clean, list(SECRET_PATTERNS), SECRET_PATTERNS)
        or near_duplicate_findings(clean, specification)[0]
    ):
        fail("self-test rejected clean generated fixtures")
    print(
        "validated PII and secret detectors, field-segment lexical overlap, "
        "and clean-fixture acceptance"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialization", type=Path)
    parser.add_argument("--specification", type=Path, default=DEFAULT_SPECIFICATION)
    parser.add_argument("--verify-ledger", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test and args.materialization is None:
        parser.error("--materialization is required unless --self-test is used")
    return args


def main() -> None:
    args = parse_args()
    try:
        if args.self_test:
            run_self_test()
            return
        if args.verify_ledger:
            verify_readiness_ledger(args.materialization, args.specification)
            print(f"verified {args.materialization.resolve()}")
            return
        ledger = audit_materialization(
            args.materialization,
            args.specification,
            write_ledger=True,
            require_pass=True,
        )
        print(
            f"content gates passed for {ledger['summary']['retained_rows_scanned']} "
            "retained rows"
        )
    except GateError as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
