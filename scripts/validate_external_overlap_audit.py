#!/usr/bin/env python3
"""Validate a sharded, recall-complete external lexical-overlap audit.

The validator never accepts an approximate candidate index as evidence of
absence.  It binds a reviewed exact engine to the materialization and gate
configuration, verifies gap-free coverage of the complete unordered pair
space, checks every shard hash, and independently validates reported pair
identities, roles, thresholds, and scores.  A disk-backed exact join then
enumerates every pair sharing a 5-gram and rejects any omitted qualifying pair;
because every positive-Jaccard pair shares a gram, this check is recall
complete for the frozen positive thresholds.  It does not print matched text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import stat
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "experiments" / "external_overlap_audit.json"
HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")
IMAGE_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
PAIR_ORDER = "record_id_ascending_row_major_upper_triangle_v1"
SHARDING_METHOD = "contiguous_pair_ordinal_ranges_v1"


class ExternalAuditError(ValueError):
    """Raised when an external overlap audit is incomplete or unbound."""


def fail(message: str) -> None:
    raise ExternalAuditError(message)


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


def digest_value(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except FileNotFoundError:
        fail(f"file not found: {path}")
    return digest.hexdigest()


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
        fail(f"could not open regular audit JSON {path}: {exc}")
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            fail(f"audit JSON must be a regular file: {path}")
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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and HEX_64_RE.fullmatch(value) is not None


def validate_contract(value: Any, path: Path) -> dict[str, Any]:
    contract_fields = {
        "schema_version",
        "artifact_type",
        "execution_status",
        "evidence_status",
        "normalization",
        "similarity_method",
        "pair_order",
        "sharding",
        "independent_verification",
        "engine",
        "required_attestations",
        "failure_policy",
        "limitations",
    }
    if (
        not isinstance(value, dict)
        or set(value) != contract_fields
        or value.get("schema_version") != 1
        or value.get("artifact_type")
        != "recall_complete_external_overlap_audit_contract"
        or value.get("normalization") != "unicode-nfkc-casefold-whitespace-v1"
        or value.get("similarity_method") != "character_5gram_jaccard"
        or value.get("pair_order") != PAIR_ORDER
    ):
        fail(f"{path}: invalid external-overlap contract")
    sharding = value.get("sharding")
    if (
        not isinstance(sharding, dict)
        or sharding.get("method") != SHARDING_METHOD
        or sharding.get("range_semantics") != "zero-based half-open"
    ):
        fail(f"{path}: unsupported pair sharding contract")
    independent = value.get("independent_verification")
    if independent != {
        "method": "disk_backed_complete_shared_5gram_join_v1",
        "proof_obligation": (
            "Every pair at positive Jaccard similarity shares at least one "
            "5-gram; exact enumeration of all shared-gram pairs is therefore "
            "recall-complete for every frozen positive threshold."
        ),
        "reported_score_rule": "exact Jaccard rounded to 12 decimal places",
    }:
        fail(f"{path}: unsupported independent exact-verification contract")
    attestations = value.get("required_attestations")
    expected_attestations = {
        "all_pairs_enumerated": True,
        "approximation_used": False,
        "early_termination": False,
        "matched_text_persisted": False,
    }
    if attestations != expected_attestations:
        fail(f"{path}: external audit attestations are not fail-closed")
    engine = value.get("engine")
    if value.get("execution_status") != "frozen":
        fail(
            f"{path}: exact external engine remains an unfrozen blocker; "
            "freeze its reviewed implementation, image, and review record first"
        )
    if (
        not isinstance(engine, dict)
        or set(engine)
        != {
            "implementation_sha256",
            "runtime_image_digest",
            "review_record",
            "required_before_freezing",
        }
        or not is_sha256(engine.get("implementation_sha256"))
        or not isinstance(engine.get("runtime_image_digest"), str)
        or IMAGE_DIGEST_RE.fullmatch(engine["runtime_image_digest"]) is None
        or not isinstance(engine.get("review_record"), str)
        or not engine["review_record"].strip()
    ):
        fail(f"{path}: frozen engine attestation is incomplete")
    return value


def expected_binding(
    *,
    paper: str,
    retained_rows: int,
    retained_record_ids_sha256: str,
    gate_specification_sha256: str,
    acquisition_manifest_sha256: str,
    content_manifest_sha256: str,
    role_files: list[dict[str, Any]],
) -> dict[str, Any]:
    if not isinstance(retained_rows, int) or retained_rows <= 0:
        fail("retained row count must be a positive integer")
    hashes = (
        retained_record_ids_sha256,
        gate_specification_sha256,
        acquisition_manifest_sha256,
        content_manifest_sha256,
    )
    if not isinstance(paper, str) or not paper or not all(is_sha256(item) for item in hashes):
        fail("external audit binding has an invalid paper or digest")
    return {
        "paper": paper,
        "retained_rows": retained_rows,
        "retained_record_ids_sha256": retained_record_ids_sha256,
        "gate_specification_sha256": gate_specification_sha256,
        "acquisition_manifest_sha256": acquisition_manifest_sha256,
        "content_manifest_sha256": content_manifest_sha256,
        "role_files_sha256": digest_value(role_files),
        "normalization": "unicode-nfkc-casefold-whitespace-v1",
        "similarity_method": "character_5gram_jaccard",
    }


def confined_bundle_path(bundle: Path, declared: Any) -> Path:
    if not isinstance(declared, str) or not declared:
        fail("external shard path must be a nonempty string")
    relative = Path(declared)
    if relative.is_absolute() or ".." in relative.parts:
        fail(f"external shard path escapes the bundle: {declared!r}")
    current = bundle
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            fail(f"external shard path traverses a symlink: {declared!r}")
    resolved = current.resolve()
    root = bundle.resolve()
    if root not in resolved.parents:
        fail(f"external shard path escapes the bundle: {declared!r}")
    return resolved


def pair_ordinal(left_index: int, right_index: int, count: int) -> int:
    if not 0 <= left_index < right_index < count:
        fail("invalid upper-triangle pair indices")
    before = left_index * (2 * count - left_index - 1) // 2
    return before + right_index - left_index - 1


def finding_class(
    left_role: str, right_role: str, gate_specification: dict[str, Any]
) -> tuple[str, float]:
    similarity = gate_specification["similarity"]
    classes = gate_specification["role_classes"]
    training = set(classes["training"])
    protected = set(classes["protected"])
    if left_role == right_role:
        return (
            "within_role_near_duplicate",
            float(similarity["within_role_threshold"]),
        )
    if (
        left_role in training
        and right_role in protected
        or right_role in training
        and left_role in protected
    ):
        return (
            "training_protected_near_duplicate",
            float(similarity["training_protected_threshold"]),
        )
    return "cross_role_near_duplicate", float(similarity["cross_role_threshold"])


def validate_finding(
    value: Any,
    *,
    shard_start: int,
    shard_stop: int,
    ordered_record_ids: list[str],
    index_by_record_id: dict[str, int],
    roles_by_record_id: dict[str, str],
    ngrams_by_record_id: dict[str, frozenset[str]],
    gate_specification: dict[str, Any],
) -> dict[str, Any]:
    fields = {
        "pair_ordinal",
        "pair_id",
        "left_record_id",
        "left_role",
        "right_record_id",
        "right_role",
        "finding_type",
        "character_5gram_jaccard",
        "threshold",
    }
    if not isinstance(value, dict) or set(value) != fields:
        fail("external overlap finding has an invalid field set")
    left_id = value["left_record_id"]
    right_id = value["right_record_id"]
    if (
        not is_sha256(left_id)
        or not is_sha256(right_id)
        or left_id >= right_id
        or left_id not in roles_by_record_id
        or right_id not in roles_by_record_id
    ):
        fail("external overlap finding has invalid or unordered record IDs")
    left_index = index_by_record_id[left_id]
    right_index = index_by_record_id[right_id]
    ordinal = pair_ordinal(left_index, right_index, len(ordered_record_ids))
    if (
        value["pair_ordinal"] != ordinal
        or value["pair_id"] != digest_value([left_id, right_id])
        or not shard_start <= ordinal < shard_stop
        or value["left_role"] != roles_by_record_id[left_id]
        or value["right_role"] != roles_by_record_id[right_id]
    ):
        fail("external overlap finding does not reproduce pair order or roles")
    expected_type, expected_threshold = finding_class(
        value["left_role"], value["right_role"], gate_specification
    )
    score = value["character_5gram_jaccard"]
    left_ngrams = ngrams_by_record_id.get(left_id)
    right_ngrams = ngrams_by_record_id.get(right_id)
    if left_ngrams is None or right_ngrams is None:
        fail("external overlap finding has no retained source text")
    expected_score = (
        len(left_ngrams.intersection(right_ngrams))
        / len(left_ngrams.union(right_ngrams))
        if left_ngrams and right_ngrams
        else 0.0
    )
    if (
        value["finding_type"] != expected_type
        or value["threshold"] != expected_threshold
        or not isinstance(score, (int, float))
        or isinstance(score, bool)
        or not math.isfinite(score)
        or not expected_threshold <= float(score) <= 1.0
        or float(score) != round(expected_score, 12)
    ):
        fail("external overlap finding has an invalid class, threshold, or score")
    return value


def independently_verify_findings(
    *,
    ordered_record_ids: list[str],
    roles_by_record_id: dict[str, str],
    ngrams_by_record_id: dict[str, frozenset[str]],
    gate_specification: dict[str, Any],
    reported_findings: list[dict[str, Any]],
) -> int:
    """Enumerate every positive-similarity pair via a disk-backed exact join."""
    candidate_pairs = 0
    finding_index = 0
    with tempfile.TemporaryDirectory(prefix="exact-overlap-verifier-") as directory:
        database_path = Path(directory) / "shared-grams.sqlite3"
        connection = sqlite3.connect(database_path)
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute(
                "CREATE TABLE grams ("
                "gram TEXT NOT NULL, record_index INTEGER NOT NULL, "
                "PRIMARY KEY (gram, record_index)"
                ") WITHOUT ROWID"
            )
            for index, record_id in enumerate(ordered_record_ids):
                connection.executemany(
                    "INSERT INTO grams (gram, record_index) VALUES (?, ?)",
                    (
                        (gram, index)
                        for gram in sorted(ngrams_by_record_id[record_id])
                    ),
                )
            connection.commit()
            cursor = connection.execute(
                "SELECT DISTINCT left_gram.record_index, right_gram.record_index "
                "FROM grams AS left_gram "
                "JOIN grams AS right_gram ON left_gram.gram = right_gram.gram "
                "WHERE left_gram.record_index < right_gram.record_index "
                "ORDER BY left_gram.record_index, right_gram.record_index"
            )
            for left_index, right_index in cursor:
                candidate_pairs += 1
                left_id = ordered_record_ids[left_index]
                right_id = ordered_record_ids[right_index]
                left_ngrams = ngrams_by_record_id[left_id]
                right_ngrams = ngrams_by_record_id[right_id]
                threshold_type, threshold = finding_class(
                    roles_by_record_id[left_id],
                    roles_by_record_id[right_id],
                    gate_specification,
                )
                length_upper_bound = min(
                    len(left_ngrams), len(right_ngrams)
                ) / max(len(left_ngrams), len(right_ngrams))
                if length_upper_bound < threshold:
                    continue
                score = len(left_ngrams.intersection(right_ngrams)) / len(
                    left_ngrams.union(right_ngrams)
                )
                if score < threshold:
                    continue
                expected = {
                    "pair_ordinal": pair_ordinal(
                        left_index, right_index, len(ordered_record_ids)
                    ),
                    "pair_id": digest_value([left_id, right_id]),
                    "left_record_id": left_id,
                    "left_role": roles_by_record_id[left_id],
                    "right_record_id": right_id,
                    "right_role": roles_by_record_id[right_id],
                    "finding_type": threshold_type,
                    "character_5gram_jaccard": round(score, 12),
                    "threshold": threshold,
                }
                if (
                    finding_index >= len(reported_findings)
                    or reported_findings[finding_index] != expected
                ):
                    fail(
                        "external overlap findings differ from the independent "
                        "disk-backed exact shared-5-gram join"
                    )
                finding_index += 1
        except sqlite3.Error as exc:
            fail(f"independent exact overlap verification failed: {exc}")
        finally:
            connection.close()
    if finding_index != len(reported_findings):
        fail(
            "external overlap findings differ from the independent disk-backed "
            "exact shared-5-gram join"
        )
    return candidate_pairs


def validate_external_overlap_bundle(
    bundle: Path,
    *,
    contract: dict[str, Any],
    binding: dict[str, Any],
    roles_by_record_id: dict[str, str],
    ngrams_by_record_id: dict[str, frozenset[str]],
    gate_specification: dict[str, Any],
) -> dict[str, Any]:
    bundle = bundle.resolve()
    manifest_path = bundle / "manifest.json"
    manifest, manifest_sha256 = read_json_with_sha256(manifest_path)
    manifest_fields = {
        "schema_version",
        "artifact_type",
        "empirical_evidence",
        "status",
        "contract_sha256",
        "binding",
        "engine",
        "attestations",
        "coverage",
        "summary",
    }
    if (
        not isinstance(manifest, dict)
        or set(manifest) != manifest_fields
        or manifest.get("schema_version") != 1
        or manifest.get("artifact_type")
        != "recall_complete_external_overlap_audit_manifest"
        or manifest.get("empirical_evidence") is not False
        or manifest.get("binding") != binding
        or manifest.get("contract_sha256") != digest_value(contract)
        or manifest.get("engine") != contract["engine"]
        or manifest.get("attestations") != contract["required_attestations"]
    ):
        fail("external overlap manifest is invalid or bound to different inputs")
    ordered_record_ids = sorted(roles_by_record_id)
    index_by_record_id = {
        record_id: index for index, record_id in enumerate(ordered_record_ids)
    }
    if (
        len(ordered_record_ids) != binding["retained_rows"]
        or digest_value(ordered_record_ids)
        != binding["retained_record_ids_sha256"]
        or set(ngrams_by_record_id) != set(ordered_record_ids)
    ):
        fail("external audit record IDs do not reproduce its input binding")
    coverage = manifest.get("coverage")
    shards = coverage.get("shards") if isinstance(coverage, dict) else None
    possible_pairs = len(ordered_record_ids) * (len(ordered_record_ids) - 1) // 2
    if (
        not isinstance(coverage, dict)
        or set(coverage)
        != {"pair_order", "sharding_method", "possible_pairs", "shards"}
        or coverage.get("pair_order") != PAIR_ORDER
        or coverage.get("sharding_method") != SHARDING_METHOD
        or coverage.get("possible_pairs") != possible_pairs
        or not isinstance(shards, list)
        or not shards
    ):
        fail("external overlap manifest has an invalid pair-space declaration")

    next_start = 0
    pairs_compared = 0
    findings: list[dict[str, Any]] = []
    seen_ordinals: set[int] = set()
    shard_hashes: list[dict[str, Any]] = []
    for expected_index, declaration in enumerate(shards):
        if (
            not isinstance(declaration, dict)
            or set(declaration)
            != {"index", "range_start", "range_stop", "path", "sha256"}
            or declaration["index"] != expected_index
            or declaration["range_start"] != next_start
            or not isinstance(declaration["range_stop"], int)
            or declaration["range_stop"] <= declaration["range_start"]
            or not is_sha256(declaration["sha256"])
        ):
            fail("external overlap shard ranges must be ordered, positive, and gap-free")
        shard_path = confined_bundle_path(bundle, declaration["path"])
        shard, observed_sha256 = read_json_with_sha256(shard_path)
        if observed_sha256 != declaration["sha256"]:
            fail(f"external overlap shard hash mismatch: {shard_path}")
        shard_findings = shard.get("findings") if isinstance(shard, dict) else None
        expected_pairs = declaration["range_stop"] - declaration["range_start"]
        shard_fields = {
            "schema_version",
            "artifact_type",
            "binding_sha256",
            "index",
            "range_start",
            "range_stop",
            "pairs_compared",
            "status",
            "approximation_used",
            "early_termination",
            "errors",
            "comparison_digest",
            "findings",
        }
        if (
            not isinstance(shard, dict)
            or set(shard) != shard_fields
            or shard.get("schema_version") != 1
            or shard.get("artifact_type")
            != "recall_complete_external_overlap_audit_shard"
            or shard.get("binding_sha256") != digest_value(binding)
            or shard.get("index") != expected_index
            or shard.get("range_start") != declaration["range_start"]
            or shard.get("range_stop") != declaration["range_stop"]
            or shard.get("pairs_compared") != expected_pairs
            or shard.get("status") != "completed"
            or shard.get("approximation_used") is not False
            or shard.get("early_termination") is not False
            or shard.get("errors") != []
            or not is_sha256(shard.get("comparison_digest"))
            or not isinstance(shard_findings, list)
        ):
            fail("external overlap shard is incomplete, approximate, or unbound")
        for finding in shard_findings:
            validated = validate_finding(
                finding,
                shard_start=declaration["range_start"],
                shard_stop=declaration["range_stop"],
                ordered_record_ids=ordered_record_ids,
                index_by_record_id=index_by_record_id,
                roles_by_record_id=roles_by_record_id,
                ngrams_by_record_id=ngrams_by_record_id,
                gate_specification=gate_specification,
            )
            ordinal = validated["pair_ordinal"]
            if ordinal in seen_ordinals:
                fail("external overlap bundle reports a pair more than once")
            seen_ordinals.add(ordinal)
            findings.append(validated)
        pairs_compared += expected_pairs
        next_start = declaration["range_stop"]
        shard_hashes.append(
            {
                "index": expected_index,
                "path": declaration["path"],
                "sha256": observed_sha256,
            }
        )
    if next_start != possible_pairs or pairs_compared != possible_pairs:
        fail("external overlap shards do not cover the complete pair space")

    findings.sort(key=lambda item: item["pair_ordinal"])
    independent_candidate_pairs = independently_verify_findings(
        ordered_record_ids=ordered_record_ids,
        roles_by_record_id=roles_by_record_id,
        ngrams_by_record_id=ngrams_by_record_id,
        gate_specification=gate_specification,
        reported_findings=findings,
    )

    summary = manifest.get("summary")
    expected_status = "pass" if not findings else "blocked"
    counts = {
        finding_type: sum(
            finding["finding_type"] == finding_type for finding in findings
        )
        for finding_type in (
            "within_role_near_duplicate",
            "cross_role_near_duplicate",
            "training_protected_near_duplicate",
        )
    }
    if (
        not isinstance(summary, dict)
        or set(summary)
        != {
            "pairs_compared",
            "qualifying_pairs",
            "findings_by_type",
            "shards_completed",
        }
        or summary.get("pairs_compared") != possible_pairs
        or summary.get("qualifying_pairs") != len(findings)
        or summary.get("findings_by_type") != counts
        or summary.get("shards_completed") != len(shards)
        or manifest.get("status") != expected_status
    ):
        fail("external overlap summary does not reproduce validated shards")
    return {
        "status": expected_status,
        "pairs_compared": possible_pairs,
        "qualifying_pairs": len(findings),
        "findings": findings,
        "independent_candidate_pairs_scored": independent_candidate_pairs,
        "manifest_sha256": manifest_sha256,
        "shards_sha256": digest_value(shard_hashes),
        "engine": contract["engine"],
    }


def run_self_test() -> None:
    contract = read_json(DEFAULT_CONTRACT)
    contract["execution_status"] = "frozen"
    contract["engine"] = {
        "implementation_sha256": digest_value("generated exact engine fixture"),
        "runtime_image_digest": "sha256:" + digest_value("generated runtime"),
        "review_record": "generated-self-test-review",
        "required_before_freezing": contract["engine"]["required_before_freezing"],
    }
    contract = validate_contract(contract, Path("<generated-self-test-contract>"))
    record_ids = [digest_value(f"generated-row-{index}") for index in range(3)]
    roles = {
        record_ids[0]: "candidate-general",
        record_ids[1]: "candidate-math",
        record_ids[2]: "sealed-code-benchmark",
    }
    ngrams = {
        record_id: frozenset({f"generated-{index}"})
        for index, record_id in enumerate(record_ids)
    }
    gate_specification = {
        "similarity": {
            "within_role_threshold": 0.85,
            "cross_role_threshold": 0.8,
            "training_protected_threshold": 0.75,
        },
        "role_classes": {
            "training": ["candidate-general", "candidate-math"],
            "protected": ["sealed-code-benchmark"],
        },
    }
    binding = expected_binding(
        paper="sft",
        retained_rows=3,
        retained_record_ids_sha256=digest_value(sorted(record_ids)),
        gate_specification_sha256=digest_value("generated gate"),
        acquisition_manifest_sha256=digest_value("generated acquisition"),
        content_manifest_sha256=digest_value("generated content"),
        role_files=[{"role": "generated", "sha256": digest_value("rows")}],
    )
    with tempfile.TemporaryDirectory(prefix="external-overlap-contract-") as directory:
        bundle = Path(directory)
        declarations = []
        for index, (start, stop) in enumerate(((0, 1), (1, 3))):
            shard_path = bundle / "shards" / f"{index:04d}.json"
            write_json(
                shard_path,
                {
                    "schema_version": 1,
                    "artifact_type": "recall_complete_external_overlap_audit_shard",
                    "binding_sha256": digest_value(binding),
                    "index": index,
                    "range_start": start,
                    "range_stop": stop,
                    "pairs_compared": stop - start,
                    "status": "completed",
                    "approximation_used": False,
                    "early_termination": False,
                    "errors": [],
                    "comparison_digest": digest_value(
                        ["generated comparisons", start, stop]
                    ),
                    "findings": [],
                },
            )
            declarations.append(
                {
                    "index": index,
                    "range_start": start,
                    "range_stop": stop,
                    "path": str(shard_path.relative_to(bundle)),
                    "sha256": digest_file(shard_path),
                }
            )
        manifest = {
            "schema_version": 1,
            "artifact_type": "recall_complete_external_overlap_audit_manifest",
            "empirical_evidence": False,
            "status": "pass",
            "contract_sha256": digest_value(contract),
            "binding": binding,
            "engine": contract["engine"],
            "attestations": contract["required_attestations"],
            "coverage": {
                "pair_order": PAIR_ORDER,
                "sharding_method": SHARDING_METHOD,
                "possible_pairs": 3,
                "shards": declarations,
            },
            "summary": {
                "pairs_compared": 3,
                "qualifying_pairs": 0,
                "findings_by_type": {
                    "within_role_near_duplicate": 0,
                    "cross_role_near_duplicate": 0,
                    "training_protected_near_duplicate": 0,
                },
                "shards_completed": 2,
            },
        }
        write_json(bundle / "manifest.json", manifest)
        validated = validate_external_overlap_bundle(
            bundle,
            contract=contract,
            binding=binding,
            roles_by_record_id=roles,
            ngrams_by_record_id=ngrams,
            gate_specification=gate_specification,
        )
        if validated["status"] != "pass" or validated["pairs_compared"] != 3:
            fail("generated complete external bundle did not pass")

        omitted_finding_ngrams = dict(ngrams)
        omitted_finding_ngrams[record_ids[1]] = ngrams[record_ids[0]]
        try:
            validate_external_overlap_bundle(
                bundle,
                contract=contract,
                binding=binding,
                roles_by_record_id=roles,
                ngrams_by_record_id=omitted_finding_ngrams,
                gate_specification=gate_specification,
            )
        except ExternalAuditError as exc:
            if "independent disk-backed exact" not in str(exc):
                raise
        else:
            fail("external overlap validator accepted an omitted qualifying pair")

        broken = dict(manifest)
        broken["coverage"] = dict(manifest["coverage"])
        broken["coverage"]["shards"] = [
            dict(declaration) for declaration in declarations
        ]
        broken["coverage"]["shards"][1]["range_start"] = 2
        write_json(bundle / "manifest.json", broken)
        try:
            validate_external_overlap_bundle(
                bundle,
                contract=contract,
                binding=binding,
                roles_by_record_id=roles,
                ngrams_by_record_id=ngrams,
                gate_specification=gate_specification,
            )
        except ExternalAuditError as exc:
            if "gap-free" not in str(exc):
                raise
        else:
            fail("external overlap validator accepted a pair-space gap")

        write_json(bundle / "manifest.json", manifest)
        shard_path = bundle / declarations[0]["path"]
        shard = read_json(shard_path)
        shard["early_termination"] = True
        write_json(shard_path, shard)
        declarations[0]["sha256"] = digest_file(shard_path)
        manifest["coverage"]["shards"] = declarations
        write_json(bundle / "manifest.json", manifest)
        try:
            validate_external_overlap_bundle(
                bundle,
                contract=contract,
                binding=binding,
                roles_by_record_id=roles,
                ngrams_by_record_id=ngrams,
                gate_specification=gate_specification,
            )
        except ExternalAuditError as exc:
            if "incomplete, approximate, or unbound" not in str(exc):
                raise
        else:
            fail("external overlap validator accepted early termination")
    print(
        "validated frozen-engine binding, complete pair-space sharding, shard "
        "hashes, exact omitted-finding detection, and fail-closed "
        "gap/termination handling"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--materialization", type=Path)
    parser.add_argument(
        "--gate-specification",
        type=Path,
        default=ROOT / "experiments" / "content_gates.json",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test and (args.bundle is None or args.materialization is None):
        parser.error("--bundle and --materialization are required")
    return args


def main() -> None:
    args = parse_args()
    try:
        if args.self_test:
            run_self_test()
            return
        from run_content_gates import (
            character_ngrams,
            load_materialized_rows,
            read_json_with_sha256 as read_gate_json_with_sha256,
            validate_specification,
        )

        gate_path = args.gate_specification.resolve()
        gate_value, gate_sha256 = read_gate_json_with_sha256(gate_path)
        gate = validate_specification(gate_value, gate_path)
        external_configuration = gate["external_overlap_audit"]
        contract_path = ROOT / external_configuration["contract_path"]
        contract_value, contract_sha256 = read_json_with_sha256(contract_path)
        if (
            contract_path.resolve() != DEFAULT_CONTRACT.resolve()
            or contract_sha256 != external_configuration["contract_sha256"]
        ):
            fail("content gate does not bind the committed external audit contract")
        rows, inputs = load_materialized_rows(
            args.materialization.resolve(), gate
        )
        if any(len(row.similarity_segments) != 1 for row in rows):
            fail(
                "record-pair external audit v1 cannot validate a multi-field "
                "materialization; use a reviewed segment-pair successor"
            )
        roles = {row.record_id: row.role for row in rows}
        ngrams = {
            row.record_id: character_ngrams(row.similarity_segments[0][2])
            for row in rows
        }
        binding = expected_binding(
            paper=inputs["paper"],
            retained_rows=len(rows),
            retained_record_ids_sha256=inputs["retained_record_ids_sha256"],
            gate_specification_sha256=gate_sha256,
            acquisition_manifest_sha256=inputs["acquisition_manifest_sha256"],
            content_manifest_sha256=inputs["content_manifest_sha256"],
            role_files=inputs["role_files"],
        )
        contract = validate_contract(contract_value, contract_path)
        result = validate_external_overlap_bundle(
            args.bundle.resolve(),
            contract=contract,
            binding=binding,
            roles_by_record_id=roles,
            ngrams_by_record_id=ngrams,
            gate_specification=gate,
        )
        print(
            "validated recall-complete qualifying-pair result for "
            f"{len(rows)} records; independently scored "
            f"{result['independent_candidate_pairs_scored']} shared-5-gram "
            f"pairs; status={result['status']}"
        )
    except ExternalAuditError as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
