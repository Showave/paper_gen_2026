#!/usr/bin/env python3
"""Materialize approved local exports with row-level provenance and deduplication.

This adapter deliberately does not download from arbitrary URLs.  A site-local
exporter must first create JSONL files at the exact revisions in the public
source registry and record their byte hashes in an input specification.  Real
materialization then fails closed unless every dataset and model referenced by
the paper has an approved manual review with a review-record identifier.

The committed synthetic fixture is the only clearance bypass.  Its outputs are
marked non-empirical and exercise hashing, normalized duplicate clustering,
cross-role quarantine, and artifact verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "experiments" / "public_source_registry.json"
DEFAULT_PREREGISTRATION = ROOT / "experiments" / "pilot_preregistration.json"
DEFAULT_FIXTURE = (
    ROOT / "experiments" / "fixtures" / "materialization" / "input_spec.json"
)
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
SPACE_RE = re.compile(r"\s+")
ROLE_RE = re.compile(r"^[a-z][a-z0-9-]*$")
CANONICAL_CONTENT_SLOTS = {
    "sft": ("dedup-text",),
    "rl": ("dedup-text",),
    "eval": ("dedup-text",),
}
NORMALIZATION = {
    "name": "unicode-nfkc-casefold-whitespace-v1",
    "unicode": "NFKC",
    "case": "casefold",
    "whitespace": "collapse and strip",
}


class ContractError(ValueError):
    """Raised when a provenance contract is incomplete or inconsistent."""


def fail(message: str) -> None:
    raise ContractError(message)


def canonical_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        fail(f"value is not canonical JSON: {exc}")
    return encoded.encode("utf-8")


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
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        fail(f"file not found: {path}")
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON in {path}: {exc}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(
                json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
                + "\n"
            )
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        fail(f"file not found: {path}")
    return parse_jsonl(text, str(path))


def parse_jsonl(text: str, source: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            fail(f"{source}:{line_number}: blank JSONL rows are forbidden")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            fail(f"{source}:{line_number}: invalid JSON: {exc}")
        if not isinstance(row, dict):
            fail(f"{source}:{line_number}: every JSONL row must be an object")
        rows.append(row)
    if not rows:
        fail(f"{source}: export is empty")
    return rows


def read_pinned_jsonl(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    try:
        payload = path.read_bytes()
    except FileNotFoundError:
        fail(f"file not found: {path}")
    observed = digest_bytes(payload)
    if observed != expected_sha256:
        fail(
            f"{path}: byte hash {observed} differs from declared "
            f"{expected_sha256}"
        )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        fail(f"{path}: export must be UTF-8: {exc}")
    return parse_jsonl(text, str(path))


def nested(value: Any, dotted_path: str) -> Any:
    current = value
    for part in dotted_path.split("."):
        if not part:
            fail(f"invalid empty component in field path {dotted_path!r}")
        if not isinstance(current, dict) or part not in current:
            fail(f"row is missing required field path {dotted_path!r}")
        current = current[part]
    return current


def normalize_text(value: str) -> str:
    return SPACE_RE.sub(" ", unicodedata.normalize("NFKC", value).casefold()).strip()


def meaningful(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (bool, int, float)):
        return True
    if isinstance(value, list):
        return bool(value) and any(meaningful(item) for item in value)
    if isinstance(value, dict):
        return bool(value) and any(meaningful(item) for item in value.values())
    return False


def project_content(
    row: dict[str, Any], projection: list[dict[str, str]]
) -> dict[str, Any]:
    projected = {item["slot"]: nested(row, item["field"]) for item in projection}
    if not all(
        isinstance(value, str) and value.strip() for value in projected.values()
    ):
        fail("canonical content projection values must be nonempty strings")
    return projected


def normalized_content(projected: dict[str, Any]) -> str:
    components: list[str] = []
    any_text = False
    for slot, value in sorted(projected.items()):
        text = normalize_text(value)
        any_text = any_text or bool(text)
        components.append(f"{slot}\u241e{text}")
    result = "\u241d".join(components)
    if not any_text:
        fail("content fields normalize to an empty value")
    return result


def referenced_registry_keys(section: dict[str, Any]) -> set[str]:
    references: set[str] = set()
    for field, value in section.items():
        if field == "source_keys" or field.endswith("_model_key") or field.endswith(
            "_model_keys"
        ):
            values = value if isinstance(value, list) else [value]
            if not all(isinstance(item, str) and item for item in values):
                fail(f"preregistration field {field!r} must contain registry keys")
            references.update(values)
    return references


def validate_registry(value: Any, path: Path) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        fail(f"{path} must contain a schema_version 1 object")
    sources = value.get("sources")
    if not isinstance(sources, list) or not sources:
        fail(f"{path} must contain a nonempty sources list")
    entries: dict[str, dict[str, Any]] = {}
    for entry in sources:
        if not isinstance(entry, dict):
            fail(f"{path}: every source must be an object")
        key = entry.get("key")
        if not isinstance(key, str) or not key or key in entries:
            fail(f"{path}: source keys must be nonempty and unique")
        if entry.get("kind") not in {"dataset", "model"}:
            fail(f"{key}: kind must be dataset or model")
        if not REVISION_RE.fullmatch(str(entry.get("revision", ""))):
            fail(f"{key}: revision must be a full 40-character commit")
        configs = entry.get("configs")
        if not isinstance(configs, list) or not configs or not all(
            isinstance(config, str) and config for config in configs
        ):
            fail(f"{key}: configs must be a nonempty string list")
        review = entry.get("manual_review")
        if not isinstance(review, dict) or review.get("status") not in {
            "pending",
            "approved",
            "rejected",
        }:
            fail(f"{key}: invalid manual_review status")
        if review["status"] == "approved" and not isinstance(
            review.get("review_record"), str
        ):
            fail(f"{key}: approved review must name a review_record")
        entries[key] = entry
    return entries


def validate_preregistration(
    value: Any, paper: str, path: Path
) -> tuple[dict[str, Any], set[str]]:
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        fail(f"{path} must contain a schema_version 1 object")
    section = value.get(paper)
    if not isinstance(section, dict):
        fail(f"{path} has no {paper!r} section")
    source_keys = section.get("source_keys")
    if not isinstance(source_keys, list) or not source_keys or not all(
        isinstance(key, str) and key for key in source_keys
    ):
        fail(f"{paper} preregistration needs source_keys")
    return section, referenced_registry_keys(section)


def require_clearance(
    referenced: set[str], entries: dict[str, dict[str, Any]]
) -> None:
    blocked: list[str] = []
    for key in sorted(referenced):
        entry = entries.get(key)
        review = entry.get("manual_review") if isinstance(entry, dict) else None
        if (
            not isinstance(review, dict)
            or review.get("status") != "approved"
            or not isinstance(review.get("review_record"), str)
            or not review["review_record"].strip()
        ):
            blocked.append(key)
    if blocked:
        fail(
            "real materialization is blocked by missing approved review records: "
            + ", ".join(blocked)
        )


def resolve_input(spec_path: Path, declared: str) -> Path:
    path = Path(declared)
    if not path.is_absolute():
        path = spec_path.parent / path
    return path.resolve()


def validate_input_spec(
    value: Any,
    spec_path: Path,
    paper: str,
    section: dict[str, Any],
    entries: dict[str, dict[str, Any]],
    synthetic_fixture: bool,
) -> list[dict[str, Any]]:
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        fail(f"{spec_path} must contain a schema_version 1 object")
    if value.get("paper") != paper:
        fail(f"{spec_path}: paper must be {paper!r}")
    if bool(value.get("synthetic_fixture")) != synthetic_fixture:
        fail("synthetic fixture flag and input specification disagree")
    declarations = value.get("sources")
    if not isinstance(declarations, list) or not declarations:
        fail(f"{spec_path}: sources must be a nonempty list")

    preregistered_sources = set(section["source_keys"])
    declared_sources: set[str] = set()
    seen_exports: set[tuple[str, str, str, str]] = set()
    for declaration in declarations:
        if not isinstance(declaration, dict):
            fail("every input source declaration must be an object")
        required_strings = ("source_key", "config", "split", "path", "sha256", "role")
        if not all(
            isinstance(declaration.get(field), str) and declaration[field]
            for field in required_strings
        ):
            fail(f"input source declaration needs string fields {required_strings}")
        key = declaration["source_key"]
        entry = entries.get(key)
        if not isinstance(entry, dict) or entry.get("kind") != "dataset":
            fail(f"{key!r} is not a registered dataset")
        if key not in preregistered_sources:
            fail(f"{key!r} is not a dataset source preregistered for {paper}")
        admitted = {
            declaration["config"],
            f"{declaration['config']}:{declaration['split']}",
        }
        if not admitted.intersection(entry["configs"]):
            fail(
                f"{key}: config/split {declaration['config']!r}/"
                f"{declaration['split']!r} is not admitted by the registry"
            )
        if not re.fullmatch(r"^[0-9a-f]{64}$", declaration["sha256"]):
            fail(f"{key}: sha256 must have 64 lowercase hexadecimal characters")
        if not ROLE_RE.fullmatch(declaration["role"]):
            fail(f"{key}: invalid role {declaration['role']!r}")
        row_id_fields = declaration.get("row_id_fields")
        if not isinstance(row_id_fields, list) or not row_id_fields or not all(
            isinstance(item, str) and item for item in row_id_fields
        ):
            fail(f"{key}: row_id_fields must be a nonempty string list")
        projection = declaration.get("content_projection")
        if not isinstance(projection, list) or not projection or not all(
            isinstance(item, dict)
            and isinstance(item.get("slot"), str)
            and ROLE_RE.fullmatch(item["slot"])
            and isinstance(item.get("field"), str)
            and item["field"]
            for item in projection
        ):
            fail(
                f"{key}: content_projection needs nonempty canonical slot/field "
                "mappings"
            )
        slots = [item["slot"] for item in projection]
        if len(slots) != len(set(slots)):
            fail(f"{key}: content_projection slots must be unique")
        expected_slots = set(CANONICAL_CONTENT_SLOTS[paper])
        if set(slots) != expected_slots:
            fail(
                f"{key}: content_projection slots must be exactly "
                f"{sorted(expected_slots)} for {paper}"
            )
        exporter = declaration.get("exporter")
        if not isinstance(exporter, dict) or not all(
            isinstance(exporter.get(field), str) and exporter[field].strip()
            for field in ("name", "version")
        ):
            fail(f"{key}: exporter must name a tool and version")
        license_fields = declaration.get("license_fields")
        if not isinstance(license_fields, list) or not license_fields or not all(
            isinstance(item, str) and item for item in license_fields
        ):
            fail(f"{key}: license_fields must be a nonempty string list")
        allowed_roles = {
            usage.split(":", 1)[1]
            for usage in entry.get("usage", [])
            if isinstance(usage, str) and usage.startswith(paper + ":")
        }
        if declaration["role"] not in allowed_roles:
            fail(
                f"{key}: role {declaration['role']!r} is not among the "
                f"registered {paper} roles {sorted(allowed_roles)}"
            )
        export_key = (
            key,
            declaration["config"],
            declaration["split"],
            declaration["role"],
        )
        if export_key in seen_exports:
            fail(f"duplicate input declaration: {export_key}")
        seen_exports.add(export_key)
        declared_sources.add(key)

        path = resolve_input(spec_path, declaration["path"])
        if synthetic_fixture:
            fixture_root = DEFAULT_FIXTURE.parent.resolve()
            if fixture_root not in path.parents:
                fail(
                    "synthetic fixture inputs must stay in the committed "
                    "fixture directory"
                )

    if not synthetic_fixture and declared_sources != preregistered_sources:
        missing = sorted(preregistered_sources - declared_sources)
        extra = sorted(declared_sources - preregistered_sources)
        fail(
            "real input spec must cover every preregistered dataset source; "
            f"missing={missing}, extra={extra}"
        )
    return declarations


def provenance_record(
    row: dict[str, Any],
    line_number: int,
    declaration: dict[str, Any],
    entry: dict[str, Any],
) -> dict[str, Any]:
    stable_id = {
        field: nested(row, field) for field in declaration["row_id_fields"]
    }
    if not all(meaningful(value) for value in stable_id.values()):
        fail("stable row identifier fields must be nonempty")
    selected_content = project_content(row, declaration["content_projection"])
    license_values = {
        field: nested(row, field) for field in declaration.get("license_fields", [])
    }
    if not all(
        isinstance(value, str) and value.strip() for value in license_values.values()
    ):
        fail("per-row license fields must be nonempty strings")
    normalized = normalized_content(selected_content)
    return {
        "schema_version": 1,
        "source_key": declaration["source_key"],
        "repo_id": entry["repo_id"],
        "revision": entry["revision"],
        "config": declaration["config"],
        "upstream_split": declaration["split"],
        "requested_role": declaration["role"],
        "export_line": line_number,
        "stable_row_id": stable_id,
        "stable_row_id_sha256": digest_value(stable_id),
        "raw_row_sha256": digest_value(row),
        "exact_content_sha256": digest_value(selected_content),
        "normalized_content_sha256": digest_bytes(normalized.encode("utf-8")),
        "normalization": NORMALIZATION["name"],
        "license_values": license_values,
    }


def materialize(
    paper: str,
    input_spec_path: Path,
    output: Path,
    registry_path: Path,
    preregistration_path: Path,
    synthetic_fixture: bool,
    force: bool,
) -> dict[str, Any]:
    registry = read_json(registry_path)
    entries = validate_registry(registry, registry_path)
    preregistration = read_json(preregistration_path)
    section, referenced = validate_preregistration(
        preregistration, paper, preregistration_path
    )
    if not referenced.issubset(entries):
        fail("preregistration references an unregistered source or model")
    if synthetic_fixture:
        if input_spec_path.resolve() != DEFAULT_FIXTURE.resolve():
            fail("the clearance bypass accepts only the committed synthetic fixture")
    else:
        require_clearance(referenced, entries)

    input_spec = read_json(input_spec_path)
    declarations = validate_input_spec(
        input_spec,
        input_spec_path,
        paper,
        section,
        entries,
        synthetic_fixture,
    )

    if output.is_symlink():
        fail(f"refusing to replace a symlink output: {output}")
    if output.exists():
        if not force:
            fail(f"output already exists (use --force to replace it): {output}")
        if not output.is_dir():
            fail(f"refusing to replace a non-directory output: {output}")
        previous = read_json(output / "acquire" / "data_manifest.json")
        if (
            not isinstance(previous, dict)
            or previous.get("artifact_type") != "row_level_source_materialization"
        ):
            fail(f"refusing to replace an unrecognized output directory: {output}")
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}-", dir=str(output.parent))
    )
    records: list[dict[str, Any]] = []
    rows_by_record: dict[str, dict[str, Any]] = {}
    input_files: list[dict[str, Any]] = []
    stable_ids: set[tuple[str, str, str, str]] = set()
    destination_names: set[str] = set()
    try:
        for declaration in declarations:
            key = declaration["source_key"]
            path = resolve_input(input_spec_path, declaration["path"])
            rows = read_pinned_jsonl(path, declaration["sha256"])
            materialized: list[dict[str, Any]] = []
            for line_number, row in enumerate(rows, start=1):
                record = provenance_record(
                    row, line_number, declaration, entries[key]
                )
                stable_key = (
                    key,
                    declaration["config"],
                    declaration["split"],
                    record["stable_row_id_sha256"],
                )
                if stable_key in stable_ids:
                    fail(
                        f"{key}: duplicate stable row id in "
                        f"{declaration['config']}:{declaration['split']}"
                    )
                stable_ids.add(stable_key)
                record_id = digest_value(
                    {
                        "source_key": key,
                        "revision": entries[key]["revision"],
                        "config": declaration["config"],
                        "split": declaration["split"],
                        "stable_row_id": record["stable_row_id"],
                    }
                )
                record["record_id"] = record_id
                records.append(record)
                rows_by_record[record_id] = row
                materialized.append({"provenance": record, "row": row})
            safe_name = re.sub(
                r"[^a-zA-Z0-9_.-]+",
                "-",
                f"{key}--{declaration['config']}--{declaration['split']}"
                f"--{declaration['role']}",
            )
            safe_name += "--" + digest_value(declaration)[:12]
            if safe_name in destination_names:
                fail(f"materialized destination name collision: {safe_name}")
            destination_names.add(safe_name)
            destination = staging / "acquire" / "materialized" / f"{safe_name}.jsonl"
            write_jsonl(destination, materialized)
            input_files.append(
                {
                    "source_key": key,
                    "repo_id": entries[key]["repo_id"],
                    "revision": entries[key]["revision"],
                    "config": declaration["config"],
                    "split": declaration["split"],
                    "role": declaration["role"],
                    "exporter": declaration["exporter"],
                    "input_sha256": declaration["sha256"],
                    "input_rows": len(rows),
                    "materialized_path": str(destination.relative_to(staging)),
                    "materialized_sha256": digest_file(destination),
                }
            )

        records.sort(key=lambda item: item["record_id"])
        write_jsonl(staging / "acquire" / "row_provenance.jsonl", records)
        acquisition = {
            "schema_version": 1,
            "paper": paper,
            "artifact_type": "row_level_source_materialization",
            "synthetic_fixture": synthetic_fixture,
            "empirical_evidence": False,
            "evidence_scope": (
                "Synthetic contract audit only; no public data were downloaded."
                if synthetic_fixture
                else (
                    "Source materialization only; not model-training or "
                    "human-study evidence."
                )
            ),
            "registry_sha256": digest_value(registry),
            "preregistration_section_sha256": digest_value(section),
            "input_spec_sha256": digest_value(input_spec),
            "normalization": NORMALIZATION,
            "manual_review_records": [
                {
                    "key": key,
                    "status": entries[key]["manual_review"]["status"],
                    "review_record": entries[key]["manual_review"].get("review_record"),
                }
                for key in sorted(referenced)
                if not synthetic_fixture
            ],
            "referenced_models": [
                {
                    "key": key,
                    "repo_id": entries[key]["repo_id"],
                    "revision": entries[key]["revision"],
                }
                for key in sorted(referenced)
                if entries[key]["kind"] == "model"
            ],
            "inputs": input_files,
            "row_count": len(records),
            "row_provenance_path": "acquire/row_provenance.jsonl",
            "row_provenance_sha256": digest_file(
                staging / "acquire" / "row_provenance.jsonl"
            ),
        }
        write_json(staging / "acquire" / "data_manifest.json", acquisition)

        clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
        exact_clusters: Counter[str] = Counter()
        for record in records:
            clusters[record["normalized_content_sha256"]].append(record)
            exact_clusters[record["exact_content_sha256"]] += 1

        retained: list[dict[str, Any]] = []
        suppressed: list[dict[str, Any]] = []
        quarantined: list[dict[str, Any]] = []
        cross_role_clusters: list[dict[str, Any]] = []
        for cluster_id, members in sorted(clusters.items()):
            roles = sorted({member["requested_role"] for member in members})
            if len(roles) > 1:
                cross_role_clusters.append(
                    {
                        "cluster_id": cluster_id,
                        "roles": roles,
                        "record_ids": sorted(member["record_id"] for member in members),
                    }
                )
                for member in members:
                    quarantined.append(
                        {
                            **member,
                            "quarantine_reason": "cross-role-normalized-duplicate",
                        }
                    )
                continue
            ordered = sorted(
                members,
                key=lambda item: (
                    item["source_key"],
                    item["stable_row_id_sha256"],
                    item["raw_row_sha256"],
                ),
            )
            retained.append({**ordered[0], "duplicate_cluster_id": cluster_id})
            for member in ordered[1:]:
                suppressed.append(
                    {
                        **member,
                        "duplicate_cluster_id": cluster_id,
                        "canonical_record_id": ordered[0]["record_id"],
                    }
                )

        retained_by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in retained:
            retained_by_role[record["requested_role"]].append(record)
        requested_roles = sorted(
            {declaration["role"] for declaration in declarations}
        )
        empty_roles = sorted(set(requested_roles) - set(retained_by_role))
        if empty_roles:
            fail(
                "deduplication/quarantine removed every row from roles: "
                + ", ".join(empty_roles)
            )

        role_files: list[dict[str, Any]] = []
        for role, role_records in sorted(retained_by_role.items()):
            path = staging / "process" / "rows" / f"{role}.jsonl"
            values = [
                {
                    "provenance": record,
                    "row": rows_by_record[record["record_id"]],
                }
                for record in sorted(role_records, key=lambda item: item["record_id"])
            ]
            write_jsonl(path, values)
            role_files.append(
                {
                    "role": role,
                    "path": str(path.relative_to(staging)),
                    "rows": len(values),
                    "sha256": digest_file(path),
                }
            )

        write_jsonl(
            staging / "process" / "duplicate_suppressed.jsonl",
            sorted(suppressed, key=lambda item: item["record_id"]),
        )
        write_jsonl(
            staging / "process" / "quarantine.jsonl",
            sorted(quarantined, key=lambda item: item["record_id"]),
        )
        process_manifest = {
            "schema_version": 1,
            "paper": paper,
            "artifact_type": "normalized_deduplication_manifest",
            "synthetic_fixture": synthetic_fixture,
            "empirical_evidence": False,
            "normalization": NORMALIZATION,
            "input_row_count": len(records),
            "retained_row_count": len(retained),
            "within_role_duplicate_rows_suppressed": len(suppressed),
            "exact_duplicate_clusters": sum(
                count > 1 for count in exact_clusters.values()
            ),
            "normalized_duplicate_clusters": sum(
                len(members) > 1 for members in clusters.values()
            ),
            "cross_role_clusters_quarantined": len(cross_role_clusters),
            "cross_role_rows_quarantined": len(quarantined),
            "cross_role_clusters": cross_role_clusters,
            "retained_role_files": role_files,
            "duplicate_suppressed_path": "process/duplicate_suppressed.jsonl",
            "duplicate_suppressed_sha256": digest_file(
                staging / "process" / "duplicate_suppressed.jsonl"
            ),
            "quarantine_path": "process/quarantine.jsonl",
            "quarantine_sha256": digest_file(
                staging / "process" / "quarantine.jsonl"
            ),
            "checks": {
                "stable_ids_unique_within_export": True,
                "input_byte_hashes_match": True,
                "retained_normalized_hashes_disjoint_across_roles": True,
                "every_requested_role_retains_rows": True,
                "cross_role_duplicates_removed_before_split_use": True,
            },
        }
        write_json(staging / "process" / "content_manifest.json", process_manifest)
        verify_output(staging)
        staging.replace(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    verify_output(output)
    return process_manifest


def verify_output(output: Path) -> None:
    acquisition_path = output / "acquire" / "data_manifest.json"
    process_path = output / "process" / "content_manifest.json"
    acquisition = read_json(acquisition_path)
    process = read_json(process_path)
    for manifest, path in ((acquisition, acquisition_path), (process, process_path)):
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
            fail(f"{path}: expected schema_version 1")

    provenance_path = output / acquisition["row_provenance_path"]
    if digest_file(provenance_path) != acquisition["row_provenance_sha256"]:
        fail("row provenance ledger hash does not match acquisition manifest")
    provenance = read_jsonl(provenance_path)
    if len(provenance) != acquisition["row_count"]:
        fail("row provenance count does not match acquisition manifest")
    materialized_count = 0
    for input_file in acquisition.get("inputs", []):
        path = output / input_file["materialized_path"]
        if digest_file(path) != input_file["materialized_sha256"]:
            fail(f"materialized acquisition file hash mismatch: {path}")
        values = read_jsonl(path)
        if len(values) != input_file["input_rows"]:
            fail(f"materialized acquisition row count mismatch: {path}")
        materialized_count += len(values)
    if materialized_count != acquisition["row_count"]:
        fail("materialized acquisition counts do not match row provenance")

    hashes_by_role: dict[str, set[str]] = {}
    retained_count = 0
    for role_file in process["retained_role_files"]:
        path = output / role_file["path"]
        if digest_file(path) != role_file["sha256"]:
            fail(f"retained role file hash mismatch: {path}")
        values = read_jsonl(path)
        if len(values) != role_file["rows"]:
            fail(f"retained role row count mismatch: {path}")
        hashes = {
            value["provenance"]["normalized_content_sha256"] for value in values
        }
        if len(hashes) != len(values):
            fail(f"retained role still contains normalized duplicates: {path}")
        hashes_by_role[role_file["role"]] = hashes
        retained_count += len(values)
    roles = sorted(hashes_by_role)
    for index, left in enumerate(roles):
        for right in roles[index + 1 :]:
            if hashes_by_role[left].intersection(hashes_by_role[right]):
                fail(f"retained roles {left!r} and {right!r} share content")
    if retained_count != process["retained_row_count"]:
        fail("retained row count does not match process manifest")

    for artifact_field, digest_field in (
        ("duplicate_suppressed_path", "duplicate_suppressed_sha256"),
        ("quarantine_path", "quarantine_sha256"),
    ):
        path = output / process[artifact_field]
        if digest_file(path) != process[digest_field]:
            fail(f"artifact hash mismatch: {path}")


def run_self_test() -> None:
    if not DEFAULT_FIXTURE.exists():
        fail(f"committed fixture not found: {DEFAULT_FIXTURE}")
    with tempfile.TemporaryDirectory(prefix="paper-data-contract-") as directory:
        output = Path(directory) / "artifact"
        summary = materialize(
            paper="sft",
            input_spec_path=DEFAULT_FIXTURE,
            output=output,
            registry_path=DEFAULT_REGISTRY,
            preregistration_path=DEFAULT_PREREGISTRATION,
            synthetic_fixture=True,
            force=False,
        )
        expected = {
            "input_row_count": 5,
            "retained_row_count": 2,
            "within_role_duplicate_rows_suppressed": 1,
            "normalized_duplicate_clusters": 2,
            "cross_role_clusters_quarantined": 1,
            "cross_role_rows_quarantined": 2,
        }
        observed = {field: summary.get(field) for field in expected}
        if observed != expected:
            fail(f"synthetic fixture summary differs: {observed} != {expected}")
        role_path = output / summary["retained_role_files"][0]["path"]
        with role_path.open("ab") as handle:
            handle.write(b"\n")
        try:
            verify_output(output)
        except ContractError as exc:
            if "hash mismatch" not in str(exc):
                raise
        else:
            fail("artifact verification accepted a tampered retained-role file")

        try:
            materialize(
                paper="sft",
                input_spec_path=DEFAULT_FIXTURE,
                output=Path(directory) / "real-must-fail",
                registry_path=DEFAULT_REGISTRY,
                preregistration_path=DEFAULT_PREREGISTRATION,
                synthetic_fixture=False,
                force=False,
            )
        except ContractError as exc:
            if "missing approved review records" not in str(exc):
                raise
        else:
            fail("real materialization ran without approved review records")
    print(
        "validated clearance blocking, row provenance, deduplication, quarantine, "
        "and tamper detection"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", choices=("sft", "rl", "eval"))
    parser.add_argument("--input-spec", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument(
        "--preregistration", type=Path, default=DEFAULT_PREREGISTRATION
    )
    parser.add_argument(
        "--synthetic-fixture",
        action="store_true",
        help="use only the committed non-empirical fixture and bypass clearances",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--verify-output",
        type=Path,
        help="verify an existing materialization artifact and exit",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run the committed synthetic contract test and exit",
    )
    args = parser.parse_args()
    if args.self_test or args.verify_output:
        return args
    if args.paper is None or args.input_spec is None or args.output is None:
        parser.error("--paper, --input-spec, and --output are required")
    return args


def main() -> None:
    args = parse_args()
    try:
        if args.self_test:
            run_self_test()
            return
        if args.verify_output:
            verify_output(args.verify_output.resolve())
            print(f"verified {args.verify_output.resolve()}")
            return
        summary = materialize(
            paper=args.paper,
            input_spec_path=args.input_spec.resolve(),
            output=args.output.absolute(),
            registry_path=args.registry.resolve(),
            preregistration_path=args.preregistration.resolve(),
            synthetic_fixture=args.synthetic_fixture,
            force=args.force,
        )
        print(
            f"materialized {summary['input_row_count']} rows; retained "
            f"{summary['retained_row_count']}, suppressed "
            f"{summary['within_role_duplicate_rows_suppressed']}, quarantined "
            f"{summary['cross_role_rows_quarantined']}"
        )
    except ContractError as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
