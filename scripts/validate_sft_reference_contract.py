#!/usr/bin/env python3
"""Validate the SFT capability map and cluster-preserving split manifest.

The self-test creates only generated identifiers.  It checks deterministic
assignment and fail-closed behavior; it is not model-training evidence.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "experiments" / "sft_reference_contract.json"
DEFAULT_REGISTRY = ROOT / "experiments" / "public_source_registry.json"
UINT256_SIZE = 1 << 256


class ContractError(ValueError):
    """Raised when a reference or split contract is inconsistent."""


def fail(message: str) -> None:
    raise ContractError(message)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        fail(f"file not found: {path}")
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON in {path}: {exc}")


def canonical_bytes(value: Any) -> bytes:
    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        fail(f"value is not canonical JSON: {exc}")
    return payload.encode("utf-8")


def digest_value(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def registry_entries(registry: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(registry, dict) or registry.get("schema_version") != 1:
        fail("source registry must be a schema-version 1 object")
    sources = registry.get("sources")
    if not isinstance(sources, list) or not sources:
        fail("source registry must contain sources")
    entries: dict[str, dict[str, Any]] = {}
    for entry in sources:
        key = entry.get("key") if isinstance(entry, dict) else None
        if not isinstance(key, str) or not key or key in entries:
            fail("source registry keys must be unique nonempty strings")
        entries[key] = entry
    return entries


def validate_contract(
    contract: Any, registry: Any
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if (
        not isinstance(contract, dict)
        or contract.get("schema_version") != 1
        or contract.get("artifact_type") != "sft_reference_and_split_contract"
    ):
        fail("SFT reference contract has an unsupported schema")
    entries = registry_entries(registry)
    capabilities = contract.get("capabilities")
    if not isinstance(capabilities, list) or len(capabilities) != 3:
        fail("SFT reference contract must define exactly three capabilities")
    by_capability: dict[str, dict[str, Any]] = {}
    for capability in capabilities:
        if not isinstance(capability, dict):
            fail("capability declarations must be objects")
        capability_id = capability.get("capability_id")
        source_key = capability.get("source_key")
        if (
            not isinstance(capability_id, str)
            or not capability_id
            or capability_id in by_capability
            or source_key not in entries
            or entries[source_key].get("kind") != "dataset"
            or not any(
                isinstance(role, str) and role.startswith("sft:candidate-")
                for role in entries[source_key].get("usage", [])
            )
        ):
            fail("capability has an invalid identifier or SFT source")
        if (
            capability.get("score_partition") != "score"
            or capability.get("gate_partition") != "gate"
            or capability.get("audit_partition") != "audit"
        ):
            fail("every capability must bind score, gate, and audit partitions")
        by_capability[capability_id] = capability

    settings = contract.get("target_settings")
    if not isinstance(settings, list) or len(settings) != 2:
        fail("SFT reference contract must define math and code settings")
    setting_ids: set[str] = set()
    for setting in settings:
        setting_id = setting.get("setting_id") if isinstance(setting, dict) else None
        target = setting.get("target_capability") if isinstance(setting, dict) else None
        protected = (
            setting.get("protected_capabilities")
            if isinstance(setting, dict)
            else None
        )
        benchmark = (
            setting.get("secondary_benchmark") if isinstance(setting, dict) else None
        )
        if (
            setting_id not in {"math", "code"}
            or setting_id in setting_ids
            or target not in by_capability
            or not isinstance(protected, list)
            or len(protected) != 2
            or len(set(protected)) != 2
            or set(protected) != set(by_capability) - {target}
            or not isinstance(benchmark, dict)
            or benchmark.get("source_key") not in entries
            or "sft:sealed-code-benchmark"
            not in entries[benchmark["source_key"]].get("usage", [])
        ):
            fail("target setting does not define one target and two protected slices")
        setting_ids.add(setting_id)
    if setting_ids != {"math", "code"}:
        fail("SFT settings must be exactly math and code")

    split = contract.get("split_assignment")
    partitions = split.get("partitions") if isinstance(split, dict) else None
    expected = [
        ("candidate", 0, 80),
        ("score", 80, 85),
        ("gate", 85, 95),
        ("audit", 95, 100),
    ]
    observed = (
        [
            (
                partition.get("name"),
                partition.get("bucket_start_inclusive"),
                partition.get("bucket_end_exclusive"),
            )
            for partition in partitions
        ]
        if isinstance(partitions, list)
        and all(isinstance(partition, dict) for partition in partitions)
        else None
    )
    if observed != expected:
        fail("SFT split buckets must be contiguous 80/5/10/5 intervals")
    gate = contract.get("gate_semantics")
    if (
        not isinstance(gate, dict)
        or gate.get("protected_slice_count_per_setting") != 2
    ):
        fail("SFT gate semantics must match the two protected capabilities")
    return by_capability, entries


def cluster_uid(semantic_audit_sha256: str, record_ids: list[str]) -> str:
    if not is_sha256(semantic_audit_sha256):
        fail("semantic audit identifier must be a SHA-256")
    if (
        not record_ids
        or len(record_ids) != len(set(record_ids))
        or not all(is_sha256(record_id) for record_id in record_ids)
    ):
        fail("cluster member record IDs must be unique SHA-256 values")
    return digest_value(
        {
            "domain": "sft-semantic-cluster-v1",
            "semantic_audit_sha256": semantic_audit_sha256,
            "sorted_member_record_ids": sorted(record_ids),
        }
    )


def assignment_digest(
    source_key: str, source_revision: str, computed_cluster_uid: str
) -> str:
    return digest_value(
        {
            "domain": "sft-split-v1",
            "source_key": source_key,
            "source_revision": source_revision,
            "cluster_uid": computed_cluster_uid,
        }
    )


def assignment_bucket(digest: str) -> int:
    if not is_sha256(digest):
        fail("assignment digest must be a SHA-256")
    return (100 * int(digest, 16)) // UINT256_SIZE


def partition_for_bucket(contract: dict[str, Any], bucket: int) -> str:
    if not isinstance(bucket, int) or not 0 <= bucket < 100:
        fail("assignment bucket must be an integer in [0, 100)")
    for partition in contract["split_assignment"]["partitions"]:
        if (
            partition["bucket_start_inclusive"]
            <= bucket
            < partition["bucket_end_exclusive"]
        ):
            return partition["name"]
    fail("assignment bucket is not covered by the contract")


def validate_manifest(
    manifest: Any,
    contract: dict[str, Any],
    registry: dict[str, Any],
) -> dict[str, Any]:
    capabilities, entries = validate_contract(contract, registry)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or manifest.get("artifact_type") != "sft_reference_split_manifest"
        or manifest.get("empirical_evidence") is not False
        or manifest.get("reference_contract_sha256") != digest_value(contract)
        or not is_sha256(manifest.get("semantic_audit_sha256"))
    ):
        fail("SFT split manifest does not bind the non-empirical contract")
    if manifest.get("execution_class") not in {"real", "synthetic-audit"}:
        fail("SFT split manifest has an invalid execution class")

    source_declarations = manifest.get("sources")
    expected_sources = {
        capability["source_key"] for capability in capabilities.values()
    }
    if not isinstance(source_declarations, list):
        fail("SFT split manifest must list its sources")
    observed_sources: set[str] = set()
    all_record_ids: set[str] = set()
    partition_counts = {
        source_key: {name: 0 for name in ("candidate", "score", "gate", "audit")}
        for source_key in expected_sources
    }
    for declaration in source_declarations:
        if not isinstance(declaration, dict):
            fail("SFT source split declarations must be objects")
        source_key = declaration.get("source_key")
        revision = declaration.get("source_revision")
        clusters = declaration.get("clusters")
        if (
            source_key not in expected_sources
            or source_key in observed_sources
            or revision != entries[source_key].get("revision")
            or not isinstance(clusters, list)
            or not clusters
        ):
            fail("SFT source split declaration is missing, duplicate, or stale")
        observed_sources.add(source_key)
        seen_clusters: set[str] = set()
        for cluster in clusters:
            if not isinstance(cluster, dict):
                fail("SFT cluster declarations must be objects")
            member_ids = cluster.get("member_record_ids")
            if not isinstance(member_ids, list):
                fail("SFT cluster must list member record IDs")
            computed_uid = cluster_uid(
                manifest["semantic_audit_sha256"], member_ids
            )
            computed_digest = assignment_digest(source_key, revision, computed_uid)
            computed_bucket = assignment_bucket(computed_digest)
            computed_partition = partition_for_bucket(contract, computed_bucket)
            if (
                cluster.get("cluster_uid") != computed_uid
                or computed_uid in seen_clusters
                or cluster.get("assignment_sha256") != computed_digest
                or cluster.get("bucket") != computed_bucket
                or cluster.get("partition") != computed_partition
            ):
                fail("SFT cluster identity or partition does not reproduce")
            if all_record_ids.intersection(member_ids):
                fail("an SFT record appears in more than one semantic cluster")
            seen_clusters.add(computed_uid)
            all_record_ids.update(member_ids)
            partition_counts[source_key][computed_partition] += len(member_ids)
    if observed_sources != expected_sources:
        fail("SFT split manifest omits a capability source")
    empty = [
        f"{source_key}:{partition}"
        for source_key, counts in partition_counts.items()
        for partition, count in counts.items()
        if count == 0
    ]
    if empty:
        fail("SFT split manifest has empty required partitions: " + ", ".join(empty))
    return {
        "execution_class": manifest["execution_class"],
        "empirical_evidence": False,
        "sources": len(observed_sources),
        "clusters": sum(
            len(declaration["clusters"]) for declaration in source_declarations
        ),
        "records": len(all_record_ids),
        "partition_counts": partition_counts,
        "reference_contract_sha256": digest_value(contract),
    }


def generated_manifest(
    contract: dict[str, Any], registry: dict[str, Any]
) -> dict[str, Any]:
    capabilities, entries = validate_contract(contract, registry)
    semantic_hash = hashlib.sha256(b"generated-semantic-audit-v1").hexdigest()
    sources = []
    for source_key in sorted(
        {capability["source_key"] for capability in capabilities.values()}
    ):
        clusters: list[dict[str, Any]] = []
        represented: set[str] = set()
        counter = 0
        while represented != {"candidate", "score", "gate", "audit"} or len(
            clusters
        ) < 32:
            record_id = hashlib.sha256(
                f"generated:{source_key}:{counter}".encode()
            ).hexdigest()
            uid = cluster_uid(semantic_hash, [record_id])
            digest = assignment_digest(source_key, entries[source_key]["revision"], uid)
            bucket = assignment_bucket(digest)
            partition = partition_for_bucket(contract, bucket)
            clusters.append(
                {
                    "cluster_uid": uid,
                    "member_record_ids": [record_id],
                    "assignment_sha256": digest,
                    "bucket": bucket,
                    "partition": partition,
                }
            )
            represented.add(partition)
            counter += 1
            if counter > 10000:
                fail("generated fixture could not populate every partition")
        sources.append(
            {
                "source_key": source_key,
                "source_revision": entries[source_key]["revision"],
                "clusters": clusters,
            }
        )
    return {
        "schema_version": 1,
        "artifact_type": "sft_reference_split_manifest",
        "execution_class": "synthetic-audit",
        "empirical_evidence": False,
        "reference_contract_sha256": digest_value(contract),
        "semantic_audit_sha256": semantic_hash,
        "sources": sources,
    }


def expect_rejection(
    manifest: dict[str, Any],
    contract: dict[str, Any],
    registry: dict[str, Any],
    description: str,
) -> None:
    try:
        validate_manifest(manifest, contract, registry)
    except ContractError:
        return
    fail(f"self-test accepted {description}")


def self_test(contract: dict[str, Any], registry: dict[str, Any]) -> dict[str, Any]:
    manifest = generated_manifest(contract, registry)
    summary = validate_manifest(manifest, contract, registry)

    wrong_partition = copy.deepcopy(manifest)
    cluster = wrong_partition["sources"][0]["clusters"][0]
    cluster["partition"] = (
        "score" if cluster["partition"] != "score" else "candidate"
    )
    expect_rejection(
        wrong_partition, contract, registry, "an outcome-moved partition"
    )

    split_cluster = copy.deepcopy(manifest)
    duplicate = copy.deepcopy(split_cluster["sources"][0]["clusters"][0])
    split_cluster["sources"][0]["clusters"].append(duplicate)
    expect_rejection(split_cluster, contract, registry, "a duplicated cluster")

    stale_contract = copy.deepcopy(manifest)
    stale_contract["reference_contract_sha256"] = "0" * 64
    expect_rejection(stale_contract, contract, registry, "a stale contract binding")
    summary["fault_injections_rejected"] = 3
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract = read_json(args.contract)
    registry = read_json(args.registry)
    validate_contract(contract, registry)
    if args.self_test:
        summary = self_test(contract, registry)
    elif args.manifest:
        summary = validate_manifest(
            read_json(args.manifest), contract, registry
        )
    else:
        summary = {
            "contract_status": "valid",
            "reference_contract_sha256": digest_value(contract),
        }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except ContractError as exc:
        raise SystemExit(f"error: {exc}") from exc
