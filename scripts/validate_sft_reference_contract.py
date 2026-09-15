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
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "experiments" / "sft_reference_contract.json"
DEFAULT_REGISTRY = ROOT / "experiments" / "public_source_registry.json"


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
        configurations = capability.get("source_configurations")
        if (
            not isinstance(configurations, dict)
            or set(configurations) - set(entries[source_key].get("configs", []))
            or not all(
                isinstance(weight, (int, float))
                and not isinstance(weight, bool)
                and math.isfinite(weight)
                and weight > 0
                for weight in configurations.values()
            )
            or not math.isclose(sum(configurations.values()), 1.0)
        ):
            fail("capability configuration weights must be positive and sum to one")
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
        ("candidate", 8000),
        ("score", 500),
        ("gate", 1000),
        ("audit", 500),
    ]
    observed = (
        [
            (
                partition.get("name"),
                partition.get("basis_points"),
            )
            for partition in partitions
        ]
        if isinstance(partitions, list)
        and all(isinstance(partition, dict) for partition in partitions)
        else None
    )
    if observed != expected:
        fail("SFT split apportionment must be 80/5/10/5 by cluster count")
    tie_order = split.get("remainder_tie_order") if isinstance(split, dict) else None
    if tie_order != [name for name, _ in expected]:
        fail("SFT largest-remainder tie order must match partition order")
    weighting = contract.get("reference_weighting")
    candidate_mass = (
        weighting.get("candidate_source_mass")
        if isinstance(weighting, dict)
        else None
    )
    if (
        not isinstance(candidate_mass, dict)
        or set(candidate_mass) != set(by_capability)
        or not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value > 0
            for value in candidate_mass.values()
        )
        or not math.isclose(sum(candidate_mass.values()), 1.0)
    ):
        fail("SFT candidate source masses must be positive and sum to one")
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


def apportion_counts(contract: dict[str, Any], cluster_count: int) -> dict[str, int]:
    if not isinstance(cluster_count, int) or cluster_count < 0:
        fail("cluster count must be a nonnegative integer")
    partitions = contract["split_assignment"]["partitions"]
    order = contract["split_assignment"]["remainder_tie_order"]
    counts = {
        partition["name"]: cluster_count * partition["basis_points"] // 10000
        for partition in partitions
    }
    remainders = {
        partition["name"]: cluster_count * partition["basis_points"] % 10000
        for partition in partitions
    }
    remaining = cluster_count - sum(counts.values())
    ranked = sorted(order, key=lambda name: (-remainders[name], order.index(name)))
    for name in ranked[:remaining]:
        counts[name] += 1
    if sum(counts.values()) != cluster_count:
        fail("largest-remainder apportionment does not sum to cluster count")
    return counts


def expected_assignments(
    contract: dict[str, Any], clusters: list[dict[str, Any]]
) -> dict[str, tuple[int, str]]:
    ordered = sorted(
        clusters,
        key=lambda cluster: (
            cluster["assignment_sha256"],
            cluster["cluster_uid"],
        ),
    )
    counts = apportion_counts(contract, len(ordered))
    result: dict[str, tuple[int, str]] = {}
    cursor = 0
    for partition in contract["split_assignment"]["remainder_tie_order"]:
        for cluster in ordered[cursor : cursor + counts[partition]]:
            result[cluster["cluster_uid"]] = (cursor, partition)
            cursor += 1
    if cursor != len(ordered):
        fail("hash-sorted assignment did not consume every cluster")
    return result


def validate_manifest(
    manifest: Any,
    contract: dict[str, Any],
    registry: dict[str, Any],
    *,
    retained_records: dict[str, dict[str, Any]] | None = None,
    partition_by_id: dict[str, str] | None = None,
    excluded_record_ids: set[str] | None = None,
    semantic_artifact: dict[str, Any] | None = None,
    semantic_artifact_sha256: str | None = None,
) -> dict[str, Any]:
    capabilities, entries = validate_contract(contract, registry)
    artifact_binding = (
        manifest.get("semantic_cluster_artifact")
        if isinstance(manifest, dict)
        else None
    )
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or manifest.get("artifact_type") != "sft_reference_split_manifest"
        or manifest.get("empirical_evidence") is not False
        or manifest.get("reference_contract_sha256") != digest_value(contract)
        or not is_sha256(manifest.get("semantic_audit_configuration_sha256"))
        or not isinstance(artifact_binding, dict)
        or artifact_binding.get("sha256") != semantic_artifact_sha256
        or not isinstance(artifact_binding.get("path"), str)
        or not artifact_binding["path"]
        or not isinstance(artifact_binding.get("bytes"), int)
        or artifact_binding["bytes"] <= 0
    ):
        fail("SFT split manifest does not bind the non-empirical contract")
    if manifest.get("execution_class") not in {"real", "synthetic-audit"}:
        fail("SFT split manifest has an invalid execution class")
    if (
        retained_records is None
        or partition_by_id is None
        or excluded_record_ids is None
        or semantic_artifact is None
        or not is_sha256(semantic_artifact_sha256)
    ):
        fail("SFT split validation requires closed-world retained-record context")
    required_semantic_status = (
        "reviewed_pass"
        if manifest["execution_class"] == "real"
        else "generated_pass"
    )
    if (
        not isinstance(semantic_artifact, dict)
        or semantic_artifact.get("schema_version") != 1
        or semantic_artifact.get("artifact_type")
        != "sft_semantic_cluster_audit"
        or semantic_artifact.get("execution_status") != required_semantic_status
        or semantic_artifact.get("empirical_evidence") is not False
        or semantic_artifact.get("audit_configuration_sha256")
        != manifest["semantic_audit_configuration_sha256"]
        or not is_sha256(semantic_artifact.get("implementation_sha256"))
        or (
            manifest["execution_class"] == "real"
            and (
                not isinstance(semantic_artifact.get("review_record"), str)
                or not semantic_artifact["review_record"].strip()
            )
        )
    ):
        fail("SFT semantic-cluster artifact is missing or not reviewed")

    source_declarations = manifest.get("sources")
    expected_sources = {
        capability["source_key"] for capability in capabilities.values()
    }
    if not isinstance(source_declarations, list):
        fail("SFT split manifest must list its sources")
    observed_sources: set[str] = set()
    included_record_ids: set[str] = set()
    excluded_cluster_record_ids: set[str] = set()
    partition_counts = {
        source_key: {name: 0 for name in ("candidate", "score", "gate", "audit")}
        for source_key in expected_sources
    }
    cluster_counts = {
        source_key: {name: 0 for name in ("candidate", "score", "gate", "audit")}
        for source_key in expected_sources
    }
    registry_role_by_source = {
        source_key: next(
            role.removeprefix("sft:")
            for role in entries[source_key]["usage"]
            if role.startswith("sft:candidate-")
        )
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
        prepared_clusters: list[dict[str, Any]] = []
        for cluster in clusters:
            if not isinstance(cluster, dict):
                fail("SFT cluster declarations must be objects")
            member_ids = cluster.get("member_record_ids")
            if not isinstance(member_ids, list):
                fail("SFT cluster must list member record IDs")
            computed_uid = cluster_uid(
                manifest["semantic_audit_configuration_sha256"], member_ids
            )
            computed_digest = assignment_digest(source_key, revision, computed_uid)
            if (
                cluster.get("cluster_uid") != computed_uid
                or computed_uid in seen_clusters
                or cluster.get("assignment_sha256") != computed_digest
                or cluster.get("disposition") != "included"
                or cluster.get("exclusion_reason") is not None
            ):
                fail("SFT included cluster identity or disposition does not reproduce")
            if included_record_ids.intersection(member_ids):
                fail("an SFT record appears in more than one semantic cluster")
            for record_id in member_ids:
                metadata = retained_records.get(record_id)
                if (
                    not isinstance(metadata, dict)
                    or metadata.get("source_key") != source_key
                    or metadata.get("revision") != revision
                    or metadata.get("requested_role")
                    != registry_role_by_source[source_key]
                    or metadata.get("config")
                    not in next(
                        capability["source_configurations"]
                        for capability in capabilities.values()
                        if capability["source_key"] == source_key
                    )
                    or record_id in excluded_record_ids
                ):
                    fail("SFT cluster member does not match retained source provenance")
            seen_clusters.add(computed_uid)
            included_record_ids.update(member_ids)
            prepared_clusters.append(cluster)
        assignments = expected_assignments(contract, prepared_clusters)
        for cluster in prepared_clusters:
            expected_rank, expected_partition = assignments[cluster["cluster_uid"]]
            if (
                cluster.get("rank") != expected_rank
                or cluster.get("partition") != expected_partition
            ):
                fail("SFT hash-sorted cluster rank or partition does not reproduce")
            for record_id in cluster["member_record_ids"]:
                if partition_by_id.get(record_id) != expected_partition:
                    fail("SFT cluster assignment differs from partition JSONL")
            partition_counts[source_key][expected_partition] += len(
                cluster["member_record_ids"]
            )
            cluster_counts[source_key][expected_partition] += 1
    if observed_sources != expected_sources:
        fail("SFT split manifest omits a capability source")
    excluded_clusters = manifest.get("excluded_clusters")
    if not isinstance(excluded_clusters, list):
        fail("SFT split manifest must list excluded semantic clusters")
    seen_excluded_clusters: set[str] = set()
    for cluster in excluded_clusters:
        if not isinstance(cluster, dict) or not isinstance(
            cluster.get("member_record_ids"), list
        ):
            fail("excluded semantic cluster is invalid")
        member_ids = cluster["member_record_ids"]
        computed_uid = cluster_uid(
            manifest["semantic_audit_configuration_sha256"], member_ids
        )
        member_metadata = [retained_records.get(record_id) for record_id in member_ids]
        if any(not isinstance(metadata, dict) for metadata in member_metadata):
            fail("excluded SFT semantic cluster contains a foreign record")
        source_keys = sorted({metadata["source_key"] for metadata in member_metadata})
        if (
            cluster.get("cluster_uid") != computed_uid
            or computed_uid in seen_excluded_clusters
            or cluster.get("disposition") != "excluded"
            or cluster.get("exclusion_reason") != "semantic-overlap"
            or cluster.get("source_keys") != source_keys
            or not set(source_keys).issubset(expected_sources)
            or included_record_ids.intersection(member_ids)
            or excluded_cluster_record_ids.intersection(member_ids)
            or any(record_id not in excluded_record_ids for record_id in member_ids)
        ):
            fail("excluded SFT semantic cluster does not reproduce")
        seen_excluded_clusters.add(computed_uid)
        excluded_cluster_record_ids.update(member_ids)
    expected_candidate_ids = {
        record_id
        for record_id, metadata in retained_records.items()
        if metadata.get("source_key") in expected_sources
    }
    if included_record_ids | excluded_cluster_record_ids != expected_candidate_ids:
        fail("SFT semantic clusters do not cover retained candidate records exactly")
    if set(partition_by_id).intersection(expected_candidate_ids) != included_record_ids:
        fail("SFT candidate partition files do not equal included semantic clusters")
    if (
        semantic_artifact.get("sources") != source_declarations
        or semantic_artifact.get("excluded_clusters") != excluded_clusters
    ):
        fail("SFT split manifest differs from its semantic-cluster artifact")
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
        )
        + len(excluded_clusters),
        "records": len(expected_candidate_ids),
        "partition_counts": partition_counts,
        "cluster_counts": cluster_counts,
        "excluded_records": len(excluded_cluster_record_ids),
        "closed_world": True,
        "reference_contract_sha256": digest_value(contract),
    }


def generated_bundle(
    contract: dict[str, Any], registry: dict[str, Any]
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, str],
    set[str],
    dict[str, Any],
    str,
]:
    capabilities, entries = validate_contract(contract, registry)
    semantic_hash = hashlib.sha256(
        b"generated-semantic-audit-configuration-v1"
    ).hexdigest()
    sources = []
    retained_records: dict[str, dict[str, Any]] = {}
    partition_by_id: dict[str, str] = {}
    for source_key in sorted(
        {capability["source_key"] for capability in capabilities.values()}
    ):
        clusters: list[dict[str, Any]] = []
        capability = next(
            item
            for item in capabilities.values()
            if item["source_key"] == source_key
        )
        requested_role = next(
            role.removeprefix("sft:")
            for role in entries[source_key]["usage"]
            if role.startswith("sft:candidate-")
        )
        for counter in range(21):
            record_id = hashlib.sha256(
                f"generated:{source_key}:{counter}".encode()
            ).hexdigest()
            uid = cluster_uid(semantic_hash, [record_id])
            digest = assignment_digest(source_key, entries[source_key]["revision"], uid)
            clusters.append(
                {
                    "cluster_uid": uid,
                    "member_record_ids": [record_id],
                    "assignment_sha256": digest,
                    "rank": None,
                    "partition": None,
                    "disposition": "included",
                    "exclusion_reason": None,
                }
            )
            retained_records[record_id] = {
                "source_key": source_key,
                "revision": entries[source_key]["revision"],
                "requested_role": requested_role,
                "config": next(iter(capability["source_configurations"])),
            }
        assignments = expected_assignments(contract, clusters)
        for cluster in clusters:
            rank, partition = assignments[cluster["cluster_uid"]]
            cluster["rank"] = rank
            cluster["partition"] = partition
            partition_by_id[cluster["member_record_ids"][0]] = partition
        sources.append(
            {
                "source_key": source_key,
                "source_revision": entries[source_key]["revision"],
                "clusters": clusters,
            }
        )
    semantic_artifact = {
        "schema_version": 1,
        "artifact_type": "sft_semantic_cluster_audit",
        "execution_status": "generated_pass",
        "empirical_evidence": False,
        "audit_configuration_sha256": semantic_hash,
        "implementation_sha256": hashlib.sha256(
            b"generated-semantic-cluster-validator"
        ).hexdigest(),
        "review_record": None,
        "sources": sources,
        "excluded_clusters": [],
    }
    semantic_artifact_sha256 = digest_value(semantic_artifact)
    semantic_artifact_bytes = len(
        json.dumps(
            semantic_artifact,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
    ) + 1
    manifest = {
        "schema_version": 1,
        "artifact_type": "sft_reference_split_manifest",
        "execution_class": "synthetic-audit",
        "empirical_evidence": False,
        "reference_contract_sha256": digest_value(contract),
        "semantic_audit_configuration_sha256": semantic_hash,
        "semantic_cluster_artifact": {
            "path": "process/generated_semantic_clusters.json",
            "sha256": semantic_artifact_sha256,
            "bytes": semantic_artifact_bytes,
        },
        "sources": sources,
        "excluded_clusters": [],
    }
    return (
        manifest,
        retained_records,
        partition_by_id,
        set(),
        semantic_artifact,
        semantic_artifact_sha256,
    )


def expect_rejection(
    manifest: dict[str, Any],
    contract: dict[str, Any],
    registry: dict[str, Any],
    description: str,
    context: tuple[
        dict[str, dict[str, Any]],
        dict[str, str],
        set[str],
        dict[str, Any],
        str,
    ],
) -> None:
    try:
        validate_manifest(
            manifest,
            contract,
            registry,
            retained_records=context[0],
            partition_by_id=context[1],
            excluded_record_ids=context[2],
            semantic_artifact=context[3],
            semantic_artifact_sha256=context[4],
        )
    except ContractError:
        return
    fail(f"self-test accepted {description}")


def self_test(contract: dict[str, Any], registry: dict[str, Any]) -> dict[str, Any]:
    (
        manifest,
        retained_records,
        partition_by_id,
        excluded_record_ids,
        semantic_artifact,
        semantic_artifact_sha256,
    ) = generated_bundle(contract, registry)
    context = (
        retained_records,
        partition_by_id,
        excluded_record_ids,
        semantic_artifact,
        semantic_artifact_sha256,
    )
    summary = validate_manifest(
        manifest,
        contract,
        registry,
        retained_records=retained_records,
        partition_by_id=partition_by_id,
        excluded_record_ids=excluded_record_ids,
        semantic_artifact=semantic_artifact,
        semantic_artifact_sha256=semantic_artifact_sha256,
    )

    wrong_partition = copy.deepcopy(manifest)
    cluster = wrong_partition["sources"][0]["clusters"][0]
    cluster["partition"] = (
        "score" if cluster["partition"] != "score" else "candidate"
    )
    wrong_context = list(copy.deepcopy(context))
    wrong_context[3]["sources"] = wrong_partition["sources"]
    wrong_context[4] = digest_value(wrong_context[3])
    wrong_partition["semantic_cluster_artifact"]["sha256"] = wrong_context[4]
    expect_rejection(wrong_partition, contract, registry, "an outcome-moved partition", tuple(wrong_context))

    split_cluster = copy.deepcopy(manifest)
    duplicate = copy.deepcopy(split_cluster["sources"][0]["clusters"][0])
    split_cluster["sources"][0]["clusters"].append(duplicate)
    duplicate_context = list(copy.deepcopy(context))
    duplicate_context[3]["sources"] = split_cluster["sources"]
    duplicate_context[4] = digest_value(duplicate_context[3])
    split_cluster["semantic_cluster_artifact"]["sha256"] = duplicate_context[4]
    expect_rejection(split_cluster, contract, registry, "a duplicated cluster", tuple(duplicate_context))

    stale_contract = copy.deepcopy(manifest)
    stale_contract["reference_contract_sha256"] = "0" * 64
    expect_rejection(stale_contract, contract, registry, "a stale contract binding", context)

    omitted = copy.deepcopy(manifest)
    omitted["sources"][0]["clusters"].pop()
    omitted_context = list(copy.deepcopy(context))
    omitted_context[3]["sources"] = omitted["sources"]
    omitted_context[4] = digest_value(omitted_context[3])
    omitted["semantic_cluster_artifact"]["sha256"] = omitted_context[4]
    expect_rejection(omitted, contract, registry, "an omitted retained cluster", tuple(omitted_context))

    relabeled_real = copy.deepcopy(manifest)
    relabeled_real["execution_class"] = "real"
    expect_rejection(relabeled_real, contract, registry, "generated identifiers relabeled real", context)

    expected_boundaries = {
        19: {"candidate": 15, "score": 1, "gate": 2, "audit": 1},
        20: {"candidate": 16, "score": 1, "gate": 2, "audit": 1},
        21: {"candidate": 17, "score": 1, "gate": 2, "audit": 1},
    }
    if any(
        apportion_counts(contract, count) != expected
        for count, expected in expected_boundaries.items()
    ):
        fail("19/20/21-cluster apportionment boundary vectors do not reproduce")
    summary["boundary_vectors_verified"] = sorted(expected_boundaries)
    summary["fault_injections_rejected"] = 5
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
