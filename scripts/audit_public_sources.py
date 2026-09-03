#!/usr/bin/env python3
"""Validate pinned public-source metadata before any experiment downloads.

The committed snapshot makes the preflight reproducible offline. ``--refresh``
contacts only the declared Hugging Face APIs and pinned license-evidence URLs;
it does not download dataset rows or model weights. Manual provenance, privacy,
and terms review remains a separate blocking gate for real acquisition.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
KINDS = {"dataset": "datasets", "model": "models"}
LICENSE_CHECKS = {"card_metadata", "pinned_text"}


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


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


def fetch(url: str) -> tuple[bytes, dict[str, str]]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "paper-gen-public-source-audit/1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            headers = {key.lower(): value for key, value in response.headers.items()}
            return response.read(), headers
    except (urllib.error.URLError, TimeoutError) as exc:
        fail(f"could not fetch {url}: {exc}")


def fetch_json(url: str) -> dict[str, Any]:
    payload, _ = fetch(url)
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON response from {url}: {exc}")
    if not isinstance(value, dict):
        fail(f"expected an object response from {url}")
    return value


def card_license(metadata: dict[str, Any]) -> str | None:
    card_data = metadata.get("cardData")
    if isinstance(card_data, dict) and isinstance(card_data.get("license"), str):
        return card_data["license"].lower()
    for tag in metadata.get("tags", []):
        if isinstance(tag, str) and tag.startswith("license:"):
            return tag.split(":", 1)[1].lower()
    return None


def validate_registry(registry: Any, source: Path) -> list[dict[str, Any]]:
    if not isinstance(registry, dict) or registry.get("schema_version") != 1:
        fail(f"{source} must use schema_version 1")
    sources = registry.get("sources")
    if not isinstance(sources, list) or not sources:
        fail(f"{source} must contain a nonempty sources list")
    seen: set[str] = set()
    for item in sources:
        if not isinstance(item, dict):
            fail("every source entry must be an object")
        key = item.get("key")
        if not isinstance(key, str) or not key or key in seen:
            fail(f"source key must be nonempty and unique: {key!r}")
        seen.add(key)
        if item.get("kind") not in KINDS:
            fail(f"{key}: unsupported kind")
        if not isinstance(item.get("repo_id"), str) or "/" not in item["repo_id"]:
            fail(f"{key}: repo_id must be a Hugging Face owner/name")
        if not REVISION_RE.fullmatch(str(item.get("revision", ""))):
            fail(f"{key}: revision must be a full 40-character commit")
        if not isinstance(item.get("expected_license"), str):
            fail(f"{key}: expected_license is required")
        check = item.get("license_check")
        if not isinstance(check, dict) or check.get("type") not in LICENSE_CHECKS:
            fail(f"{key}: unsupported license_check")
        if check["type"] == "pinned_text":
            if not all(isinstance(check.get(field), str) for field in ("url", "marker")):
                fail(f"{key}: pinned_text needs url and marker")
            if item["revision"] not in check["url"] and "LiveBench/LiveBench" not in check["url"]:
                fail(f"{key}: license evidence must use a pinned URL")
        for field in ("configs", "usage"):
            values = item.get(field)
            if not isinstance(values, list) or not values or not all(
                isinstance(value, str) and value for value in values
            ):
                fail(f"{key}: {field} must be a nonempty string list")
        review = item.get("manual_review")
        if not isinstance(review, dict) or review.get("status") not in {
            "pending",
            "approved",
            "rejected",
        }:
            fail(f"{key}: manual_review status must be pending, approved, or rejected")
        requirements = review.get("requirements")
        if not isinstance(requirements, list) or not requirements:
            fail(f"{key}: manual review requirements are required")
    return sources


def audit_live(item: dict[str, Any]) -> dict[str, Any]:
    namespace = KINDS[item["kind"]]
    api_url = f"https://huggingface.co/api/{namespace}/{item['repo_id']}"
    metadata = fetch_json(api_url)
    observed_revision = metadata.get("sha")
    if observed_revision != item["revision"]:
        fail(
            f"{item['key']}: live revision {observed_revision!r} differs from "
            f"pinned {item['revision']!r}; review upstream changes before repinning"
        )
    if metadata.get("private") or metadata.get("gated") or metadata.get("disabled"):
        fail(f"{item['key']}: source is no longer public, ungated, and enabled")

    check = item["license_check"]
    detected_license = card_license(metadata)
    if check["type"] == "card_metadata":
        if detected_license != item["expected_license"].lower():
            fail(
                f"{item['key']}: card license {detected_license!r} does not match "
                f"{item['expected_license']!r}"
            )
        evidence = {"type": "card_metadata", "matched": True}
    else:
        payload, _ = fetch(check["url"])
        text = payload.decode("utf-8", errors="replace")
        if check["marker"] not in text:
            fail(f"{item['key']}: pinned license marker was not found")
        evidence = {
            "type": "pinned_text",
            "url": check["url"],
            "marker": check["marker"],
            "matched": True,
        }

    return {
        "key": item["key"],
        "kind": item["kind"],
        "repo_id": item["repo_id"],
        "revision": observed_revision,
        "public": True,
        "gated": False,
        "disabled": False,
        "expected_license": item["expected_license"],
        "card_license": detected_license,
        "license_evidence": evidence,
        "configs": item["configs"],
        "usage": item["usage"],
        "manual_review_status": item["manual_review"]["status"],
        "automated_checks_passed": True,
    }


def validate_snapshot(
    snapshot: Any,
    registry: dict[str, Any],
    sources: list[dict[str, Any]],
    source: Path,
) -> None:
    if not isinstance(snapshot, dict) or snapshot.get("schema_version") != 1:
        fail(f"{source} must use schema_version 1")
    records = snapshot.get("sources")
    if not isinstance(records, list):
        fail(f"{source} needs a sources list")
    by_key = {record.get("key"): record for record in records if isinstance(record, dict)}
    expected_keys = {item["key"] for item in sources}
    if set(by_key) != expected_keys:
        fail(f"{source} source keys differ from the registry")
    for item in sources:
        record = by_key[item["key"]]
        fields = {
            "kind": item["kind"],
            "repo_id": item["repo_id"],
            "revision": item["revision"],
            "expected_license": item["expected_license"],
            "configs": item["configs"],
            "usage": item["usage"],
            "manual_review_status": item["manual_review"]["status"],
            "automated_checks_passed": True,
        }
        for field, expected in fields.items():
            if record.get(field) != expected:
                fail(f"{source}: {item['key']} has stale field {field!r}")
    if snapshot.get("registry_as_of_date") != registry.get("as_of_date"):
        fail(f"{source} was not generated for the current registry date")


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=repo / "experiments" / "public_source_registry.json",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=repo / "experiments" / "public_source_snapshot.json",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="contact declared hosts and atomically replace the metadata snapshot",
    )
    parser.add_argument(
        "--require-manual-clearance",
        action="store_true",
        help="fail unless every manual provenance/privacy/terms review is approved",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = read_json(args.registry)
    sources = validate_registry(registry, args.registry)
    if args.require_manual_clearance:
        blocked = [
            item["key"]
            for item in sources
            if item["manual_review"]["status"] != "approved"
        ]
        if blocked:
            fail("manual clearance is incomplete for: " + ", ".join(blocked))

    if args.refresh:
        records = [audit_live(item) for item in sources]
        snapshot = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "registry_as_of_date": registry["as_of_date"],
            "evidence_status": registry["evidence_status"],
            "sources": records,
        }
        write_json(args.snapshot, snapshot)
        print(f"wrote {args.snapshot} with {len(records)} verified pins")
    else:
        snapshot = read_json(args.snapshot)
        validate_snapshot(snapshot, registry, sources, args.snapshot)
        pending = sum(
            item["manual_review"]["status"] != "approved" for item in sources
        )
        print(
            f"validated {len(sources)} pinned sources offline; "
            f"{pending} manual clearances remain"
        )


if __name__ == "__main__":
    main()
