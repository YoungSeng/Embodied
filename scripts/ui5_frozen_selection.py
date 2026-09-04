#!/usr/bin/env python3
"""Validate one immutable UI5 rollout selection and expose its formal size.

The formal curriculum launchers use this module instead of a hand-maintained
``EXPECTED_HARD_GROUPS`` value.  The count is accepted only after the frozen
publication marker and every inventoried file have been verified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from merge_ui5_rollout_selections import SCHEMA_VERSION, _validate_frozen


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer, got {value!r}")
    return value


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _valid_sha256(value: Any, *, label: str) -> str:
    digest = str(value or "").lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{label} is not a SHA-256 digest")
    return digest


def _formal_hard_ids(path: Path) -> tuple[list[str], int, int]:
    """Reconstruct formal membership from the authoritative complete8."""

    hard_ids: list[str] = []
    all_ids: set[str] = set()
    eligible_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, Mapping):
                raise ValueError(f"expected an object at {path}:{line_number}")
            sample_id = row.get("_ui5_sample_id") or row.get("sample_id") or row.get(
                "record_id"
            )
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"complete8 row lacks a stable sample ID at line {line_number}")
            if sample_id in all_ids:
                raise ValueError(f"duplicate complete8 sample ID: {sample_id}")
            all_ids.add(sample_id)
            eligible = row.get("grpo_source_eligible") is True and not any(
                row.get(field) is True
                for field in (
                    "pipeline_coverage_failure",
                    "annotation_anomaly",
                    "coordinate_transform_anomaly",
                )
            )
            eligible_count += int(eligible)
            crop_correct = row.get("crop_correct_count")
            if isinstance(crop_correct, bool) or not isinstance(crop_correct, int):
                raise ValueError(
                    f"complete8 row {sample_id} has invalid crop_correct_count="
                    f"{crop_correct!r}"
                )
            if eligible and crop_correct == 0:
                hard_ids.append(sample_id)
    return sorted(hard_ids), eligible_count, len(all_ids)


def resolve_frozen_selection(root: Path) -> dict[str, Any]:
    """Return a digest-bound formal selection identity after strict validation."""

    root = root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"frozen selection is not a directory: {root}")
    manifest_path = root / "manifest.json"
    summary_path = root / "summary.json"
    complete8_path = root / "complete8.jsonl"
    success_path = root / "_SUCCESS"
    for label, path in (
        ("manifest", manifest_path),
        ("summary", summary_path),
        ("complete8", complete8_path),
        ("success marker", success_path),
    ):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"frozen selection {label} is missing/empty: {path}")

    signatures_before = {
        name: _sha256(path)
        for name, path in (
            ("manifest", manifest_path),
            ("summary", summary_path),
            ("complete8", complete8_path),
            ("success", success_path),
        )
    }

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError(f"frozen selection manifest must be an object: {manifest_path}")
    # This verifies the publication contract, source-set hash, exact file set,
    # byte sizes, per-file SHA-256 values, JSONL row counts, and stable reads.
    publication = _validate_frozen(root, manifest)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, Mapping) or summary.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"frozen selection summary contract is invalid: {summary_path}")
    if summary.get("source_set_sha256") != manifest.get("source_set_sha256"):
        raise ValueError("frozen selection summary/manifest source-set digests differ")
    unique_count = _positive_integer(
        summary.get("unique_complete8_samples"),
        label="summary.unique_complete8_samples",
    )
    formal_eligible = _positive_integer(
        summary.get("formal_eligible_groups"),
        label="summary.formal_eligible_groups",
    )
    hard_groups = _positive_integer(
        summary.get("formal_crop_hard_groups"),
        label="summary.formal_crop_hard_groups",
    )
    if hard_groups > formal_eligible or formal_eligible > unique_count:
        raise ValueError(
            "frozen selection group counts are inconsistent: "
            f"hard={hard_groups}, eligible={formal_eligible}, unique={unique_count}"
        )

    hard_ids, observed_eligible, observed_unique = _formal_hard_ids(complete8_path)
    if observed_unique != unique_count or observed_eligible != formal_eligible:
        raise ValueError(
            "frozen selection summary counts differ from authoritative complete8: "
            f"unique={unique_count}/{observed_unique}, "
            f"eligible={formal_eligible}/{observed_eligible}"
        )
    if len(hard_ids) != hard_groups:
        raise ValueError(
            "frozen selection hard count differs from authoritative complete8: "
            f"declared={hard_groups}, observed={len(hard_ids)}"
        )
    hard_ids_digest = _canonical_json_sha256(hard_ids)
    declared_ids = summary.get("formal_crop_hard_sample_ids")
    if declared_ids is not None and declared_ids != hard_ids:
        raise ValueError(
            "frozen selection summary hard IDs differ from authoritative complete8"
        )
    declared_ids_digest = summary.get("formal_crop_hard_sample_ids_sha256")
    if declared_ids_digest is not None and _valid_sha256(
        declared_ids_digest,
        label="summary.formal_crop_hard_sample_ids_sha256",
    ) != hard_ids_digest:
        raise ValueError(
            "frozen selection summary hard-ID digest differs from authoritative complete8"
        )

    signatures_after = {
        name: _sha256(path)
        for name, path in (
            ("manifest", manifest_path),
            ("summary", summary_path),
            ("complete8", complete8_path),
            ("success", success_path),
        )
    }
    if signatures_after != signatures_before:
        raise RuntimeError("frozen selection changed while its identity was resolved")

    return {
        "schema_version": 1,
        "root": str(root),
        "formal_crop_hard_groups": hard_groups,
        "formal_crop_hard_sample_ids_sha256": hard_ids_digest,
        "formal_eligible_groups": formal_eligible,
        "unique_complete8_samples": unique_count,
        "source_set_sha256": str(manifest["source_set_sha256"]),
        "manifest_path": str(manifest_path),
        "manifest_sha256": signatures_before["manifest"],
        "summary_path": str(summary_path),
        "summary_sha256": signatures_before["summary"],
        "complete8_path": str(complete8_path),
        "complete8_sha256": signatures_before["complete8"],
        "success_path": str(success_path),
        "success_sha256": signatures_before["success"],
        "publication": publication,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-selection", type=Path, required=True)
    parser.add_argument(
        "--field",
        choices=("formal_crop_hard_groups",),
        help="Print only one scalar for shell command substitution.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = resolve_frozen_selection(args.frozen_selection)
    if args.field:
        print(result[args.field])
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
