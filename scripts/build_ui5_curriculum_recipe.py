#!/usr/bin/env python3
"""Build the three-pool UI5 continuation curriculum from frozen rollout results.

The hard unit is an image/task group named by an immutable frozen-selection
summary after its crop checkpoint completed all four rollouts and got 0/4
exactly correct.  A deterministic 4/4 anchor is matched one-to-one by task and
polarity for every hard group, preferring equal GT-box and base-crop counts.
Every structurally eligible group in the complete rollout bundle that belongs
to neither set forms the global replay pool; replay membership does not depend
on whether the group appeared in the frozen complete8 rollout selection.

Formal hard/anchor groups must also be source-eligible and free of every
structural anomaly recorded by the authoritative snapshot.  Multiple original
annotations for any trained image/task are emitted as one snapshot-verified
union-GT record, never as contradictory bbox subsets.

The output recipe uses absolute media paths and recipe-relative annotations so
it can be consumed unchanged after the training process is restarted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from PIL import Image

try:
    from ui5_curriculum_progress import BuildProgress
except ModuleNotFoundError as exc:
    if exc.name != "ui5_curriculum_progress":
        raise
    from scripts.ui5_curriculum_progress import BuildProgress


SCHEMA_VERSION = 4
POOLS = ("hard", "matched_anchor", "global_replay")
TASKS = (
    "occlusion",
    "cropping",
    "text_overflow",
    "text_ellipsis",
    "content_missing",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-recipe",
        type=Path,
        help=(
            "Deprecated and rejected for formal builds: all pools must use the "
            "portable original training records in the complete rollout bundle."
        ),
    )
    parser.add_argument("--rollout-difficulty", type=Path, required=True)
    parser.add_argument("--rollout-bundle-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reuse-crops-from", type=Path,
        help=(
            "Reuse ALL verified PNGs from a completed schema-v4 curriculum for the "
            "same immutable bundle. Hard-link only; missing/mismatched assets or "
            "cross-filesystem links fail, never fall back to recropping."
        ),
    )
    parser.add_argument(
        "--expected-hard-groups",
        type=int,
        default=None,
        help=(
            "Optional assertion only. Formal hard membership/count are read from "
            "the frozen selection summary.json, never configured independently."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--progress-interval-seconds", type=float, default=10.0,
        help="Heartbeat/progress interval; ETA is estimated for the current stage.",
    )
    parser.add_argument(
        "--print-full-summary", action="store_true",
        help="Also print the full manifest; it is always saved in the output directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing complete curriculum instead of verifying/reusing it.",
    )
    return parser.parse_args(argv)


def _progress_items(
    progress: BuildProgress | None,
    name: str,
    items: Iterable[Any],
    *,
    total: int | None = None,
    unit: str = "items",
    detail: Callable[[Any], str] | None = None,
) -> Iterator[Any]:
    """Count completed items, keeping a heartbeat active inside each slow item."""
    if progress is None:
        yield from items
        return
    if total is None and hasattr(items, "__len__"):
        total = len(items)
    with progress.stage(name, total=total, unit=unit) as stage:
        for completed, item in enumerate(items, 1):
            if detail is not None:
                stage.set_detail(detail(item))
            yield item
            stage.update(completed)


def _json_digest(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_sha256(value: Any, *, label: str) -> str:
    digest = str(value or "").lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{label} is not a SHA-256 digest")
    return digest


def _bundle_file(bundle: Path, value: Any, *, label: str) -> Path:
    relative = Path(str(value or ""))
    if not str(value or "") or relative.is_absolute():
        raise ValueError(f"{label} must be a non-empty bundle-relative path")
    try:
        resolved = (bundle / relative).resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} is missing: {bundle / relative}") from exc
    try:
        resolved.relative_to(bundle)
    except ValueError as exc:
        raise ValueError(f"{label} escapes rollout bundle root: {value!r}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is not a file: {resolved}")
    return resolved


def _verify_rollout_bundle(
    bundle: Path,
    progress: BuildProgress | None = None,
) -> tuple[dict[str, Any], dict[Path, str]]:
    """Fail closed unless every declared bundle input and image is immutable."""

    manifest_path = bundle / "bundle_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"rollout bundle is incomplete: {manifest_path}")
    manifest_digest_before = _sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping) or manifest.get("complete") is not True:
        raise RuntimeError(f"rollout bundle manifest is not complete: {manifest_path}")
    declared_files = manifest.get("files")
    if not isinstance(declared_files, Mapping) or not declared_files:
        raise ValueError("rollout bundle manifest lacks a non-empty files inventory")
    required_files = {
        "base_scan_plans.json",
        "manifest/crop_samples.jsonl",
        "manifest/source_records.jsonl",
        "manifest/task_samples.jsonl",
        "manifest/unique_images.jsonl",
    }
    missing_declarations = sorted(required_files - set(map(str, declared_files)))
    if missing_declarations:
        raise ValueError(
            "rollout bundle manifest does not declare required files: "
            f"{missing_declarations}"
        )

    verified_declared: dict[str, str] = {}
    for relative, raw_metadata in _progress_items(
        progress, "verify_bundle_files", declared_files.items(), unit="files",
        detail=lambda item: str(item[0]),
    ):
        label = f"rollout bundle declared file {relative!r}"
        if not isinstance(raw_metadata, Mapping):
            raise ValueError(f"{label} metadata is not an object")
        path = _bundle_file(bundle, relative, label=label)
        expected_bytes = raw_metadata.get("bytes")
        if (
            isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or expected_bytes < 0
        ):
            raise ValueError(f"{label} has invalid byte count")
        if path.stat().st_size != expected_bytes:
            raise RuntimeError(
                f"{label} size mismatch: expected={expected_bytes}, "
                f"actual={path.stat().st_size}"
            )
        expected_digest = _validated_sha256(
            raw_metadata.get("sha256"), label=f"{label} sha256"
        )
        actual_digest = _sha256_file(path)
        if actual_digest != expected_digest:
            raise RuntimeError(
                f"{label} SHA-256 mismatch: expected={expected_digest}, "
                f"actual={actual_digest}"
            )
        verified_declared[str(relative)] = actual_digest

    unique_path = _bundle_file(
        bundle,
        "manifest/unique_images.jsonl",
        label="rollout bundle unique-images manifest",
    )
    unique_rows = _read_jsonl(unique_path)
    if not unique_rows:
        raise ValueError("rollout bundle unique-images manifest is empty")
    declared_image_count = manifest.get("unique_images")
    if (
        isinstance(declared_image_count, bool)
        or not isinstance(declared_image_count, int)
        or declared_image_count != len(unique_rows)
    ):
        raise ValueError(
            "rollout bundle unique image count mismatch: "
            f"declared={declared_image_count!r}, observed={len(unique_rows)}"
        )

    verified_images: dict[Path, str] = {}
    seen_image_ids: set[str] = set()
    for row_number, row in enumerate(_progress_items(
        progress, "verify_bundle_images", unique_rows, unit="images",
        detail=lambda row: str(row.get("image_relpath", "")),
    ), 1):
        image_id = str(row.get("image_id") or "")
        if not image_id:
            raise ValueError(f"unique_images.jsonl row {row_number} lacks image_id")
        if image_id in seen_image_ids:
            raise ValueError(f"duplicate rollout bundle image_id: {image_id}")
        seen_image_ids.add(image_id)
        image = _bundle_file(
            bundle,
            row.get("image_relpath"),
            label=f"rollout bundle image for {image_id}",
        )
        if image in verified_images:
            raise ValueError(f"duplicate rollout bundle image path: {image}")
        expected_digest = _validated_sha256(
            row.get("sha256"), label=f"rollout bundle image {image_id} sha256"
        )
        actual_digest = _sha256_file(image)
        if actual_digest != expected_digest:
            raise RuntimeError(
                f"rollout bundle image SHA-256 mismatch for {image_id}: "
                f"expected={expected_digest}, actual={actual_digest}"
            )
        verified_images[image] = actual_digest

    manifest_digest = _sha256_file(manifest_path)
    if manifest_digest != manifest_digest_before:
        raise RuntimeError("rollout bundle manifest changed during verification")
    identity = {
        "bundle_manifest_sha256": manifest_digest,
        "declared_files": verified_declared,
        "unique_images_sha256": verified_declared["manifest/unique_images.jsonl"],
    }
    return (
        {
            "root": str(bundle),
            "manifest_sha256": manifest_digest,
            "identity_digest": _json_digest(identity),
            "verified_declared_files": len(verified_declared),
            "verified_unique_images": len(verified_images),
        },
        verified_images,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _task(value: Any) -> str:
    task = str(value or "")
    if task.startswith("ui_"):
        task = task[3:]
    if task not in TASKS:
        raise ValueError(f"unknown UI5 task: {value!r}")
    return task


def _sample_id(row: Mapping[str, Any]) -> str:
    value = row.get("_ui5_sample_id") or row.get("sample_id") or row.get("record_id")
    if not value:
        raise ValueError("training/rollout row lacks a stable sample id")
    return str(value)


AUTHORITATIVE_DIFFICULTY_FIELDS = (
    "task",
    "m31_complete4",
    "crop_correct_count",
    "crop_complete4",
    "cross_model_complete8",
    "technical_error_free",
    "runtime_error_count",
    "parse_error_count",
    "gt_global",
    "grpo_source_eligible",
    "pipeline_coverage_failure",
    "annotation_anomaly",
    "coordinate_transform_anomaly",
)
PROJECTION_MATCH_FIELDS = ("task", "crop_correct_count", "crop_complete4")
FORMAL_ANOMALY_FIELDS = (
    "pipeline_coverage_failure",
    "annotation_anomaly",
    "coordinate_transform_anomaly",
)


def _unique_rows_by_sample(
    rows: Sequence[Mapping[str, Any]], *, label: str
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        sample_id = _sample_id(row)
        if sample_id in indexed:
            raise ValueError(f"duplicate {label} sample_id: {sample_id}")
        indexed[sample_id] = row
    return indexed


def _strict_int_field(row: Mapping[str, Any], key: str, *, label: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} has invalid {key}: {value!r}")
    return value


def _strict_bbox_list(
    value: Any, *, label: str, maximum: int | None = None
) -> list[list[int]]:
    if not isinstance(value, list):
        raise ValueError(f"{label} is not a list")
    normalized: list[list[int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for index, raw_box in enumerate(value):
        if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
            raise ValueError(f"{label}[{index}] is not an xyxy box: {raw_box!r}")
        if any(
            isinstance(item, bool) or not isinstance(item, int) for item in raw_box
        ):
            raise ValueError(f"{label}[{index}] contains non-integer coordinates")
        box = tuple(int(item) for item in raw_box)
        if any(item < 0 or (maximum is not None and item > maximum) for item in box):
            raise ValueError(f"{label}[{index}] is out of range: {list(box)}")
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"{label}[{index}] has non-positive area: {list(box)}")
        if box in seen:
            raise ValueError(f"{label} contains duplicate box: {list(box)}")
        seen.add(box)
        normalized.append(list(box))
    return normalized


def _validate_authoritative_difficulty(
    rows: Sequence[Mapping[str, Any]], *, path: Path
) -> dict[str, Mapping[str, Any]]:
    if not rows:
        raise ValueError(f"authoritative rollout difficulty is empty: {path}")
    indexed = _unique_rows_by_sample(rows, label="authoritative rollout difficulty")
    for sample_id, row in indexed.items():
        missing = [key for key in AUTHORITATIVE_DIFFICULTY_FIELDS if key not in row]
        if missing:
            raise ValueError(
                f"authoritative rollout row {sample_id} lacks fields {missing}: {path}"
            )
        _task(row.get("task"))
        correct = _strict_int_field(
            row, "crop_correct_count", label=f"authoritative rollout row {sample_id}"
        )
        if not 0 <= correct <= 4:
            raise ValueError(
                f"authoritative rollout row {sample_id} has invalid "
                f"crop_correct_count={correct}"
            )
        for key in ("m31_complete4", "crop_complete4", "cross_model_complete8"):
            if not isinstance(row.get(key), bool):
                raise ValueError(
                    f"authoritative rollout row {sample_id} has invalid {key}"
                )
        if not isinstance(row.get("technical_error_free"), bool):
            raise ValueError(
                f"authoritative rollout row {sample_id} has invalid "
                "technical_error_free"
            )
        if not isinstance(row.get("grpo_source_eligible"), bool):
            raise ValueError(
                f"authoritative rollout row {sample_id} has invalid "
                "grpo_source_eligible"
            )
        for key in FORMAL_ANOMALY_FIELDS:
            if not isinstance(row.get(key), bool):
                raise ValueError(
                    f"authoritative rollout row {sample_id} has invalid {key}"
                )
        if row["grpo_source_eligible"] and any(
            row[key] for key in FORMAL_ANOMALY_FIELDS
        ):
            raise ValueError(
                f"authoritative rollout row {sample_id} marks an anomalous "
                "sample grpo_source_eligible=true"
            )
        for key in ("runtime_error_count", "parse_error_count"):
            count = _strict_int_field(
                row, key, label=f"authoritative rollout row {sample_id}"
            )
            if count < 0:
                raise ValueError(
                    f"authoritative rollout row {sample_id} has negative {key}"
                )
        _strict_bbox_list(
            row.get("gt_global"),
            label=f"authoritative rollout row {sample_id}.gt_global",
        )
    return indexed


def _load_difficulty_rows(
    requested_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    requested_digest = _sha256_file(requested_path)
    requested_rows = _read_jsonl(requested_path)
    if not requested_rows:
        raise ValueError(f"rollout difficulty is empty: {requested_path}")
    has_authoritative_fields = all(
        all(key in row for key in AUTHORITATIVE_DIFFICULTY_FIELDS)
        for row in requested_rows
    )
    authoritative_path = requested_path
    projection_verified = False
    if not has_authoritative_fields:
        authoritative_path = requested_path.parent / "complete8.jsonl"
        if not authoritative_path.is_file():
            raise FileNotFoundError(
                "projected rollout difficulty requires authoritative sibling "
                f"complete8.jsonl: {authoritative_path}"
            )
        authoritative_path = authoritative_path.resolve(strict=True)

    authoritative_digest = (
        requested_digest
        if authoritative_path == requested_path
        else _sha256_file(authoritative_path)
    )
    authoritative_rows = (
        requested_rows
        if authoritative_path == requested_path
        else _read_jsonl(authoritative_path)
    )
    authoritative_by_id = _validate_authoritative_difficulty(
        authoritative_rows, path=authoritative_path
    )
    if authoritative_path != requested_path:
        projected_by_id = _unique_rows_by_sample(
            requested_rows, label="projected rollout difficulty"
        )
        if set(projected_by_id) != set(authoritative_by_id):
            missing = sorted(set(authoritative_by_id) - set(projected_by_id))
            extra = sorted(set(projected_by_id) - set(authoritative_by_id))
            raise ValueError(
                "projected/authoritative rollout difficulty sample IDs differ: "
                f"missing={missing[:10]}, extra={extra[:10]}"
            )
        for sample_id, projected in projected_by_id.items():
            authoritative = authoritative_by_id[sample_id]
            for key in PROJECTION_MATCH_FIELDS:
                if key == "task":
                    projected_value = _task(projected.get(key))
                    authoritative_value = _task(authoritative.get(key))
                elif key == "crop_correct_count":
                    projected_value = _strict_int_field(
                        projected,
                        key,
                        label=f"projected rollout row {sample_id}",
                    )
                    authoritative_value = _strict_int_field(
                        authoritative,
                        key,
                        label=f"authoritative rollout row {sample_id}",
                    )
                else:
                    projected_value = projected.get(key)
                    authoritative_value = authoritative.get(key)
                    if not isinstance(projected_value, bool):
                        raise ValueError(
                            f"projected rollout row {sample_id} has invalid {key}"
                        )
                if projected_value != authoritative_value:
                    raise ValueError(
                        "projected/authoritative rollout difficulty mismatch for "
                        f"{sample_id}.{key}: projected={projected_value!r}, "
                        f"authoritative={authoritative_value!r}"
                    )
        projection_verified = True

    if _sha256_file(requested_path) != requested_digest:
        raise RuntimeError("rollout difficulty changed during verification")
    if _sha256_file(authoritative_path) != authoritative_digest:
        raise RuntimeError(
            "authoritative rollout difficulty changed during verification"
        )
    return [dict(row) for row in authoritative_rows], {
        "requested_path": str(requested_path),
        "requested_sha256": requested_digest,
        "authoritative_path": str(authoritative_path),
        "authoritative_sha256": authoritative_digest,
        "projection_verified": projection_verified,
        "rows": len(authoritative_rows),
    }


def _formal_selection_eligible(row: Mapping[str, Any]) -> bool:
    """Mirror the frozen-selection summary's formal structural predicate."""

    return row.get("grpo_source_eligible") is True and not any(
        row.get(key) is True for key in FORMAL_ANOMALY_FIELDS
    )


def _formal_crop_hard_ids(
    rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    return sorted(
        _sample_id(row)
        for row in rows
        if _formal_selection_eligible(row)
        and _strict_int_field(
            row,
            "crop_correct_count",
            label=f"authoritative rollout row {_sample_id(row)}",
        )
        == 0
    )


def _load_frozen_selection_summary(
    difficulty_state: Mapping[str, Any],
    authoritative_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[str], dict[str, Any]]:
    """Bind formal hard membership to the immutable frozen selection.

    Freezer v1 summaries exposed only the formal hard count.  For those already
    published selections we deterministically reconstruct the IDs from the
    sibling authoritative complete8 file and bind both its SHA-256 and the
    resulting sorted-ID digest.  New summaries carry the IDs and digest and are
    checked item-for-item.
    """

    authoritative_path = Path(str(difficulty_state["authoritative_path"]))
    summary_path = authoritative_path.parent / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            "frozen rollout selection requires sibling summary.json: "
            f"{summary_path}"
        )
    summary_digest_before = _sha256_file(summary_path)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"frozen selection summary is not an object: {summary_path}")

    declared_total = payload.get("unique_complete8_samples")
    if (
        isinstance(declared_total, bool)
        or not isinstance(declared_total, int)
        or declared_total != len(authoritative_rows)
    ):
        raise ValueError(
            "frozen summary unique_complete8_samples differs from authoritative "
            f"selection: declared={declared_total!r}, observed={len(authoritative_rows)}"
        )

    reconstructed = _formal_crop_hard_ids(authoritative_rows)
    declared_count = payload.get("formal_crop_hard_groups")
    if (
        isinstance(declared_count, bool)
        or not isinstance(declared_count, int)
        or declared_count != len(reconstructed)
    ):
        raise ValueError(
            "frozen summary formal_crop_hard_groups differs from authoritative "
            f"selection: declared={declared_count!r}, observed={len(reconstructed)}"
        )

    declared_ids = payload.get("formal_crop_hard_sample_ids")
    membership_source = "reconstructed_from_authoritative_complete8"
    if declared_ids is not None:
        if (
            not isinstance(declared_ids, list)
            or any(not isinstance(value, str) or not value for value in declared_ids)
            or declared_ids != sorted(set(declared_ids))
        ):
            raise ValueError(
                "frozen summary formal_crop_hard_sample_ids must be sorted, unique, "
                "non-empty strings"
            )
        if declared_ids != reconstructed:
            missing = sorted(set(reconstructed) - set(declared_ids))
            extra = sorted(set(declared_ids) - set(reconstructed))
            raise ValueError(
                "frozen summary formal hard IDs differ from authoritative complete8: "
                f"missing={missing[:10]}, extra={extra[:10]}"
            )
        membership_source = "explicit_summary_ids"

    hard_ids_digest = _json_digest(reconstructed)
    declared_ids_digest = payload.get("formal_crop_hard_sample_ids_sha256")
    if declared_ids_digest is not None:
        declared_ids_digest = _validated_sha256(
            declared_ids_digest,
            label="frozen summary formal_crop_hard_sample_ids_sha256",
        )
        if declared_ids_digest != hard_ids_digest:
            raise ValueError(
                "frozen summary formal hard ID digest differs from authoritative "
                "complete8 reconstruction"
            )

    declared_eligible = payload.get("formal_eligible_groups")
    observed_eligible = sum(_formal_selection_eligible(row) for row in authoritative_rows)
    if declared_eligible is not None and (
        isinstance(declared_eligible, bool)
        or not isinstance(declared_eligible, int)
        or declared_eligible != observed_eligible
    ):
        raise ValueError(
            "frozen summary formal_eligible_groups differs from authoritative "
            f"selection: declared={declared_eligible!r}, observed={observed_eligible}"
        )

    if _sha256_file(summary_path) != summary_digest_before:
        raise RuntimeError("frozen selection summary changed during verification")
    authoritative_digest = str(difficulty_state["authoritative_sha256"])
    if _sha256_file(authoritative_path) != authoritative_digest:
        raise RuntimeError("authoritative complete8 changed during summary verification")
    return reconstructed, {
        "path": str(summary_path.resolve()),
        "sha256": summary_digest_before,
        "formal_crop_hard_groups": len(reconstructed),
        "formal_crop_hard_sample_ids_sha256": hard_ids_digest,
        "membership_source": membership_source,
        "authoritative_complete8_path": str(authoritative_path.resolve()),
        "authoritative_complete8_sha256": authoritative_digest,
    }


def _assistant_text(record: Mapping[str, Any]) -> str:
    conversations = record.get("conversations")
    if not isinstance(conversations, list):
        return ""
    for item in reversed(conversations):
        if isinstance(item, Mapping) and item.get("from") in {"gpt", "assistant"}:
            return str(item.get("value") or "")
    return ""


def _human_text(record: Mapping[str, Any]) -> str:
    conversations = record.get("conversations")
    if not isinstance(conversations, list):
        return ""
    for item in conversations:
        if isinstance(item, Mapping) and item.get("from") in {"human", "user"}:
            return str(item.get("value") or "")
    return ""


_BOX_TAG_RE = re.compile(r"<box>.*?</box>", re.IGNORECASE | re.DOTALL)
_NUMERIC_BOX_TAG_RE = re.compile(
    r"<box>\s*<\s*(-?\d+)\s*>\s*<\s*(-?\d+)\s*>\s*"
    r"<\s*(-?\d+)\s*>\s*<\s*(-?\d+)\s*>\s*</box>",
    re.IGNORECASE | re.DOTALL,
)
_NONE_BOX_TAG_RE = re.compile(r"<box>\s*none\s*</box>", re.IGNORECASE)


def _answer_contract(answer: str, *, label: str) -> tuple[str, list[list[int]]]:
    """Return the non-box prefix and strict norm1000 boxes in a LocateAnything answer."""

    matches = list(_BOX_TAG_RE.finditer(answer))
    if not matches:
        raise ValueError(f"{label} has no <box> supervision")
    prefix = answer[: matches[0].start()].strip()
    cursor = matches[0].start()
    boxes: list[list[int]] = []
    saw_none = False
    for match in matches:
        if answer[cursor : match.start()].strip():
            raise ValueError(f"{label} has text interleaved with box supervision")
        tag = match.group(0)
        numeric = _NUMERIC_BOX_TAG_RE.fullmatch(tag)
        if numeric is not None:
            boxes.append([int(value) for value in numeric.groups()])
        elif _NONE_BOX_TAG_RE.fullmatch(tag) is not None:
            saw_none = True
        else:
            raise ValueError(f"{label} has an unsupported <box> value: {tag!r}")
        cursor = match.end()
    if answer[cursor:].strip():
        raise ValueError(f"{label} has trailing text after box supervision")
    if saw_none and (boxes or len(matches) != 1):
        raise ValueError(f"{label} mixes <box>none</box> with other supervision")
    normalized = _strict_bbox_list(boxes, label=f"{label} boxes", maximum=1000)
    return prefix, normalized


def _replace_assistant_text(record: Mapping[str, Any], answer: str) -> dict[str, Any]:
    output = json.loads(json.dumps(record, ensure_ascii=False))
    conversations = output.get("conversations")
    if not isinstance(conversations, list):
        raise ValueError("training record conversations are not a list")
    assistant_indices = [
        index
        for index, item in enumerate(conversations)
        if isinstance(item, Mapping) and item.get("from") in {"gpt", "assistant"}
    ]
    if len(assistant_indices) != 1:
        raise ValueError(
            "union supervision requires exactly one assistant conversation turn"
        )
    conversations[assistant_indices[0]]["value"] = answer
    return output


def _media_signature(record: Mapping[str, Any]) -> str:
    return _json_digest(
        {
            key: record.get(key)
            for key in ("image", "images", "image_list", "video", "videos", "video_list")
            if key in record
        }
    )


def _explicit_polarity(row: Mapping[str, Any]) -> str | None:
    candidates: list[tuple[str, str]] = []
    for key in ("_ui5_positive", "positive"):
        explicit = row.get(key)
        if explicit is not None:
            candidates.append(
                (key, "positive" if bool(explicit) else "negative")
            )
    gt = row.get("gt_global")
    if isinstance(gt, list):
        candidates.append(("gt_global", "positive" if gt else "negative"))
    answer = _assistant_text(row)
    if answer:
        candidates.append(
            (
                "assistant_answer",
                "positive"
                if "<box>" in answer and "<box>none</box>" not in answer
                else "negative",
            )
        )
    values = {value for _, value in candidates}
    if len(values) > 1:
        sample_id = row.get("_ui5_sample_id") or row.get("sample_id") or "<unknown>"
        raise ValueError(
            f"conflicting explicit polarity for sample {sample_id}: {candidates}"
        )
    return next(iter(values), None)


def _polarity(row: Mapping[str, Any]) -> str:
    polarity = _explicit_polarity(row)
    if polarity is None:
        raise ValueError(f"cannot determine polarity for sample {_sample_id(row)}")
    return polarity


def _resolve_path(value: Any, *, recipe: Path, relative: bool) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve(strict=True)
    preferred = recipe.parent / path if relative else Path.cwd() / path
    fallback = Path.cwd() / path if relative else recipe.parent / path
    for candidate in (preferred, fallback):
        if candidate.exists():
            return candidate.resolve(strict=True)
    raise FileNotFoundError(
        f"cannot resolve {value!r}; tried {preferred} and {fallback}"
    )


def _absolutize_media(record: dict[str, Any], root: Path | None) -> dict[str, Any]:
    output = dict(record)
    for key in ("image", "images", "image_list", "video", "videos", "video_list"):
        raw = output.get(key)
        if raw is None:
            continue
        values = raw if isinstance(raw, list) else [raw]
        normalized: list[Any] = []
        for value in values:
            if not isinstance(value, str):
                normalized.append(value)
                continue
            path = Path(value).expanduser()
            if not path.is_absolute():
                if root is None:
                    raise ValueError(
                        f"relative media path {value!r} has no recipe root"
                    )
                path = root / path
            normalized.append(str(path.resolve(strict=True)))
        output[key] = normalized if isinstance(raw, list) else normalized[0]
    return output


def _base_records(recipe: Path) -> list[dict[str, Any]]:
    payload = json.loads(recipe.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError("base recipe must be a non-empty dataset mapping")
    records: list[dict[str, Any]] = []
    for dataset_name, raw_entry in payload.items():
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"recipe entry {dataset_name!r} is not an object")
        relative = bool(raw_entry.get("paths_relative_to_meta", False))
        annotations = raw_entry.get("annotation")
        if not annotations:
            raise ValueError(f"recipe entry {dataset_name!r} lacks annotation")
        annotation_values = (
            list(annotations) if isinstance(annotations, (list, tuple)) else [annotations]
        )
        root_value = str(raw_entry.get("root") or "")
        root = (
            _resolve_path(root_value, recipe=recipe, relative=relative)
            if root_value
            else None
        )
        for annotation_value in annotation_values:
            annotation = _resolve_path(
                annotation_value, recipe=recipe, relative=relative
            )
            for raw_record in _read_jsonl(annotation):
                record = _absolutize_media(raw_record, root)
                record["_curriculum_source_dataset"] = str(dataset_name)
                records.append(record)
    if not records:
        raise ValueError("base recipe contains no records")
    return records


def _verify_bundle_record_images(
    records: Sequence[Mapping[str, Any]], verified_images: Mapping[Path, str]
) -> None:
    unverified: list[str] = []
    for record in records:
        record_images: list[Any] = []
        for key in ("image", "images", "image_list"):
            raw = record.get(key)
            if raw is None:
                continue
            record_images.extend(raw if isinstance(raw, list) else [raw])
        if not record_images:
            unverified.append(f"{_sample_id(record)}:<missing>")
            continue
        for raw_image in record_images:
            if not isinstance(raw_image, str):
                unverified.append(f"{_sample_id(record)}:{raw_image!r}")
                continue
            image = Path(raw_image).resolve(strict=True)
            if image not in verified_images:
                unverified.append(f"{_sample_id(record)}:{raw_image}")
    if unverified:
        raise ValueError(
            "bundle training record images are not verified by "
            "manifest/unique_images.jsonl; "
            f"first={unverified[:10]}"
        )


def _bundle_records(bundle: Path) -> list[dict[str, Any]]:
    source_path = bundle / "manifest" / "source_records.jsonl"
    if not source_path.is_file():
        raise FileNotFoundError(f"rollout bundle source records are missing: {source_path}")
    records: list[dict[str, Any]] = []
    seen_source_record_ids: set[str] = set()
    for source in _read_jsonl(source_path):
        raw = source.get("portable_training_record") or source.get(
            "original_training_record"
        )
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"bundle source record lacks portable training data: {source.get('source_record_id')}"
            )
        source_record_id = str(source.get("source_record_id") or "")
        if not source_record_id:
            raise ValueError("bundle source record lacks source_record_id")
        if source_record_id in seen_source_record_ids:
            raise ValueError(
                f"duplicate bundle source_record_id: {source_record_id}"
            )
        seen_source_record_ids.add(source_record_id)
        source_gt_global = _strict_bbox_list(
            source.get("gt_boxes_global_xyxy"),
            label=f"bundle source record {source_record_id}.gt_boxes_global_xyxy",
        )
        source_gt_1000 = _strict_bbox_list(
            source.get("gt_boxes_1000"),
            label=f"bundle source record {source_record_id}.gt_boxes_1000",
            maximum=1000,
        )
        if len(source_gt_global) != len(source_gt_1000):
            raise ValueError(
                f"bundle source record {source_record_id} has mismatched "
                "global/norm1000 GT counts"
            )
        record = _absolutize_media(dict(raw), bundle)
        record.update(
            {
                "_ui5_sample_id": str(source.get("sample_id") or ""),
                "_ui5_image_id": str(source.get("image_id") or ""),
                "_ui5_task": _task(source.get("task")),
                "_ui5_positive": bool(source_gt_global),
                "_ui5_source_record_id": source_record_id,
                "_ui5_source_gt_global": source_gt_global,
                "_ui5_source_gt_1000": source_gt_1000,
                "_ui5_record_kind": str(
                    record.get("_ui5_record_kind") or "full_image"
                ),
                "_curriculum_source_dataset": "rollout_bundle_source_records",
            }
        )
        _sample_id(record)
        records.append(record)
    if not records:
        raise ValueError("rollout bundle contains no portable source records")
    return records


def _bundle_group_catalog(
    bundle: Path,
    crop_rows_by_sample: Mapping[str, Sequence[Mapping[str, Any]]],
    record_truth: Mapping[str, Mapping[str, str]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Validate the complete bundle group universe and return all/eligible rows."""

    path = bundle / "manifest" / "task_samples.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"rollout bundle task samples are missing: {path}")
    all_groups: dict[str, dict[str, Any]] = {}
    eligible: dict[str, dict[str, Any]] = {}
    required_booleans = (
        "positive",
        "grpo_eligible",
        "annotation_anomaly",
        "coordinate_transform_anomaly",
        "pipeline_coverage_failure",
    )
    for row_number, raw in enumerate(_read_jsonl(path), 1):
        row = dict(raw)
        sample_id = _sample_id(row)
        if sample_id in all_groups:
            raise ValueError(f"duplicate bundle task sample: {sample_id}")
        task = _task(row.get("task"))
        for key in required_booleans:
            if not isinstance(row.get(key), bool):
                raise ValueError(
                    f"bundle task sample {sample_id} has invalid {key}: {row.get(key)!r}"
                )
        gt_global = _strict_bbox_list(
            row.get("gt_global"), label=f"bundle task sample {sample_id}.gt_global"
        )
        if row["positive"] is not bool(gt_global):
            raise ValueError(f"bundle task sample {sample_id} polarity differs from GT")
        anomaly = any(
            row[key]
            for key in (
                "annotation_anomaly",
                "coordinate_transform_anomaly",
                "pipeline_coverage_failure",
            )
        )
        if row["grpo_eligible"] is not (not anomaly):
            raise ValueError(
                f"bundle task sample {sample_id} grpo_eligible differs from anomaly flags"
            )

        crop_rows = list(crop_rows_by_sample.get(sample_id, ()))
        if not crop_rows:
            raise ValueError(f"bundle task sample has no crop rows: {sample_id}")
        declared_crop_ids = row.get("crop_ids")
        observed_crop_ids = [str(crop["crop_id"]) for crop in crop_rows]
        if (
            not isinstance(declared_crop_ids, list)
            or any(not isinstance(value, str) or not value for value in declared_crop_ids)
            or declared_crop_ids != observed_crop_ids
        ):
            raise ValueError(
                f"bundle task sample {sample_id} crop_ids differ from crop manifest"
            )
        crop_count = _strict_nonnegative_int(
            row.get("crop_count"), label=f"bundle task sample {sample_id}.crop_count"
        )
        if crop_count != len(crop_rows):
            raise ValueError(
                f"bundle task sample {sample_id} crop_count differs from crop manifest"
            )
        for crop in crop_rows:
            if crop["task"] != task:
                raise ValueError(f"bundle task/crop task conflict for {sample_id}")
            if crop.get("sample_gt_global") != gt_global:
                raise ValueError(f"bundle task/crop GT conflict for {sample_id}")
            for key in ("source_image_id", "image_relpath", "prompt"):
                if str(crop.get(key) or "") != str(row.get(key) or ""):
                    raise ValueError(
                        f"bundle task/crop {key} conflict for {sample_id}"
                    )
        truth = record_truth.get(sample_id)
        if truth is None:
            raise ValueError(f"bundle task sample has no source training record: {sample_id}")
        polarity = "positive" if gt_global else "negative"
        row.update(
            {
                "sample_id": sample_id,
                "record_id": str(row.get("record_id") or sample_id),
                "task": task,
                "gt_global": gt_global,
                "polarity": polarity,
                "crop_count": crop_count,
            }
        )
        all_groups[sample_id] = row
        if row["grpo_eligible"] and not anomaly:
            if truth != {"task": task, "polarity": polarity}:
                raise ValueError(
                    f"eligible bundle task/source truth conflict for {sample_id}: "
                    f"task_sample={(task, polarity)}, source={truth}"
                )
            if any(crop.get("partial_gt_indices") != [] for crop in crop_rows):
                raise ValueError(
                    f"eligible bundle task sample has partial crop GT: {sample_id}"
                )
            eligible[sample_id] = row
    if set(all_groups) != set(crop_rows_by_sample):
        missing = sorted(set(crop_rows_by_sample) - set(all_groups))
        extra = sorted(set(all_groups) - set(crop_rows_by_sample))
        raise ValueError(
            "bundle task/crop sample ID sets differ: "
            f"missing_task_samples={missing[:10]}, extra_task_samples={extra[:10]}"
        )
    if set(all_groups) != set(record_truth):
        missing = sorted(set(record_truth) - set(all_groups))
        extra = sorted(set(all_groups) - set(record_truth))
        raise ValueError(
            "bundle task/source sample ID sets differ: "
            f"missing_task_samples={missing[:10]}, extra_task_samples={extra[:10]}"
        )
    return all_groups, eligible


def _record_group_truth(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, str]]:
    tasks: dict[str, str] = {}
    has_positive: dict[str, bool] = defaultdict(bool)
    for record in records:
        sample_id = _sample_id(record)
        raw_task = record.get("_ui5_task") or record.get("task")
        task = _task(raw_task)
        polarity = _polarity(record)
        previous_task = tasks.get(sample_id)
        if previous_task is not None and previous_task != task:
            raise ValueError(
                f"conflicting training-record truth for sample {sample_id}: "
                f"first_task={previous_task}, observed_task={task}"
            )
        tasks[sample_id] = task
        has_positive[sample_id] = has_positive[sample_id] or polarity == "positive"
    return {
        sample_id: {
            "task": task,
            # One rollout sample represents the union of all source-record GT.
            # A positive source subset therefore makes the whole group positive.
            "polarity": "positive" if has_positive[sample_id] else "negative",
        }
        for sample_id, task in tasks.items()
    }


def _canonical_selected_supervision(
    sample_id: str,
    records: Sequence[Mapping[str, Any]],
    rollout_row: Mapping[str, Any],
) -> dict[str, Any]:
    """Return exactly one non-contradictory record for a selected group.

    Bundle records retain the paired pixel/norm1000 GT for every original
    source row.  That lets us reconstruct the exact union answer evaluated by
    rollout instead of training on mutually exclusive source subsets.  Older
    base recipes do not carry that pairing, so a selected group with more than
    one record is rejected rather than guessed at.
    """

    if not records:
        raise ValueError(f"selected sample has no training records: {sample_id}")
    authoritative_gt = _strict_bbox_list(
        rollout_row.get("gt_global"),
        label=f"rollout row {sample_id}.gt_global",
    )
    metadata_keys = (
        "_ui5_source_record_id",
        "_ui5_source_gt_global",
        "_ui5_source_gt_1000",
    )
    has_source_metadata = [
        all(key in record for key in metadata_keys) for record in records
    ]
    if any(has_source_metadata) and not all(has_source_metadata):
        raise ValueError(
            f"selected sample {sample_id} mixes records with and without "
            "source GT provenance"
        )

    if all(has_source_metadata):
        global_to_norm: dict[tuple[int, int, int, int], tuple[int, int, int, int]] = {}
        prefixes: set[str] = set()
        prompts: set[str] = set()
        media: set[str] = set()
        source_record_ids: set[str] = set()
        for index, record in enumerate(records):
            source_record_id = str(record["_ui5_source_record_id"])
            if not source_record_id or source_record_id in source_record_ids:
                raise ValueError(
                    f"selected sample {sample_id} has duplicate/empty source_record_id: "
                    f"{source_record_id!r}"
                )
            source_record_ids.add(source_record_id)
            global_boxes = _strict_bbox_list(
                record["_ui5_source_gt_global"],
                label=f"selected sample {sample_id} source[{index}].global GT",
            )
            norm_boxes = _strict_bbox_list(
                record["_ui5_source_gt_1000"],
                label=f"selected sample {sample_id} source[{index}].norm1000 GT",
                maximum=1000,
            )
            if len(global_boxes) != len(norm_boxes):
                raise ValueError(
                    f"selected sample {sample_id} source {source_record_id} has "
                    "mismatched global/norm1000 GT counts"
                )
            prefix, answer_boxes = _answer_contract(
                _assistant_text(record),
                label=f"selected sample {sample_id} source {source_record_id} answer",
            )
            if answer_boxes != norm_boxes:
                raise ValueError(
                    f"selected sample {sample_id} source {source_record_id} answer "
                    "does not match its declared norm1000 GT"
                )
            prefixes.add(prefix)
            prompts.add(_human_text(record))
            media.add(_media_signature(record))
            for global_box, norm_box in zip(global_boxes, norm_boxes):
                global_key = tuple(global_box)
                norm_key = tuple(norm_box)
                previous = global_to_norm.get(global_key)
                if previous is not None and previous != norm_key:
                    raise ValueError(
                        f"selected sample {sample_id} maps global box "
                        f"{list(global_key)} to conflicting norm1000 boxes"
                    )
                global_to_norm[global_key] = norm_key

        if len(prefixes) != 1 or len(prompts) != 1 or "" in prompts or len(media) != 1:
            raise ValueError(
                f"selected sample {sample_id} cannot form one lossless union "
                "record because source prompt/ref/media differ"
            )
        authoritative_keys = [tuple(box) for box in authoritative_gt]
        if set(global_to_norm) != set(authoritative_keys):
            missing = sorted(set(authoritative_keys) - set(global_to_norm))
            extra = sorted(set(global_to_norm) - set(authoritative_keys))
            raise ValueError(
                f"selected sample {sample_id} source GT union does not match "
                f"snapshot gt_global: missing={missing}, extra={extra}"
            )
        norm_union = [list(global_to_norm[key]) for key in authoritative_keys]
        if len({tuple(box) for box in norm_union}) != len(norm_union):
            raise ValueError(
                f"selected sample {sample_id} has non-bijective global/norm1000 GT"
            )
        prefix = next(iter(prefixes))
        answer = prefix + (
            "".join(
                f"<box><{box[0]}><{box[1]}><{box[2]}><{box[3]}></box>"
                for box in norm_union
            )
            if norm_union
            else "<box>none</box>"
        )
        output = _replace_assistant_text(records[0], answer)
        for key in metadata_keys:
            # Do not leave the representative source subset next to the union
            # answer; publish only group-level supervision/provenance.
            output.pop(key, None)
        output.update(
            {
                "_ui5_positive": bool(authoritative_gt),
                "_ui5_union_gt_global": authoritative_gt,
                "_ui5_union_gt_1000": norm_union,
                "_ui5_union_source_record_ids": sorted(source_record_ids),
                "_ui5_union_source_record_count": len(source_record_ids),
                "_ui5_supervision_policy": "snapshot_verified_union_gt",
            }
        )
        return output

    if len(records) > 1:
        raise ValueError(
            f"selected sample {sample_id} has multiple training records without "
            "source GT provenance; cannot form a verified union"
        )
    output = json.loads(json.dumps(records[0], ensure_ascii=False))
    output["_ui5_supervision_policy"] = "single_record"
    output["_ui5_union_source_record_count"] = len(records)
    return output


def _strict_nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return int(value)


def _strict_image_box(
    value: Any, *, label: str, width: int, height: int
) -> list[int]:
    boxes = _strict_bbox_list([value], label=label)
    box = boxes[0]
    if box[2] > width or box[3] > height:
        raise ValueError(
            f"{label} is outside image bounds {width}x{height}: {box}"
        )
    return box


def _strict_local_boxes(
    value: Any, *, label: str, width: int, height: int
) -> list[list[int]]:
    boxes = _strict_bbox_list(value, label=label)
    for box in boxes:
        if box[2] > width or box[3] > height:
            raise ValueError(
                f"{label} is outside crop bounds {width}x{height}: {box}"
            )
    return boxes


def _strict_vertical_partition(
    value: Any, *, label: str, width: int, height: int
) -> list[list[int]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty tile list")
    tiles = [
        _strict_image_box(tile, label=f"{label}[{index}]", width=width, height=height)
        for index, tile in enumerate(value)
    ]
    if any(tile[0] != 0 or tile[2] != width for tile in tiles):
        raise ValueError(f"{label} is not a full-width detector-scan partition")
    if tiles[0][1] != 0 or tiles[-1][3] != height:
        raise ValueError(f"{label} does not span the full image height")
    for left, right in zip(tiles, tiles[1:]):
        if left[3] != right[1]:
            raise ValueError(f"{label} has a gap or overlap between tiles")
    if len({tuple(tile) for tile in tiles}) != len(tiles):
        raise ValueError(f"{label} contains duplicate tiles")
    return tiles


def _bundle_crop_geometry(
    bundle: Path,
    verified_images: Mapping[Path, str],
    progress: BuildProgress | None = None,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    """Load only immutable, GT-free detector-scan geometry from the bundle."""

    unique_rows = _read_jsonl(bundle / "manifest" / "unique_images.jsonl")
    images_by_id: dict[str, dict[str, Any]] = {}
    for row_number, raw in enumerate(_progress_items(
        progress, "index_image_geometry", unique_rows, unit="images",
        detail=lambda row: str(row.get("image_id", "")),
    ), 1):
        image_id = str(raw.get("image_id") or "")
        if not image_id or image_id in images_by_id:
            raise ValueError(
                f"unique-images row {row_number} has duplicate/empty image_id"
            )
        image = _bundle_file(
            bundle,
            raw.get("image_relpath"),
            label=f"unique image {image_id}",
        )
        if image not in verified_images:
            raise ValueError(f"unique image is not bundle-verified: {image_id}")
        width = _strict_nonnegative_int(
            raw.get("width"), label=f"unique image {image_id}.width"
        )
        height = _strict_nonnegative_int(
            raw.get("height"), label=f"unique image {image_id}.height"
        )
        if width == 0 or height == 0:
            raise ValueError(f"verified bundle image has invalid size: {image_id}")
        images_by_id[image_id] = {
            "path": image,
            "relative": str(raw["image_relpath"]),
            "sha256": verified_images[image],
            "width": width,
            "height": height,
        }

    plans_payload = json.loads(
        (bundle / "base_scan_plans.json").read_text(encoding="utf-8")
    )
    if not isinstance(plans_payload, Mapping):
        raise ValueError("base_scan_plans.json is not an object")
    plans: dict[str, dict[str, Any]] = {}
    for raw_image_id, raw_plan in _progress_items(
        progress, "validate_base_scan_plans", plans_payload.items(), unit="images",
        detail=lambda item: str(item[0]),
    ):
        image_id = str(raw_image_id)
        if image_id not in images_by_id:
            raise ValueError(f"base scan plan references unknown image_id={image_id}")
        if not isinstance(raw_plan, Mapping):
            raise ValueError(f"base scan plan is not an object: image_id={image_id}")
        image_state = images_by_id[image_id]
        width, height = image_state["width"], image_state["height"]
        for key, actual in (("width", width), ("height", height)):
            declared = raw_plan.get(key)
            if (
                isinstance(declared, bool)
                or not isinstance(declared, int)
                or int(declared) != actual
            ):
                raise ValueError(
                    f"base scan plan {image_id} {key} mismatch: "
                    f"declared={declared!r}, decoded={actual}"
                )
        if raw_plan.get("gt_used") is not False:
            raise ValueError(
                f"base scan plan {image_id} is not explicitly GT-free"
            )
        tiles = _strict_vertical_partition(
            raw_plan.get("base_tiles"),
            label=f"base scan plan {image_id}.base_tiles",
            width=width,
            height=height,
        )
        plans[image_id] = {**dict(raw_plan), "base_tiles": tiles}

    rows_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_crop_ids: set[str] = set()
    seen_sample_indices: set[tuple[str, int]] = set()
    for row_number, raw in enumerate(_progress_items(
        progress, "validate_crop_geometry",
        _read_jsonl(bundle / "manifest" / "crop_samples.jsonl"), unit="crops",
        detail=lambda row: str(row.get("crop_id", "")),
    ), 1):
        row = dict(raw)
        sample_id = _sample_id(row)
        image_id = str(row.get("source_image_id") or "")
        if image_id not in images_by_id:
            raise ValueError(
                f"crop_samples row {row_number} references unknown image_id={image_id!r}"
            )
        task = _task(row.get("task"))
        crop_id = str(row.get("crop_id") or "")
        if not crop_id or crop_id in seen_crop_ids:
            raise ValueError(
                f"crop_samples row {row_number} has duplicate/empty crop_id={crop_id!r}"
            )
        seen_crop_ids.add(crop_id)
        crop_index = _strict_nonnegative_int(
            row.get("crop_index"), label=f"crop {crop_id}.crop_index"
        )
        index_key = (sample_id, crop_index)
        if index_key in seen_sample_indices:
            raise ValueError(
                f"crop sample {sample_id} has duplicate crop_index={crop_index}"
            )
        seen_sample_indices.add(index_key)
        image_state = images_by_id[image_id]
        width, height = image_state["width"], image_state["height"]
        crop = _strict_image_box(
            row.get("crop_xyxy"),
            label=f"crop {crop_id}.crop_xyxy",
            width=width,
            height=height,
        )
        crop_width, crop_height = crop[2] - crop[0], crop[3] - crop[1]
        if row.get("crop_size") != [crop_width, crop_height]:
            raise ValueError(f"crop {crop_id} has inconsistent crop_size")
        if row.get("geometry_source") != "base_scan_plans.base_tiles":
            raise ValueError(f"crop {crop_id} has an untrusted geometry source")
        if row.get("gt_used_for_geometry") is not False:
            raise ValueError(f"crop {crop_id} geometry is not explicitly GT-free")
        if str(row.get("image_relpath") or "") != image_state["relative"]:
            raise ValueError(f"crop {crop_id} image_relpath differs from unique-images")

        if task == "content_missing":
            expected_tiles = [[0, 0, width, height]]
        else:
            if image_id not in plans:
                raise ValueError(
                    f"crop sample {sample_id} has no GT-free base scan plan"
                )
            expected_tiles = plans[image_id]["base_tiles"]
        if crop_index >= len(expected_tiles) or crop != expected_tiles[crop_index]:
            raise ValueError(
                f"crop {crop_id} does not equal base tile index {crop_index}"
            )
        row.update(
            {
                "sample_id": sample_id,
                "task": task,
                "crop_id": crop_id,
                "crop_index": crop_index,
                "crop_xyxy": crop,
                "_source_image": str(image_state["path"]),
                "_source_image_sha256": image_state["sha256"],
                "_source_width": width,
                "_source_height": height,
                "_base_tiles": expected_tiles,
            }
        )
        rows_by_sample[sample_id].append(row)

    for sample_id, rows in rows_by_sample.items():
        rows.sort(key=lambda row: int(row["crop_index"]))
        expected = list(range(len(rows[0]["_base_tiles"])))
        observed = [int(row["crop_index"]) for row in rows]
        if observed != expected or len(rows) != len(rows[0]["_base_tiles"]):
            raise ValueError(
                f"crop sample {sample_id} does not contain every base tile exactly once: "
                f"expected={expected}, observed={observed}"
            )
        if any(row["_base_tiles"] != rows[0]["_base_tiles"] for row in rows):
            raise ValueError(f"crop sample {sample_id} mixes base scan plans")
    if not rows_by_sample:
        raise ValueError("rollout bundle contains no crop_samples records")
    return dict(rows_by_sample), plans, images_by_id


def _validated_selected_crop_rows(
    sample_id: str,
    rollout_row: Mapping[str, Any],
    rows_by_sample: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    expected_prompt: str,
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in rows_by_sample.get(sample_id, ())]
    if not rows:
        raise ValueError(f"selected region group has no base-tile rows: {sample_id}")
    task = _task(rollout_row.get("task"))
    if task == "content_missing":
        raise ValueError("content_missing must not be converted to crop supervision")
    authoritative = _strict_bbox_list(
        rollout_row.get("gt_global"), label=f"selected group {sample_id}.gt_global"
    )
    occurrences: Counter[tuple[int, int, int, int]] = Counter()
    for row in rows:
        crop_id = str(row["crop_id"])
        if row["task"] != task:
            raise ValueError(f"crop {crop_id} task differs from selected group")
        if str(row.get("source_image_id") or "") != str(
            rollout_row.get("source_image_id") or rollout_row.get("image_id") or ""
        ):
            raise ValueError(f"crop {crop_id} source image differs from selected group")
        if str(row.get("image_relpath") or "") != str(
            rollout_row.get("image_relpath") or ""
        ):
            raise ValueError(f"crop {crop_id} image path differs from selected group")
        if str(row.get("record_id") or "") != sample_id:
            raise ValueError(f"crop {crop_id} record_id differs from selected group")
        if str(row.get("prompt") or "") != expected_prompt:
            raise ValueError(f"crop {crop_id} prompt differs from training record")
        if row.get("pipeline_coverage_failure") is not False:
            raise ValueError(f"crop {crop_id} belongs to a coverage-failure sample")
        if row.get("coordinate_transform_anomaly") is not False:
            raise ValueError(f"crop {crop_id} has a coordinate-transform anomaly")
        partial = row.get("partial_gt_indices")
        if partial != []:
            raise ValueError(
                f"crop {crop_id} has partial_gt_indices={partial!r}; "
                "seam-crossing supervision is not learnable"
            )
        sample_gt = _strict_bbox_list(
            row.get("sample_gt_global"), label=f"crop {crop_id}.sample_gt_global"
        )
        if sample_gt != authoritative:
            raise ValueError(
                f"crop {crop_id} sample_gt_global differs from authoritative union"
            )
        crop = row["crop_xyxy"]
        crop_width, crop_height = crop[2] - crop[0], crop[3] - crop[1]
        global_boxes = _strict_bbox_list(
            row.get("gt_global"), label=f"crop {crop_id}.gt_global"
        )
        local_boxes = _strict_local_boxes(
            row.get("gt_local"),
            label=f"crop {crop_id}.gt_local",
            width=crop_width,
            height=crop_height,
        )
        norm_boxes = _strict_bbox_list(
            row.get("gt_local_1000"),
            label=f"crop {crop_id}.gt_local_1000",
            maximum=1000,
        )
        if not (len(global_boxes) == len(local_boxes) == len(norm_boxes)):
            raise ValueError(f"crop {crop_id} has mismatched GT coordinate counts")
        transforms = row.get("coordinate_transforms")
        if not isinstance(transforms, list) or len(transforms) != len(global_boxes):
            raise ValueError(f"crop {crop_id} has mismatched coordinate_transforms")
        for index, (global_box, local_box, norm_box, transform) in enumerate(
            zip(global_boxes, local_boxes, norm_boxes, transforms)
        ):
            if not (
                crop[0] <= global_box[0] < global_box[2] <= crop[2]
                and crop[1] <= global_box[1] < global_box[3] <= crop[3]
            ):
                raise ValueError(f"crop {crop_id} GT is not fully contained")
            expected_local = [
                global_box[0] - crop[0],
                global_box[1] - crop[1],
                global_box[2] - crop[0],
                global_box[3] - crop[1],
            ]
            expected_norm = [
                round(expected_local[0] / crop_width * 1000),
                round(expected_local[1] / crop_height * 1000),
                round(expected_local[2] / crop_width * 1000),
                round(expected_local[3] / crop_height * 1000),
            ]
            if local_box != expected_local or norm_box != expected_norm:
                raise ValueError(f"crop {crop_id} local coordinate mapping differs")
            if not isinstance(transform, Mapping) or any(
                transform.get(key) != expected
                for key, expected in (
                    ("global_bbox_xyxy", global_box),
                    ("local_bbox_xyxy", local_box),
                    ("local_bbox_1000", norm_box),
                )
            ):
                raise ValueError(
                    f"crop {crop_id} coordinate_transforms[{index}] differs"
                )
            occurrences[tuple(global_box)] += 1
    expected_occurrences = Counter(tuple(box) for box in authoritative)
    if occurrences != expected_occurrences or any(
        count != 1 for count in occurrences.values()
    ):
        raise ValueError(
            f"selected group {sample_id} tile GT union/count differs from "
            f"authoritative union: expected={expected_occurrences}, observed={occurrences}"
        )
    return rows


def _asset_relative_path(asset_namespace: str, crop_id: str) -> str:
    filename = hashlib.sha256(crop_id.encode("utf-8")).hexdigest()[:32] + ".png"
    return (Path(asset_namespace) / filename).as_posix()


def _selected_crop_supervision(
    sample_id: str,
    canonical: Mapping[str, Any],
    rollout_row: Mapping[str, Any],
    rows_by_sample: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    asset_namespace: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prompt = _human_text(canonical)
    prefix, _ = _answer_contract(
        _assistant_text(canonical), label=f"selected group {sample_id} union answer"
    )
    rows = _validated_selected_crop_rows(
        sample_id, rollout_row, rows_by_sample, expected_prompt=prompt
    )
    output: list[dict[str, Any]] = []
    assets: list[dict[str, Any]] = []
    for row in rows:
        norm_boxes = row["gt_local_1000"]
        answer = prefix + (
            "".join(
                f"<box><{box[0]}><{box[1]}><{box[2]}><{box[3]}></box>"
                for box in norm_boxes
            )
            if norm_boxes
            else "<box>none</box>"
        )
        record = _replace_assistant_text(canonical, answer)
        source_image = record.get("image")
        if not isinstance(source_image, str) or any(
            key in record for key in ("video", "videos", "video_list")
        ):
            raise ValueError(
                f"selected crop group {sample_id} must have exactly one image field"
            )
        # The portable bundle deliberately preserves legacy ``images`` aliases
        # when they existed in the source record.  Canonical media consistency
        # was already checked above; collapse an equivalent one-image alias,
        # but reject multiple or conflicting media rather than guessing.
        for alias in ("images", "image_list"):
            if alias not in record:
                continue
            raw_alias = record[alias]
            alias_values = raw_alias if isinstance(raw_alias, list) else [raw_alias]
            if alias_values != [source_image]:
                raise ValueError(
                    f"selected crop group {sample_id} has conflicting {alias} media"
                )
            record.pop(alias)
        relative = _asset_relative_path(asset_namespace, str(row["crop_id"]))
        # build() replaces this recipe-relative placeholder after the whole
        # image tree is atomically published.
        record["image"] = relative
        record["_ui5_crop_asset_relpath"] = relative
        record.pop("_ui5_union_gt_1000", None)
        record.update(
            {
                "_ui5_record_kind": "crop",
                "_ui5_crop_source": "gt_free_detector_scan_base_tile",
                "_ui5_training_eligible": True,
                "_ui5_positive": bool(row["gt_local_1000"]),
                "_ui5_crop_id": row["crop_id"],
                "_ui5_crop_bbox": row["crop_xyxy"],
                "_ui5_crop_index": row["crop_index"],
                "_ui5_base_tile_count": len(row["_base_tiles"]),
                "_ui5_crop_gt_local": row["gt_local"],
                "_ui5_crop_gt_local_1000": row["gt_local_1000"],
                "_ui5_crop_gt_global": row["gt_global"],
                "_ui5_sample_gt_global": row["sample_gt_global"],
                "_ui5_partial_gt_indices": [],
                "_ui5_geometry_source": "base_scan_plans.base_tiles",
                "_ui5_gt_used_for_geometry": False,
            }
        )
        output.append(record)
        assets.append(
            {
                "relative_path": relative,
                "crop_id": row["crop_id"],
                "sample_id": sample_id,
                "source_image": row["_source_image"],
                "source_image_sha256": row["_source_image_sha256"],
                "source_size": [row["_source_width"], row["_source_height"]],
                "crop_xyxy": row["crop_xyxy"],
                "crop_size": row["crop_size"],
            }
        )
    return output, assets


def _load_crop_reuse_inventory(
    source_dir: Path, output_dir: Path, bundle_state: Mapping[str, Any],
    assets: Sequence[Mapping[str, Any]], progress: BuildProgress | None,
) -> dict[str, Any]:
    """Import only pixel assets, never the old selection, labels or pool membership."""
    source_dir = source_dir.resolve(strict=True)
    output_dir = output_dir.resolve()
    if source_dir == output_dir or source_dir in output_dir.parents or output_dir in source_dir.parents:
        raise ValueError("crop reuse requires separate, non-nested curriculum directories")
    manifest_path = source_dir / "curriculum_manifest.json"
    success_path = source_dir / "_SUCCESS.json"
    if not manifest_path.is_file() or not success_path.is_file():
        raise RuntimeError("crop reuse source is incomplete; wait for its _SUCCESS.json")
    with (progress.stage("load_crop_reuse_inventory", unit="files")
          if progress else nullcontext()):
        signatures = {p.name: _sha256_file(p) for p in (manifest_path, success_path)}
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        success = json.loads(success_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or not isinstance(success, dict):
            raise RuntimeError("crop reuse source has invalid publication metadata")
        if manifest.get("schema_version") != SCHEMA_VERSION or success.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError("crop reuse source schema differs from the verified v4 crop contract")
        identity_payload = dict(manifest)
        identity = identity_payload.pop("identity_digest", None)
        if (
            success.get("complete") is not True
            or identity != _json_digest(identity_payload)
            or success.get("identity_digest") != identity
        ):
            raise RuntimeError("crop reuse source publication identity mismatch")
        old_bundle = (manifest.get("inputs") or {}).get("rollout_bundle")
        if not isinstance(old_bundle, Mapping) or any(
            old_bundle.get(key) != bundle_state.get(key)
            for key in ("root", "manifest_sha256")
        ):
            raise RuntimeError("crop reuse source bundle identity differs; no recropping fallback")
        policy = manifest.get("training_view_policy") or {}
        if policy.get("tile_selection_uses_gt") is not False or policy.get("partial_gt_allowed") is not False:
            raise RuntimeError("crop reuse source has an incompatible crop policy")
        rows = manifest.get("crop_assets")
        if not isinstance(rows, list) or not rows:
            raise RuntimeError("crop reuse source has no crop inventory")
        source_by_id: dict[str, dict[str, Any]] = {}
        source_relatives: set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                raise RuntimeError("invalid crop reuse inventory row")
            crop_id = str(row.get("crop_id") or "")
            relative = Path(str(row.get("relative_path") or ""))
            if (
                not crop_id or crop_id in source_by_id
                or relative.is_absolute() or len(relative.parts) != 2
                or not relative.parts[0].startswith("training_crops-")
                or relative.as_posix() != _asset_relative_path(relative.parts[0], crop_id)
                or relative.as_posix() in source_relatives
            ):
                raise RuntimeError("unsafe or duplicate crop reuse inventory path/ID")
            _validated_sha256(row.get("sha256"), label="reused PNG sha256")
            _validated_sha256(row.get("source_image_sha256"), label="reused source image sha256")
            source_by_id[crop_id] = row
            source_relatives.add(relative.as_posix())
        files = success.get("files")
        non_image_files = {
            "ui5_crop_rollout4_curriculum.json", "hard.jsonl", "matched_anchor.jsonl",
            "global_replay.jsonl", "hard_groups.jsonl", "matched_anchor_groups.jsonl",
            "crop_assets.jsonl",
        }
        if not isinstance(files, dict) or set(files) != non_image_files | source_relatives:
            raise RuntimeError("crop reuse source success file inventory differs")
        for relative in _progress_items(
            progress, "verify_crop_reuse_metadata", sorted(non_image_files), unit="files",
            detail=str,
        ):
            path = source_dir / relative
            metadata = files[relative]
            if (
                not isinstance(metadata, dict) or not path.is_file()
                or path.stat().st_size != metadata.get("bytes")
                or _sha256_file(path) != metadata.get("sha256")
            ):
                raise RuntimeError(f"crop reuse source artifact changed: {path}")
        if _read_jsonl(source_dir / "crop_assets.jsonl") != rows:
            raise RuntimeError("crop reuse JSONL differs from the bound manifest inventory")
        if success.get("recipe_sha256") != files["ui5_crop_rollout4_curriculum.json"]["sha256"]:
            raise RuntimeError("crop reuse source recipe digest differs")
        expected_ids = {str(row["crop_id"]) for row in assets}
        if len(expected_ids) != len(assets) or expected_ids != set(source_by_id):
            raise RuntimeError("crop reuse ID set differs from the complete target bundle; no recropping fallback")
        for raw in assets:
            old = source_by_id[str(raw["crop_id"])]
            expected = {
                "sample_id": str(raw["sample_id"]),
                "source_image_sha256": str(raw["source_image_sha256"]),
                "crop_xyxy": list(raw["crop_xyxy"]),
                "width": raw["crop_size"][0], "height": raw["crop_size"][1],
            }
            if any(old.get(key) != value for key, value in expected.items()):
                raise RuntimeError(f"crop reuse geometry/source mismatch: {raw['crop_id']}")
            if files[old["relative_path"]] != {"bytes": old["bytes"], "sha256": old["sha256"]}:
                raise RuntimeError(f"crop reuse PNG digest is not bound by success: {raw['crop_id']}")
        if any(_sha256_file(source_dir / name) != value for name, value in signatures.items()):
            raise RuntimeError("crop reuse publication changed during validation")
        return {
            "root": source_dir, "by_id": source_by_id,
            "audit": {
                "mode": "verified_hardlink_all", "source_curriculum_dir": str(source_dir),
                "source_curriculum_identity": identity,
                "source_manifest_sha256": signatures[manifest_path.name],
                "source_success_sha256": signatures[success_path.name],
                "reused_crop_assets": len(assets), "generated_crop_assets": 0,
            },
        }


def _materialize_crop_assets(
    output_dir: Path, asset_namespace: str, assets: Sequence[Mapping[str, Any]],
    progress: BuildProgress | None = None,
    reuse: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Publish every selected crop as one atomically-renamed image tree."""

    if not assets:
        raise ValueError("curriculum selection produced zero region crop assets")
    target_dir = output_dir / asset_namespace
    staging = Path(
        tempfile.mkdtemp(prefix=f".{asset_namespace}.staging-", dir=output_dir)
    )
    inventory: list[dict[str, Any]] = []
    try:
        seen_relative: set[str] = set()
        for raw in _progress_items(
            progress, "reuse_crop_pngs" if reuse else "materialize_crop_pngs",
            sorted(assets, key=lambda row: str(row["relative_path"])), unit="crops",
            detail=lambda row: f"crop_id={row['crop_id']} source={row['source_image']}",
        ):
            relative = Path(str(raw["relative_path"]))
            if relative.parts[0] != asset_namespace or len(relative.parts) != 2:
                raise ValueError(f"unsafe crop asset path: {relative}")
            if relative.as_posix() in seen_relative:
                raise ValueError(f"duplicate crop asset path: {relative}")
            seen_relative.add(relative.as_posix())
            crop = [int(value) for value in raw["crop_xyxy"]]
            destination = staging / relative.name
            if reuse:
                old = reuse["by_id"][str(raw["crop_id"])]
                source_png = (reuse["root"] / old["relative_path"]).resolve(strict=True)
                if not source_png.is_relative_to(reuse["root"]):
                    raise RuntimeError(f"crop reuse path escapes source directory: {source_png}")
                if source_png.stat().st_size != old["bytes"] or _sha256_file(source_png) != old["sha256"]:
                    raise RuntimeError(f"crop reuse PNG changed: {source_png}")
                try:
                    os.link(source_png, destination)
                except OSError as exc:
                    raise RuntimeError(
                        "crop reuse requires hard-link support on the same filesystem; "
                        "no copy/recropping fallback"
                    ) from exc
            else:
                source = Path(str(raw["source_image"])).resolve(strict=True)
                if _sha256_file(source) != str(raw["source_image_sha256"]):
                    raise RuntimeError(f"verified source image changed: {source}")
                with Image.open(source) as handle:
                    handle.load()
                    if list(map(int, handle.size)) != list(raw["source_size"]):
                        raise RuntimeError(
                            f"verified source image dimensions changed: {source}"
                        )
                    materialized = handle.crop(tuple(crop))
                    if materialized.mode not in {"1", "L", "LA", "P", "RGB", "RGBA", "I", "I;16"}:
                        materialized = materialized.convert("RGB")
                    materialized.save(destination, format="PNG")
                with destination.open("rb+") as handle:
                    handle.flush()
                    os.fsync(handle.fileno())
            expected_size = tuple(int(value) for value in raw["crop_size"])
            with Image.open(destination) as handle:
                # Reused bytes already match a fully decoded, published PNG.
                # Check its header dimensions without decoding the image again.
                if not reuse:
                    handle.load()
                if tuple(map(int, handle.size)) != expected_size:
                    raise RuntimeError(
                        f"materialized crop size mismatch: {relative.as_posix()}"
                    )
            inventory.append(
                {
                    "relative_path": relative.as_posix(),
                    "crop_id": str(raw["crop_id"]),
                    "sample_id": str(raw["sample_id"]),
                    "source_image_sha256": str(raw["source_image_sha256"]),
                    "crop_xyxy": crop,
                    "width": expected_size[0],
                    "height": expected_size[1],
                    "bytes": destination.stat().st_size,
                    "sha256": _sha256_file(destination),
                }
            )
            if reuse and (
                inventory[-1]["bytes"] != old["bytes"]
                or inventory[-1]["sha256"] != old["sha256"]
            ):
                raise RuntimeError(f"crop reuse PNG changed while linking: {source_png}")

        expected_names = {Path(row["relative_path"]).name for row in inventory}
        if target_dir.exists():
            observed_names = {
                path.name for path in target_dir.iterdir() if path.is_file()
            }
            if observed_names != expected_names or any(
                (target_dir / Path(row["relative_path"]).name).stat().st_size
                != int(row["bytes"])
                or _sha256_file(target_dir / Path(row["relative_path"]).name)
                != row["sha256"]
                for row in _progress_items(
                    progress, "verify_existing_crop_directory", inventory, unit="crops",
                    detail=lambda row: str(row["relative_path"]),
                )
            ):
                raise RuntimeError(
                    f"existing crop asset directory differs: {target_dir}"
                )
            shutil.rmtree(staging)
        else:
            os.replace(staging, target_dir)
        for row in _progress_items(
            progress, "verify_published_crop_pngs", inventory, unit="crops",
            detail=lambda row: str(row["relative_path"]),
        ):
            published = output_dir / row["relative_path"]
            if (
                not published.is_file()
                or published.stat().st_size != int(row["bytes"])
                or _sha256_file(published) != row["sha256"]
            ):
                raise RuntimeError(f"published crop asset changed: {published}")
        return inventory
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _global_view_supervision(
    canonical: Mapping[str, Any], *, retention: bool
) -> dict[str, Any]:
    record = json.loads(json.dumps(canonical, ensure_ascii=False))
    record.update(
        {
            "_ui5_record_kind": "full_image" if retention else "global_view",
            "_ui5_crop_source": (
                "global_replay_retention" if retention else "content_missing_global"
            ),
            "_ui5_training_eligible": True,
            "_ui5_retention_view": bool(retention),
        }
    )
    return record


def _eligible_difficulty(
    rows: Sequence[Mapping[str, Any]],
    record_truth: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    seen_sample_ids: set[str] = set()
    for raw in rows:
        sample_id = _sample_id(raw)
        if sample_id in seen_sample_ids:
            raise ValueError(f"duplicate rollout difficulty group: {sample_id}")
        seen_sample_ids.add(sample_id)
        row = dict(raw)
        # Formal hard/anchor membership is permitted only for source samples
        # that the immutable snapshot marked structurally eligible.  Check this
        # before consulting potentially anomalous task/polarity annotations.
        if (
            row.get("grpo_source_eligible") is not True
            or any(row.get(key) is True for key in FORMAL_ANOMALY_FIELDS)
            or row.get("m31_complete4") is not True
            or row.get("crop_complete4") is not True
            or row.get("cross_model_complete8") is not True
            or row.get("technical_error_free") is not True
            or row.get("runtime_error_count") != 0
            or row.get("parse_error_count") != 0
        ):
            continue
        truth = record_truth.get(sample_id) if record_truth is not None else None
        if record_truth is not None and truth is None:
            raise ValueError(
                f"rollout difficulty sample has no training-record truth: {sample_id}"
            )
        explicit_task = _task(row.get("task"))
        if truth is not None and explicit_task != truth["task"]:
            raise ValueError(
                f"rollout/training task conflict for sample {sample_id}: "
                f"rollout={explicit_task}, training={truth['task']}"
            )
        row["task"] = explicit_task
        explicit_polarity = _explicit_polarity(row)
        if explicit_polarity is None:
            if truth is None:
                raise ValueError(f"cannot determine polarity for sample {sample_id}")
            polarity = truth["polarity"]
        else:
            polarity = explicit_polarity
            if truth is not None and polarity != truth["polarity"]:
                raise ValueError(
                    f"rollout/training polarity conflict for sample {sample_id}: "
                    f"rollout={polarity}, training={truth['polarity']}"
                )
        value = row.get("crop_correct_count")
        if value is None:
            continue
        value = _strict_int_field(
            row, "crop_correct_count", label=f"rollout row {sample_id}"
        )
        if not 0 <= value <= 4:
            raise ValueError(f"invalid crop_correct_count={value} for {sample_id}")
        row["crop_correct_count"] = value
        row["sample_id"] = sample_id
        row["record_id"] = str(row.get("record_id") or sample_id)
        row["positive"] = polarity == "positive"
        row["polarity"] = polarity
        by_id[sample_id] = row
    return by_id


def _minimum_cost_rectangular_assignment(costs: Sequence[Sequence[int]]) -> list[int]:
    """Return one distinct column per row for a rectangular integer cost matrix."""

    row_count = len(costs)
    if row_count == 0:
        return []
    column_count = len(costs[0])
    if column_count < row_count or any(len(row) != column_count for row in costs):
        raise ValueError("anchor assignment requires a rectangular rows<=columns matrix")
    # Hungarian shortest-augmenting-path form.  Iteration order and integer
    # costs make ties reproducible on every Python/platform combination.
    u = [0] * (row_count + 1)
    v = [0] * (column_count + 1)
    matched_row = [0] * (column_count + 1)
    way = [0] * (column_count + 1)
    for row_index in range(1, row_count + 1):
        matched_row[0] = row_index
        min_value: list[int | None] = [None] * (column_count + 1)
        used = [False] * (column_count + 1)
        column = 0
        while True:
            used[column] = True
            current_row = matched_row[column]
            delta: int | None = None
            next_column = 0
            for candidate in range(1, column_count + 1):
                if used[candidate]:
                    continue
                reduced = (
                    int(costs[current_row - 1][candidate - 1])
                    - u[current_row]
                    - v[candidate]
                )
                if min_value[candidate] is None or reduced < min_value[candidate]:
                    min_value[candidate] = reduced
                    way[candidate] = column
                if delta is None or min_value[candidate] < delta:
                    delta = min_value[candidate]
                    next_column = candidate
            if delta is None:
                raise RuntimeError("anchor assignment could not find an augmenting path")
            for candidate in range(column_count + 1):
                if used[candidate]:
                    u[matched_row[candidate]] += delta
                    v[candidate] -= delta
                elif min_value[candidate] is not None:
                    min_value[candidate] -= delta
            column = next_column
            if matched_row[column] == 0:
                break
        while True:
            previous = way[column]
            matched_row[column] = matched_row[previous]
            column = previous
            if column == 0:
                break
    assignment = [-1] * row_count
    for column in range(1, column_count + 1):
        if matched_row[column]:
            assignment[matched_row[column] - 1] = column - 1
    if any(column < 0 for column in assignment) or len(set(assignment)) != row_count:
        raise RuntimeError("anchor assignment is incomplete or non-unique")
    return assignment


def _match_anchors(
    hard: Sequence[Mapping[str, Any]],
    eligible: Mapping[str, Mapping[str, Any]],
    crop_rows_by_sample: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    seed: int,
    progress: BuildProgress | None = None,
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in eligible.values():
        if int(row["crop_correct_count"]) != 4:
            continue
        buckets[(str(row["task"]), str(row["polarity"]))].append(dict(row))
    hard_buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in hard:
        hard_buckets[(str(row["task"]), str(row["polarity"]))].append(dict(row))

    by_hard_id: dict[str, dict[str, Any]] = {}
    for key in _progress_items(
        progress, "match_anchor_strata", sorted(hard_buckets), unit="strata",
        detail=lambda key: (
            f"task={key[0]} polarity={key[1]} hard={len(hard_buckets[key])} "
            f"anchor_candidates={len(buckets.get(key, []))}"
        ),
    ):
        hard_rows = sorted(hard_buckets[key], key=lambda row: str(row["sample_id"]))
        candidates = sorted(buckets.get(key, []), key=lambda row: str(row["sample_id"]))
        if len(candidates) < len(hard_rows):
            raise ValueError(
                "not enough distinct 4/4 anchors for stratum "
                f"task={key[0]} polarity={key[1]}: "
                f"hard={len(hard_rows)}, candidates={len(candidates)}"
            )
        edge_order = sorted(
            (
                _json_digest([seed, *key, hard_row["sample_id"], candidate["sample_id"]]),
                str(hard_row["sample_id"]),
                str(candidate["sample_id"]),
            )
            for hard_row in hard_rows
            for candidate in candidates
        )
        tie_rank = {
            (hard_id, candidate_id): rank
            for rank, (_, hard_id, candidate_id) in enumerate(edge_order)
        }
        max_crop_delta = max(
            abs(
                len(crop_rows_by_sample[str(hard_row["sample_id"])])
                - len(crop_rows_by_sample[str(candidate["sample_id"])])
            )
            for hard_row in hard_rows
            for candidate in candidates
        )
        max_tie_total = len(hard_rows) * max(1, len(edge_order))
        max_crop_total = len(hard_rows) * max_crop_delta
        tie_scale = max_tie_total + 1
        gt_scale = (max_crop_total + 1) * tie_scale
        costs: list[list[int]] = []
        for hard_row in hard_rows:
            hard_id = str(hard_row["sample_id"])
            hard_gt_count = len(
                _strict_bbox_list(
                    hard_row.get("gt_global"), label=f"hard group {hard_id}.gt_global"
                )
            )
            hard_crop_count = len(crop_rows_by_sample[hard_id])
            row_costs = []
            for candidate in candidates:
                candidate_id = str(candidate["sample_id"])
                candidate_gt_count = len(
                    _strict_bbox_list(
                        candidate.get("gt_global"),
                        label=f"anchor candidate {candidate_id}.gt_global",
                    )
                )
                candidate_crop_count = len(crop_rows_by_sample[candidate_id])
                row_costs.append(
                    abs(hard_gt_count - candidate_gt_count) * gt_scale
                    + abs(hard_crop_count - candidate_crop_count) * tie_scale
                    + tie_rank[(hard_id, candidate_id)]
                )
            costs.append(row_costs)
        assignment = _minimum_cost_rectangular_assignment(costs)
        for hard_row, candidate_index in zip(hard_rows, assignment):
            anchor = dict(candidates[candidate_index])
            hard_id = str(hard_row["sample_id"])
            anchor_id = str(anchor["sample_id"])
            hard_gt_count = len(hard_row["gt_global"])
            anchor_gt_count = len(anchor["gt_global"])
            hard_crop_count = len(crop_rows_by_sample[hard_id])
            anchor_crop_count = len(crop_rows_by_sample[anchor_id])
            anchor.update(
                {
                    "matched_hard_sample_id": hard_id,
                    "matched_hard_gt_count": hard_gt_count,
                    "anchor_gt_count": anchor_gt_count,
                    "matched_gt_count_delta": abs(hard_gt_count - anchor_gt_count),
                    "matched_hard_crop_count": hard_crop_count,
                    "anchor_crop_count": anchor_crop_count,
                    "matched_crop_count_delta": abs(
                        hard_crop_count - anchor_crop_count
                    ),
                    "anchor_match_policy": (
                        "task_polarity_min_total_gt_delta_then_crop_delta"
                    ),
                }
            )
            by_hard_id[hard_id] = anchor
    ordered_hard_ids = [str(row["sample_id"]) for row in hard]
    if set(by_hard_id) != set(ordered_hard_ids):
        raise RuntimeError("anchor assignment does not cover every hard group exactly once")
    selected = [by_hard_id[sample_id] for sample_id in ordered_hard_ids]
    anchor_ids = [str(row["sample_id"]) for row in selected]
    if len(set(anchor_ids)) != len(anchor_ids):
        raise RuntimeError("anchor assignment reused a sample")
    return selected


def _pool_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    samples = {_sample_id(row) for row in records}
    by_task = Counter(_task(row.get("_ui5_task") or row.get("task")) for row in records)
    by_polarity = Counter(_polarity(row) for row in records)
    by_view = Counter(str(row.get("_ui5_record_kind") or "unknown") for row in records)
    return {
        # ``training_records`` is the durable launcher/preflight contract;
        # retain ``records`` for readers of the v1 manifest.
        "training_records": len(records),
        "records": len(records),
        "sample_groups": len(samples),
        "records_by_task": dict(sorted(by_task.items())),
        "records_by_polarity": dict(sorted(by_polarity.items())),
        "records_by_view": dict(sorted(by_view.items())),
        "crop_training_records": int(by_view.get("crop", 0)),
        "content_missing_global_records": sum(
            _task(row.get("_ui5_task") or row.get("task")) == "content_missing"
            and row.get("_ui5_record_kind") == "global_view"
            for row in records
        ),
        "retention_full_image_records": sum(
            row.get("_ui5_record_kind") == "full_image"
            and row.get("_ui5_retention_view") is True
            for row in records
        ),
    }


def _group_stratum_counts(
    groups: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    counts: Counter[tuple[str, str]] = Counter(
        (_task(row.get("task") or row.get("_ui5_task")), _polarity(row))
        for row in groups
    )
    return {
        task: {
            polarity: int(counts[(task, polarity)])
            for polarity in ("positive", "negative")
        }
        for task in TASKS
    }


def _require_all_ui5_strata(
    counts: Mapping[str, Mapping[str, int]], *, label: str
) -> None:
    missing = [
        f"{task}/{polarity}"
        for task in TASKS
        for polarity in ("positive", "negative")
        if int(counts.get(task, {}).get(polarity, 0)) <= 0
    ]
    if missing:
        raise ValueError(f"{label} lacks required UI5 task/polarity strata: {missing}")


def build(args: argparse.Namespace) -> dict[str, Any]:
    with BuildProgress(
        args.output_dir,
        interval_seconds=float(getattr(args, "progress_interval_seconds", 10.0)),
    ) as progress:
        with progress.stage("curriculum_build", unit="stages") as status:
            return _build(args, progress, status)


def _build(args: argparse.Namespace, progress: BuildProgress, status: Any) -> dict[str, Any]:
    status.set_detail("resolving frozen selection, bundle and output paths")
    base_recipe = (
        args.base_recipe.expanduser().resolve(strict=True)
        if args.base_recipe is not None
        else None
    )
    difficulty_path = args.rollout_difficulty.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    configured_expected_hard = getattr(args, "expected_hard_groups", None)
    if configured_expected_hard is not None and int(configured_expected_hard) <= 0:
        raise ValueError("--expected-hard-groups must be positive when provided")

    bundle = (
        args.rollout_bundle_root.expanduser().resolve(strict=True)
        if args.rollout_bundle_root is not None
        else None
    )
    if bundle is None:
        raise ValueError(
            "--rollout-bundle-root is required for GT-free crop-aligned curriculum"
        )
    if base_recipe is not None:
        raise ValueError(
            "--base-recipe is incompatible with the formal curriculum: all three "
            "pools must be reconstructed from the complete rollout bundle"
        )

    bundle_state: dict[str, Any] | None = None
    verified_bundle_images: dict[Path, str] = {}
    with progress.stage("load_frozen_selection", unit="records"):
        difficulty_rows, difficulty_state = _load_difficulty_rows(difficulty_path)
        formal_hard_ids, frozen_summary_state = _load_frozen_selection_summary(
            difficulty_state, difficulty_rows
        )
    if configured_expected_hard is not None and int(configured_expected_hard) != len(
        formal_hard_ids
    ):
        raise ValueError(
            "configured hard group assertion differs from frozen summary: "
            f"configured={configured_expected_hard}, frozen={len(formal_hard_ids)}"
        )
    status.set_detail("reading rollout bundle inventory and validating all source bytes")
    bundle_state, verified_bundle_images = _verify_rollout_bundle(bundle, progress)
    status.set_detail("loading crop manifests and detector scan geometry")
    crop_rows_by_sample, _, _ = _bundle_crop_geometry(bundle, verified_bundle_images, progress)

    existing_manifest_path = output_dir / "curriculum_manifest.json"
    existing_success_path = output_dir / "_SUCCESS.json"
    if existing_manifest_path.is_file() and existing_success_path.is_file() and not bool(
        getattr(args, "force", False)
    ):
        status.set_detail("complete curriculum found; verifying identity for reuse (no crop generation)")
        existing = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
        success = json.loads(existing_success_path.read_text(encoding="utf-8"))
        if int(existing.get("schema_version", -1)) != SCHEMA_VERSION or int(
            success.get("schema_version", -1)
        ) != SCHEMA_VERSION:
            raise RuntimeError(
                "existing curriculum builder schema differs; use a new output "
                "directory or --force"
            )
        expected_inputs = {
            "base_recipe_sha256": None,
            "rollout_difficulty_sha256": _sha256_file(difficulty_path),
            "rollout_difficulty_authoritative": difficulty_state,
            "frozen_selection_summary": frozen_summary_state,
        }
        actual_inputs = existing.get("inputs") or {}
        for key, expected in expected_inputs.items():
            if actual_inputs.get(key) != expected:
                raise RuntimeError(
                    f"existing curriculum input changed ({key}); use a new output directory"
                )
        actual_bundle = actual_inputs.get("rollout_bundle")
        if bundle_state is None:
            if actual_bundle is not None:
                raise RuntimeError(
                    "existing curriculum input changed (rollout_bundle); "
                    "use a new output directory"
                )
        elif not isinstance(actual_bundle, Mapping) or any(
            actual_bundle.get(key) != bundle_state[key]
            for key in ("root", "manifest_sha256")
        ):
            raise RuntimeError(
                "existing curriculum input changed (rollout_bundle); "
                "use a new output directory"
            )
        if int(existing.get("seed", -1)) != int(args.seed) or int(
            existing.get("expected_hard_groups", -1)
        ) != len(formal_hard_ids):
            raise RuntimeError(
                "existing curriculum seed/expected-hard-groups differs; "
                "use a new output directory"
            )
        if success.get("identity_digest") != existing.get("identity_digest"):
            raise RuntimeError("existing curriculum success marker identity mismatch")
        identity_payload = dict(existing)
        identity_digest = identity_payload.pop("identity_digest", None)
        if identity_digest != _json_digest(identity_payload):
            raise RuntimeError("existing curriculum manifest identity mismatch")
        if success.get("complete") is not True:
            raise RuntimeError("existing curriculum success marker is not complete")
        success_files = success.get("files")
        required_success_files = {
            "ui5_crop_rollout4_curriculum.json",
            "hard.jsonl",
            "matched_anchor.jsonl",
            "global_replay.jsonl",
            "hard_groups.jsonl",
            "matched_anchor_groups.jsonl",
            "crop_assets.jsonl",
        }
        expected_asset_files = {
            str(row.get("relative_path") or "")
            for row in existing.get("crop_assets", [])
            if isinstance(row, Mapping)
        }
        if (
            not isinstance(success_files, Mapping)
            or "" in expected_asset_files
            or set(map(str, success_files))
            != required_success_files | expected_asset_files
        ):
            raise RuntimeError(
                "existing curriculum success marker artifact hashes do not match"
            )
        for relative, metadata in _progress_items(
            progress, "verify_existing_curriculum", success_files.items(), unit="files",
            detail=lambda item: str(item[0]),
        ):
            path = output_dir / str(relative)
            if not isinstance(metadata, Mapping):
                raise RuntimeError(
                    f"existing curriculum artifact metadata is invalid: {relative}"
                )
            if (
                not path.is_file()
                or path.stat().st_size != metadata.get("bytes")
                or _sha256_file(path) != metadata.get("sha256")
            ):
                raise RuntimeError(f"existing curriculum artifact changed: {path}")
        status.set_detail("existing curriculum verified; reusing all published data")
        return existing

    with progress.stage("load_original_groups", unit="groups") as group_status:
        group_status.set_detail("loading original records, union GT and full-bundle group catalog")
        records = _bundle_records(bundle)
        _verify_bundle_record_images(records, verified_bundle_images)
        record_truth = _record_group_truth(records)
        records_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            records_by_sample[_sample_id(record)].append(record)
        bundle_groups, eligible_bundle_groups = _bundle_group_catalog(
            bundle, crop_rows_by_sample, record_truth
        )
        difficulty = _eligible_difficulty(difficulty_rows, record_truth)
    status.set_detail("validating frozen hard membership against original sample groups")
    missing_difficulty_bundle = sorted(set(difficulty) - set(eligible_bundle_groups))
    if missing_difficulty_bundle:
        raise ValueError(
            "formally eligible complete8 groups are not eligible in the bound "
            f"rollout bundle: {missing_difficulty_bundle[:10]}"
        )
    mismatched_difficulty_gt = sorted(
        sample_id
        for sample_id, row in difficulty.items()
        if row.get("gt_global") != eligible_bundle_groups[sample_id].get("gt_global")
    )
    if mismatched_difficulty_gt:
        raise ValueError(
            "formally eligible complete8 GT differs from the bound rollout bundle: "
            f"{mismatched_difficulty_gt[:10]}"
        )
    missing_formal_hard = sorted(set(formal_hard_ids) - set(difficulty))
    extra_recomputed_hard = sorted(
        sample_id
        for sample_id, row in difficulty.items()
        if int(row["crop_correct_count"]) == 0
        and sample_id not in set(formal_hard_ids)
    )
    if missing_formal_hard or extra_recomputed_hard:
        raise ValueError(
            "frozen formal hard membership differs after full technical/source "
            "eligibility validation: "
            f"missing={missing_formal_hard[:10]}, extra={extra_recomputed_hard[:10]}"
        )
    hard = sorted(
        (dict(difficulty[sample_id]) for sample_id in formal_hard_ids),
        key=lambda row: (str(row["task"]), str(row["sample_id"])),
    )
    anchors = _match_anchors(
        hard, difficulty, crop_rows_by_sample, seed=int(args.seed), progress=progress,
    )
    hard_ids = {str(row["sample_id"]) for row in hard}
    anchor_ids = {str(row["sample_id"]) for row in anchors}
    missing_selected_bundle = sorted(
        (hard_ids | anchor_ids) - set(eligible_bundle_groups)
    )
    if missing_selected_bundle:
        raise ValueError(
            "hard/anchor groups are not eligible original bundle crop samples: "
            f"{missing_selected_bundle[:10]}"
        )
    replay_ids = set(eligible_bundle_groups) - hard_ids - anchor_ids
    if hard_ids & anchor_ids:
        raise AssertionError("hard and matched-anchor groups overlap")
    if hard_ids & replay_ids or anchor_ids & replay_ids:
        raise AssertionError("curriculum sample groups are not disjoint")
    bundle_all_strata = _group_stratum_counts(bundle_groups.values())
    bundle_eligible_strata = _group_stratum_counts(eligible_bundle_groups.values())
    hard_strata = _group_stratum_counts(hard)
    anchor_strata = _group_stratum_counts(anchors)
    replay_strata = _group_stratum_counts(
        eligible_bundle_groups[sample_id] for sample_id in sorted(replay_ids)
    )
    _require_all_ui5_strata(
        bundle_eligible_strata, label="eligible complete rollout bundle"
    )
    _require_all_ui5_strata(replay_strata, label="global replay after hard/anchor exclusion")

    missing_hard = sorted(hard_ids - set(records_by_sample))
    missing_anchors = sorted(anchor_ids - set(records_by_sample))
    missing_replay = sorted(replay_ids - set(records_by_sample))
    if missing_hard or missing_anchors or missing_replay:
        raise ValueError(
            "rollout groups do not map to base training records: "
            f"missing_hard={missing_hard[:10]}, missing_anchor={missing_anchors[:10]}, "
            f"missing_replay={missing_replay[:10]}"
        )
    if bundle is not None:
        unverified_group_images: list[str] = []
        for row in (*hard, *anchors):
            try:
                image = _bundle_file(
                    bundle,
                    row.get("image_relpath"),
                    label=f"rollout difficulty image for {_sample_id(row)}",
                )
            except (FileNotFoundError, ValueError):
                unverified_group_images.append(str(row.get("image_relpath")))
                continue
            if image not in verified_bundle_images:
                unverified_group_images.append(str(row.get("image_relpath")))
        if unverified_group_images:
            raise ValueError(
                "hard/anchor images are not verified by manifest/unique_images.jsonl; "
                f"first={unverified_group_images[:10]}"
            )

    selected_rollout_rows = dict(difficulty)
    selected_rollout_rows.update(
        {sample_id: eligible_bundle_groups[sample_id] for sample_id in replay_ids}
    )
    pool_group_ids = {
        "hard": hard_ids,
        "matched_anchor": anchor_ids,
        "global_replay": replay_ids,
    }
    status.set_detail(
        f"planning training views: hard={len(hard_ids)} anchor={len(anchor_ids)} "
        f"global_replay={len(replay_ids)} groups"
    )
    selected_crop_identity = {
        pool: [
            {
                "sample_id": sid,
                "task": selected_rollout_rows[sid]["task"],
                "crop_ids": [
                    str(row["crop_id"])
                    for row in crop_rows_by_sample.get(sid, ())
                ],
            }
            for sid in sorted(sample_ids)
        ]
        for pool, sample_ids in pool_group_ids.items()
    }
    asset_namespace = (
        "training_crops-" + _json_digest(selected_crop_identity)[:16]
    )
    crop_assets: list[dict[str, Any]] = []
    pool_records: dict[str, list[dict[str, Any]]] = {
        "hard": [],
        "matched_anchor": [],
        "global_replay": [],
    }
    for pool, sample_ids in pool_group_ids.items():
        for sid in _progress_items(
            progress, f"plan_{pool}_views", sorted(sample_ids), unit="groups",
            detail=lambda sid: str(sid),
        ):
            rollout_row = selected_rollout_rows[sid]
            canonical = _canonical_selected_supervision(
                sid, records_by_sample[sid], rollout_row
            )
            if _task(rollout_row["task"]) == "content_missing":
                pool_records[pool].append(
                    _global_view_supervision(
                        canonical, retention=pool == "global_replay"
                    )
                )
                continue
            crop_records, assets = _selected_crop_supervision(
                sid,
                canonical,
                rollout_row,
                crop_rows_by_sample,
                asset_namespace=asset_namespace,
            )
            if not crop_records:
                raise AssertionError(
                    f"selected region group emitted no base-tile records: {sid}"
                )
            pool_records[pool].extend(crop_records)
            crop_assets.extend(assets)
    if any(not pool_records[pool] for pool in POOLS):
        empty = [pool for pool in POOLS if not pool_records[pool]]
        raise ValueError(f"curriculum pool is empty: {empty}")
    emitted_id_sets = {
        pool: {_sample_id(row) for row in pool_records[pool]} for pool in POOLS
    }
    expected_id_sets = pool_group_ids
    if emitted_id_sets != expected_id_sets:
        raise AssertionError(
            f"curriculum emitted group partition differs: {emitted_id_sets}"
        )
    if set().union(*emitted_id_sets.values()) != set(eligible_bundle_groups):
        raise AssertionError(
            "curriculum pools do not exactly partition eligible bundle groups"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    reuse = None
    if getattr(args, "reuse_crops_from", None) is not None:
        reuse = _load_crop_reuse_inventory(
            args.reuse_crops_from, output_dir, bundle_state, crop_assets, progress,
        )
    asset_inventory = _materialize_crop_assets(
        output_dir, asset_namespace, crop_assets, progress, reuse=reuse,
    )
    print(
        f"[CROP ASSETS] total={len(asset_inventory)} "
        f"reused={len(asset_inventory) if reuse else 0} "
        f"generated={0 if reuse else len(asset_inventory)} "
        f"mode={'verified_hardlink_all' if reuse else 'materialize'}",
        file=os.sys.stderr, flush=True,
    )
    status.set_detail("binding generated crop assets to training annotations")
    assets_by_relative = {
        str(row["relative_path"]): row for row in asset_inventory
    }
    for pool in _progress_items(progress, "bind_crop_assets", POOLS, unit="pools"):
        for record in pool_records[pool]:
            relative = record.pop("_ui5_crop_asset_relpath", None)
            if relative is None:
                if _task(record.get("_ui5_task") or record.get("task")) != (
                    "content_missing"
                ) or record.get("_ui5_record_kind") not in {
                    "global_view",
                    "full_image",
                }:
                    raise AssertionError(
                        f"non-crop selected record is not content_missing: {record}"
                    )
                continue
            asset = assets_by_relative.get(str(relative))
            if asset is None:
                raise AssertionError(f"crop record has no published asset: {relative}")
            path = (output_dir / str(relative)).resolve(strict=True)
            record["image"] = str(path)
            record["_ui5_crop_asset_sha256"] = asset["sha256"]
            record["_ui5_crop_asset_bytes"] = asset["bytes"]

    crop_assets_path = output_dir / "crop_assets.jsonl"
    _atomic_jsonl(crop_assets_path, asset_inventory)
    annotation_paths: dict[str, Path] = {}
    for pool in POOLS:
        annotation = output_dir / f"{pool}.jsonl"
        tagged = []
        for raw in pool_records[pool]:
            row = dict(raw)
            row["_ui5_curriculum_pool"] = pool
            tagged.append(row)
        _atomic_jsonl(annotation, tagged)
        annotation_paths[pool] = annotation

    _atomic_jsonl(output_dir / "hard_groups.jsonl", hard)
    _atomic_jsonl(output_dir / "matched_anchor_groups.jsonl", anchors)

    recipe: dict[str, Any] = {}
    for pool in _progress_items(progress, "publish_pool_annotations", POOLS, unit="pools"):
        crop_recipe = any(
            row.get("_ui5_record_kind") == "crop" for row in pool_records[pool]
        )
        if not crop_recipe:
            raise ValueError(f"{pool} contains no region crop supervision")
        recipe[f"ui5_curriculum_{pool}"] = {
            "annotation": [annotation_paths[pool].name],
            "root": "",
            "repeat_time": 1.0,
            "sampling_weight": 1.0,
            "data_augment": False,
            "paths_relative_to_meta": True,
            "ui5_crop_recipe": crop_recipe,
            "ui5_retention_recipe": pool == "global_replay",
            "ui_sampling_mode": "fixed_ratio",
            "curriculum_pool": pool,
        }
    recipe_path = output_dir / "ui5_crop_rollout4_curriculum.json"
    status.set_detail("publishing recipe and full curriculum manifest")
    _atomic_json(recipe_path, recipe)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "seed": int(args.seed),
        "hard_definition": (
            "exact formal_crop_hard membership bound to frozen summary.json and "
            "its authoritative complete8 SHA-256, followed by full technical/source "
            "eligibility validation"
        ),
        "anchor_definition": (
            "distinct formally eligible crop_complete4 4/4 group matched "
            "one-to-one by task and polarity, minimizing total GT-count delta then "
            "base-crop-count delta with a seeded stable tie-break"
        ),
        "selected_supervision_policy": (
            "all region groups emit every immutable GT-free base tile with verified "
            "local labels; content_missing emits one global/full-image view; source "
            "subsets are first replaced by authoritative union GT"
        ),
        "global_replay_definition": (
            "every structurally eligible original rollout-bundle group not selected "
            "as hard/anchor; region tasks retain every immutable GT-free base crop "
            "and content_missing retains one union-GT full-image view"
        ),
        "training_view_policy": {
            "hard": "all_gt_free_detector_scan_base_tiles",
            "matched_anchor": "all_gt_free_detector_scan_base_tiles",
            "content_missing": "full_image_global_view",
            "global_replay": (
                "all_gt_free_detector_scan_base_tiles_except_content_missing_full_image"
            ),
            "tile_selection_uses_gt": False,
            "partial_gt_allowed": False,
        },
        "expected_hard_groups": len(formal_hard_ids),
        "configured_expected_hard_groups": (
            int(configured_expected_hard)
            if configured_expected_hard is not None
            else None
        ),
        "hard_groups": len(hard),
        "matched_anchor_groups": len(anchors),
        "base_sample_groups": len(records_by_sample),
        "base_training_records": len(records),
        "formal_eligibility": {
            "authoritative_groups": len(difficulty_rows),
            "source_eligible_groups": sum(
                row.get("grpo_source_eligible") is True for row in difficulty_rows
            ),
            "structural_anomaly_groups": sum(
                any(row.get(key) is True for key in FORMAL_ANOMALY_FIELDS)
                for row in difficulty_rows
            ),
            "fully_eligible_rollout_groups": len(difficulty),
        },
        "bundle_group_selection": {
            "all_source_groups": len(bundle_groups),
            "eligible_source_groups": len(eligible_bundle_groups),
            "ineligible_source_groups": len(bundle_groups) - len(eligible_bundle_groups),
            "hard_selected_groups": len(hard_ids),
            "anchor_selected_groups": len(anchor_ids),
            "global_replay_selected_groups": len(replay_ids),
            "all_source_strata": bundle_all_strata,
            "eligible_source_strata": bundle_eligible_strata,
            "selected_pool_strata": {
                "hard": hard_strata,
                "matched_anchor": anchor_strata,
                "global_replay": replay_strata,
            },
            "global_replay_selected_strata": replay_strata,
            "selection_policy": (
                "retain every eligible bundle group; exclude hard/anchor from replay; "
                "no task/polarity resampling in recipe selection"
            ),
        },
        "pools": {pool: _pool_counts(pool_records[pool]) for pool in POOLS},
        "crop_assets": asset_inventory,
        "inputs": {
            "base_recipe": str(base_recipe) if base_recipe is not None else None,
            "base_recipe_sha256": (
                _sha256_file(base_recipe) if base_recipe is not None else None
            ),
            "rollout_difficulty": str(difficulty_path),
            "rollout_difficulty_sha256": _sha256_file(difficulty_path),
            "rollout_difficulty_authoritative": difficulty_state,
            "frozen_selection_summary": frozen_summary_state,
            "rollout_bundle": bundle_state,
        },
        "outputs": {
            "recipe": str(recipe_path),
            "hard_groups": str(output_dir / "hard_groups.jsonl"),
            "matched_anchor_groups": str(output_dir / "matched_anchor_groups.jsonl"),
            "crop_assets_manifest": str(crop_assets_path),
            "crop_asset_namespace": asset_namespace,
        },
    }
    if reuse:
        summary["crop_asset_reuse"] = reuse["audit"]
    summary["identity_digest"] = _json_digest(summary)
    _atomic_json(output_dir / "curriculum_manifest.json", summary)

    # Read every published artifact before declaring success.  This catches a
    # truncated network-volume write before the two-GPU job starts.
    published_counts = {
        pool: len(_read_jsonl(annotation_paths[pool]))
        for pool in _progress_items(progress, "verify_pool_annotations", POOLS, unit="pools")
    }
    if published_counts != {
        pool: len(pool_records[pool]) for pool in POOLS
    }:
        raise RuntimeError(f"published curriculum count mismatch: {published_counts}")
    durable_paths = [
        recipe_path,
        output_dir / "hard.jsonl",
        output_dir / "matched_anchor.jsonl",
        output_dir / "global_replay.jsonl",
        output_dir / "hard_groups.jsonl",
        output_dir / "matched_anchor_groups.jsonl",
        crop_assets_path,
        *(output_dir / row["relative_path"] for row in asset_inventory),
    ]
    success_files = {
        path.relative_to(output_dir).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in _progress_items(
            progress, "verify_final_artifacts", durable_paths, unit="files",
            detail=lambda path: str(path.relative_to(output_dir)),
        )
    }
    if len(success_files) != len(durable_paths):
        raise AssertionError("curriculum durable artifact paths are not unique")
    status.set_detail("all artifacts verified; publishing _SUCCESS.json")
    _atomic_json(
        output_dir / "_SUCCESS.json",
        {
            "schema_version": SCHEMA_VERSION,
            "identity_digest": summary["identity_digest"],
            "recipe_sha256": _sha256_file(recipe_path),
            "files": success_files,
            "complete": True,
        },
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        summary = build(args)
        if args.print_full_summary:
            print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        else:
            print(json.dumps({
                "complete": True,
                "curriculum_manifest": str(args.output_dir.resolve() / "curriculum_manifest.json"),
                "progress_path": str(args.output_dir.resolve() / "progress" / "build_progress.json"),
                "identity_digest": summary["identity_digest"],
                "hard_groups": summary["hard_groups"],
                "matched_anchor_groups": summary["matched_anchor_groups"],
                "base_sample_groups": summary["base_sample_groups"],
                "crop_assets": len(summary["crop_assets"]),
                "pools": summary["pools"],
            }, ensure_ascii=False, indent=2), flush=True)
    except Exception as exc:
        print(f"[curriculum-recipe:error] {type(exc).__name__}: {exc}", file=os.sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
