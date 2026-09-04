#!/usr/bin/env python3
"""Build the three-pool UI5 continuation curriculum from frozen rollout results.

The hard unit is an image/task group whose crop checkpoint completed all four
rollouts and got 0/4 exactly correct.  A deterministic 4/4 anchor is matched by
task and polarity for every hard group.  Training records belonging to neither
set form the global replay pool only when their snapshot group is likewise
source-eligible, anomaly-free, and technically complete across all eight routes.

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
import random
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image


SCHEMA_VERSION = 3
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
            "Optional audited training recipe. When omitted, use every portable "
            "original training record in the rollout bundle."
        ),
    )
    parser.add_argument("--rollout-difficulty", type=Path, required=True)
    parser.add_argument("--rollout-bundle-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-hard-groups", type=int, default=72)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing complete curriculum instead of verifying/reusing it.",
    )
    return parser.parse_args(argv)


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
        "manifest/unique_images.jsonl",
    }
    missing_declarations = sorted(required_files - set(map(str, declared_files)))
    if missing_declarations:
        raise ValueError(
            "rollout bundle manifest does not declare required files: "
            f"{missing_declarations}"
        )

    verified_declared: dict[str, str] = {}
    for relative, raw_metadata in declared_files.items():
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
    for row_number, row in enumerate(unique_rows, 1):
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
    """Return exactly one non-contradictory record for a hard/anchor group.

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
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    """Load only immutable, GT-free detector-scan geometry from the bundle."""

    unique_rows = _read_jsonl(bundle / "manifest" / "unique_images.jsonl")
    images_by_id: dict[str, dict[str, Any]] = {}
    for row_number, raw in enumerate(unique_rows, 1):
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
    for raw_image_id, raw_plan in plans_payload.items():
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
    for row_number, raw in enumerate(
        _read_jsonl(bundle / "manifest" / "crop_samples.jsonl"), 1
    ):
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


def _materialize_crop_assets(
    output_dir: Path, asset_namespace: str, assets: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Publish every selected crop as one atomically-renamed image tree."""

    if not assets:
        raise ValueError("hard/anchor selection produced zero region crop assets")
    target_dir = output_dir / asset_namespace
    staging = Path(
        tempfile.mkdtemp(prefix=f".{asset_namespace}.staging-", dir=output_dir)
    )
    inventory: list[dict[str, Any]] = []
    try:
        seen_relative: set[str] = set()
        for raw in sorted(assets, key=lambda row: str(row["relative_path"])):
            relative = Path(str(raw["relative_path"]))
            if relative.parts[0] != asset_namespace or len(relative.parts) != 2:
                raise ValueError(f"unsafe crop asset path: {relative}")
            if relative.as_posix() in seen_relative:
                raise ValueError(f"duplicate crop asset path: {relative}")
            seen_relative.add(relative.as_posix())
            source = Path(str(raw["source_image"])).resolve(strict=True)
            if _sha256_file(source) != str(raw["source_image_sha256"]):
                raise RuntimeError(f"verified source image changed: {source}")
            crop = [int(value) for value in raw["crop_xyxy"]]
            destination = staging / relative.name
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
                for row in inventory
            ):
                raise RuntimeError(
                    f"existing crop asset directory differs: {target_dir}"
                )
            shutil.rmtree(staging)
        else:
            os.replace(staging, target_dir)
        for row in inventory:
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


def _match_anchors(
    hard: Sequence[Mapping[str, Any]],
    eligible: Mapping[str, Mapping[str, Any]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in eligible.values():
        if int(row["crop_correct_count"]) != 4:
            continue
        buckets[(str(row["task"]), str(row["polarity"]))].append(dict(row))
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for hard_row in sorted(hard, key=lambda row: str(row["sample_id"])):
        key = (str(hard_row["task"]), str(hard_row["polarity"]))
        candidates = sorted(buckets.get(key, []), key=lambda row: str(row["sample_id"]))
        random.Random(_json_digest([seed, *key, hard_row["sample_id"]])).shuffle(candidates)
        anchor = next(
            (row for row in candidates if str(row["sample_id"]) not in used), None
        )
        if anchor is None:
            raise ValueError(
                "not enough distinct 4/4 anchors for stratum "
                f"task={key[0]} polarity={key[1]}"
            )
        anchor_id = str(anchor["sample_id"])
        used.add(anchor_id)
        anchor["matched_hard_sample_id"] = str(hard_row["sample_id"])
        selected.append(anchor)
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


def build(args: argparse.Namespace) -> dict[str, Any]:
    base_recipe = (
        args.base_recipe.expanduser().resolve(strict=True)
        if args.base_recipe is not None
        else None
    )
    difficulty_path = args.rollout_difficulty.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    if int(args.expected_hard_groups) <= 0:
        raise ValueError("--expected-hard-groups must be positive")

    bundle = (
        args.rollout_bundle_root.expanduser().resolve(strict=True)
        if args.rollout_bundle_root is not None
        else None
    )
    if bundle is None:
        raise ValueError(
            "--rollout-bundle-root is required for GT-free crop-aligned curriculum"
        )

    bundle_state: dict[str, Any] | None = None
    verified_bundle_images: dict[Path, str] = {}
    difficulty_rows, difficulty_state = _load_difficulty_rows(difficulty_path)
    bundle_state, verified_bundle_images = _verify_rollout_bundle(bundle)
    crop_rows_by_sample, _, _ = _bundle_crop_geometry(
        bundle, verified_bundle_images
    )

    existing_manifest_path = output_dir / "curriculum_manifest.json"
    existing_success_path = output_dir / "_SUCCESS.json"
    if existing_manifest_path.is_file() and existing_success_path.is_file() and not bool(
        getattr(args, "force", False)
    ):
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
            "base_recipe_sha256": (
                _sha256_file(base_recipe) if base_recipe is not None else None
            ),
            "rollout_difficulty_sha256": _sha256_file(difficulty_path),
            "rollout_difficulty_authoritative": difficulty_state,
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
        ) != int(args.expected_hard_groups):
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
        for relative, metadata in success_files.items():
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
        return existing

    records = _base_records(base_recipe) if base_recipe is not None else _bundle_records(bundle)
    if base_recipe is None:
        _verify_bundle_record_images(records, verified_bundle_images)
    record_truth = _record_group_truth(records)
    difficulty = _eligible_difficulty(difficulty_rows, record_truth)
    hard = sorted(
        (dict(row) for row in difficulty.values() if row["crop_correct_count"] == 0),
        key=lambda row: (str(row["task"]), str(row["sample_id"])),
    )
    if len(hard) != int(args.expected_hard_groups):
        raise ValueError(
            "0/4 hard group count mismatch: "
            f"expected={args.expected_hard_groups}, observed={len(hard)}"
        )
    anchors = _match_anchors(hard, difficulty, seed=int(args.seed))
    hard_ids = {str(row["sample_id"]) for row in hard}
    anchor_ids = {str(row["sample_id"]) for row in anchors}
    replay_ids = set(difficulty) - hard_ids - anchor_ids
    if hard_ids & anchor_ids:
        raise AssertionError("hard and matched-anchor groups overlap")
    if hard_ids & replay_ids or anchor_ids & replay_ids:
        raise AssertionError("curriculum sample groups are not disjoint")

    records_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        records_by_sample[_sample_id(record)].append(record)
    missing_hard = sorted(hard_ids - set(records_by_sample))
    missing_anchors = sorted(anchor_ids - set(records_by_sample))
    if missing_hard or missing_anchors:
        raise ValueError(
            "rollout groups do not map to base training records: "
            f"missing_hard={missing_hard[:10]}, missing_anchor={missing_anchors[:10]}"
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
        for pool, sample_ids in (("hard", hard_ids), ("matched_anchor", anchor_ids))
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
    for pool, sample_ids in (("hard", hard_ids), ("matched_anchor", anchor_ids)):
        for sid in sorted(sample_ids):
            rollout_row = selected_rollout_rows[sid]
            canonical = _canonical_selected_supervision(
                sid, records_by_sample[sid], rollout_row
            )
            if _task(rollout_row["task"]) == "content_missing":
                pool_records[pool].append(
                    _global_view_supervision(canonical, retention=False)
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
    pool_records["global_replay"] = [
        _global_view_supervision(
            _canonical_selected_supervision(
                sid, records_by_sample[sid], selected_rollout_rows[sid]
            ),
            retention=True,
        )
        for sid in sorted(replay_ids)
    ]
    if any(not pool_records[pool] for pool in POOLS):
        empty = [pool for pool in POOLS if not pool_records[pool]]
        raise ValueError(f"curriculum pool is empty: {empty}")
    emitted_id_sets = {
        pool: {_sample_id(row) for row in pool_records[pool]} for pool in POOLS
    }
    expected_id_sets = {
        "hard": hard_ids,
        "matched_anchor": anchor_ids,
        "global_replay": replay_ids,
    }
    if emitted_id_sets != expected_id_sets:
        raise AssertionError(
            f"curriculum emitted group partition differs: {emitted_id_sets}"
        )
    if set().union(*emitted_id_sets.values()) != set(difficulty):
        raise AssertionError(
            "curriculum pools do not exactly partition formally eligible groups"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    asset_inventory = _materialize_crop_assets(
        output_dir, asset_namespace, crop_assets
    )
    assets_by_relative = {
        str(row["relative_path"]): row for row in asset_inventory
    }
    for pool in ("hard", "matched_anchor"):
        for record in pool_records[pool]:
            relative = record.pop("_ui5_crop_asset_relpath", None)
            if relative is None:
                if record.get("_ui5_record_kind") != "global_view" or _task(
                    record.get("_ui5_task") or record.get("task")
                ) != "content_missing":
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
    for pool in POOLS:
        crop_recipe = any(
            row.get("_ui5_record_kind") == "crop" for row in pool_records[pool]
        )
        if pool in {"hard", "matched_anchor"} and not crop_recipe:
            raise ValueError(f"{pool} contains no region crop supervision")
        if pool == "global_replay" and crop_recipe:
            raise AssertionError("global replay must remain a full-image retention pool")
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
    _atomic_json(recipe_path, recipe)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "seed": int(args.seed),
        "hard_definition": (
            "grpo_source_eligible == true, all structural anomaly flags false, "
            "cross_model_complete8 == true, crop_correct_count == 0, and no "
            "runtime/parse error"
        ),
        "anchor_definition": (
            "distinct formally eligible crop_complete4 4/4 group matched "
            "one-to-one by task and polarity"
        ),
        "selected_supervision_policy": (
            "hard/matched-anchor region groups emit every immutable GT-free base "
            "tile with verified local labels; content_missing emits one global view; "
            "source subsets are first replaced by snapshot-verified union GT"
        ),
        "global_replay_definition": (
            "all remaining source-eligible, anomaly-free, technically clean "
            "complete8 groups; one snapshot-verified full-image union-GT retention "
            "record per sample"
        ),
        "training_view_policy": {
            "hard": "all_gt_free_detector_scan_base_tiles",
            "matched_anchor": "all_gt_free_detector_scan_base_tiles",
            "content_missing": "full_image_global_view",
            "global_replay": "full_image_retention",
            "tile_selection_uses_gt": False,
            "partial_gt_allowed": False,
        },
        "expected_hard_groups": int(args.expected_hard_groups),
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
    summary["identity_digest"] = _json_digest(summary)
    _atomic_json(output_dir / "curriculum_manifest.json", summary)

    # Read every published artifact before declaring success.  This catches a
    # truncated network-volume write before the two-GPU job starts.
    published_counts = {
        pool: len(_read_jsonl(annotation_paths[pool])) for pool in POOLS
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
        for path in durable_paths
    }
    if len(success_files) != len(durable_paths):
        raise AssertionError("curriculum durable artifact paths are not unique")
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
        print(json.dumps(build(parse_args(argv)), ensure_ascii=False, indent=2))
    except Exception as exc:
        print(f"[curriculum-recipe:error] {type(exc).__name__}: {exc}", file=os.sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
