#!/usr/bin/env python3
"""Build the three-pool UI5 continuation curriculum from frozen rollout results.

The hard unit is an image/task group whose crop checkpoint completed all four
rollouts and got 0/4 exactly correct.  A deterministic 4/4 anchor is matched by
task and polarity for every hard group.  Training records belonging to neither
set form the global replay pool.

The output recipe uses absolute media paths and recipe-relative annotations so
it can be consumed unchanged after the training process is restarted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
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
    "crop_correct_count",
    "crop_complete4",
    "technical_error_free",
    "runtime_error_count",
    "parse_error_count",
    "gt_global",
)
PROJECTION_MATCH_FIELDS = ("task", "crop_correct_count", "crop_complete4")


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
        if not isinstance(row.get("crop_complete4"), bool):
            raise ValueError(
                f"authoritative rollout row {sample_id} has invalid crop_complete4"
            )
        if not isinstance(row.get("technical_error_free"), bool):
            raise ValueError(
                f"authoritative rollout row {sample_id} has invalid "
                "technical_error_free"
            )
        for key in ("runtime_error_count", "parse_error_count"):
            count = _strict_int_field(
                row, key, label=f"authoritative rollout row {sample_id}"
            )
            if count < 0:
                raise ValueError(
                    f"authoritative rollout row {sample_id} has negative {key}"
                )
        if not isinstance(row.get("gt_global"), list):
            raise ValueError(
                f"authoritative rollout row {sample_id} has invalid gt_global"
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
    for source in _read_jsonl(source_path):
        raw = source.get("portable_training_record") or source.get(
            "original_training_record"
        )
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"bundle source record lacks portable training data: {source.get('source_record_id')}"
            )
        record = _absolutize_media(dict(raw), bundle)
        record.update(
            {
                "_ui5_sample_id": str(source.get("sample_id") or ""),
                "_ui5_image_id": str(source.get("image_id") or ""),
                "_ui5_task": _task(source.get("task")),
                "_ui5_positive": bool(source.get("gt_boxes_global_xyxy")),
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
    truth: dict[str, dict[str, str]] = {}
    for record in records:
        sample_id = _sample_id(record)
        raw_task = record.get("_ui5_task") or record.get("task")
        task = _task(raw_task)
        polarity = _polarity(record)
        observed = {"task": task, "polarity": polarity}
        previous = truth.get(sample_id)
        if previous is not None and previous != observed:
            raise ValueError(
                f"conflicting training-record truth for sample {sample_id}: "
                f"first={previous}, observed={observed}"
            )
        truth[sample_id] = observed
    return truth


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
        if row.get("crop_complete4") is not True:
            continue
        if row.get("technical_error_free") is not True:
            continue
        if _strict_int_field(
            row, "runtime_error_count", label=f"rollout row {sample_id}"
        ) != 0:
            continue
        if _strict_int_field(
            row, "parse_error_count", label=f"rollout row {sample_id}"
        ) != 0:
            continue
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
    return {
        "records": len(records),
        "sample_groups": len(samples),
        "records_by_task": dict(sorted(by_task.items())),
        "records_by_polarity": dict(sorted(by_polarity.items())),
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
    if base_recipe is None and bundle is None:
        raise ValueError("provide --base-recipe or --rollout-bundle-root")

    bundle_state: dict[str, Any] | None = None
    verified_bundle_images: dict[Path, str] = {}
    difficulty_rows, difficulty_state = _load_difficulty_rows(difficulty_path)
    if bundle is not None:
        bundle_state, verified_bundle_images = _verify_rollout_bundle(bundle)

    existing_manifest_path = output_dir / "curriculum_manifest.json"
    existing_success_path = output_dir / "_SUCCESS.json"
    if existing_manifest_path.is_file() and existing_success_path.is_file() and not bool(
        getattr(args, "force", False)
    ):
        existing = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
        success = json.loads(existing_success_path.read_text(encoding="utf-8"))
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
        }
        if not isinstance(success_files, Mapping) or set(
            map(str, success_files)
        ) != required_success_files:
            raise RuntimeError(
                "existing curriculum success marker artifact hashes do not match"
            )
        for relative, expected_hash in success_files.items():
            path = output_dir / str(relative)
            if not path.is_file() or _sha256_file(path) != expected_hash:
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
    if hard_ids & anchor_ids:
        raise AssertionError("hard and matched-anchor groups overlap")

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

    pool_records = {
        "hard": [row for sid in sorted(hard_ids) for row in records_by_sample[sid]],
        "matched_anchor": [
            row for sid in sorted(anchor_ids) for row in records_by_sample[sid]
        ],
        "global_replay": [
            row
            for sid in sorted(records_by_sample)
            if sid not in hard_ids and sid not in anchor_ids
            for row in records_by_sample[sid]
        ],
    }
    if any(not pool_records[pool] for pool in POOLS):
        empty = [pool for pool in POOLS if not pool_records[pool]]
        raise ValueError(f"curriculum pool is empty: {empty}")

    output_dir.mkdir(parents=True, exist_ok=True)
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
        recipe[f"ui5_curriculum_{pool}"] = {
            "annotation": [annotation_paths[pool].name],
            "root": "",
            "repeat_time": 1.0,
            "sampling_weight": 1.0,
            "data_augment": False,
            "paths_relative_to_meta": True,
            "ui5_crop_recipe": crop_recipe,
            "ui_sampling_mode": "fixed_ratio",
            "curriculum_pool": pool,
        }
    recipe_path = output_dir / "ui5_crop_rollout4_curriculum.json"
    _atomic_json(recipe_path, recipe)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "seed": int(args.seed),
        "hard_definition": (
            "crop_complete4 == true and crop_correct_count == 0 with no runtime/parse error"
        ),
        "anchor_definition": (
            "distinct crop_complete4 4/4 group matched one-to-one by task and polarity"
        ),
        "global_replay_definition": "all remaining base-recipe sample groups",
        "expected_hard_groups": int(args.expected_hard_groups),
        "hard_groups": len(hard),
        "matched_anchor_groups": len(anchors),
        "base_sample_groups": len(records_by_sample),
        "base_training_records": len(records),
        "pools": {pool: _pool_counts(pool_records[pool]) for pool in POOLS},
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
    _atomic_json(
        output_dir / "_SUCCESS.json",
        {
            "schema_version": SCHEMA_VERSION,
            "identity_digest": summary["identity_digest"],
            "recipe_sha256": _sha256_file(recipe_path),
            "files": {
                path.name: _sha256_file(path)
                for path in (
                    recipe_path,
                    output_dir / "hard.jsonl",
                    output_dir / "matched_anchor.jsonl",
                    output_dir / "global_replay.jsonl",
                    output_dir / "hard_groups.jsonl",
                    output_dir / "matched_anchor_groups.jsonl",
                )
            },
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
