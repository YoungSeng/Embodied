#!/usr/bin/env python3
"""Validate or relocate a standalone UI5 scan manifest using image bytes.

CPU-only; requires Pillow but never imports a model runtime.  Relocation reads
the five destination test JSONLs, matches the source BLAKE2b-20 content IDs,
and changes only image_path/image_paths.  The source is never overwritten.
Without --output-manifest this performs read-only runtime coverage validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image

from locany_ui5_common import DEFAULT_UI5_FULL_TEST_UNIQUE_IMAGES, TASK_JSONL
from ui5_lossless_tiling import (
    assert_lossless_coverage,
    build_raw_detector_edge_geometry,
    detector_boundary_cut_count,
    strict_vertical_partition_metrics,
)


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def _read_rows(path: Path) -> tuple[list[dict[str, Any]], str]:
    payload = path.read_bytes()
    rows = []
    for number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line, object_pairs_hook=_object, parse_constant=_constant)
            if not isinstance(row, dict):
                raise ValueError("expected a JSON object")
        except ValueError as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc
        rows.append(row)
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows, hashlib.sha256(payload).hexdigest()


def _aliases(row: dict[str, Any]) -> list[str]:
    values = row.get("image_paths", [])
    if not isinstance(values, list):
        raise ValueError("image_paths must be a list")
    values = [row.get("image_path"), *values]
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError("manifest image_path/image_paths must be nonempty strings")
    return list(dict.fromkeys(values))


def _key(value: str | Path) -> str:
    return str(Path(value).expanduser().resolve(strict=False))


def _validate_geometry(row: dict[str, Any]) -> None:
    if row.get("mode") != "detector_scan" or row.get("gt_used") is not False:
        raise ValueError("manifest must be GT-free detector_scan")
    width, height = row.get("width"), row.get("height")
    if any(type(value) is not int or value <= 0 for value in (width, height)):
        raise ValueError("invalid manifest dimensions")
    tiles = row.get("tiles")
    if not isinstance(tiles, list) or not tiles or any(
        not isinstance(tile, list) or len(tile) != 4
        or any(type(value) is not int for value in tile) for tile in tiles
    ):
        raise ValueError("invalid integer tile geometry")
    try:
        assert_lossless_coverage(width, height, tiles)
    except (AssertionError, ValueError) as exc:
        raise ValueError(f"invalid lossless tile geometry: {exc}") from exc
    if not strict_vertical_partition_metrics(width, height, tiles)["strict_vertical_partition"]:
        raise ValueError("manifest is not a strict nonoverlapping horizontal partition")
    if row.get("detector_boundary_cut_count") != 0:
        raise ValueError("manifest cuts detector boxes")
    if row.get("every_seam_is_raw_detector_edge") is not True:
        raise ValueError("manifest does not declare raw detector edge alignment")
    detector_items = []
    for source, field in (("text", "text_detections"), ("icon", "icon_detections")):
        items = row.get(field)
        if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
            raise ValueError(f"manifest must embed {field} for geometry validation")
        detector_items.extend({**item, "source": source} for item in items)
    geometry = build_raw_detector_edge_geometry(width, height, detector_items)
    if detector_boundary_cut_count(tiles, geometry["raw_boxes"]):
        raise ValueError("tile geometry cuts embedded detector boxes")
    if any(tile[3] not in geometry["safe_raw_edge_candidates"] for tile in tiles[:-1]):
        raise ValueError("tile geometry has a non-safe raw detector edge seam")


def _image_info(path: Path) -> tuple[str, tuple[int, int]]:
    before = path.stat()
    digest = hashlib.blake2b(digest_size=20)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    with Image.open(path) as opened:
        # Match inference's Image.open(...).convert('RGB'), without EXIF rotation.
        size = opened.size
        opened.verify()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"image changed during validation: {path}")
    return digest.hexdigest(), size


def prepare_manifest(
    manifest: Path,
    input_dir: Path,
    *,
    output_manifest: Path | None = None,
    expected_unique_images: int = DEFAULT_UI5_FULL_TEST_UNIQUE_IMAGES,
) -> dict[str, Any]:
    manifest = manifest.expanduser().resolve(strict=True)
    input_dir = input_dir.expanduser().resolve(strict=True)
    if expected_unique_images <= 0:
        raise ValueError("expected_unique_images must be positive")
    output = output_manifest.expanduser().resolve(strict=False) if output_manifest else None
    if output is not None and (
        output == manifest or (output.exists() and output.samefile(manifest))
    ):
        raise ValueError("output must be a new manifest, never the source")
    rows, source_sha = _read_rows(manifest)
    by_content: dict[str, dict[str, Any]] = {}
    source_aliases: dict[str, str] = {}
    image_ids: set[str] = set()
    for row in rows:
        _validate_geometry(row)
        content_id, image_id = row.get("content_id"), row.get("image_id")
        if not isinstance(content_id, str) or len(content_id) != 40 or any(
            char not in "0123456789abcdef" for char in content_id
        ):
            raise ValueError("manifest content_id must be BLAKE2b-20 of image bytes")
        if not isinstance(image_id, str) or not image_id:
            raise ValueError("manifest image_id must be nonempty")
        if content_id in by_content or image_id in image_ids:
            raise ValueError(f"duplicate manifest content_id or image_id: {image_id}")
        by_content[content_id] = row
        image_ids.add(image_id)
        for alias in _aliases(row):
            key = _key(alias)
            if key in source_aliases and source_aliases[key] != content_id:
                raise ValueError(f"manifest path alias collision: {alias}")
            source_aliases[key] = content_id

    local_aliases: dict[str, set[str]] = {}
    image_info: dict[str, tuple[str, tuple[int, int]]] = {}
    task_digests: dict[str, str] = {}
    by_task: dict[str, int] = {}
    missing_aliases: list[str] = []
    print(f"[EVAL MANIFEST CHECK] source_rows={len(rows)} input_dir={input_dir}", flush=True)
    for task, filename in TASK_JSONL.items():
        task_rows, task_sha = _read_rows(input_dir / filename)
        task_digests[filename] = task_sha
        task_paths: set[str] = set()
        for number, row in enumerate(task_rows, 1):
            images = row.get("images", row.get("image"))
            if isinstance(images, (str, dict)):
                images = [images]
            if not isinstance(images, list) or not images:
                raise ValueError(f"{filename}:{number}: missing images/image")
            for image in images:
                value = image.get("path") if isinstance(image, dict) else image
                if not isinstance(value, str) or not value:
                    raise ValueError(f"{filename}:{number}: invalid image reference")
                path = Path(value).expanduser()
                if ":" in path.name:  # Same Figma exclusion as formal inference.
                    continue
                if not path.is_absolute():
                    path = input_dir / path
                path = path.resolve(strict=True)
                key = str(path)
                task_paths.add(key)
                if key not in image_info:
                    image_info[key] = _image_info(path)
                    if len(image_info) % 100 == 0:
                        print(f"[EVAL MANIFEST CHECK] hashed_images={len(image_info)}", flush=True)
                content_id, size = image_info[key]
                scan = by_content.get(content_id)
                if scan is None:
                    raise ValueError(f"image content is absent from source manifest: {path}")
                if size != (scan["width"], scan["height"]):
                    raise ValueError(f"image dimensions differ from source manifest: {path}")
                local_aliases.setdefault(content_id, set()).add(key)
                if key in source_aliases and source_aliases[key] != content_id:
                    raise ValueError(f"destination alias collision with different image content: {key}")
                if task != "content_missing" and source_aliases.get(key) != content_id:
                    missing_aliases.append(key)
        if not task_paths:
            raise ValueError(f"no evaluation images for task {task}")
        by_task[task] = len(task_paths)
    if set(local_aliases) != set(by_content):
        raise ValueError("source manifest and destination test image content sets differ")
    if len(local_aliases) != expected_unique_images:
        raise ValueError(
            f"full-test unique image count mismatch: {len(local_aliases)} != {expected_unique_images}"
        )
    if output is None and missing_aliases:
        raise ValueError(
            "destination image path aliases are missing from the scan manifest; "
            "run this script with --output-manifest to relocate by content. Examples: "
            + ", ".join(sorted(set(missing_aliases))[:3])
        )

    for path, expected_sha in [(manifest, source_sha), *(
        (input_dir / filename, sha) for filename, sha in task_digests.items()
    )]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
            raise ValueError(f"input changed during validation: {path}")
    changed = 0
    published_sha = source_sha
    if output is not None:
        relocated = []
        for row in rows:
            aliases = sorted(set(_aliases(row)) | local_aliases[row["content_id"]])
            updated = dict(row)
            updated["image_paths"] = aliases
            updated["image_path"] = sorted(local_aliases[row["content_id"]])[0]
            changed += updated != row
            relocated.append(updated)
        payload = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in relocated
        ).encode("utf-8")
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            handle = output.open("xb")
        except FileExistsError:
            if output.read_bytes() != payload:
                raise ValueError(f"existing output differs; use a new output manifest: {output}")
        else:
            try:
                with handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
            except BaseException:
                output.unlink(missing_ok=True)
                raise
        published_sha = hashlib.sha256(payload).hexdigest()
    return {
        "valid": True,
        "source_manifest": str(manifest),
        "source_sha256": source_sha,
        "manifest": str(output or manifest),
        "manifest_sha256": published_sha,
        "content_unique_images": len(local_aliases),
        "by_task": by_task,
        "task_jsonl_sha256": task_digests,
        "relocated_rows": changed,
        "geometry_changed": False,
        "content_missing_mode": "full_image",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument("--expected-unique-images", type=int, default=DEFAULT_UI5_FULL_TEST_UNIQUE_IMAGES)
    args = parser.parse_args()
    report = prepare_manifest(
        args.manifest, args.input_dir,
        output_manifest=args.output_manifest,
        expected_unique_images=args.expected_unique_images,
    )
    print("[EVAL MANIFEST PASS] " + json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
