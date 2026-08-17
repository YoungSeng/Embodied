#!/usr/bin/env python3
"""Export ten approved real records into a portable LocateAnything sample set.

For each of the five processed UI-defect JSONL files, select one positive and
one negative record with a deterministic reservoir sample, copy the referenced
images, and rewrite all paths so the result can move with the repository.

This script requires an explicit authorization flag because internal UI images
may contain private or restricted information. It performs no de-identification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError


TASKS = (
    ("ui_occlusion", "overlapping elements"),
    ("ui_cropping", "cropped element"),
    ("ui_text_overflow", "text overflow"),
    ("ui_text_ellipsis", "abnormal text ellipsis"),
    ("ui_content_missing", "missing content"),
)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--source-data-dir", type=Path, required=True)
    parser.add_argument(
        "--source-image-root",
        type=Path,
        default=Path("/"),
        help="Recipe root used to resolve non-absolute image paths. Default: /",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument(
        "--confirm-authorized-export",
        action="store_true",
        help="Confirm that every selected record may be moved outside its source environment.",
    )
    return parser.parse_args()


def resolve_image(raw: str, source_image_root: Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = source_image_root / path
    return path.resolve(strict=False)


def is_positive(record: dict[str, Any]) -> bool:
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or len(conversations) < 2:
        raise ValueError("record has no valid conversations")
    answer = conversations[-1]
    if not isinstance(answer, dict) or answer.get("from") != "gpt":
        raise ValueError("record has no final gpt answer")
    value = answer.get("value")
    if not isinstance(value, str):
        raise ValueError("gpt answer is not a string")
    return value != "<box>none</box>"


def reservoir_pick(
    input_path: Path,
    *,
    source_image_root: Path,
    rng: random.Random,
    used_images: set[Path],
) -> dict[str, tuple[dict[str, Any], Path]]:
    selected: dict[str, tuple[dict[str, Any], Path]] = {}
    seen = {"positive": 0, "negative": 0}

    with input_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                kind = "positive" if is_positive(record) else "negative"
                raw_image = record.get("image")
                if not isinstance(raw_image, str) or not raw_image:
                    continue
                image_path = resolve_image(raw_image, source_image_root)
                if image_path in used_images or not image_path.is_file():
                    continue
            except (json.JSONDecodeError, OSError, ValueError):
                continue

            seen[kind] += 1
            if rng.randrange(seen[kind]) == 0:
                selected[kind] = (record, image_path)

    missing = {kind for kind in ("positive", "negative") if kind not in selected}
    if missing:
        raise ValueError(
            f"{input_path}: could not find authorized candidates for: {sorted(missing)}"
        )
    return selected


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact_json(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"


def main() -> int:
    args = parse_args()
    if not args.confirm_authorized_export:
        raise SystemExit(
            "Refusing to copy real UI data without --confirm-authorized-export.\n"
            "The flag confirms authorization; it does not de-identify or approve the data."
        )

    project_root = args.project_root.resolve()
    source_data_dir = args.source_data_dir.resolve()
    source_image_root = args.source_image_root.resolve()
    output_dir = args.output_dir.resolve()

    if not source_data_dir.is_dir():
        raise SystemExit(f"Source data directory does not exist: {source_data_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(
            f"Refusing to overwrite non-empty output directory: {output_dir}"
        )

    image_dir = output_dir / "images"
    annotation_dir = output_dir / "annotations"
    recipe_dir = output_dir / "recipe"
    image_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)
    recipe_dir.mkdir(parents=True, exist_ok=True)

    try:
        output_rel = output_dir.relative_to(project_root).as_posix()
    except ValueError:
        output_rel = output_dir.as_posix()

    used_images: set[Path] = set()
    annotation_paths: list[str] = []
    manifest_samples: list[dict[str, Any]] = []

    for task_idx, (task_name, expected_label) in enumerate(TASKS):
        input_path = source_data_dir / f"{task_name}_train.jsonl"
        if not input_path.is_file():
            raise FileNotFoundError(f"missing processed annotation: {input_path}")

        selected = reservoir_pick(
            input_path,
            source_image_root=source_image_root,
            rng=random.Random(args.seed + task_idx),
            used_images=used_images,
        )
        output_records: list[dict[str, Any]] = []

        for kind in ("positive", "negative"):
            record, source_image = selected[kind]
            used_images.add(source_image)
            suffix = source_image.suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}:
                raise ValueError(
                    f"unsupported selected image extension {suffix!r}: {source_image}"
                )
            image_name = f"{task_name}_{kind}{suffix}"
            destination = image_dir / image_name

            try:
                with Image.open(source_image) as image:
                    image.verify()
            except (OSError, UnidentifiedImageError) as exc:
                raise ValueError(f"unreadable selected image: {source_image}") from exc

            shutil.copy2(source_image, destination)
            portable_record = dict(record)
            portable_record.pop("images", None)
            portable_record["image"] = image_name
            output_records.append(portable_record)
            manifest_samples.append(
                {
                    "task": task_name,
                    "expected_label": expected_label,
                    "kind": kind,
                    "image": image_name,
                    "sha256": sha256(destination),
                }
            )

        annotation_path = annotation_dir / f"{task_name}_train.jsonl"
        annotation_path.write_text(
            "".join(compact_json(record) for record in output_records),
            encoding="utf-8",
        )
        annotation_paths.append(
            f"{output_rel}/annotations/{task_name}_train.jsonl"
        )

    recipe = {
        "ui_defect_5class_smoke_real": {
            "annotation": annotation_paths,
            "root": f"{output_rel}/images",
            "repeat_time": 1.0,
            "data_augment": False,
        }
    }
    recipe_path = recipe_dir / "ui_defect_5class_train.json"
    recipe_path.write_text(
        json.dumps(recipe, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "provenance": "real records exported after explicit authorization confirmation",
        "warning": "No automatic de-identification was performed; manual review is required.",
        "seed": args.seed,
        "num_samples": len(manifest_samples),
        "samples": manifest_samples,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Exported {len(manifest_samples)} records to: {output_dir}")
    print(f"Recipe: {recipe_path}")
    print("Manual privacy, policy, copyright, and licensing review is still required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
