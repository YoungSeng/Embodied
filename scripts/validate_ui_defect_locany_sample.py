#!/usr/bin/env python3
"""Validate the portable ten-sample LocateAnything UI-defect dataset."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


EXPECTED_LABELS = {
    "ui_occlusion": "overlapping elements",
    "ui_cropping": "cropped element",
    "ui_text_overflow": "text overflow",
    "ui_text_ellipsis": "abnormal text ellipsis",
    "ui_content_missing": "missing content",
}
BOX_PATTERN = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument(
        "--recipe",
        type=Path,
        default=None,
        help=(
            "Default: <project-root>/samples/ui_defect_locany_smoke/recipe/"
            "ui_defect_5class_train.json"
        ),
    )
    return parser.parse_args()


def resolve_from_project(project_root: Path, raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    resolved = (project_root / path).resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(f"path escapes project root: {raw}") from exc
    return resolved


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_no}: record must be an object")
            yield line_no, record


def validate_record(
    record: dict[str, Any],
    *,
    annotation_path: Path,
    line_no: int,
    task_name: str,
    image_root: Path,
) -> str:
    prefix = f"{annotation_path}:{line_no}"
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or len(conversations) != 2:
        raise ValueError(f"{prefix}: expected exactly two conversation turns")

    human, gpt = conversations
    if human.get("from") != "human" or gpt.get("from") != "gpt":
        raise ValueError(f"{prefix}: invalid conversation roles")

    label = EXPECTED_LABELS[task_name]
    expected_prompt = (
        "Locate all the instances that match the following description: "
        f"{label}."
    )
    if human.get("value") != expected_prompt:
        raise ValueError(f"{prefix}: prompt does not match production template")

    image_value = record.get("image")
    if not isinstance(image_value, str) or not image_value:
        raise ValueError(f"{prefix}: missing image path")
    if Path(image_value).is_absolute() or ".." in Path(image_value).parts:
        raise ValueError(f"{prefix}: sample image path must be portable: {image_value}")

    image_path = (image_root / image_value).resolve()
    try:
        image_path.relative_to(image_root.resolve())
    except ValueError as exc:
        raise ValueError(f"{prefix}: image path escapes recipe root") from exc
    if not image_path.is_file():
        raise ValueError(f"{prefix}: missing image: {image_path}")
    with Image.open(image_path) as image:
        if image.format not in {"PNG", "JPEG", "WEBP", "BMP", "TIFF"}:
            raise ValueError(f"{prefix}: unsupported image format: {image.format}")
        if image.width <= 0 or image.height <= 0:
            raise ValueError(f"{prefix}: invalid image dimensions")
        image.verify()

    answer = gpt.get("value")
    if answer == "<box>none</box>":
        return "negative"
    if not isinstance(answer, str) or not answer.startswith(f"<ref>{label}</ref>"):
        raise ValueError(f"{prefix}: positive answer has the wrong <ref> label")

    boxes = BOX_PATTERN.findall(answer)
    if not boxes:
        raise ValueError(f"{prefix}: positive answer contains no valid box")
    for raw_box in boxes:
        x1, y1, x2, y2 = map(int, raw_box)
        if not all(0 <= value <= 1000 for value in (x1, y1, x2, y2)):
            raise ValueError(f"{prefix}: coordinate outside [0, 1000]: {raw_box}")
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"{prefix}: zero/reversed box: {raw_box}")
    return "positive"


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    recipe_path = (
        args.recipe.resolve()
        if args.recipe is not None
        else project_root
        / "samples"
        / "ui_defect_locany_smoke"
        / "recipe"
        / "ui_defect_5class_train.json"
    )
    if not recipe_path.is_file():
        raise SystemExit(f"Recipe does not exist: {recipe_path}")

    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    if not isinstance(recipe, dict) or len(recipe) != 1:
        raise ValueError("smoke recipe must contain exactly one dataset")
    config = next(iter(recipe.values()))
    annotation_values = config.get("annotation")
    if not isinstance(annotation_values, list) or len(annotation_values) != 5:
        raise ValueError("smoke recipe must list five annotation files")
    image_root = resolve_from_project(project_root, config.get("root", ""))
    if not image_root.is_dir():
        raise ValueError(f"recipe image root is not a directory: {image_root}")

    by_task: dict[str, Counter[str]] = defaultdict(Counter)
    image_names: list[str] = []
    total = 0

    for raw_annotation in annotation_values:
        annotation_path = resolve_from_project(project_root, raw_annotation)
        if not annotation_path.is_file():
            raise ValueError(f"missing annotation file: {annotation_path}")
        task_name = annotation_path.name.removesuffix("_train.jsonl")
        if task_name not in EXPECTED_LABELS:
            raise ValueError(f"unexpected task annotation: {annotation_path.name}")

        for line_no, record in iter_jsonl(annotation_path):
            kind = validate_record(
                record,
                annotation_path=annotation_path,
                line_no=line_no,
                task_name=task_name,
                image_root=image_root,
            )
            by_task[task_name][kind] += 1
            image_names.append(record["image"])
            total += 1

    if total != 10:
        raise ValueError(f"expected 10 records, found {total}")
    if len(set(image_names)) != 10:
        raise ValueError("the smoke dataset must contain ten unique image paths")

    expected_balance = Counter({"positive": 1, "negative": 1})
    for task_name in EXPECTED_LABELS:
        if by_task[task_name] != expected_balance:
            raise ValueError(
                f"{task_name}: expected one positive and one negative, "
                f"found {dict(by_task[task_name])}"
            )

    print(f"[OK] recipe: {recipe_path}")
    print(f"[OK] image root: {image_root}")
    print("[OK] 10 records, 10 images, 5 tasks, balanced positive/negative samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

