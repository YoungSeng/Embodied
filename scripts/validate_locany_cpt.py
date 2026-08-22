#!/usr/bin/env python3
"""Fail-fast validation for a LocateAnything CPT recipe."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eaglevl.train.dataset_sampling import resolve_recipe_entry_paths
from prepare_locany_cpt import FORBIDDEN_MARKERS, LOCANY_BOX_RE, NormalizeError, validate_locany_text


def iter_image_paths(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from iter_image_paths(item)


def validate_record(record: dict[str, Any], root: Path, check_images: bool) -> None:
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or len(conversations) < 2:
        raise NormalizeError("missing conversations")
    if conversations[0].get("from") != "human" or conversations[-1].get("from") != "gpt":
        raise NormalizeError("conversation must start with human and end with gpt")
    for index, turn in enumerate(conversations):
        expected = "human" if index % 2 == 0 else "gpt"
        if turn.get("from") != expected:
            raise NormalizeError(f"unexpected role at turn {index}: {turn.get('from')!r}")
        text = turn.get("value")
        if not isinstance(text, str) or not text.strip():
            raise NormalizeError(f"empty text at turn {index}")
        validate_locany_text(text, expected)
        if any(marker in text for marker in FORBIDDEN_MARKERS):
            raise NormalizeError(f"Qwen marker remains at turn {index}")

    raw_images = list(iter_image_paths(record.get("image")))
    if not raw_images:
        raise NormalizeError("missing image")
    prompt = "\n".join(turn["value"] for turn in conversations if turn["from"] == "human")
    if prompt.count("<image>") + len(re.findall(r"<image-\d+>", prompt)) < len(raw_images):
        raise NormalizeError("fewer image placeholders than image paths")
    if check_images:
        for raw in raw_images:
            path = Path(raw)
            resolved = path if path.is_absolute() else root / path
            if not resolved.is_file():
                raise NormalizeError(f"image does not exist: {resolved}")

    for turn in conversations:
        for match in LOCANY_BOX_RE.finditer(turn["value"]):
            coords = tuple(int(value) for value in match.groups())
            if not (0 <= coords[0] < coords[2] <= 1000 and 0 <= coords[1] < coords[3] <= 1000):
                raise NormalizeError(f"invalid bbox: {match.group(0)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument(
        "--records-per-dataset",
        type=int,
        default=32,
        help="0 validates every row",
    )
    parser.add_argument("--skip-image-check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.records_per_dataset < 0:
        raise SystemExit("--records-per-dataset cannot be negative")
    recipe_path = args.recipe.expanduser().resolve()
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    if not isinstance(recipe, dict) or not recipe:
        raise SystemExit("recipe must be a non-empty JSON object")

    weights = []
    checked_total = 0
    for dataset_name, meta in recipe.items():
        if not isinstance(meta, dict):
            raise SystemExit(f"{dataset_name}: metadata is not an object")
        meta = resolve_recipe_entry_paths(meta, recipe_path)
        weight = float(meta.get("sampling_weight", 0.0))
        if not math.isfinite(weight) or weight <= 0:
            raise SystemExit(f"{dataset_name}: invalid sampling_weight={weight!r}")
        weights.append(weight)
        root_value = str(meta.get("root", ""))
        root = Path(root_value) if root_value else Path("/")
        annotations = meta.get("annotation")
        if isinstance(annotations, str):
            annotations = [annotations]
        if not isinstance(annotations, list) or not annotations:
            raise SystemExit(f"{dataset_name}: missing annotation files")

        checked_dataset = 0
        for annotation_value in annotations:
            annotation = Path(annotation_value)
            if not annotation.is_file():
                raise SystemExit(f"{dataset_name}: annotation does not exist: {annotation}")
            with annotation.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if args.records_per_dataset and checked_dataset >= args.records_per_dataset:
                        break
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        if not isinstance(record, dict):
                            raise NormalizeError("row is not an object")
                        validate_record(record, root=root, check_images=not args.skip_image_check)
                    except Exception as exc:
                        raise SystemExit(
                            f"{dataset_name}: {annotation}:{line_number}: "
                            f"{type(exc).__name__}: {exc}"
                        ) from exc
                    checked_dataset += 1
                    checked_total += 1
        if checked_dataset == 0:
            raise SystemExit(f"{dataset_name}: no records found")
        print(f"{dataset_name:32s} checked={checked_dataset:6,d} weight={weight:g}")

    if max(weights) - min(weights) > 1e-12:
        raise SystemExit(f"CPT task weights are not equal: {weights}")
    probability = 1.0 / len(weights)
    print(f"OK: datasets={len(weights)}, checked={checked_total:,}, probability_per_task={probability:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
