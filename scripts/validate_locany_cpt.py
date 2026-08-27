#!/usr/bin/env python3
"""Fail-fast validation for a LocateAnything CPT recipe."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from collections import defaultdict
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


def validate_split_manifest(path: Path) -> dict[str, Any]:
    """Fail when any stable identity crosses train/held-out boundaries."""
    identities: dict[str, dict[str, set[str]]] = {
        "group_id": defaultdict(set),
        "record_id": defaultdict(set),
        "record_id_hash": defaultdict(set),
        "group_id_hash": defaultdict(set),
        "image_sha256": defaultdict(set),
        "normalized_image_path": defaultdict(set),
    }
    rows = 0
    split_groups: dict[str, set[str]] = defaultdict(set)
    task_rows: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    record_occurrences: dict[str, int] = defaultdict(int)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise NormalizeError(f"{path}:{line_number}: manifest row is not an object")
            split = str(value.get("split", ""))
            if split not in {"train", "heldout"}:
                raise NormalizeError(f"{path}:{line_number}: invalid split={split!r}")
            group_id = str(value.get("group_id", ""))
            record_id = str(value.get("record_id", ""))
            if not group_id or not record_id:
                raise NormalizeError(f"{path}:{line_number}: missing group_id/record_id")
            identities["group_id"][group_id].add(split)
            identities["record_id"][record_id].add(split)
            for hash_key in ("record_id_hash", "group_id_hash"):
                if value.get(hash_key) is not None:
                    identities[hash_key][str(value[hash_key])].add(split)
            record_occurrences[record_id] += 1
            task_rows[str(value.get("task", ""))][split] += 1
            split_groups[split].add(group_id)
            for digest in value.get("image_sha256", []):
                identities["image_sha256"][str(digest)].add(split)
            for image in value.get("image", []):
                normalized = Path(str(image)).as_posix().casefold()
                identities["normalized_image_path"][normalized].add(split)
            rows += 1

    leaks = {
        kind: sorted(identity for identity, splits in values.items() if len(splits) > 1)
        for kind, values in identities.items()
    }
    leaks = {kind: values for kind, values in leaks.items() if values}
    intersection = split_groups["train"] & split_groups["heldout"]
    if intersection:
        leaks["train_val_group_intersection"] = sorted(intersection)
    duplicate_record_ids = sorted(
        record_id for record_id, count in record_occurrences.items() if count != 1
    )
    if duplicate_record_ids:
        leaks["duplicate_record_ids"] = duplicate_record_ids
    if leaks:
        preview = {kind: values[:10] for kind, values in leaks.items()}
        raise NormalizeError(f"CPT split leakage detected: {preview}")
    return {
        "rows": rows,
        "train_groups": len(split_groups["train"]),
        "val_groups": len(split_groups["heldout"]),
        "group_intersection": 0,
        "task_rows": {
            task: dict(sorted(values.items()))
            for task, values in sorted(task_rows.items())
        },
    }


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
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help="defaults to <recipe parent>/../diagnostics/split_manifest.jsonl when present",
    )
    parser.add_argument(
        "--require-split",
        choices=("train", "heldout"),
        default=None,
        help="require every checked row to carry this cpt_split",
    )
    parser.add_argument(
        "--require-equal-weights",
        action="store_true",
        help="enforce sample-equal recipe weights (alternative fixed modes may differ)",
    )
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
    recipe_task_rows = {}
    checked_total = 0
    for dataset_name, meta in recipe.items():
        if not isinstance(meta, dict):
            raise SystemExit(f"{dataset_name}: metadata is not an object")
        meta = resolve_recipe_entry_paths(meta, recipe_path)
        weight = float(meta.get("sampling_weight", 0.0))
        if not math.isfinite(weight) or weight <= 0:
            raise SystemExit(f"{dataset_name}: invalid sampling_weight={weight!r}")
        weights.append(weight)
        task = str(meta.get("cpt_task") or dataset_name.removeprefix("locany_cpt_"))
        recipe_task_rows[task] = int(meta.get("dataset_rows", 0) or 0)
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
                        if args.require_split is not None:
                            actual_split = str(record.get("cpt_split", ""))
                            if actual_split != args.require_split:
                                raise NormalizeError(
                                    f"expected cpt_split={args.require_split!r}, got {actual_split!r}"
                                )
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

    if args.require_equal_weights and max(weights) - min(weights) > 1e-12:
        raise SystemExit(f"CPT task weights are not equal: {weights}")
    total_weight = sum(weights)
    probabilities = [weight / total_weight for weight in weights]

    manifest_path = args.split_manifest
    if manifest_path is None:
        candidate = recipe_path.parent.parent / "diagnostics" / "split_manifest.jsonl"
        manifest_path = candidate if candidate.is_file() else None
    manifest_report = None
    if manifest_path is not None:
        manifest_report = validate_split_manifest(manifest_path.expanduser().resolve())
        if args.require_split is not None:
            manifest_split = "heldout" if args.require_split == "heldout" else "train"
            mismatches = {}
            for task, expected_rows in recipe_task_rows.items():
                manifest_rows = int(
                    manifest_report["task_rows"].get(task, {}).get(manifest_split, 0)
                )
                if expected_rows and expected_rows != manifest_rows:
                    mismatches[task] = {
                        "recipe_rows": expected_rows,
                        "manifest_rows": manifest_rows,
                    }
            if mismatches:
                raise SystemExit(
                    f"recipe/manifest row-count mismatch for {manifest_split}: {mismatches}"
                )
        print(f"split_manifest={manifest_path} report={manifest_report}")
    elif args.require_split is not None:
        raise SystemExit(
            "group-level split validation requires split_manifest.jsonl; "
            "pass --split-manifest explicitly"
        )

    print(
        f"OK: datasets={len(weights)}, checked={checked_total:,}, "
        f"probability_range={min(probabilities):.2%}..{max(probabilities):.2%}, "
        f"zero_leakage={manifest_report is not None}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
