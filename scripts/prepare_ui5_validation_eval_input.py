#!/usr/bin/env python3
"""Stage five held-out validation JSONLs under scorer-compatible filenames."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from locany_ui5_common import TASK_JSONL, TASKS
from run_ui5_crop_audit import atomic_write_json


VALIDATION_JSONL = {task: f"ui_{task}_val.jsonl" for task in TASKS}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _absolute_image_value(value: Any, base: Path) -> Any:
    if isinstance(value, str):
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = base / path
        return str(path.resolve(strict=True))
    if isinstance(value, dict):
        output = dict(value)
        if isinstance(output.get("path"), str):
            output["path"] = _absolute_image_value(output["path"], base)
        return output
    if isinstance(value, list):
        return [_absolute_image_value(item, base) for item in value]
    return value


def build(args: argparse.Namespace) -> dict[str, Any]:
    source_dir = args.source_dir.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"schema_version": 1, "source_dir": str(source_dir), "tasks": {}}
    content_ids: set[str] = set()
    image_paths: set[str] = set()
    for task in TASKS:
        source = source_dir / VALIDATION_JSONL[task]
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = output_dir / TASK_JSONL[task]
        temporary = destination.with_name(f".{destination.name}.tmp")
        count = 0
        with source.open("r", encoding="utf-8") as reader, temporary.open(
            "w", encoding="utf-8"
        ) as writer:
            for line in reader:
                if not line.strip():
                    continue
                row = json.loads(line)
                if "images" in row:
                    row["images"] = _absolute_image_value(row["images"], source.parent)
                    image_value = row["images"]
                elif "image" in row:
                    row["image"] = _absolute_image_value(row["image"], source.parent)
                    image_value = row["image"]
                else:
                    raise ValueError(f"validation row lacks image field: {source}:{count + 1}")
                if isinstance(image_value, list):
                    image_value = image_value[0] if image_value else None
                if isinstance(image_value, dict):
                    image_value = image_value.get("path")
                if not isinstance(image_value, str):
                    raise ValueError(f"validation row lacks one image: {source}:{count + 1}")
                image_path = Path(image_value).resolve(strict=True)
                image_paths.add(str(image_path))
                digest = hashlib.sha256()
                with image_path.open("rb") as image_handle:
                    for chunk in iter(lambda: image_handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                content_ids.add(digest.hexdigest())
                writer.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
        temporary.replace(destination)
        summary["tasks"][task] = {
            "source": str(source),
            "staged": str(destination),
            "records": count,
        }
    summary["total_records"] = sum(row["records"] for row in summary["tasks"].values())
    summary["path_unique_images"] = len(image_paths)
    summary["content_unique_images"] = len(content_ids)
    summary["expected_unique_images"] = len(content_ids)
    atomic_write_json(output_dir / "validation_staging_summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(build(parse_args(argv)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
