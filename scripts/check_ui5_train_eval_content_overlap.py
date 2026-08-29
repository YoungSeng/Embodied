#!/usr/bin/env python3
"""Fail closed when crop-only train images overlap validation or test by bytes."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

from analyze_ui5_source_overlap import content_fingerprint
from locany_ui5_common import TASK_JSONL
from run_ui5_crop_audit import atomic_write_json, read_jsonl


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-unique-manifest", type=Path, required=True)
    parser.add_argument("--validation-data-dir", type=Path, required=True)
    parser.add_argument("--test-data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _image_value(record: dict[str, Any]) -> str:
    value = record.get("images", record.get("image"))
    if isinstance(value, list):
        value = value[0] if value else None
    if isinstance(value, dict):
        value = value.get("path")
    if not isinstance(value, str):
        raise ValueError("JSONL record has no single image path")
    return value


def _content_ids(files: Sequence[Path]) -> tuple[set[str], int]:
    ids = set()
    records = 0
    path_cache: dict[Path, str] = {}
    for source in files:
        if not source.is_file():
            raise FileNotFoundError(source)
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                records += 1
                record = json.loads(line)
                image = Path(_image_value(record)).expanduser()
                if not image.is_absolute():
                    image = source.parent / image
                image = image.resolve(strict=True)
                digest = path_cache.get(image)
                if digest is None:
                    digest = content_fingerprint(image)
                    path_cache[image] = digest
                ids.add(digest)
    return ids, records


def build(args: argparse.Namespace) -> dict[str, Any]:
    train_manifest = args.train_unique_manifest.expanduser().resolve(strict=True)
    validation_dir = args.validation_data_dir.expanduser().resolve(strict=True)
    test_dir = args.test_data_dir.expanduser().resolve(strict=True)
    train_rows = read_jsonl(train_manifest)
    train_ids = {
        str(row.get("content_id") or "") for row in train_rows if row.get("content_id")
    }
    if len(train_ids) != len(train_rows):
        raise ValueError("training unique manifest lacks one content_id per unique image")
    validation_files = sorted(validation_dir.glob("ui_*_val.jsonl"))
    if len(validation_files) != 5:
        raise RuntimeError(
            f"expected five validation JSONL files under {validation_dir}, found {len(validation_files)}"
        )
    test_files = [test_dir / name for name in TASK_JSONL.values()]
    validation_ids, validation_records = _content_ids(validation_files)
    test_ids, test_records = _content_ids(test_files)
    train_validation_overlap = len(train_ids & validation_ids)
    train_test_overlap = len(train_ids & test_ids)
    validation_test_overlap = len(validation_ids & test_ids)
    payload = {
        "schema_version": 1,
        "train_unique_images": len(train_ids),
        "validation_unique_images": len(validation_ids),
        "test_unique_images": len(test_ids),
        "validation_records": validation_records,
        "test_records": test_records,
        "train_validation_content_overlap_count": train_validation_overlap,
        "train_test_content_overlap_count": train_test_overlap,
        "validation_test_content_overlap_count": validation_test_overlap,
        "passes": (
            train_validation_overlap == 0
            and train_test_overlap == 0
            and validation_test_overlap == 0
        ),
        "inputs": {
            "train_unique_manifest": str(train_manifest),
            "validation_files": [str(path.resolve()) for path in validation_files],
            "test_files": [str(path.resolve()) for path in test_files],
        },
    }
    atomic_write_json(args.output.expanduser().resolve(), payload)
    if not payload["passes"]:
        raise RuntimeError(
            "train/eval content leakage detected: "
            f"train-val={payload['train_validation_content_overlap_count']}, "
            f"train-test={payload['train_test_content_overlap_count']}, "
            f"validation-test={payload['validation_test_content_overlap_count']}"
        )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(build(parse_args(argv)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
