#!/usr/bin/env python3
"""Audit path/content overlap for the five UI5 source and training datasets.

Identity is never inferred from a basename.  Canonical absolute paths and a fast
content fingerprint are both retained because the two answer different audit
questions: path reuse and byte-identical images stored at different locations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from prepare_ui_defect_locany import (
    TASKS,
    choose_split,
    extract_image_path,
    iter_numeric_boxes,
)


BOX_PATTERN = re.compile(
    r"<box>\s*<(-?\d+(?:\.\d+)?)>\s*<(-?\d+(?:\.\d+)?)>\s*"
    r"<(-?\d+(?:\.\d+)?)>\s*<(-?\d+(?:\.\d+)?)>\s*</box>"
)
TASK_NAMES = tuple(task["name"] for task in TASKS)
ProgressCallback = Callable[[str, int, int], None]


def _count_nonempty_lines(paths: Iterable[Path]) -> int:
    total = 0
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            total += sum(1 for line in handle if line.strip())
    return total


def _print_progress(label: str, completed: int, total: int, started: float) -> None:
    elapsed = max(0.001, time.monotonic() - started)
    rate = completed / elapsed if completed else 0.0
    remaining = max(0, total - completed)
    eta = remaining / rate if rate else None

    def duration(seconds: float | None) -> str:
        if seconds is None:
            return "--:--:--"
        value = int(round(seconds))
        hours, remainder = divmod(value, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    percent = completed / total if total else 1.0
    print(
        f"[进度 prepare/overlap {label}] {completed}/{total} ({percent:.1%}) | "
        f"已耗时 {duration(elapsed)} | 速度 {rate:.2f} records/s | ETA {duration(eta)}",
        flush=True,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare UI5 source JSONL and effective LocateAnything data."
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--locany-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260728)
    return parser.parse_args(argv)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def content_fingerprint(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    """Return a stable, fast byte-content identity without loading the whole file."""
    digest = hashlib.blake2b(digest_size=16)
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return "blake2b128:" + digest.hexdigest()


def canonical_existing_path(path: Path) -> Path:
    try:
        return path.expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"referenced image does not exist: {path}") from exc


def resolve_training_image(raw: str, source_dir: Path, locany_data_dir: Path) -> Path:
    path = Path(raw).expanduser()
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        # LocateAnything recipes use root="/" and therefore often store an
        # absolute POSIX path without the leading slash.
        if raw.startswith(("mnt/", "data/", "home/", "tmp/")):
            candidates.append(Path("/") / path)
        candidates.extend((source_dir / path, locany_data_dir / path, Path.cwd() / path))
    for candidate in candidates:
        if candidate.is_file():
            return canonical_existing_path(candidate)
    rendered = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"cannot resolve training image {raw!r}; tried: {rendered}")


def assistant_answer(record: Mapping[str, Any]) -> str:
    answer = ""
    for message in record.get("conversations", []):
        if isinstance(message, Mapping) and message.get("from") in {"gpt", "assistant"}:
            answer = str(message.get("value", ""))
    return answer


def source_gt_count(record: Mapping[str, Any]) -> int:
    objects = record.get("objects") or {}
    boxes = objects.get("bbox", []) if isinstance(objects, Mapping) else []
    return len(list(iter_numeric_boxes(boxes, "xyxy")))


def make_row(
    *,
    task: str,
    source_file: Path,
    line_no: int,
    image_path: str,
    canonical_path: Path,
    content_id: str,
    positive: bool,
    gt_count: int,
    split: str,
) -> dict[str, Any]:
    return {
        "task": task,
        "source_file": str(source_file),
        "line_no": line_no,
        "image_path": image_path,
        "canonical_path": str(canonical_path),
        "content_id": content_id,
        "positive": bool(positive),
        "gt_count": int(gt_count),
        "split": split,
    }


def load_source_rows(
    source_dir: Path,
    *,
    val_ratio: float,
    seed: int,
    fingerprint_cache: dict[str, str],
    progress_callback: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_paths = [source_dir / task["file"] for task in TASKS]
    missing = [path for path in source_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing source JSONL: " + ", ".join(map(str, missing)))
    total = _count_nonempty_lines(source_paths)
    started = time.monotonic()
    if progress_callback:
        progress_callback("source", 0, total)
    else:
        _print_progress("source", 0, total, started)
    completed = 0
    for task in TASKS:
        source_file = source_dir / task["file"]
        if not source_file.is_file():
            raise FileNotFoundError(f"missing source JSONL: {source_file}")
        with source_file.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                image = canonical_existing_path(extract_image_path(record, source_dir))
                canonical = str(image)
                if canonical not in fingerprint_cache:
                    fingerprint_cache[canonical] = content_fingerprint(image)
                content_id = fingerprint_cache[canonical]
                gt_count = source_gt_count(record)
                rows.append(
                    make_row(
                        task=task["name"],
                        source_file=source_file,
                        line_no=line_no,
                        image_path=str((record.get("images") or [record.get("image")])[0]),
                        canonical_path=image,
                        content_id=content_id,
                        positive=gt_count > 0,
                        gt_count=gt_count,
                        split=choose_split(image, val_ratio, seed),
                    )
                )
                completed += 1
                if completed % 250 == 0 or completed == total:
                    if progress_callback:
                        progress_callback("source", completed, total)
                    else:
                        _print_progress("source", completed, total, started)
    return rows


def load_locany_rows(
    source_dir: Path,
    locany_data_dir: Path,
    *,
    fingerprint_cache: dict[str, str],
    include_val: bool,
    progress_callback: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    splits = ("train", "val") if include_val else ("train",)
    locany_paths = [
        locany_data_dir / f"{task['name']}_{split}.jsonl"
        for task in TASKS
        for split in splits
        if (locany_data_dir / f"{task['name']}_{split}.jsonl").is_file()
    ]
    total = _count_nonempty_lines(locany_paths)
    started = time.monotonic()
    if progress_callback:
        progress_callback("train/val", 0, total)
    else:
        _print_progress("train/val", 0, total, started)
    completed = 0
    for task in TASKS:
        for split in splits:
            source_file = locany_data_dir / f"{task['name']}_{split}.jsonl"
            if not source_file.is_file():
                if split == "val":
                    continue
                raise FileNotFoundError(f"missing training JSONL: {source_file}")
            with source_file.open("r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    raw_image = str(record["image"])
                    image = resolve_training_image(raw_image, source_dir, locany_data_dir)
                    canonical = str(image)
                    if canonical not in fingerprint_cache:
                        fingerprint_cache[canonical] = content_fingerprint(image)
                    content_id = fingerprint_cache[canonical]
                    boxes = BOX_PATTERN.findall(assistant_answer(record))
                    rows.append(
                        make_row(
                            task=task["name"],
                            source_file=source_file,
                            line_no=line_no,
                            image_path=raw_image,
                            canonical_path=image,
                            content_id=content_id,
                            positive=bool(boxes),
                            gt_count=len(boxes),
                            split=split,
                        )
                    )
                    completed += 1
                    if completed % 250 == 0 or completed == total:
                        if progress_callback:
                            progress_callback("train/val", completed, total)
                        else:
                            _print_progress("train/val", completed, total, started)
    return rows


def identity_sets(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, set[str]]:
    result = {task: set() for task in TASK_NAMES}
    for row in rows:
        result[str(row["task"])].add(str(row[key]))
    return result


def overlap_matrix(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    sets = identity_sets(rows, key)
    counts: dict[str, dict[str, int]] = {}
    jaccard: dict[str, dict[str, float]] = {}
    for left in TASK_NAMES:
        counts[left] = {}
        jaccard[left] = {}
        for right in TASK_NAMES:
            intersection = sets[left] & sets[right]
            union = sets[left] | sets[right]
            counts[left][right] = len(intersection)
            jaccard[left][right] = round(len(intersection) / len(union), 8) if union else 0.0
    return {"counts": counts, "jaccard": jaccard}


def task_cardinality(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    tasks_by_identity: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        tasks_by_identity[str(row[key])].add(str(row["task"]))
    counts = Counter(len(tasks) for tasks in tasks_by_identity.values())
    return {str(size): counts.get(size, 0) for size in range(1, 6)}


def polarity_conflicts(
    rows: Sequence[Mapping[str, Any]], key: str, identity_type: str
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    conflicts = []
    for identity, members in sorted(grouped.items()):
        statuses = {bool(member["positive"]) for member in members}
        tasks = {str(member["task"]) for member in members}
        if len(statuses) > 1 and len(tasks) > 1:
            conflicts.append(
                {
                    "identity_type": identity_type,
                    "identity": identity,
                    "tasks": sorted(tasks),
                    "records": [
                        {
                            "task": member["task"],
                            "positive": member["positive"],
                            "canonical_path": member["canonical_path"],
                            "source_file": member["source_file"],
                            "line_no": member["line_no"],
                        }
                        for member in members
                    ],
                }
            )
    return conflicts


def basename_conflicts(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[Path(str(row["canonical_path"])).name].append(row)
    conflicts = []
    for basename, members in sorted(grouped.items()):
        paths = sorted({str(member["canonical_path"]) for member in members})
        contents = sorted({str(member["content_id"]) for member in members})
        if len(paths) > 1:
            conflicts.append(
                {
                    "basename": basename,
                    "different_path": True,
                    "different_content": len(contents) > 1,
                    "canonical_paths": paths,
                    "content_ids": contents,
                    "tasks": sorted({str(member["task"]) for member in members}),
                }
            )
    return conflicts


def dataset_statistics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    per_task: dict[str, Any] = {}
    for task in TASK_NAMES:
        selected = [row for row in rows if row["task"] == task]
        per_task[task] = {
            "records": len(selected),
            "positive_samples": sum(bool(row["positive"]) for row in selected),
            "negative_samples": sum(not bool(row["positive"]) for row in selected),
            "gt_count": sum(int(row["gt_count"]) for row in selected),
            "unique_images_by_path": len({row["canonical_path"] for row in selected}),
            "unique_images_by_content": len({row["content_id"] for row in selected}),
        }
    path_conflicts = polarity_conflicts(rows, "canonical_path", "path")
    content_conflicts = polarity_conflicts(rows, "content_id", "content")
    basename = basename_conflicts(rows)
    return {
        "per_task": per_task,
        "path_overlap": overlap_matrix(rows, "canonical_path"),
        "content_overlap": overlap_matrix(rows, "content_id"),
        "task_cardinality_by_path": task_cardinality(rows, "canonical_path"),
        "task_cardinality_by_content": task_cardinality(rows, "content_id"),
        "positive_negative_conflicts": {
            "by_path_count": len(path_conflicts),
            "by_content_count": len(content_conflicts),
            "by_path": path_conflicts,
            "by_content": content_conflicts,
        },
        "basename_conflicts": {
            "count": len(basename),
            "different_path_count": sum(item["different_path"] for item in basename),
            "different_content_count": sum(item["different_content"] for item in basename),
            "details": basename,
        },
    }


def cross_split_content(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["content_id"])].append(row)
    conflicts = []
    for content_id, members in sorted(grouped.items()):
        splits = {str(member["split"]) for member in members}
        if {"train", "val"}.issubset(splits):
            conflicts.append(
                {
                    "content_id": content_id,
                    "splits": sorted(splits),
                    "paths": sorted({str(member["canonical_path"]) for member in members}),
                    "tasks": sorted({str(member["task"]) for member in members}),
                }
            )
    return conflicts


def analyze(
    source_dir: Path,
    locany_data_dir: Path,
    output_dir: Path,
    *,
    val_ratio: float = 0.02,
    seed: int = 20260728,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be in [0, 1)")
    source_dir = source_dir.resolve(strict=True)
    locany_data_dir = locany_data_dir.resolve(strict=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    fingerprints: dict[str, str] = {}
    source_rows = load_source_rows(
        source_dir,
        val_ratio=val_ratio,
        seed=seed,
        fingerprint_cache=fingerprints,
        progress_callback=progress_callback,
    )
    all_locany_rows = load_locany_rows(
        source_dir,
        locany_data_dir,
        fingerprint_cache=fingerprints,
        include_val=True,
        progress_callback=progress_callback,
    )
    training_rows = [row for row in all_locany_rows if row["split"] == "train"]
    split_conflicts = cross_split_content(all_locany_rows)
    result = {
        "source_data": dataset_statistics(source_rows),
        "actual_training_data": dataset_statistics(training_rows),
        "same_content_cross_train_val": {
            "count": len(split_conflicts),
            "details": split_conflicts,
        },
        "definitions": {
            "path_identity": "resolved absolute canonical_path",
            "content_identity": "blake2b-128 digest of image bytes",
            "basename_identity": "never used; warning only",
        },
    }
    write_jsonl(output_dir / "source_records.jsonl", source_rows)
    write_jsonl(output_dir / "training_records.jsonl", training_rows)
    write_json(output_dir / "source_overlap.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = analyze(
        args.source_dir,
        args.locany_data_dir,
        args.output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
