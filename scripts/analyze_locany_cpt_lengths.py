#!/usr/bin/env python3
"""Measure exact processor/MTP lengths for a prepared CPT recipe.

Run this in the training environment because it deliberately uses the same
processor and ``LazySupervisedDatasetMTP`` implementation as formal training.
It performs no truncation and stops on the first unexpected sample failure.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def percentile(values: list[int], probability: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * probability) - 1))
    return int(ordered[index])


def describe(values: list[int]) -> dict[str, int | float | None]:
    return {
        "count": len(values),
        "mean": sum(values) / len(values) if values else None,
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", delete=False
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    # Import heavy training dependencies only after argument parsing so --help
    # remains useful in lightweight environments.
    os.environ["LOCANY_CPT_MODE"] = "1"
    from transformers import AutoProcessor

    from eaglevl.train.locany_finetune_magi_stream import (
        LazySupervisedDatasetMTP,
        resolve_recipe_entry_paths,
    )

    recipe_path = args.recipe.expanduser().resolve()
    with recipe_path.open("r", encoding="utf-8") as handle:
        recipe = json.load(handle)
    processor = AutoProcessor.from_pretrained(
        args.processor,
        trust_remote_code=True,
        use_fast=True,
    )
    per_task: dict[str, dict[str, Any]] = {}
    source_counts: dict[str, Counter] = defaultdict(Counter)
    oversize_records: list[dict[str, Any]] = []
    for name, raw_meta in recipe.items():
        meta = resolve_recipe_entry_paths(dict(raw_meta), str(recipe_path))
        meta["data_augment"] = False
        task = str(meta.get("cpt_task") or name.removeprefix("locany_cpt_"))
        dataset = LazySupervisedDatasetMTP(
            name,
            meta,
            processor,
            block_size=args.block_size,
            repeat_time=1,
            balance_ui_defects=False,
        )
        lengths: dict[str, list[int]] = defaultdict(list)
        oversize = Counter()
        group_ids = set()
        for logical_index, real_index in enumerate(dataset.active_indices):
            raw = dataset.lazy_loader[real_index]
            try:
                sample = dataset._get_item_once(real_index)
            except Exception as exc:
                record_id = raw.get("cpt_record_id") or raw.get("id") or f"{name}:{real_index}"
                raise RuntimeError(
                    f"length analysis failed: task={task}, record_id={record_id}, "
                    f"source={raw.get('cpt_source')}, line={raw.get('cpt_source_line')}"
                ) from exc
            for field in (
                "raw_text_tokens",
                "vision_tokens",
                "pre_mtp_seq_len",
                "post_mtp_seq_len",
                "main_supervised_tokens",
                "mtp_supervised_tokens",
                "total_supervised_tokens",
            ):
                lengths[field].append(int(sample[f"_sample_{field}"][0]))
            pre = lengths["pre_mtp_seq_len"][-1]
            post = lengths["post_mtp_seq_len"][-1]
            source = str(raw.get("cpt_source") or "<unknown>")
            if post > args.max_num_tokens_per_sample:
                reason = "pre_mtp_already_oversize" if pre > args.max_num_tokens_per_sample else "mtp_expansion"
                oversize[reason] += 1
                source_counts[source][reason] += 1
                oversize_records.append(
                    {
                        "task": task,
                        "record_id": str(
                            raw.get("cpt_record_id")
                            or raw.get("id")
                            or f"{name}:{real_index}"
                        ),
                        "group_id": str(raw.get("cpt_group_id") or ""),
                        "source": source,
                        "line": raw.get("cpt_source_line"),
                        "pre_mtp_seq_len": pre,
                        "post_mtp_seq_len": post,
                        "main_supervised_tokens": lengths["main_supervised_tokens"][-1],
                        "mtp_supervised_tokens": lengths["mtp_supervised_tokens"][-1],
                        "total_supervised_tokens": lengths["total_supervised_tokens"][-1],
                        "reason": reason,
                        "max_num_tokens_per_sample": args.max_num_tokens_per_sample,
                    }
                )
            group_ids.add(str(raw.get("cpt_group_id")))
            if args.progress_every and (logical_index + 1) % args.progress_every == 0:
                print(f"[lengths] task={task} processed={logical_index + 1}/{len(dataset)}", flush=True)
        rows = len(dataset)
        task_payload = {
            "rows": rows,
            "groups": len(group_ids),
            "max_num_tokens_per_sample": args.max_num_tokens_per_sample,
            "oversize_samples": sum(oversize.values()),
            "oversize_rate": sum(oversize.values()) / rows if rows else 0.0,
            "oversize_reasons": dict(sorted(oversize.items())),
            "lengths": {field: describe(values) for field, values in sorted(lengths.items())},
        }
        per_task[task] = task_payload
        raw_meta["dataset_rows"] = rows
        raw_meta["dataset_groups"] = len(group_ids)
        raw_meta["mean_total_supervised_tokens"] = task_payload["lengths"]["total_supervised_tokens"]["mean"]
        raw_meta["length_stats_source"] = str(args.output.expanduser().resolve())

    output = args.output.expanduser().resolve()
    oversize_path = output.parent / "oversize_samples.jsonl"
    payload = {
        "schema_version": 1,
        "recipe": str(recipe_path),
        "processor": args.processor,
        "block_size": args.block_size,
        "max_num_tokens_per_sample": args.max_num_tokens_per_sample,
        "oversize_records_file": str(oversize_path),
        "oversize_records": len(oversize_records),
        "tasks": per_task,
        "oversize_by_source": {
            source: dict(sorted(counts.items()))
            for source, counts in sorted(source_counts.items())
        },
    }
    atomic_json(output, payload)
    atomic_jsonl(oversize_path, oversize_records)
    # cpt_data_stats.json is the canonical source consumed by the workbook;
    # retain data_length_stats.json as the requested compatibility artifact.
    for alias in (
        output.parent / "cpt_data_stats.json",
        output.parent / "data_length_stats.json",
    ):
        if alias != output:
            atomic_json(alias, payload)
    if not args.no_update_recipe:
        atomic_json(recipe_path, recipe)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--processor", required=True, help="processor/checkpoint path")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=6)
    parser.add_argument("--max-num-tokens-per-sample", type=int, default=16384)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--no-update-recipe", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = analyze(args)
    print(json.dumps({"output": str(args.output), "tasks": payload["tasks"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
