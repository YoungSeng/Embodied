#!/usr/bin/env python3
"""Summarize UI5 rollout OOM retries from raw JSONL without jq."""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def raw_rows(root: Path):
    for path in sorted((root / "raw").glob("*/rollout_*/part-*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
                yield row


def run(output_root: Path) -> dict[str, Any]:
    affected_rows = []
    unique: dict[str, dict[str, Any]] = {}
    by_model_rollout: Counter[tuple[str, int]] = Counter()
    raw_record_count = 0
    oom_total = 0
    recovered = 0
    final_failed = 0
    for row in raw_rows(output_root):
        raw_record_count += 1
        events = int(row.get("oom_events", 0))
        if events <= 0 and not row.get("oom_recovered") and not row.get("oom_final_failure"):
            continue
        oom_total += events
        recovered += int(bool(row.get("oom_recovered")))
        final_failed += int(bool(row.get("oom_final_failure")))
        model = str(row.get("model_id"))
        rollout = int(row.get("rollout_id", -1))
        by_model_rollout[(model, rollout)] += events
        affected_rows.append(
            {
                key: row.get(key)
                for key in (
                    "model_id",
                    "rollout_id",
                    "record_id",
                    "sample_id",
                    "source_image_id",
                    "task",
                    "image_relpath",
                    "oom_events",
                    "oom_recovered",
                    "oom_final_failure",
                    "oom_retry",
                    "runtime_error",
                )
            }
        )
        record_id = str(row.get("record_id"))
        item = unique.setdefault(
            record_id,
            {
                "record_id": record_id,
                "sample_id": row.get("sample_id"),
                "source_image_id": row.get("source_image_id"),
                "task": row.get("task"),
                "image_relpath": row.get("image_relpath"),
                "oom_events": 0,
                "recovered_occurrences": 0,
                "final_failed_occurrences": 0,
                "model_rollouts": [],
            },
        )
        item["oom_events"] += events
        item["recovered_occurrences"] += int(bool(row.get("oom_recovered")))
        item["final_failed_occurrences"] += int(bool(row.get("oom_final_failure")))
        item["model_rollouts"].append({"model_id": model, "rollout_id": rollout})
    unique_rows = sorted(unique.values(), key=lambda row: row["record_id"])
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "raw_records_scanned": raw_record_count,
        "oom_total_count": oom_total,
        "oom_recovered_count": recovered,
        "oom_final_failed_count": final_failed,
        "oom_affected_rollout_records": len(affected_rows),
        "unique_oom_record_ids": len(unique_rows),
        "by_model_rollout_oom_events": [
            {"model_id": model, "rollout_id": rollout, "oom_events": count}
            for (model, rollout), count in sorted(by_model_rollout.items())
        ],
        "unique_oom_samples": unique_rows,
    }
    atomic_json(output_root / "reports" / "oom_summary.json", summary)
    atomic_jsonl(output_root / "selection" / "oom_samples.jsonl", unique_rows)
    atomic_jsonl(output_root / "reports" / "oom_affected_rollouts.jsonl", affected_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> int:
    run(parse_args().output_root.expanduser().resolve(strict=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
