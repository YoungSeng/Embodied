"""Resume-safe atomic persistence for UI5 sampling coverage diagnostics."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping


MONOTONIC_FIELDS = (
    "samples_drawn_with_repetition",
    "seen_unique_records",
    "seen_unique_crops",
    "seen_unique_source_images",
    "manual_repair_seen",
)


def is_monotonic_coverage(
    previous: Mapping[str, Any], current: Mapping[str, Any]
) -> bool:
    previous_datasets = previous.get("datasets", [])
    current_datasets = current.get("datasets", [])
    if len(previous_datasets) != len(current_datasets):
        return False
    return all(
        int(current_row.get(field, 0)) >= int(previous_row.get(field, 0))
        for previous_row, current_row in zip(previous_datasets, current_datasets)
        for field in MONOTONIC_FIELDS
    )


def write_sampling_coverage_atomic(
    destination: Path, payload: Mapping[str, Any]
) -> bool:
    """Publish coverage unless it would regress an existing same-step record."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        try:
            previous = json.loads(destination.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"existing sampling coverage is unreadable: {destination}: {exc}"
            ) from exc
        if not is_monotonic_coverage(previous, payload):
            return False
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return True
