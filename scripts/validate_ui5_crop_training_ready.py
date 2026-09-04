#!/usr/bin/env python3
"""Fail closed unless a UI5 crop audit marker matches all live report inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from run_ui5_crop_audit import validate_training_ready_marker


TRAIN_MODES = ("full_only", "crop_only")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument(
        "--recipe",
        type=Path,
        default=None,
        help="Require this exact audited recipe path and digest (v4)",
    )
    parser.add_argument(
        "--expected-train-mode",
        choices=TRAIN_MODES,
        required=True,
        help="Require the audited recipe to match this crop training mode exactly",
    )
    return parser.parse_args(argv)


def validate_expected_train_mode(
    result: Mapping[str, Any], expected_train_mode: str
) -> None:
    """Reject a valid audited recipe when it is for a different train mode."""

    audited_train_mode = result.get("crop_train_mode")
    if audited_train_mode != expected_train_mode:
        raise RuntimeError(
            "audited crop recipe train mode mismatch: "
            f"expected {expected_train_mode!r}, got {audited_train_mode!r}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = validate_training_ready_marker(args.audit_dir, recipe_path=args.recipe)
    validate_expected_train_mode(result, args.expected_train_mode)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("crop_train_mode") == "crop_only":
        print(f"crop records={result['crop_only_region_records']}")
        print(f"active crop retention={result['active_crop_retention_policy']}")
        print(f"negative crop records={result['crop_only_negative_records']}")
        print(
            "full image records by local task="
            f"{result['crop_only_local_task_full_image_records']}"
        )
        print(
            "content_missing global records="
            f"{result['crop_only_content_missing_global_records']}"
        )
        print(
            "five-task positive/negative="
            + json.dumps(
                result["crop_only_positive_negative_by_task"],
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
