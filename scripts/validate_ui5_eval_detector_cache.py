#!/usr/bin/env python3
"""Validate a completed UI5 evaluation detector/scan cache without models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from ui5_eval_detector_cache import validate_eval_detector_cache


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--scan-name", required=True)
    parser.add_argument("--expected-unique-images", type=int, default=0)
    parser.add_argument("--cache-scope", choices=("preview", "full_test"), default=None)
    parser.add_argument("--require-strict-nonoverlap", action="store_true")
    parser.add_argument("--require-detector-edge-alignment", action="store_true")
    parser.add_argument(
        "--require-guarded-bbox-unique-containment", action="store_true"
    )
    parser.add_argument("--require-ready", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    marker = validate_eval_detector_cache(
        args.cache_dir,
        scan_name=args.scan_name,
        expected_unique_images=args.expected_unique_images,
        require_ready=args.require_ready,
        required_cache_scope=args.cache_scope,
        require_strict_nonoverlap=args.require_strict_nonoverlap,
        require_detector_edge_alignment=args.require_detector_edge_alignment,
        require_guarded_bbox_unique_containment=(
            args.require_guarded_bbox_unique_containment
        ),
    )
    print(
        json.dumps(
            {
                "valid": True,
                "scan_name": args.scan_name,
                "content_unique_images": marker.get("dataset", {}).get(
                    "content_unique_images"
                ),
                "cache_scope": marker.get("cache_scope"),
                "strict_vertical_partition": marker.get("geometry", {}).get(
                    "strict_vertical_partition"
                ),
                "detector_edge_aligned": marker.get("geometry", {}).get(
                    "detector_edge_aligned"
                ),
                "guarded_bbox_unique_containment": marker.get("geometry", {}).get(
                    "guarded_bbox_unique_containment"
                ),
                "scan_manifest": marker.get("geometry", {}).get("scan_manifest", {}).get(
                    "path"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
