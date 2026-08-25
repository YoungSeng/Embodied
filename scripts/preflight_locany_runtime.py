#!/usr/bin/env python3
"""Fail-fast runtime check for UI5 inference dependencies and processor loading."""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path


EXIT_IMPORT_FAILED = 41
EXIT_LIBGL_MISSING = 42
EXIT_PROCESSOR_FAILED = 43


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processor-path", type=Path, required=True)
    parser.add_argument("--skip-processor", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("===== LocateAnything inference runtime preflight =====", flush=True)
    try:
        import cv2
    except Exception as exc:
        traceback.print_exc()
        print(
            f"[PREFLIGHT ERROR] cv2 import failed: {type(exc).__name__}: {exc}",
            flush=True,
        )
        if "libGL.so.1" in str(exc):
            print(
                "[PREFLIGHT ERROR] Missing libGL.so.1; install libgl1 and "
                "libglib2.0-0 in this task container.",
                flush=True,
            )
            return EXIT_LIBGL_MISSING
        return EXIT_IMPORT_FAILED

    print(f"cv2 version              : {cv2.__version__}", flush=True)
    print(f"cv2 module               : {cv2.__file__}", flush=True)
    if args.skip_processor:
        print("processor check          : skipped", flush=True)
        print("======================================================", flush=True)
        return 0

    processor_path = args.processor_path.expanduser().resolve()
    try:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            processor_path,
            trust_remote_code=True,
            local_files_only=True,
        )
    except Exception as exc:
        traceback.print_exc()
        print(
            "[PREFLIGHT ERROR] AutoProcessor load failed: "
            f"path={processor_path}; {type(exc).__name__}: {exc}",
            flush=True,
        )
        return EXIT_PROCESSOR_FAILED

    print(f"processor path           : {processor_path}", flush=True)
    print(
        "processor class          : "
        f"{type(processor).__module__}.{type(processor).__name__}",
        flush=True,
    )
    print("runtime preflight        : PASSED", flush=True)
    print("======================================================", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
