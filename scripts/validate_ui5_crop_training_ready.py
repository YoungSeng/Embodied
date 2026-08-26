#!/usr/bin/env python3
"""Fail closed unless a UI5 crop audit marker matches all live report inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from run_ui5_crop_audit import validate_training_ready_marker


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = validate_training_ready_marker(args.audit_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
