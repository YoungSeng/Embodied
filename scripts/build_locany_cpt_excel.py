#!/usr/bin/env python3
"""Rebuild the optional three-sheet CPT workbook from JSON/JSONL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eaglevl.train.cpt_excel import build_cpt_workbook


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    written = build_cpt_workbook(args.diagnostics_dir, args.output)
    print(f"CPT workbook written={written}")
    # Excel is deliberately optional; failure must not make orchestration mark
    # an otherwise valid CPT run as failed.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
