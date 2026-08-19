#!/usr/bin/env python3
"""Resolve and print final LocateAnything UI5 runtime configuration."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

from locany_ui5_common import DEFAULT_CONFIG_PATH, resolve_runtime_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--format", choices=("json", "shell"), default="json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    resolved = resolve_runtime_config(config_path=args.config)
    if args.format == "json":
        print(json.dumps(resolved, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        for key in sorted(resolved):
            print(f"export {key}={shlex.quote(str(resolved[key]))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
