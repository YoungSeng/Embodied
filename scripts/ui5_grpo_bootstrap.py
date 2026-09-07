#!/usr/bin/env python3
"""Stdlib-only handoff to the recorded conda before importing training tools."""
import argparse
import json
import os
from pathlib import Path
import sys

PREVIOUS = Path("/mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace/gui_logs/ui5_curriculum/"
                "locany-ui5-crop-rollout4-curriculum-hour021-h20x2-sdpa7268-20260906T073110Z-746aed")


def recorded_source(previous):
    directory = previous.resolve(strict=True)
    root, seen = directory.parent, set()
    while True:
        if directory in seen:
            raise ValueError("submission successor cycle")
        seen.add(directory)
        names = ("caption-retry.started", "detail-audit-restart.started", "storage-restart.started",
                 "curriculum-v3-text-v3-1.started")
        pointers = [directory / name for name in names if (directory / name).exists()]
        if not pointers:
            return directory / "snapshot-switch.json"
        if len(pointers) != 1:
            raise ValueError("ambiguous recorded submission successor")
        following = Path(pointers[0].read_text(encoding="utf-8").strip()).resolve(strict=True)
        state = json.loads(following.read_text(encoding="utf-8"))
        current = directory / "snapshot-switch.json"
        parent_key = "source_submission" if pointers[0].name == "curriculum-v3-text-v3-1.started" else "retry_of"
        if (following.name != "snapshot-switch.json" or following.parent.parent != root
                or Path(state[parent_key]).resolve() != current.resolve()
                or state["snapshot"] != json.loads(current.read_text(encoding="utf-8"))["snapshot"]):
            raise ValueError("submission successor changed its source/frozen cutoff")
        directory = following.parent


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--previous-submission-dir", type=Path, default=PREVIOUS)
    args, _ = parser.parse_known_args()
    source = recorded_source(args.previous_submission_dir)
    env = json.loads(source.read_text(encoding="utf-8"))["runtime"]
    python = env.get("PYTHON_BIN") or str(Path(env["ENV_DIR"]) / "bin/python")
    os.environ["PATH"] = str(Path(env["ENV_DIR"]) / "bin") + os.pathsep + os.environ.get("PATH", "")
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(python, [python, str(Path(__file__).with_name("submit_ui5_grpo_mixed.py")), *sys.argv[1:]])


if __name__ == "__main__":
    main()
