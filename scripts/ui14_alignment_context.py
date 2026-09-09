#!/usr/bin/env python3
"""Isolated UI14 alignment diagnosis, decoding evaluation and formal training."""
import argparse
from pathlib import Path
import subprocess
import sys
import os
from ui14_common import read_json, write_json, INIT_CHECKPOINT, PROJECT_ROOT
from ui14_alignment_common import *
from ui14_progress import ProgressSession
from ui14_verification import preparation_lock, verification_session

STAGES = ("audit-errors", "prepare", "check", "eval-prepare", "eval-existing", "render", "submit", "status")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=STAGES)
    p.add_argument("--data-root", default=os.environ.get("UI14_DATA_ROOT", DATA))
    p.add_argument("--parent-root", default=os.environ.get("UI14_PARENT_DATA_ROOT", NEG_DATA))
    p.set_defaults(output_dir=OUTPUT)
    p.add_argument("--old-run", default=OLD_OUTPUT)
    p.add_argument("--old-manifest", default=OLD_MANIFEST)
    p.add_argument("--task", nargs="+", default=["ui_alignment"])
    p.add_argument("--steps", nargs="+", type=int, default=[2000, 4000])
    p.add_argument("--examples", type=int, default=5)
    p.add_argument("--gpu-devices", default="0", help="Independent evaluation allocation only; formal training remains 4 A800")
    p.add_argument("--resource-group", default="aiai_locate")
    return p.parse_args(argv)


def run(args):
    root = independent_output(args.data_root, args.parent_root, args.old_run, NEG_OUTPUT, args.output_dir)
    independent_output(args.output_dir, args.parent_root, args.old_run, NEG_OUTPUT)
    if args.stage == "status":
        result = {stage: read_json(root / "stage_summaries" / f"{stage}.json")
                  if (root / "stage_summaries" / f"{stage}.json").is_file() else {"status": "not_run"} for stage in STAGES}
        import json
        print(json.dumps(result, ensure_ascii=False, indent=2)); return result
    with preparation_lock(root, filename=".alignment-stage.lock"), ProgressSession("alignment-" + args.stage, root):
        try:
            if args.stage in ("audit-errors", "prepare", "eval-prepare"):
                with verification_session(root):
                    if args.stage == "audit-errors":
                        from ui14_alignment_audit import audit_runs
                        result = audit_runs(args.old_run, args.old_manifest, args.task, args.steps,
                                            root / "historical_audit", examples=args.examples)
                        if any(r["status"] != "complete" for r in result):
                            raise ValueError("Audit/scorer discrepancy; inspect historical_audit/summary.csv and per-step summary.json")
                    elif args.stage == "prepare":
                        from ui14_alignment_data import prepare_frozen_data
                        result = prepare_frozen_data(args.parent_root, root, args.output_dir)
                    else:
                        from ui14_alignment_eval import prepare_comparisons
                        result = prepare_comparisons(args.old_run, args.old_manifest, args.task, args.steps, root)
            elif args.stage == "eval-existing":
                from ui14_alignment_eval import run_comparison
                plan = root / "comparison_plan.json"
                if not plan.is_file(): raise ValueError("Run eval-prepare on CPU before allocating an evaluation GPU")
                selected = []
                for folder in read_json(plan)["directories"]:
                    binding = read_json(Path(folder) / "binding.json")
                    if binding["step"] in args.steps and binding["tasks"] == args.task: selected.append(folder)
                if len(selected) != len(set(args.steps)): raise ValueError("CPU comparison plan does not cover requested tasks/steps")
                result = [run_comparison(folder, gpus=args.gpu_devices) for folder in selected]
            elif args.stage == "check":
                from ui14_profile import validate_prepared_profile
                validate_prepared_profile({"UI14_DATA_ROOT": str(root), "INIT_CHECKPOINT": INIT_CHECKPOINT,
                                           "OUTPUT_DIR": args.output_dir})
                result = {"status": "ready", "report": str(root / "cpu_check_report.json")}
            else:
                command = [sys.executable, str(PROJECT_ROOT / "scripts/submit_locany_ui5.py"),
                           "--profile", PROFILE, "--machine", "a800", "--resource-group", args.resource_group,
                           "--gpus", "4", "--ui14-data-root", str(root)]
                if args.stage == "render": command.append("--render-only")
                subprocess.run(command, check=True)
                result = {"command": command, "render_only": args.stage == "render"}
            write_json(root / "stage_summaries" / f"{args.stage}.json", {"status": "complete", "result": result})
            return result
        except BaseException as exc:
            write_json(root / "stage_summaries" / f"{args.stage}.json",
                       {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            raise


if __name__ == "__main__":
    run(parse_args())
