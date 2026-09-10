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
CROP_STAGES = ("cache-prepare", "cache", "cache-finalize", "finalize")


def parse_args(argv=None, *, alignment_crops_run=False):
    p = argparse.ArgumentParser(description=__doc__)
    data_root, parent_root = alignment_default_roots()
    profile, output = PROFILE, OUTPUT
    if alignment_crops_run:
        from ui14_alignment_crops_common import default_roots, PROFILE as CROPS_PROFILE, OUTPUT as CROPS_OUTPUT
        data_root, parent_root = default_roots()
        profile, output = CROPS_PROFILE, CROPS_OUTPUT
    p.add_argument("stage", choices=(*STAGES, *CROP_STAGES) if alignment_crops_run else STAGES)
    p.add_argument("--data-root", default=data_root)
    p.add_argument("--parent-root", default=parent_root)
    p.set_defaults(output_dir=output, profile=profile, alignment_crops_run=alignment_crops_run)
    p.add_argument("--old-run", default=OLD_OUTPUT)
    p.add_argument("--old-manifest", default=OLD_MANIFEST)
    p.add_argument("--task", nargs="+", default=["ui_alignment"])
    p.add_argument("--steps", nargs="+", type=int, default=[2000, 4000])
    p.add_argument("--examples", type=int, default=5)
    p.add_argument("--reuse-audit-root", type=Path,
                   help="Read-only import of completed historical_audit results; source files are never moved or edited")
    p.add_argument("--gpu-devices", default="0", help="Independent evaluation allocation only; formal training remains 4 A800")
    p.add_argument("--answer-grammar", choices=("legacy", "ui14_answer_v1"), default="legacy",
                   help="Independent comparison decoder; default matches the original formal hybrid inference")
    p.add_argument("--resource-group", default="aiai_locate")
    return p.parse_args(argv)


def run(args):
    if getattr(args, "alignment_crops_run", False):
        from ui14_alignment_crops_common import protect_context
        protect_context(args.data_root, args.output_dir)
    print(f"[alignment paths] stage={args.stage}\ndata_root={args.data_root}\n"
          f"parent_root={args.parent_root}\nold_run={args.old_run}\n"
          f"old_manifest={args.old_manifest}\noutput_dir={args.output_dir}", flush=True)
    for name, selected in (("UI14_DATA_ROOT", args.data_root), ("UI14_PARENT_DATA_ROOT", args.parent_root)):
        inherited = os.environ.get(name)
        if inherited and Path(inherited).resolve() != Path(selected).resolve():
            print(f"[alignment paths] ignored inherited {name}={inherited}; "
                  "use UI14_ALIGNMENT_* or explicit --data-root/--parent-root", flush=True)
    root = alignment_destination(args.data_root, args.parent_root, args.old_run, args.output_dir)
    independent_output(args.output_dir, args.parent_root, args.old_run, NEG_OUTPUT)
    if args.stage == "prepare": require_neg11_parent(args.parent_root)
    if args.reuse_audit_root and args.stage != "audit-errors":
        raise ValueError("--reuse-audit-root is only valid for audit-errors")
    if args.stage == "status":
        stages = (*STAGES, *CROP_STAGES) if getattr(args, "alignment_crops_run", False) else STAGES
        result = {stage: read_json(root / "stage_summaries" / f"{stage}.json")
                  if (root / "stage_summaries" / f"{stage}.json").is_file() else {"status": "not_run"} for stage in stages}
        import json
        print(json.dumps(result, ensure_ascii=False, indent=2)); return result
    with preparation_lock(root, filename=".alignment-stage.lock"), ProgressSession("alignment-" + args.stage, root) as progress:
        try:
            if args.stage in CROP_STAGES:
                if args.stage == "finalize":
                    from ui14_alignment_crops_data import finalize_crops
                    with verification_session(root): result = finalize_crops(root, args.output_dir)
                else:
                    report = read_json(root / "cpu_check_report.json")
                    if report.get("profile") != args.profile: raise ValueError("Run this experiment's CPU prepare first")
                    if any(Path(args.output_dir).glob("checkpoint-*")): raise ValueError("Cannot rebuild input caches after this run started training")
                    steps = {"cache-prepare": "prepare", "cache": "detect", "cache-finalize": "crops"}
                    command = [sys.executable, str(PROJECT_ROOT / "scripts/prepare_ui14_detector_crops.py"),
                               "--stage", steps[args.stage], "--data-root", str(root),
                               "--tasks", "ui_alignment", "--local-task-inputs", "--gpus", "0,1,2,3"]
                    progress.delegated = True
                    try: subprocess.run(command, check=True)
                    finally: progress.delegated = False
                    result = {"command": command, "tasks": ["ui_alignment"], "splits": ["train", "test"]}
            elif args.stage in ("audit-errors", "prepare", "eval-prepare"):
                with verification_session(root):
                    if args.stage == "audit-errors":
                        if args.reuse_audit_root:
                            from ui14_alignment_audit_reuse import import_completed_audits
                            imported = import_completed_audits(args.reuse_audit_root, root / "historical_audit",
                                                               args.task, args.steps)
                            write_json(root / "historical_audit/import_summary.json", imported)
                        from ui14_alignment_audit import audit_runs
                        result = audit_runs(args.old_run, args.old_manifest, args.task, args.steps,
                                            root / "historical_audit", examples=args.examples)
                        if any(r["status"] != "complete" for r in result):
                            raise ValueError("Audit/scorer discrepancy; inspect historical_audit/summary.csv and per-step summary.json")
                    elif args.stage == "prepare":
                        from ui14_alignment_data import prepare_frozen_data
                        result = prepare_frozen_data(args.parent_root, root, args.output_dir,
                                                     profile=getattr(args, "profile", PROFILE))
                    else:
                        from ui14_alignment_eval import prepare_comparisons
                        result = prepare_comparisons(args.old_run, args.old_manifest, args.task, args.steps, root,
                                                     grammar=args.answer_grammar)
            elif args.stage == "eval-existing":
                from ui14_alignment_eval import run_comparison
                plan = root / "comparison_plan.json"
                if not plan.is_file(): raise ValueError("Run eval-prepare on CPU before allocating an evaluation GPU")
                selected = []
                for folder in read_json(plan)["directories"]:
                    binding = read_json(Path(folder) / "binding.json")
                    if (binding["step"] in args.steps and binding["tasks"] == args.task
                            and binding["decoder_contract"]["policy"] == args.answer_grammar): selected.append(folder)
                if len(selected) != len(set(args.steps)): raise ValueError("CPU comparison plan does not cover requested tasks/steps")
                result = [run_comparison(folder, gpus=args.gpu_devices) for folder in selected]
            elif args.stage == "check":
                from ui14_profile import validate_prepared_profile
                validate_prepared_profile({"UI14_DATA_ROOT": str(root), "INIT_CHECKPOINT": INIT_CHECKPOINT,
                                           "OUTPUT_DIR": args.output_dir})
                result = {"status": "ready", "report": str(root / "cpu_check_report.json")}
            else:
                command = [sys.executable, str(PROJECT_ROOT / "scripts/submit_locany_ui5.py"),
                           "--profile", getattr(args, "profile", PROFILE), "--machine", "a800", "--resource-group", args.resource_group,
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
