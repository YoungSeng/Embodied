#!/usr/bin/env python3
"""CPU-only recovery of the model-only step-0 startup ordering failure.

No checkpoint, evaluation result, or workbook is modified here. Legacy result
reuse across commits requires unchanged inference/scoring code and runtime
configuration, plus an untouched deterministic checkpoint-0 export.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import re
import subprocess

from ui14_common import PROJECT_ROOT, UI_TASKS, read_json, write_json


def validate_initial_export(runtime, snapshot, checkpoint):
    from eaglevl.train.ui5_checkpoint_utils import has_model_weights
    from eaglevl.ui_task_registry import load_registry, validate_registry

    checkpoint = Path(checkpoint)
    error = "Unbound checkpoint-0 is not a matching CPT9000 model-only export"
    try:
        config = read_json(checkpoint / "config.json")
        exported = read_json(checkpoint / "ui5_checkpoint0_manifest.json")
        base = Path(runtime["INIT_CHECKPOINT"]).resolve()
        if checkpoint.name != "checkpoint-0" or not has_model_weights(checkpoint):
            raise ValueError("incomplete model weights")
        for record in (config, exported):
            if record.get("init_cpt_step") != 9000 or Path(record["init_checkpoint"]).resolve() != base:
                raise ValueError("CPT initialization differs")
        if Path(exported["checkpoint"]).resolve() != checkpoint.resolve():
            raise ValueError("export destination differs")
        if (config.get("ui_relation_initialization_seed") != 42
                or config.get("ui_relation_initialization_reason") != "checkpoint-0-export"):
            raise ValueError("missing deterministic export provenance")
        current = load_registry(runtime["UI_TASK_REGISTRY"])
        saved = validate_registry(config.get("ui_task_registry"), config.get("ui_num_tasks"))
        if config.get("ui_num_tasks") != 14 or saved != current:
            raise ValueError("checkpoint/current task registry differs")
        for task in current[5:]:
            for key in ("normalization_id", "repair_run_id"):
                if task.get(key) != snapshot[key]:
                    raise ValueError(f"task registry {key} differs")
        # Even a step-0 directory may contain an interrupted training state.
        patterns = ("trainer_state.json", "training_args.bin", "optimizer*", "scheduler*",
                    "global_step*", "rng_state*", "dataloader_state*")
        if any(any(checkpoint.glob(pattern)) for pattern in patterns):
            raise ValueError("training state present; not a model-only export")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(f"{error}: {exc}") from exc


# These files only control startup/validation, not predictions or scoring.
# All other tracked changes (including model/processor, prompts, sampler,
# export, evaluator, scorer and formal configuration) invalidate legacy reuse.
STARTUP_FILES = frozenset({
    "scripts/ui14_profile.py", "scripts/validate_ui14_ready.py",
    "scripts/ui14_run_recovery.py", "shell/run_locany_ui5_pipeline.sh",
})

PIPELINE_BINDING_BLOCK = '''# Publish the UI14 data binding before checkpoint-0 export/evaluation. This is
# intentionally after the eval-only branch: external evaluation is read-only.
if [[ -n "${UI_TASK_REGISTRY:-}" ]]; then
  echo "[PIPELINE] validating and binding UI14 data before initial evaluation/training"
  "${PIPELINE_PYTHON}" "${PROJECT_ROOT}/scripts/validate_ui14_ready.py"
fi

'''


def pipeline_evaluation_code(source):
    # Only remove this exact binding block and completion-check substitution.
    # The pipeline also builds inference commands: changes to those must not
    # slip through the startup-file allowlist with an unchanged config hash.
    return source.replace("\r\n", "\n").replace(PIPELINE_BINDING_BLOCK, "").replace(
        'scripts/ui14_run_recovery.py" --output-dir', 'scripts/run_ui14_eval.py" --output-dir')


def startup_only_changes(project, previous, current):
    if not all(re.fullmatch(r"[0-9a-fA-F]{40,64}", str(ref)) for ref in (previous, current)):
        raise ValueError("missing full evaluation/current Git commit")

    def git(*args):
        return subprocess.run(["git", "-C", str(project), *args], check=True,
                              text=True, encoding="utf-8", capture_output=True, timeout=30).stdout

    # Compare both committed changes and local tracked modifications. Git errors
    # (e.g. a shallow clone missing the old commit) never grant reuse.
    changed = set(git("diff", "--name-only", previous, current, "--").splitlines())
    dirty = set(git("diff", "--name-only", current, "--").splitlines())
    def relevant(path):
        return path not in STARTUP_FILES and not path.startswith(("tests/", "docs/"))
    unsafe = sorted(p for p in changed | dirty if relevant(p))
    pipeline = "shell/run_locany_ui5_pipeline.sh"
    if pipeline in changed | dirty:
        before = pipeline_evaluation_code(git("show", f"{previous}:{pipeline}"))
        committed = pipeline_evaluation_code(git("show", f"{current}:{pipeline}"))
        working = pipeline_evaluation_code((Path(project) / pipeline).read_text(encoding="utf-8"))
        if before != committed or before != working:
            unsafe.append(pipeline + " (changes beyond initial data binding/completion check)")
    if unsafe:
        raise ValueError("inference/runtime code may have changed: " + ", ".join(unsafe[:8]))
    return sorted(changed | dirty)


def completed_evaluation(output, step, manifest, checkpoint, *, runtime=None, project=PROJECT_ROOT):
    from run_ui14_eval import evaluation_identity, is_complete
    from eaglevl.train.ui5_excel_logger import UI5ExcelLogger

    output, checkpoint = Path(output), Path(checkpoint)
    if is_complete(output, step, manifest, checkpoint):
        print(f"[UI14 eval] reused complete step={step}: 14/14 tasks and 36 Excel rows", flush=True)
        return True
    path = output / "evaluation" / f"ui14-step-{step}.json"
    if step != 0 or not path.is_file():
        return False
    try:
        saved = read_json(path)
        keys = {t.task_key for t in UI_TASKS}
        if saved.get("status") != "success" or set(saved.get("tasks", {})) != keys:
            raise ValueError("step-0 evaluation is incomplete")
        old, current = saved["identity"], evaluation_identity(manifest, checkpoint)
        if not old.get("config_hash") or {k: v for k, v in old.items() if k != "git_commit"} != {
                k: v for k, v in current.items() if k != "git_commit"}:
            raise ValueError("evaluation data/model/runtime identity differs")
        changed = startup_only_changes(project, old.get("git_commit"), current.get("git_commit"))
        env = os.environ if runtime is None else runtime
        snapshot = read_json(Path(env["UI14_DATA_ROOT"]) / "source_snapshot.json")
        validate_initial_export(env, snapshot, checkpoint)
        if any(saved.get(k) != snapshot[k] for k in ("normalization_id", "repair_run_id")):
            raise ValueError("evaluation repair/data binding differs")
        if Path(saved["init_checkpoint"]).resolve() != Path(env["INIT_CHECKPOINT"]).resolve():
            raise ValueError("evaluation CPT initialization differs")
        # Legacy results contain no weight digest. Do not use the architecture
        # signature alone as evidence of unchanged weights: require the entire
        # export to predate inference, including file change times. No big-file
        # reread is necessary; replacing weights after evaluation rejects reuse.
        started = datetime.fromisoformat(saved["started"]).timestamp()
        if datetime.fromisoformat(saved["finished"]).timestamp() < started:
            raise ValueError("invalid evaluation timestamps")
        files = {}
        for file in checkpoint.rglob("*"):
            if not file.is_file() or "__pycache__" in file.parts:
                continue
            stat = file.stat()
            if max(stat.st_mtime, stat.st_ctime) > started:
                raise ValueError(f"checkpoint file changed since evaluation started: {file.name}")
            files[str(file.relative_to(checkpoint))] = {
                "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns}
        workbook = UI5ExcelLogger(output / "diagnostics/ui5_training_evaluation.xlsx", sorted(keys))
        if not workbook.has_eval_step(0):
            raise ValueError("step-0 Excel task set is incomplete")
        # A separate receipt preserves the original evaluation Git provenance,
        # scores, best-checkpoint history, and Excel bytes.
        receipt = {"schema_version": 1, "reason": "startup-only-code-change",
                   "sft_step": 0, "source_identity": old, "current_identity": current,
                   "changed_files": changed, "checkpoint_files": files,
                   "evaluation": str(path), "normalization_id": snapshot["normalization_id"]}
        destination = output / "evaluation/ui14-step-0-startup-reuse.json"
        if not destination.is_file() or read_json(destination) != receipt:
            write_json(destination, receipt)
        print("[UI14 eval] reused complete step=0: 14/14 tasks, 36 Excel rows; "
              "startup-only fix verified; inference skipped", flush=True)
        return True
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"[UI14 eval] step-0 reuse unavailable: {exc}", flush=True)
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    raise SystemExit(0 if completed_evaluation(args.output_dir, args.step, args.manifest, args.checkpoint) else 1)
