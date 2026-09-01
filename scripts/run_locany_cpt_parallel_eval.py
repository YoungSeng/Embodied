#!/usr/bin/env python3
"""Evaluate the ten CPT held-out tasks with one dynamic worker per GPU."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for import_root in (PROJECT_ROOT, SCRIPTS_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from eaglevl.train.cpt_checkpoint_files import ensure_local_checkpoint_files  # noqa: E402
from eaglevl.train.cpt_observability import CPT_TASKS  # noqa: E402
from locany_ui5_common import parse_gpu_devices  # noqa: E402
from run_ui5_parallel_inference import (  # noqa: E402
    atomic_write_json,
    print_failure_log,
    run_priority_gpu_tasks,
)
import eval_locany_cpt_learning as evaluator  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--processor-path", type=Path, default=None)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-devices", required=True)
    parser.add_argument("--tasks", nargs="+", choices=CPT_TASKS, default=list(CPT_TASKS))
    parser.add_argument("--samples-per-task", type=int, default=200)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--evaluator-script",
        type=Path,
        default=PROJECT_ROOT / "scripts/eval_locany_cpt_learning.py",
    )
    parser.add_argument("--metrics-jsonl", type=Path, default=None)
    parser.add_argument("--train-metrics-jsonl", type=Path, default=None)
    parser.add_argument("--base-cache-dir", type=Path, default=None)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--vision-attn-implementation", default="flash_attention_2")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--iou-threshold", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--task-retries", type=int, default=1)
    parser.add_argument("--failure-log-lines", type=int, default=160)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def valid_fragment(path: Path, task: str, step: int) -> bool:
    try:
        value = read_json(path)
        summary = value["summary"]
        return (
            value.get("status") == "complete"
            and value.get("evaluator_protocol_version")
            == evaluator.EVALUATOR_PROTOCOL_VERSION
            and value.get("tasks") == [task]
            and int(summary.get("step")) == step
            and set(summary["checkpoint_metrics"]["per_task"]) == {task}
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def merge_model_summaries(
    fragments: list[dict[str, Any]], model_key: str, split: str, iou_threshold: float
) -> dict[str, Any]:
    per_task: dict[str, Any] = {}
    errors: Counter[str] = Counter()
    examples = successful = 0
    for fragment in fragments:
        model = fragment["summary"][model_key]
        per_task.update(model.get("per_task", {}))
        errors.update(model.get("errors", {}))
        examples += int(model.get("examples") or 0)
        successful += int(model.get("successful") or 0)
    loss_sum = sum(
        float(metrics.get("eval_main_loss_sum") or 0.0) for metrics in per_task.values()
    )
    loss_tokens = sum(
        int(metrics.get("eval_main_loss_tokens") or 0) for metrics in per_task.values()
    )
    task_ces = [
        float(metrics["eval_main_token_ce"])
        for metrics in per_task.values()
        if isinstance(metrics.get("eval_main_token_ce"), (int, float))
    ]
    macro = evaluator.task_macro_primary(per_task)
    return {
        "split": split,
        "iou_threshold": iou_threshold,
        "examples": examples,
        "successful": successful,
        "errors": dict(sorted(errors.items())),
        "micro_primary": evaluator.micro_primary(per_task),
        "eval_main_token_ce": loss_sum / loss_tokens if loss_tokens else None,
        "eval_main_loss_tokens": loss_tokens,
        "task_macro_eval_main_token_ce": (
            sum(task_ces) / len(task_ces) if task_ces else None
        ),
        "heldout_task_macro_primary": macro if split == "heldout" else None,
        "train_pool_task_macro_primary": macro if split == "train_pool" else None,
        "per_task": per_task,
    }


def concatenate(paths: list[Path], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            for path in paths:
                if not path.is_file():
                    continue
                payload = path.read_text(encoding="utf-8")
                output.write(payload)
                if payload and not payload.endswith("\n"):
                    output.write("\n")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    started = time.time()
    if args.checkpoint_step < 0 or args.samples_per_task <= 0 or args.task_retries < 0:
        raise ValueError("step, samples, and retries must be non-negative (samples > 0)")
    for name in (
        "checkpoint", "base_model", "recipe", "manifest", "run_dir", "output_dir",
        "evaluator_script",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.processor_path = (args.processor_path or args.base_model).expanduser().resolve()
    args.metrics_jsonl = (
        args.metrics_jsonl.expanduser().resolve()
        if args.metrics_jsonl is not None
        else args.run_dir / "diagnostics/cpt_eval_metrics.jsonl"
    )
    args.train_metrics_jsonl = (
        args.train_metrics_jsonl.expanduser().resolve()
        if args.train_metrics_jsonl is not None
        else args.run_dir / "diagnostics/cpt_train_metrics.jsonl"
    )
    args.base_cache_dir = (
        args.base_cache_dir.expanduser().resolve()
        if args.base_cache_dir is not None
        else args.run_dir / "eval/base_cache"
    )
    for path, kind in (
        (args.checkpoint, "dir"), (args.base_model, "dir"),
        (args.processor_path, "dir"), (args.recipe, "file"),
        (args.manifest, "file"), (args.evaluator_script, "file"),
    ):
        exists = path.is_dir() if kind == "dir" else path.is_file()
        if not exists:
            raise FileNotFoundError(path)
    # Repair checkpoint Python/config files once, before concurrent workers.
    ensure_local_checkpoint_files(args.checkpoint, args.base_model)
    gpus = parse_gpu_devices(args.gpu_devices)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fragment_dir = args.output_dir / "fragments"
    worker_dir = args.output_dir / "worker_tasks"
    log_dir = args.output_dir / "worker_logs"
    for path in (fragment_dir, worker_dir, log_dir, args.base_cache_dir):
        path.mkdir(parents=True, exist_ok=True)

    requested = list(dict.fromkeys(args.tasks))
    pending = []
    fragments: dict[str, Path] = {}
    for task in requested:
        fragment = fragment_dir / f"{task}.json"
        fragments[task] = fragment
        if args.force or not valid_fragment(fragment, task, args.checkpoint_step):
            pending.append(task)
    # Long tasks are queued first; prior measured wall time refines later checkpoints.
    runtime_profile_path = args.run_dir / "diagnostics/cpt_heldout_runtime_profile.json"
    runtime_profile: dict[str, Any] = {}
    if runtime_profile_path.is_file():
        try:
            runtime_profile = read_json(runtime_profile_path)
        except (OSError, ValueError, json.JSONDecodeError):
            runtime_profile = {}
    estimates = {
        task: float(
            runtime_profile.get("tasks", {}).get(task, {}).get(
                "elapsed_seconds", args.samples_per_task
            )
        )
        for task in pending
    }

    def run_task(task: str, gpu: str, attempt: int) -> dict[str, Any]:
        task_output = worker_dir / task
        fragment = fragments[task]
        log_path = log_dir / f"{task}.log"
        command = [
            str(args.python), str(args.evaluator_script),
            "--checkpoint", str(args.checkpoint),
            "--checkpoint-step", str(args.checkpoint_step),
            "--base-model", str(args.base_model),
            "--processor-path", str(args.processor_path),
            "--recipe", str(args.recipe),
            "--manifest", str(args.manifest),
            "--eval-split", "heldout",
            "--subset-strategy", "hash",
            "--samples-per-task", str(args.samples_per_task),
            "--tasks", task,
            "--output-dir", str(task_output),
            "--output-fragment", str(fragment),
            "--gpu-device", gpu,
            "--base-cache-dir", str(args.base_cache_dir),
            "--train-metrics-jsonl", str(args.train_metrics_jsonl),
            "--metrics-jsonl", str(args.metrics_jsonl),
            "--device", "cuda:0",
            "--dtype", args.dtype,
            "--attn-implementation", args.attn_implementation,
            "--vision-attn-implementation", args.vision_attn_implementation,
            "--max-new-tokens", str(args.max_new_tokens),
            "--iou-threshold", str(args.iou_threshold),
            "--seed", str(args.seed),
            "--teacher-forced", "--fail-fast-inference-errors",
        ]
        if args.checkpoint_step > 0:
            command.append("--skip-base-if-cached")
        print(
            f"[CPT EVAL START] task={task} attempt={attempt} physical_gpu={gpu} "
            f"command={shlex.join(command)}",
            flush=True,
        )
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["PYTHONUNBUFFERED"] = "1"
        task_started = time.time()
        with log_path.open("a", encoding="utf-8") as log:
            log.write(
                f"\n===== {utc_now()} task={task} gpu={gpu} attempt={attempt} =====\n"
            )
            log.write(shlex.join(command) + "\n")
            log.flush()
            process = subprocess.run(
                command, cwd=PROJECT_ROOT, env=env, stdout=log,
                stderr=subprocess.STDOUT, check=False,
            )
        return_code = process.returncode
        if return_code == 0 and not valid_fragment(fragment, task, args.checkpoint_step):
            return_code = 90
        tail = ""
        if return_code:
            tail = print_failure_log(
                f"CPT task={task} GPU={gpu}", log_path, args.failure_log_lines
            )
        return {
            "task": task,
            "physical_gpu": gpu,
            "logical_device": "cuda:0",
            "return_code": return_code,
            "command": command,
            "log_path": str(log_path),
            "fragment": str(fragment),
            "elapsed_seconds": time.time() - task_started,
            "log_tail": tail,
        }

    results = run_priority_gpu_tasks(
        tasks=pending,
        gpu_devices=gpus,
        estimates=estimates,
        runner=run_task,
        retries=args.task_retries,
        continue_on_failure=True,
    ) if pending else {}

    completed = [
        task for task in requested
        if valid_fragment(fragments[task], task, args.checkpoint_step)
    ]
    failed = [task for task in requested if task not in completed]
    fragment_values = [read_json(fragments[task]) for task in completed]
    summary: dict[str, Any] | None = None
    metric_rows: list[dict[str, Any]] = []
    if fragment_values:
        first = fragment_values[0]["summary"]
        base = merge_model_summaries(fragment_values, "base", "heldout", args.iou_threshold)
        checkpoint = merge_model_summaries(
            fragment_values, "checkpoint_metrics", "heldout", args.iou_threshold
        )
        summary = {
            "schema_version": 4,
            "evaluation_kind": "heldout_generalization",
            "eligible_for_best_checkpoint": not failed,
            "split": "heldout",
            "manifest": str(args.manifest),
            "manifest_id": first["manifest_id"],
            "subset_strategy": "hash",
            "seed": args.seed,
            "samples_per_task": args.samples_per_task,
            "iou_threshold": args.iou_threshold,
            "task_counts": {
                task: int(fragment_values[index]["summary"]["task_counts"].get(task, 0))
                for index, task in enumerate(completed)
            },
            "base_model": str(args.base_model),
            "checkpoint": str(args.checkpoint),
            "step": args.checkpoint_step,
            "teacher_forced": True,
            "base": base,
            "checkpoint_metrics": checkpoint,
            "checkpoint_minus_base": evaluator.metric_delta(base, checkpoint),
            "failed_tasks": failed,
            "complete_ten_task_heldout": set(completed) == set(CPT_TASKS),
            "eval_wall_time_seconds": time.time() - started,
        }
        metric_args = SimpleNamespace(
            checkpoint_step=args.checkpoint_step,
            checkpoint=str(args.checkpoint),
            train_metrics_jsonl=args.train_metrics_jsonl,
            eval_split="heldout",
            subset_strategy="hash",
            seed=args.seed,
            samples_per_task=args.samples_per_task,
            max_new_tokens=args.max_new_tokens,
            teacher_forced=True,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
            vision_attn_implementation=args.vision_attn_implementation,
            iou_threshold=args.iou_threshold,
            processor_path=str(args.processor_path),
            base_model=str(args.base_model),
        )
        history = evaluator._read_jsonl_rows(args.metrics_jsonl)
        metric_rows, selection = evaluator.build_eval_metric_rows(
            metric_args, summary, history
        )
        summary["checkpoint_selection"] = selection
        for row in metric_rows:
            row["eval_wall_time_seconds"] = summary["eval_wall_time_seconds"]
            row["evaluation_status"] = "complete" if not failed else "incomplete"
            row["failed_tasks"] = failed
        summary["evaluation_protocol_id"] = metric_rows[0]["evaluation_protocol_id"]
        concatenate(
            [worker_dir / task / "predictions.jsonl" for task in completed],
            args.output_dir / "predictions.jsonl",
        )
        concatenate(
            [worker_dir / task / "errors_by_task.jsonl" for task in completed],
            args.output_dir / "errors_by_task.jsonl",
        )
        concatenate(
            [worker_dir / task / "qualitative_samples.md" for task in completed],
            args.output_dir / "qualitative_samples.md",
        )
        referring_review = worker_dir / "referring_kg" / "manual_review_referring_kg.jsonl"
        if referring_review.is_file():
            concatenate(
                [referring_review],
                args.output_dir / "manual_review_referring_kg.jsonl",
            )
        evaluator.write_eval_metric_rows(args.output_dir, metric_rows, args.metrics_jsonl)
        atomic_write_json(args.output_dir / "summary.json", summary)

    status = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint),
        "step": args.checkpoint_step,
        "gpu_devices": gpus,
        "requested_tasks": requested,
        "completed_tasks": completed,
        "failed_tasks": failed,
        "tasks": results,
        "complete_ten_task_heldout": set(completed) == set(CPT_TASKS),
        "success": not failed,
        "summary": str(args.output_dir / "summary.json") if summary else None,
        "metrics_jsonl": str(args.metrics_jsonl),
        "eval_wall_time_seconds": time.time() - started,
        "finished_at": utc_now(),
    }
    atomic_write_json(args.output_dir / "parallel_heldout_status.json", status)
    profile_tasks = dict(runtime_profile.get("tasks", {}))
    for task, result in results.items():
        if int(result.get("return_code", 1)) == 0:
            profile_tasks[task] = {
                "elapsed_seconds": result.get("elapsed_seconds"),
                "sample_count": args.samples_per_task,
                "physical_gpu": result.get("physical_gpu"),
                "updated_at": utc_now(),
            }
    atomic_write_json(
        runtime_profile_path,
        {"schema_version": 1, "tasks": profile_tasks},
    )
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
