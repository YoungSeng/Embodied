#!/usr/bin/env python3
"""Run five LocateAnything UI tasks on a configurable pool of physical GPUs."""

from __future__ import annotations

import argparse
import json
import os
import queue
import shlex
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from locany_ui5_common import TASK_JSONL, TASKS, parse_gpu_devices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--processor-path", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-devices", required=True)
    parser.add_argument(
        "--attn-implementation",
        choices=("sdpa", "magi", "flash_attention_2", "eager", "auto"),
        required=True,
    )
    parser.add_argument("--inference-script", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument(
        "--vision-attn-implementation",
        choices=("sdpa", "flash_attention_2", "eager"),
        default="flash_attention_2",
    )
    parser.add_argument(
        "--generation-mode", choices=("fast", "slow", "hybrid"), default="hybrid"
    )
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--n-future-tokens", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--runtime-profile", type=Path, default=None)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--max-images-per-task", type=int, default=0)
    parser.add_argument(
        "--relation-gate-mode", choices=("observe", "hard", "soft"), default="observe"
    )
    parser.add_argument("--relation-gate-threshold", type=float, default=None)
    parser.add_argument(
        "--enable-pbd",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="独立控制 PBD/coordinate bridge，供同 checkpoint 消融使用",
    )
    parser.add_argument(
        "--enable-ui-relation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--save-raw-answer", action="store_true")
    parser.add_argument(
        "--model-load-preflight",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "可选：并行 worker 启动前先在第一张 GPU 上单进程加载一次模型；默认关闭，"
            "直接启动任务 worker"
        ),
    )
    parser.add_argument(
        "--failure-log-lines",
        type=int,
        default=160,
        help="worker 或模型预检失败时回显到主日志的末尾行数",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--task-retries",
        type=int,
        default=0,
        help="retry a failed task this many times on the same physical GPU",
    )
    parser.add_argument(
        "--continue-on-task-failure",
        action="store_true",
        help="allow other queued tasks to finish after a task exhausts retries",
    )
    return parser.parse_args()


def count_jsonl_records(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def load_runtime_profile(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_command(
    args: argparse.Namespace,
    task: str,
    gpu: str,
    summary: Path,
    *,
    load_only: bool = False,
) -> list[str]:
    command = [
        args.python,
        str(args.inference_script),
        "--checkpoint",
        str(args.checkpoint),
        "--processor-path",
        str(args.processor_path),
        "--input-dir",
        str(args.input_dir),
        "--output-dir",
        str(args.output_dir),
        "--summary-path",
        str(summary),
        "--cuda-visible-devices",
        gpu,
        "--device",
        "cuda:0",
        "--attn-implementation",
        args.attn_implementation,
        "--vision-attn-implementation",
        args.vision_attn_implementation,
        "--dtype",
        args.dtype,
        "--generation-mode",
        args.generation_mode,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--n-future-tokens",
        str(args.n_future_tokens),
        "--seed",
        str(args.seed),
        "--tasks",
        task,
        "--skip-figma",
        "--fail-fast",
        "--relation-gate-mode",
        args.relation_gate_mode,
    ]
    command.append("--enable-ui-relation" if args.enable_ui_relation else "--no-enable-ui-relation")
    command.append("--enable-pbd" if args.enable_pbd else "--no-enable-pbd")
    if args.save_raw_answer:
        command.append("--save-raw-answer")
    if args.relation_gate_threshold is not None:
        command.extend(
            ["--relation-gate-threshold", str(args.relation_gate_threshold)]
        )
    if args.overwrite:
        command.append("--overwrite")
    if load_only:
        command.append("--load-only")
    if args.max_images_per_task:
        command.extend(["--max-images-per-task", str(args.max_images_per_task)])
    return command


def read_log_tail(path: Path, line_count: int) -> str:
    """Read a bounded log tail without loading a potentially huge inference log."""
    if line_count <= 0 or not path.is_file():
        return ""
    max_bytes = 256 * 1024
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes), os.SEEK_SET)
            payload = handle.read()
        text = payload.decode("utf-8", errors="replace")
        if size > max_bytes:
            text = text.split("\n", 1)[-1]
        return "\n".join(text.splitlines()[-line_count:])
    except OSError as exc:
        return f"<failed to read log tail: {type(exc).__name__}: {exc}>"


def print_failure_log(label: str, path: Path, line_count: int) -> str:
    tail = read_log_tail(path, line_count)
    print(f"===== BEGIN FAILURE LOG: {label} ({path}) =====", file=sys.stderr, flush=True)
    print(tail or "<log is empty or missing>", file=sys.stderr, flush=True)
    print(f"===== END FAILURE LOG: {label} =====", file=sys.stderr, flush=True)
    return tail


def run_priority_gpu_tasks(
    *,
    tasks: list[str],
    gpu_devices: list[str],
    estimates: dict[str, float],
    runner: Callable[[str, str, int], dict[str, Any]],
    retries: int = 0,
    continue_on_failure: bool = True,
) -> dict[str, dict[str, Any]]:
    """Run dynamically assigned tasks with one subprocess at a time per GPU.

    This is the shared scheduler used by both UI5 and CPT held-out evaluation.
    A retry is deliberately performed inside the same worker so the failed
    task remains on its original physical GPU.
    """
    if retries < 0:
        raise ValueError("retries cannot be negative")
    work_queue: queue.PriorityQueue[tuple[float, str]] = queue.PriorityQueue()
    for task in tasks:
        work_queue.put((-float(estimates.get(task, 0.0)), task))
    lock = threading.Lock()
    stop_event = threading.Event()
    results: dict[str, dict[str, Any]] = {}

    def worker(gpu: str) -> None:
        while not stop_event.is_set():
            try:
                _, task = work_queue.get_nowait()
            except queue.Empty:
                return
            attempts: list[dict[str, Any]] = []
            final: dict[str, Any] | None = None
            try:
                for attempt in range(1, retries + 2):
                    try:
                        current = dict(runner(task, gpu, attempt))
                    except Exception as exc:
                        current = {
                            "return_code": 99,
                            "error": f"scheduler runner raised {type(exc).__name__}: {exc}",
                        }
                    current.setdefault("task", task)
                    current.setdefault("physical_gpu", gpu)
                    current["attempt"] = attempt
                    attempts.append(current)
                    final = current
                    if int(current.get("return_code", 1)) == 0:
                        break
                assert final is not None
                final = dict(final)
                final["attempts"] = attempts
                final["retry_count"] = max(0, len(attempts) - 1)
                with lock:
                    results[task] = final
                if int(final.get("return_code", 1)) != 0 and not continue_on_failure:
                    stop_event.set()
            finally:
                work_queue.task_done()

    threads = [
        threading.Thread(target=worker, args=(gpu,), name=f"gpu-{gpu}", daemon=False)
        for gpu in gpu_devices
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results


def main() -> int:
    args = parse_args()
    if args.max_images_per_task < 0:
        raise ValueError("--max-images-per-task cannot be negative")
    if args.failure_log_lines < 0:
        raise ValueError("--failure-log-lines cannot be negative")
    if args.task_retries < 0:
        raise ValueError("--task-retries cannot be negative")
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.processor_path = args.processor_path.expanduser().resolve()
    args.input_dir = args.input_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.inference_script = args.inference_script.expanduser().resolve()
    if args.runtime_profile is not None:
        args.runtime_profile = args.runtime_profile.expanduser().resolve()
    for path, label in (
        (args.checkpoint, "checkpoint"),
        (args.processor_path, "processor path"),
        (args.input_dir, "input directory"),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if not args.inference_script.is_file():
        raise FileNotFoundError(f"inference script does not exist: {args.inference_script}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite and not args.dry_run:
        removed = 0
        for task in args.tasks:
            task_dir = args.output_dir / task
            if task_dir.is_dir():
                for artifact in task_dir.rglob("*"):
                    if artifact.is_file():
                        artifact.unlink()
                        removed += 1
        for name in ("_run_manifest.json", "_summary.json", "parallel_inference_status.json"):
            target = args.output_dir / name
            if target.is_file():
                target.unlink()
                removed += 1
        print(f"[OVERWRITE] removed {removed} stale inference artifacts before workers")

    gpus = parse_gpu_devices(args.gpu_devices)
    counts: dict[str, int] = {}
    for task in args.tasks:
        jsonl_path = args.input_dir / TASK_JSONL[task]
        if not jsonl_path.is_file():
            raise FileNotFoundError(f"Missing test JSONL for {task}: {jsonl_path}")
        counts[task] = count_jsonl_records(jsonl_path)
        if counts[task] <= 0:
            raise RuntimeError(f"Test JSONL is empty for {task}: {jsonl_path}")

    previous_profile = load_runtime_profile(args.runtime_profile)
    estimates: dict[str, float] = {}
    for task, count in counts.items():
        old = previous_profile.get("tasks", {}).get(task, {})
        previous_elapsed = old.get("elapsed_seconds")
        previous_count = old.get("sample_count")
        if isinstance(previous_elapsed, (int, float)) and previous_elapsed > 0:
            if isinstance(previous_count, int) and previous_count > 0:
                estimates[task] = float(previous_elapsed) * count / previous_count
            else:
                estimates[task] = float(previous_elapsed)
        else:
            estimates[task] = float(count)

    ordered = sorted(args.tasks, key=lambda task: (-estimates[task], task))
    print("===== UI5 parallel inference scheduler =====")
    print(f"physical GPUs       : {','.join(gpus)}")
    print(f"logical device      : cuda:0 in every subprocess")
    for task in ordered:
        source = "runtime profile" if previous_profile.get("tasks", {}).get(task) else "sample count"
        print(
            f"{task:16s}: records={counts[task]:6d}, "
            f"estimated={estimates[task]:.2f} ({source})"
        )

    summaries_dir = args.output_dir / "_worker_summaries"
    logs_dir = args.output_dir / "_worker_logs"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    preflight_result: dict[str, Any] | None = None
    if args.model_load_preflight:
        preflight_gpu = gpus[0]
        preflight_log = logs_dir / "model_load_preflight.log"
        preflight_summary = summaries_dir / "model_load_preflight.json"
        preflight_command = build_command(
            args,
            ordered[0],
            preflight_gpu,
            preflight_summary,
            load_only=True,
        )
        print(
            f"[PREFLIGHT] model load physical_gpu={preflight_gpu} "
            f"logical_device=cuda:0 command={shlex.join(preflight_command)}",
            flush=True,
        )
        started = time.time()
        preflight_return_code = 0
        preflight_tail = ""
        if not args.dry_run:
            child_env = dict(os.environ)
            child_env["CUDA_VISIBLE_DEVICES"] = preflight_gpu
            child_env["PYTHONUNBUFFERED"] = "1"
            with preflight_log.open("a", encoding="utf-8") as log_handle:
                log_handle.write(
                    "\n===== "
                    f"{datetime.now(timezone.utc).isoformat()} model_load_preflight "
                    f"gpu={preflight_gpu} =====\n"
                )
                log_handle.write(shlex.join(preflight_command) + "\n")
                log_handle.flush()
                process = subprocess.run(
                    preflight_command,
                    env=child_env,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            preflight_return_code = process.returncode
            if preflight_return_code != 0:
                preflight_tail = print_failure_log(
                    "model_load_preflight",
                    preflight_log,
                    args.failure_log_lines,
                )
        preflight_result = {
            "physical_gpu": preflight_gpu,
            "logical_device": "cuda:0",
            "command": preflight_command,
            "return_code": preflight_return_code,
            "elapsed_seconds": round(time.time() - started, 6),
            "log_path": str(preflight_log),
            "log_tail": preflight_tail,
        }
        if preflight_return_code != 0:
            status = {
                "schema_version": 2,
                "checkpoint": str(args.checkpoint),
                "input_dir": str(args.input_dir),
                "output_dir": str(args.output_dir),
                "gpu_devices": gpus,
                "counts": counts,
                "model_load_preflight": preflight_result,
                "tasks": {},
                "missing_tasks": list(args.tasks),
                "success": False,
                "dry_run": args.dry_run,
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
            atomic_write_json(args.output_dir / "parallel_inference_status.json", status)
            print(
                f"[ERROR] model load preflight failed: GPU={preflight_gpu} "
                f"exit_code={preflight_return_code} log={preflight_log}",
                file=sys.stderr,
                flush=True,
            )
            return 1
        print(
            f"[PREFLIGHT DONE] model load elapsed={preflight_result['elapsed_seconds']:.1f}s "
            f"log={preflight_log}",
            flush=True,
        )

    def run_task(task: str, gpu: str, attempt: int) -> dict[str, Any]:
            summary_path = summaries_dir / f"{task}.json"
            log_path = logs_dir / f"{task}.log"
            command = build_command(args, task, gpu, summary_path)
            print(
                f"[START] task={task} attempt={attempt} physical_gpu={gpu} logical_device=cuda:0 "
                f"command={shlex.join(command)}",
                flush=True,
            )
            started = time.time()
            return_code = 0
            error = ""
            if not args.dry_run:
                child_env = dict(os.environ)
                child_env["CUDA_VISIBLE_DEVICES"] = gpu
                child_env["PYTHONUNBUFFERED"] = "1"
                with log_path.open("a", encoding="utf-8") as log_handle:
                    log_handle.write(
                        f"\n===== {datetime.now(timezone.utc).isoformat()} task={task} "
                        f"gpu={gpu} attempt={attempt} =====\n"
                    )
                    log_handle.write(shlex.join(command) + "\n")
                    log_handle.flush()
                    process = subprocess.run(
                        command,
                        env=child_env,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                return_code = process.returncode
                if return_code != 0:
                    error = f"inference subprocess exited with code {return_code}"
            elapsed = time.time() - started
            log_tail = ""
            if return_code != 0:
                log_tail = print_failure_log(
                    f"task={task} GPU={gpu}",
                    log_path,
                    args.failure_log_lines,
                )
            result: dict[str, Any] = {
                "task": task,
                "physical_gpu": gpu,
                "logical_device": "cuda:0",
                "command": command,
                "return_code": return_code,
                "elapsed_seconds": round(elapsed, 6),
                "sample_count": counts[task],
                "log_path": str(log_path),
                "summary_path": str(summary_path),
                "error": error,
                "log_tail": log_tail,
            }
            if return_code != 0:
                print(
                    f"[FAILED] task={task} attempt={attempt} GPU={gpu} exit_code={return_code} "
                    f"command={shlex.join(command)} log={log_path} error={error}",
                    flush=True,
                )
            else:
                print(
                    f"[DONE] task={task} GPU={gpu} elapsed={elapsed:.1f}s log={log_path}",
                    flush=True,
                )
            return result

    results = run_priority_gpu_tasks(
        tasks=ordered,
        gpu_devices=gpus,
        estimates=estimates,
        runner=run_task,
        retries=args.task_retries,
        continue_on_failure=args.continue_on_task_failure,
    )

    missing_tasks = [task for task in args.tasks if task not in results]
    failures = [result for result in results.values() if result["return_code"] != 0]
    status = {
        "schema_version": 2,
        "checkpoint": str(args.checkpoint),
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "gpu_devices": gpus,
        "counts": counts,
        "model_load_preflight": preflight_result,
        "tasks": results,
        "missing_tasks": missing_tasks,
        "success": not failures and not missing_tasks,
        "dry_run": args.dry_run,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(args.output_dir / "parallel_inference_status.json", status)

    if args.runtime_profile is not None and not args.dry_run:
        profile_tasks = dict(previous_profile.get("tasks", {}))
        for task, result in results.items():
            if result["return_code"] == 0:
                profile_tasks[task] = {
                    "elapsed_seconds": result["elapsed_seconds"],
                    "sample_count": result["sample_count"],
                    "seconds_per_item": result["elapsed_seconds"]
                    / max(1, result["sample_count"]),
                    "physical_gpu": result["physical_gpu"],
                    "updated_at": status["finished_at"],
                }
        atomic_write_json(
            args.runtime_profile,
            {"schema_version": 1, "tasks": profile_tasks},
        )

    if failures or missing_tasks:
        print(
            f"[ERROR] parallel inference failed: failures={len(failures)}, "
            f"not_started={missing_tasks}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
