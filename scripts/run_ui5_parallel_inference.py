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
from typing import Any

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
    parser.add_argument("--runtime-profile", type=Path, default=None)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--max-images-per-task", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
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


def build_command(args: argparse.Namespace, task: str, gpu: str, summary: Path) -> list[str]:
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
        "--generation-mode",
        "hybrid",
        "--tasks",
        task,
        "--skip-figma",
        "--fail-fast",
    ]
    if args.max_images_per_task:
        command.extend(["--max-images-per-task", str(args.max_images_per_task)])
    return command


def main() -> int:
    args = parse_args()
    if args.max_images_per_task < 0:
        raise ValueError("--max-images-per-task cannot be negative")
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

    work_queue: queue.PriorityQueue[tuple[float, str]] = queue.PriorityQueue()
    for task in ordered:
        work_queue.put((-estimates[task], task))

    summaries_dir = args.output_dir / "_worker_summaries"
    logs_dir = args.output_dir / "_worker_logs"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    stop_event = threading.Event()
    results: dict[str, dict[str, Any]] = {}

    def run_worker(gpu: str) -> None:
        while not stop_event.is_set():
            try:
                _, task = work_queue.get_nowait()
            except queue.Empty:
                return
            summary_path = summaries_dir / f"{task}.json"
            log_path = logs_dir / f"{task}.log"
            command = build_command(args, task, gpu, summary_path)
            print(
                f"[START] task={task} physical_gpu={gpu} logical_device=cuda:0 "
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
                        f"\n===== {datetime.now(timezone.utc).isoformat()} task={task} gpu={gpu} =====\n"
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
                task_dir = args.output_dir / task
                prediction_files = list(task_dir.glob("*.json")) if task_dir.is_dir() else []
                if return_code == 0 and not prediction_files:
                    return_code = 90
                    error = f"no prediction JSON files found under {task_dir}"
            elapsed = time.time() - started
            result = {
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
            }
            with lock:
                results[task] = result
            if return_code != 0:
                stop_event.set()
                print(
                    f"[FAILED] task={task} GPU={gpu} exit_code={return_code} "
                    f"command={shlex.join(command)} log={log_path} error={error}",
                    flush=True,
                )
            else:
                print(
                    f"[DONE] task={task} GPU={gpu} elapsed={elapsed:.1f}s log={log_path}",
                    flush=True,
                )
            work_queue.task_done()

    threads = [
        threading.Thread(target=run_worker, args=(gpu,), name=f"gpu-{gpu}", daemon=False)
        for gpu in gpus
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    missing_tasks = [task for task in args.tasks if task not in results]
    failures = [result for result in results.values() if result["return_code"] != 0]
    status = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint),
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "gpu_devices": gpus,
        "counts": counts,
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
