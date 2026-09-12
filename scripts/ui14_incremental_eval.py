"""CPU completion consumer. Only this coordinator scores and writes Excel."""
from __future__ import annotations

import importlib.util
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

from ui14_common import read_json


def describe_failure(event):
    return (f"task={event.get('task')} GPU={event.get('physical_gpu')} "
            f"exclusive={event.get('exclusive_gpu')} exit={event.get('return_code')} "
            f"reason={event.get('failure_reason') or event.get('error') or 'worker failed; inspect log'} "
            f"log={event.get('log_path')}")


class InferenceFailure(subprocess.CalledProcessError):
    def __init__(self, returncode, command, failures, progress):
        super().__init__(returncode, command)
        self.failures, self.progress = failures, progress

    def __str__(self):
        details = " | ".join(describe_failure(e) for e in self.failures) or "No failed task event; inspect scheduler log"
        return (f"UI14 inference failed (exit={self.returncode}): {details}; "
                f"not_started={self.progress.get('pending', [])}")


def progress_message(events, progress):
    successes = sum(e.get("return_code") == 0 for e in events)
    failures = [e for e in events if e.get("return_code") != 0]
    running = [f"{task}@GPU{gpu}" for gpu, slots in progress.get("running", {}).items()
               for task in slots.values()]
    message = (f"inference succeeded={successes} failed={len(failures)} | "
               f"running={running} | pending={progress.get('pending', [])}")
    if progress.get("stopped"):
        message += " | queue stopped; waiting for active workers and saving their results"
    if failures:
        message += " | " + " | ".join(describe_failure(e) for e in failures)
    return message


def run_incremental_inference(command, *, cwd, completion_dir, on_complete):
    """Drain atomic task events, including successes after another worker failed."""
    completion_dir = Path(completion_dir)
    completion_dir.mkdir(parents=True, exist_ok=False)
    command = [*command, "--completion-dir", str(completion_dir)]
    seen, errors = set(), []
    events, progress = [], {}
    started = last_progress = time.monotonic()
    with subprocess.Popen(command, cwd=cwd) as process:
        while True:
            finished = process.poll() is not None
            for path in sorted(completion_dir.glob("*.json")):
                if path.name in seen:
                    continue
                event = read_json(path)
                seen.add(path.name)
                events.append(event)
                if event.get("return_code") != 0:
                    print(f"[UI14 eval] inference failure: {describe_failure(event)}", flush=True)
                try:
                    on_complete(event)
                except Exception as exc:
                    # Do not lose other finished tasks because one scorer failed.
                    errors.append(exc)
                    print(f"[UI14 eval] task scoring failed: {event.get('task')}: {exc}", flush=True)
            progress_path = completion_dir / "_scheduler/status.json"
            if progress_path.is_file():
                progress = read_json(progress_path)
            if finished:
                break
            if time.monotonic() - last_progress >= 10:
                print(f"[UI14 eval] {progress_message(events, progress)} | "
                      f"elapsed={time.monotonic()-started:.0f}s | scoring failures={len(errors)} | "
                      f"events={completion_dir}", flush=True)
                last_progress = time.monotonic()
            time.sleep(.2)
    if errors:
        raise errors[0]
    if process.returncode:
        raise InferenceFailure(process.returncode, command,
                               [e for e in events if e.get('return_code') != 0], progress)


def score_ui5_task(spec, prediction, destination, scorer_root):
    """Call the configured UI5 scorer's existing single-task path, including no_figma."""
    path = Path(scorer_root) / "qwen3vl_merge_and_score_fixed_5tasks.py"
    module_spec = importlib.util.spec_from_file_location("ui14_incremental_ui5_scorer", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    args = SimpleNamespace(input_mode="yolo_dir", yolo_bbox_format="xyxy",
                           iou_thresh=.1, merge_only=False)
    return module.run_one_task(args, spec["task_key"], spec["test"],
                               str(prediction / spec["task_key"]), destination)
