"""CPU completion consumer. Only this coordinator scores and writes Excel."""
from __future__ import annotations

import importlib.util
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

from ui14_common import read_json


def run_incremental_inference(command, *, cwd, completion_dir, on_complete):
    """Drain atomic task events, including successes after another worker failed."""
    completion_dir = Path(completion_dir)
    completion_dir.mkdir(parents=True, exist_ok=False)
    command = [*command, "--completion-dir", str(completion_dir)]
    seen, errors = set(), []
    started = last_progress = time.monotonic()
    with subprocess.Popen(command, cwd=cwd) as process:
        while True:
            finished = process.poll() is not None
            for path in sorted(completion_dir.glob("*.json")):
                if path.name in seen:
                    continue
                event = read_json(path)
                seen.add(path.name)
                try:
                    on_complete(event)
                except Exception as exc:
                    # Do not lose other finished tasks because one scorer failed.
                    errors.append(exc)
                    print(f"[UI14 eval] task scoring failed: {event.get('task')}: {exc}", flush=True)
            if finished:
                break
            if time.monotonic() - last_progress >= 10:
                print(f"[UI14 eval] consumed {len(seen)} task completions | "
                      f"elapsed={time.monotonic()-started:.0f}s | scoring failures={len(errors)} | "
                      f"events={completion_dir}", flush=True)
                last_progress = time.monotonic()
            time.sleep(.2)
    if errors:
        raise errors[0]
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)


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
