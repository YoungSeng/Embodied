#!/usr/bin/env python3
"""GPU cache stage: only the seven new crop tasks, isolated by task and split."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
from ui14_common import *
from ui14_repair import validate_normalization
from ui14_progress import ProgressSession, detector_status, phase, track


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--parser-root", default=WORKSPACE + "/code/Eagle_LocateUI5_v4/ui-region-parser")
    parser.add_argument("--ui5-cache", default=WORKSPACE + "/code/Eagle_LocateUI5_v4/Embodied-ui5-det-crop/work_dirs/ui5_eval_detector_cache_horizontal_v5")
    parser.add_argument("--text-python", default=None)
    parser.add_argument("--icon-python", default=WORKSPACE + "/conda_envs/LocateAnything/bin/python")
    parser.add_argument("--text-model-dir", default=None)
    parser.add_argument("--icon-model", default=None)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--progress-interval-seconds", type=float,
                        default=os.environ.get("UI14_PROGRESS_INTERVAL_SECONDS", "10"))
    args = parser.parse_args()
    with ProgressSession("cache", args.data_root, args.progress_interval_seconds):
        return run(args)


def run(args):
    root = Path(args.data_root).resolve(strict=True)
    validate_normalization(root)
    write_json(root / "cpu_check_report.json", {**read_json(root / "cpu_check_report.json"),
               "ready": False, "stage": "cache_pending_final_cpu_check"})
    registry = load_registry(root / "task_registry.json")
    # Reuse the audited detector configuration, including its separate Paddle environment.
    config = read_json(Path(args.ui5_cache) / "detections" / "detector_config.json")
    stage_summary = Path(args.ui5_cache) / "detections" / "text" / "stage_summary.json"
    runtime = read_json(stage_summary).get("runtime", {}) if stage_summary.is_file() else {}
    text_python = args.text_python or runtime.get("python") or WORKSPACE + "/conda_envs/UI5PaddleOCR/bin/python"
    for path in (text_python, args.icon_python, args.parser_root):
        if not Path(path).exists(): raise FileNotFoundError(f"Detector runtime is unreadable: {path}")
    jobs = [(get_task(spec["task_id"]), split) for spec in registry[5:]
            if get_task(spec["task_id"]).view_policy == "crops" for split in ("train", "test")]
    for task, split in track(jobs, "七个 crop 任务 × train/test", unit="任务/split",
                            detail=lambda job: f"{job[0].task_key}/{job[1]}", estimate=False):
        with phase(f"{task.task_key}/{split} detector 与 crop cache"):
            paths = paths_for(root, task.task_key, split)
            records = list(read_jsonl(paths["normalized"]))
            count = len({file_digest(r["source_image"]) for r in track(
                records, "统计 detector 图片身份", unit="原图", detail=lambda r: r["source_image"])})
            command = [args.icon_python, "-u", str(PROJECT_ROOT / "scripts" / "prepare_ui5_eval_detector_crops.py"),
                "--stage", "all", "--input-dir", str(paths["detector_input"].parent),
                "--task-input-manifest", str(paths["detector_inputs"]), "--data-split", split,
                "--output-dir", str(paths["cache"]), "--parser-root", args.parser_root,
                "--gpus", args.gpus, "--workers-per-gpu", "1", "--text-python", text_python,
                "--icon-python", args.icon_python, "--scan-name", SCAN_NAME,
                "--cache-scope", "full_test" if split == "test" else "full_train",
                "--expected-unique-images", str(count), "--no-skip-figma", "--resume",
                "--progress-interval-seconds", str(args.progress_interval_seconds)]
            command += ["--text-long-side", str(config["text"]["long_side"]),
                        "--text-box-threshold", str(config["text"]["box_threshold"]),
                        "--icon-long-side", str(config["icon"]["long_side"]),
                        "--icon-confidence", str(config["icon"]["confidence"])]
            if config["text"].get("enable_mkldnn"): command.append("--enable-mkldnn")
            text_model = args.text_model_dir or config.get("text", {}).get("model_dir")
            icon_model = args.icon_model or config.get("icon", {}).get("model")
            if text_model: command += ["--text-model-dir", text_model]
            if icon_model: command += ["--icon-model", icon_model]
            with phase("PP-OCRv5 → OmniParser → 横向裁剪计划"), detector_status(paths["cache"] / "run_status.json"):
                subprocess.run(command, check=True)
            from prepare_ui14_sft import crop_annotations
            crop_annotations(root, task, split, records)
    validate_normalization(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
