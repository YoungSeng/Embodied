#!/usr/bin/env python3
"""Separate CPU preparation, GPU detection and CPU crops for UI14 caches."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
from ui14_common import *
from ui14_repair import validate_normalization
from ui14_progress import ProgressSession, detector_status, phase, track
from ui14_cache_prepare import ImageInfoJournal, publish_prepared, validate_prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "detect", "crops", "all"), default="detect",
                        help="prepare/crops are CPU-only; detect never scans images before GPU workers")
    parser.add_argument("--prepare-workers", type=int, default=int(os.environ.get("UI14_PREPARE_WORKERS", "16")))
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
    name = {"prepare": "cache-prepare", "detect": "cache", "crops": "cache-finalize", "all": "cache-all"}[args.stage]
    with ProgressSession(name, args.data_root, args.progress_interval_seconds):
        return run(args)


def run(args):
    root = Path(args.data_root).resolve(strict=True)
    binding = validate_normalization(root)
    write_json(root / "cpu_check_report.json", {**read_json(root / "cpu_check_report.json"),
               "ready": False, "stage": "cache_pending_final_cpu_check"})
    registry = load_registry(root / "task_registry.json")
    if len(registry) != 14:
        raise ValueError("UI14 cache requires the complete 14-task registry")
    # Reuse the audited detector configuration, including its separate Paddle environment.
    config = read_json(Path(args.ui5_cache) / "detections" / "detector_config.json")
    stage_summary = Path(args.ui5_cache) / "detections" / "text" / "stage_summary.json"
    runtime = read_json(stage_summary).get("runtime", {}) if stage_summary.is_file() else {}
    text_python = args.text_python or runtime.get("python") or WORKSPACE + "/conda_envs/UI5PaddleOCR/bin/python"
    jobs = [(get_task(spec["task_id"]), split) for spec in registry[5:]
            if get_task(spec["task_id"]).view_policy == "crops" for split in ("train", "test")]
    stage = getattr(args, "stage", "detect")

    def options(paths, split, step, count=1):
        command = ["--stage", step, "--input-dir", str(paths["detector_input"].parent),
                "--task-input-manifest", str(paths["detector_inputs"]), "--data-split", split,
                "--output-dir", str(paths["cache"]), "--parser-root", args.parser_root,
                "--gpus", args.gpus, "--workers-per-gpu", "1", "--scan-name", SCAN_NAME,
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
        if step in ("text", "icon"):
            command += ["--text-python", text_python, "--icon-python", args.icon_python]
        return command

    import prepare_ui5_eval_detector_crops as detector
    plans = [(task, split, paths_for(root, task.task_key, split)) for task, split in jobs]
    configs = {(task.task_key, split): detector.detector_config(detector.parse_args(options(paths, split, "prepare")))
               for task, split, paths in plans}
    if stage in ("prepare", "all"):
        journal = ImageInfoJournal(root / "cache_preparation/image_info.jsonl", getattr(args, "prepare_workers", 16))
        for task, split, paths in track(plans, "CPU 准备七个 crop 任务 × train/test", unit="任务/split", estimate=False):
            with phase(f"{task.task_key}/{split} CPU 图片扫描与分片"):
                expected_config = configs[task.task_key, split]
                # Reject incompatible legacy detector settings before touching its manifests.
                detector.ensure_detector_config(paths["cache"] / "detections/detector_config.json", expected_config)
                marker = paths["cache"] / "manifest/ui14_prepare_ready.json"
                marker.unlink(missing_ok=True)
                prepared_args = detector.parse_args(options(paths, split, "prepare"))
                rows = detector.prepare_manifest(prepared_args, image_info_loader=journal.load)
                publish_prepared(paths, binding["normalization_id"], expected_config, len(rows))
                print(f"[cache-prepare ready] {task.task_key}/{split}: {len(rows)} unique images", flush=True)
        write_json(root / "cache_preparation/summary.json", {**binding, **journal.totals, "splits": len(plans), "cpu_only": True})

    if stage in ("detect", "crops", "all"):
        # Verify ALL split handoffs before starting any GPU subprocess. Never
        # silently fall back to CPU scanning on an allocated GPU node.
        counts = {(task.task_key, split): validate_prepared(paths, binding["normalization_id"], configs[task.task_key, split])
                  for task, split, paths in plans}
        if stage in ("detect", "all"):
            for path in (text_python, args.icon_python, args.parser_root):
                if not Path(path).exists(): raise FileNotFoundError(f"Detector runtime is unreadable: {path}")
        phases = (["text", "icon"] if stage == "detect" else ["merge", "crop"] if stage == "crops"
                  else ["text", "icon", "merge", "crop"])
        # Finish GPU work for every split before starting any CPU geometry.
        for step in phases:
            for task, split, paths in track(plans, f"{step}: 七个 crop 任务 × train/test", unit="任务/split", estimate=False):
                with phase(f"{task.task_key}/{split} {step}"), detector_status(paths["cache"] / "run_status.json"):
                    command = [args.icon_python, "-u", str(PROJECT_ROOT / "scripts/prepare_ui5_eval_detector_crops.py")]
                    command += options(paths, split, step, counts[task.task_key, split])
                    subprocess.run(command, check=True)
                if step == "crop":
                    from prepare_ui14_sft import crop_annotations
                    crop_annotations(root, task, split, list(read_jsonl(paths["normalized"])))
    validate_normalization(root)
    if stage == "detect":
        print("[cache] GPU detection complete. Release the GPU allocation; run cache-finalize and finalize on CPU.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
