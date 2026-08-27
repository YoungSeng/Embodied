#!/usr/bin/env python3
"""Patch, infer, score, and record one LocateAnything UI5 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from locany_ui5_common import PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--machine-type", choices=("a800", "h20"), required=True)
    parser.add_argument("--gpu-count", type=int, required=True)
    parser.add_argument("--max-num-tokens", type=int, required=True)
    parser.add_argument("--eval-gpu-devices", required=True)
    parser.add_argument("--attn-implementation", required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scorer-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--max-images-per-task", type=int, default=0)
    parser.add_argument(
        "--inference-crop-mode",
        choices=("full_image", "lossless_tiling", "detector_scan"),
        default="full_image",
    )
    parser.add_argument("--tile-max-count", type=int, default=10)
    parser.add_argument("--tile-target-long-side", type=int, default=1600)
    parser.add_argument("--tile-overlap-ratio", type=float, default=0.10)
    parser.add_argument("--tile-nms-iou", type=float, default=0.50)
    parser.add_argument("--eval-parser-root", type=Path, default=None)
    parser.add_argument("--eval-detector-cache", type=Path, default=None)
    parser.add_argument("--eval-text-python", default=None)
    parser.add_argument("--eval-icon-python", default=None)
    parser.add_argument("--eval-text-model-dir", type=Path, default=None)
    parser.add_argument("--eval-icon-model", type=Path, default=None)
    parser.add_argument("--eval-detector-workers-per-gpu", type=int, choices=(1, 2), default=1)
    parser.add_argument("--scan-target-height", type=int, default=960)
    parser.add_argument("--scan-vertical-link-ratio", type=float, default=0.025)
    parser.add_argument("--scan-context-ratio", type=float, default=0.20)
    parser.add_argument("--scan-min-context-image-ratio", type=float, default=0.015)
    parser.add_argument("--scan-dense-band-ratio", type=float, default=0.80)
    parser.add_argument("--scan-visualization-samples", type=int, default=20)
    parser.add_argument(
        "--relation-gate-mode", choices=("observe", "hard"), default="observe"
    )
    parser.add_argument("--relation-gate-threshold", type=float, default=None)
    parser.add_argument("--skip-patch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def run_checked(command: list[str], *, cwd: Path | None, stage: str) -> None:
    print(f"[EVAL:{stage}] command={shlex.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=cwd, check=False)
    if completed.returncode != 0:
        error = subprocess.CalledProcessError(completed.returncode, command)
        error.stage = stage  # type: ignore[attr-defined]
        raise error


def record_history(
    *,
    args: argparse.Namespace,
    history_dir: Path,
    metrics_json: Path | None,
    start_time: str,
    end_time: str,
    status: str,
    prediction_dir: Path,
    evaluation_run_dir: Path | None,
    error: str,
) -> None:
    command = [
        sys.executable,
        str(args.project_root / "scripts" / "collect_ui5_metrics.py"),
        "record",
        "--history-dir",
        str(history_dir),
        "--step",
        str(args.step),
        "--machine-type",
        args.machine_type,
        "--gpu-count",
        str(args.gpu_count),
        "--max-num-tokens",
        str(args.max_num_tokens),
        "--max-num-tokens-scope",
        "per_rank_packed_batch",
        "--relation-gate-mode",
        args.relation_gate_mode,
        "--checkpoint",
        str(args.checkpoint),
        "--start-time",
        start_time,
        "--end-time",
        end_time,
        "--status",
        status,
        "--prediction-dir",
        str(prediction_dir),
        "--gt-dir",
        str(args.input_dir),
        "--scorer-root",
        str(args.scorer_root),
        "--error",
        error,
    ]
    if metrics_json is not None and metrics_json.is_file():
        command.extend(["--metrics-json", str(metrics_json)])
    if evaluation_run_dir is not None:
        command.extend(["--evaluation-run-dir", str(evaluation_run_dir)])
    run_checked(command, cwd=args.project_root, stage="history")


def main() -> int:
    args = parse_args()
    if args.step < 0:
        raise ValueError("--step cannot be negative")
    if not 1 <= args.tile_max_count <= 10:
        raise ValueError("--tile-max-count must be in [1, 10]")
    if not 0 < args.tile_overlap_ratio < 1:
        raise ValueError("--tile-overlap-ratio must be in (0, 1)")
    if args.scan_target_height <= 0:
        raise ValueError("--scan-target-height must be positive")
    if min(
        args.scan_vertical_link_ratio,
        args.scan_context_ratio,
        args.scan_min_context_image_ratio,
    ) < 0:
        raise ValueError("scan link/context ratios cannot be negative")
    if not 0 < args.scan_dense_band_ratio <= 1:
        raise ValueError("--scan-dense-band-ratio must be in (0, 1]")
    if args.scan_visualization_samples < 0:
        raise ValueError("--scan-visualization-samples cannot be negative")
    args.project_root = args.project_root.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.base_model = args.base_model.expanduser().resolve()
    args.input_dir = args.input_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.scorer_root = args.scorer_root.expanduser().resolve()
    args.eval_parser_root = (
        args.eval_parser_root.expanduser().resolve(strict=False)
        if args.eval_parser_root is not None
        else (args.project_root.parent / "ui-region-parser").resolve(strict=False)
    )
    args.eval_detector_cache = (
        args.eval_detector_cache.expanduser().resolve(strict=False)
        if args.eval_detector_cache is not None
        else (args.output_dir / "evaluation" / "detector_scan_cache").resolve(strict=False)
    )

    prediction_suffix = {
        "full_image": "full",
        "lossless_tiling": "lossless-tiling",
        "detector_scan": "detector-scan",
    }[args.inference_crop_mode]
    prediction_dir = args.output_dir / f"inference-checkpoint-{args.step}-{prediction_suffix}"
    history_dir = args.output_dir / "evaluation"
    raw_evaluation_root = history_dir / "raw"
    attempt_stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_name = f"checkpoint-{args.step}-{attempt_stamp}"
    evaluation_run_dir = raw_evaluation_root / run_name
    runtime_profile = history_dir / "task_runtime_profile.json"
    attempts_dir = history_dir / "attempts"
    metadata_path = attempts_dir / f"{run_name}.json"
    start_time = utc_now()
    current_stage = "preflight"
    current_command: list[str] = []
    metrics_json: Path | None = None

    metadata: dict[str, Any] = {
        "schema_version": 1,
        "step": args.step,
        "machine_type": args.machine_type,
        "gpu_count": args.gpu_count,
        "max_num_tokens": args.max_num_tokens,
        "max_num_tokens_scope": "per_rank_packed_batch",
        "relation_gate_mode": args.relation_gate_mode,
        "inference_crop_mode": args.inference_crop_mode,
        "gt_repair_used_for_inference": False,
        "detector_crop_cache": (
            str(args.eval_detector_cache)
            if args.inference_crop_mode == "detector_scan"
            else None
        ),
        "checkpoint": str(args.checkpoint),
        "prediction_dir": str(prediction_dir),
        "evaluation_run_dir": str(evaluation_run_dir),
        "evaluation_start_time": start_time,
        "status": "running",
    }
    atomic_write_json(metadata_path, metadata)

    try:
        for path, label in (
            (args.checkpoint, "checkpoint"),
            (args.base_model, "base model"),
            (args.input_dir, "evaluation input directory"),
            (args.scorer_root, "scorer root"),
        ):
            if not path.is_dir():
                raise FileNotFoundError(f"{label} does not exist: {path}")

        detector_crop_manifest: Path | None = None
        detector_crop_manifest_digest: str | None = None
        if args.inference_crop_mode == "detector_scan":
            detector_crop_manifest = (
                args.eval_detector_cache / "scan_crops" / "detector_scan_crops.jsonl"
            )
            if not args.eval_parser_root.is_dir():
                raise FileNotFoundError(
                    f"ui-region-parser does not exist: {args.eval_parser_root}"
                )
            current_stage = "prepare_detector_scan"
            current_command = [
                sys.executable,
                str(args.project_root / "scripts" / "prepare_ui5_eval_detector_crops.py"),
                "--stage", "all",
                "--input-dir", str(args.input_dir),
                "--output-dir", str(args.eval_detector_cache),
                "--parser-root", str(args.eval_parser_root),
                "--gpus", args.eval_gpu_devices,
                "--workers-per-gpu", str(args.eval_detector_workers_per_gpu),
                "--scan-max-crops", str(args.tile_max_count),
                "--scan-target-height", str(args.scan_target_height),
                "--scan-overlap-ratio", str(args.tile_overlap_ratio),
                "--scan-vertical-link-ratio", str(args.scan_vertical_link_ratio),
                "--scan-context-ratio", str(args.scan_context_ratio),
                "--scan-min-context-image-ratio", str(args.scan_min_context_image_ratio),
                "--scan-dense-band-ratio", str(args.scan_dense_band_ratio),
                "--visualization-samples", str(args.scan_visualization_samples),
                "--resume",
            ]
            if args.max_images_per_task:
                current_command.extend(
                    ["--max-images-per-task", str(args.max_images_per_task)]
                )
            if args.eval_text_python:
                current_command.extend(["--text-python", args.eval_text_python])
            if args.eval_icon_python:
                current_command.extend(["--icon-python", args.eval_icon_python])
            if args.eval_text_model_dir:
                current_command.extend(
                    ["--text-model-dir", str(args.eval_text_model_dir)]
                )
            if args.eval_icon_model:
                current_command.extend(["--icon-model", str(args.eval_icon_model)])
            if args.eval_detector_workers_per_gpu == 2:
                current_command.append("--allow-two-processes-per-gpu")
            if args.dry_run:
                print(f"[DRY RUN:{current_stage}] {shlex.join(current_command)}")
            else:
                run_checked(current_command, cwd=args.project_root, stage=current_stage)
                if not detector_crop_manifest.is_file():
                    raise FileNotFoundError(
                        f"detector scan preparation did not write {detector_crop_manifest}"
                    )
                detector_crop_manifest_digest = hashlib.sha256(
                    detector_crop_manifest.read_bytes()
                ).hexdigest()

        if not args.skip_patch:
            current_stage = "patch"
            current_command = [
                sys.executable,
                str(args.project_root / "scripts" / "patch_locany_checkpoint.py"),
                "--base-model",
                str(args.base_model),
                "--checkpoint",
                str(args.checkpoint),
                "--project-root",
                str(args.project_root),
                "--force",
                "--validate-relation-weights",
            ]
            if args.relation_gate_mode == "observe":
                current_command.append("--allow-legacy-slot-gate")
            if not args.dry_run:
                run_checked(current_command, cwd=args.project_root, stage=current_stage)
            else:
                print(f"[DRY RUN:{current_stage}] {shlex.join(current_command)}")

        current_stage = "parallel_inference"
        current_command = [
            sys.executable,
            str(args.project_root / "scripts" / "run_ui5_parallel_inference.py"),
            "--checkpoint",
            str(args.checkpoint),
            "--processor-path",
            str(args.base_model),
            "--input-dir",
            str(args.input_dir),
            "--output-dir",
            str(prediction_dir),
            "--gpu-devices",
            args.eval_gpu_devices,
            "--attn-implementation",
            args.attn_implementation,
            "--inference-script",
            str(args.project_root / "scripts" / "inference_ui_defect_locany.py"),
            "--runtime-profile",
            str(runtime_profile),
            "--relation-gate-mode",
            args.relation_gate_mode,
            "--inference-crop-mode",
            args.inference_crop_mode,
            "--tile-max-count",
            str(args.tile_max_count),
            "--tile-target-long-side",
            str(args.tile_target_long_side),
            "--tile-overlap-ratio",
            str(args.tile_overlap_ratio),
            "--tile-nms-iou",
            str(args.tile_nms_iou),
        ]
        if detector_crop_manifest is not None:
            current_command.extend(
                ["--detector-crop-manifest", str(detector_crop_manifest)]
            )
        existing_manifest = prediction_dir / "_run_manifest.json"
        overwrite_for_gate_mode_change = False
        if existing_manifest.is_file():
            try:
                previous_manifest = json.loads(existing_manifest.read_text(encoding="utf-8"))
                previous_mode = previous_manifest.get("generation", {}).get(
                    "relation_gate_mode"
                )
                previous_crop = previous_manifest.get("inference_crop", {})
                expected_crop = {
                    "mode": args.inference_crop_mode,
                    "max_tiles": args.tile_max_count,
                    "target_long_side": args.tile_target_long_side,
                    "overlap_ratio": args.tile_overlap_ratio,
                    "nms_iou": args.tile_nms_iou,
                    "detector_crop_manifest": (
                        str(detector_crop_manifest)
                        if detector_crop_manifest is not None
                        else None
                    ),
                    "detector_crop_manifest_digest": detector_crop_manifest_digest,
                    "gt_repair_allowed": False,
                }
                overwrite_for_gate_mode_change = (
                    previous_mode != args.relation_gate_mode
                    or previous_crop != expected_crop
                )
            except (OSError, json.JSONDecodeError):
                overwrite_for_gate_mode_change = True
        if overwrite_for_gate_mode_change:
            current_command.append("--overwrite")
        if args.relation_gate_threshold is not None:
            current_command.extend(
                ["--relation-gate-threshold", str(args.relation_gate_threshold)]
            )
        if args.max_images_per_task:
            current_command.extend(
                ["--max-images-per-task", str(args.max_images_per_task)]
            )
        if args.dry_run:
            print(f"[DRY RUN:{current_stage}] {shlex.join(current_command)}")
        else:
            run_checked(current_command, cwd=args.project_root, stage=current_stage)

        scorer_script = args.scorer_root / "qwen3vl_merge_and_score_fixed_5tasks.py"
        if not scorer_script.is_file():
            raise FileNotFoundError(f"Scorer script does not exist: {scorer_script}")
        current_stage = "score"
        current_command = [
            sys.executable,
            str(scorer_script),
            "--all_tasks",
            "--input_mode",
            "yolo_dir",
            "--gt_dir",
            str(args.input_dir),
            "--pred_root",
            str(prediction_dir),
            "--output_root",
            str(raw_evaluation_root),
            "--run_name",
            run_name,
            "--yolo_bbox_format",
            "xyxy",
        ]
        if args.dry_run:
            print(f"[DRY RUN:{current_stage}] {shlex.join(current_command)}")
            metadata.update(
                {"status": "dry_run", "evaluation_end_time": utc_now()}
            )
            atomic_write_json(metadata_path, metadata)
            return 0
        run_checked(current_command, cwd=args.scorer_root, stage=current_stage)

        metrics_json = evaluation_run_dir / "all_tasks_evaluation.json"
        if not metrics_json.is_file():
            legacy_report = evaluation_run_dir / "all_tasks_evaluation.txt"
            if legacy_report.is_file():
                current_stage = "convert_legacy_score_report"
                current_command = [
                    sys.executable,
                    str(args.project_root / "scripts" / "collect_ui5_metrics.py"),
                    "convert-report",
                    "--report",
                    str(legacy_report),
                    "--output",
                    str(metrics_json),
                ]
                run_checked(current_command, cwd=args.project_root, stage=current_stage)
        if not metrics_json.is_file():
            raise FileNotFoundError(
                f"Scorer succeeded but metric JSON was not generated: {metrics_json}"
            )
        end_time = utc_now()
        current_stage = "history"
        record_history(
            args=args,
            history_dir=history_dir,
            metrics_json=metrics_json,
            start_time=start_time,
            end_time=end_time,
            status="success",
            prediction_dir=prediction_dir,
            evaluation_run_dir=evaluation_run_dir,
            error="",
        )
        metadata.update(
            {
                "status": "success",
                "evaluation_end_time": end_time,
                "metrics_json": str(metrics_json),
            }
        )
        atomic_write_json(metadata_path, metadata)
        print(
            f"[EVAL SUCCESS] step={args.step} checkpoint={args.checkpoint} "
            f"metrics={metrics_json}"
        )
        return 0
    except Exception as exc:
        end_time = utc_now()
        exit_code = exc.returncode if isinstance(exc, subprocess.CalledProcessError) else 1
        error_text = f"{type(exc).__name__}: {exc}"
        print(
            f"[EVAL FAILED] step={args.step} stage={current_stage} "
            f"checkpoint={args.checkpoint} command={shlex.join(current_command) if current_command else '<none>'} "
            f"exit_code={exit_code} error={error_text}",
            file=sys.stderr,
        )
        traceback.print_exc()
        try:
            record_history(
                args=args,
                history_dir=history_dir,
                metrics_json=metrics_json,
                start_time=start_time,
                end_time=end_time,
                status="failed",
                prediction_dir=prediction_dir,
                evaluation_run_dir=evaluation_run_dir,
                error=f"stage={current_stage}; exit_code={exit_code}; {error_text}",
            )
        except Exception as history_exc:
            print(f"[EVAL FAILED] additionally failed to record history: {history_exc}", file=sys.stderr)
        metadata.update(
            {
                "status": "failed",
                "evaluation_end_time": end_time,
                "failed_stage": current_stage,
                "failed_command": current_command,
                "exit_code": exit_code,
                "error": error_text,
            }
        )
        atomic_write_json(metadata_path, metadata)
        return int(exit_code) if int(exit_code) != 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
