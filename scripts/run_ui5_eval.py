#!/usr/bin/env python3
"""Patch, infer, score, and record one LocateAnything UI5 checkpoint."""

from __future__ import annotations

import argparse
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
    args.project_root = args.project_root.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.base_model = args.base_model.expanduser().resolve()
    args.input_dir = args.input_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.scorer_root = args.scorer_root.expanduser().resolve()

    prediction_dir = args.output_dir / f"inference-checkpoint-{args.step}-full"
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
        ]
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
