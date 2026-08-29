#!/usr/bin/env python3
"""Run a short UI5 training smoke directly on already-allocated local GPUs."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Start LocateAnything UI5 training directly on visible GPUs, "
            "without step-0 or periodic evaluation"
        )
    )
    parser.add_argument("--machine", choices=("a800", "h20"), default="a800")
    parser.add_argument(
        "--gpus", "--gpu", dest="gpus", type=int, choices=(4, 8), default=4
    )
    parser.add_argument("--cuda-devices", default=None)
    parser.add_argument("--max-num-tokens", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--learning-rate", default="2e-5")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--env-dir", type=Path, default=None)
    parser.add_argument("--base-model", type=Path, default=None)
    parser.add_argument("--meta-path", type=Path, default=None)
    parser.add_argument("--use-detection-crops", action="store_true")
    parser.add_argument("--crop-audit-dir", type=Path, default=None)
    parser.add_argument(
        "--crop-train-mode",
        choices=("full_only", "full_plus_crop", "crop_only"),
        default=None,
    )
    parser.add_argument("--crop-meta-path", type=Path, default=None)
    parser.add_argument(
        "--ui-sampling-mode",
        choices=("fixed_ratio", "task_balanced_all_records"),
        default=None,
    )
    parser.add_argument("--training-data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-base", type=Path, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved direct-launch command without starting training",
    )
    args = parser.parse_args(argv)
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.save_steps is not None and args.save_steps <= 0:
        parser.error("--save-steps must be positive")
    if args.max_num_tokens is not None and args.max_num_tokens <= 0:
        parser.error("--max-num-tokens must be positive")
    return args


def build_environment(
    args: argparse.Namespace,
    *,
    base_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    project_root = args.project_root.expanduser().resolve()
    cuda_devices = args.cuda_devices or ",".join(
        str(index) for index in range(args.gpus)
    )
    devices = [value.strip() for value in cuda_devices.split(",") if value.strip()]
    if len(devices) != args.gpus or len(set(devices)) != args.gpus:
        raise ValueError(
            f"--gpus={args.gpus}, but --cuda-devices resolves to {devices}"
        )

    max_num_tokens = args.max_num_tokens or (12800 if args.gpus == 4 else 25600)
    save_steps = args.save_steps or args.max_steps
    run_name = args.run_name or (
        f"locany-ui5-v4-local-{args.machine}x{args.gpus}-smoke{args.max_steps}-"
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    output_base = (args.output_base or project_root / "work_dirs").expanduser().resolve()

    # These values deliberately select the same training entrypoint and 4/8-GPU
    # schedule as a formal job.  Only evaluation is disabled for fast debugging.
    env.update(
        {
            "PROJECT_ROOT": str(project_root),
            "MACHINE_TYPE": args.machine,
            "GPU_COUNT": str(args.gpus),
            "GPUS": str(args.gpus),
            "CUDA_DEVICES": cuda_devices,
            "CUDA_VISIBLE_DEVICES": cuda_devices,
            "EVAL_GPU_DEVICES": ",".join(devices[: min(4, len(devices))]),
            "MAX_NUM_TOKENS": str(max_num_tokens),
            "MAX_STEPS": str(args.max_steps),
            "SAVE_STEPS": str(save_steps),
            "WARMUP_STEPS": str(args.warmup_steps),
            "LEARNING_RATE": str(args.learning_rate),
            "GRADIENT_ACCUMULATION_STEPS": "2" if args.gpus == 4 else "1",
            "ENABLE_EVAL": "0",
            "EVAL_AT_START": "0",
            "PIPELINE_MODE": "train",
            "RUN_NAME": run_name,
            "OUTPUT_BASE": str(output_base),
            "UI5_USE_DETECTION_CROPS": "1" if args.use_detection_crops else "0",
            "UI5_CROP_TRAIN_MODE": args.crop_train_mode
            or ("full_plus_crop" if args.use_detection_crops else "full_only"),
            "UI5_UI_SAMPLING_MODE": args.ui_sampling_mode
            or (
                "task_balanced_all_records"
                if args.crop_train_mode == "crop_only"
                else "fixed_ratio"
            ),
            "UI5_CROP_AUDIT_DIR": "",
            "UI5_CROP_META_PATH": "",
        }
    )
    if args.use_detection_crops and args.crop_audit_dir is None:
        raise ValueError("--use-detection-crops requires --crop-audit-dir")
    optional_paths = {
        "ENV_DIR": args.env_dir,
        "BASE_MODEL": args.base_model,
        "MODEL_PATH": args.base_model,
        "META_PATH": args.meta_path,
        "TRAINING_DATA_DIR": args.training_data_dir,
        "OUTPUT_DIR": args.output_dir,
        "UI5_CROP_AUDIT_DIR": args.crop_audit_dir,
        "UI5_CROP_META_PATH": args.crop_meta_path,
    }
    for key, value in optional_paths.items():
        if value is not None:
            env[key] = str(value.expanduser().resolve())

    if "ENV_DIR" not in env:
        # Running from the already activated LocateAnything environment is the
        # least surprising behavior on an interactive GPU node.
        env["ENV_DIR"] = env.get("CONDA_PREFIX", sys.prefix)
    return env


def build_command(args: argparse.Namespace) -> list[str]:
    project_root = args.project_root.expanduser().resolve()
    return ["bash", str(project_root / "shell" / "run_locany_ui5_pipeline.sh")]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    env = build_environment(args)
    command = build_command(args)
    print("===== LocateAnything UI5 local training debug =====")
    print(f"command                       : {shlex.join(command)}")
    for key in (
        "PROJECT_ROOT",
        "ENV_DIR",
        "MACHINE_TYPE",
        "GPU_COUNT",
        "CUDA_DEVICES",
        "MAX_NUM_TOKENS",
        "GRADIENT_ACCUMULATION_STEPS",
        "MAX_STEPS",
        "SAVE_STEPS",
        "ENABLE_EVAL",
        "UI5_USE_DETECTION_CROPS",
        "UI5_CROP_AUDIT_DIR",
        "UI5_CROP_TRAIN_MODE",
        "UI5_CROP_META_PATH",
        "UI5_UI_SAMPLING_MODE",
        "RUN_NAME",
        "OUTPUT_BASE",
    ):
        print(f"{key:<30}: {env[key]}")
    print("=====================================================")
    if args.dry_run:
        return 0
    completed = subprocess.run(
        command,
        cwd=str(args.project_root.expanduser().resolve()),
        env=env,
        check=False,
    )
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
