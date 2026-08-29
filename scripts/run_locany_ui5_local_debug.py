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
    parser.add_argument(
        "--tc-msed-stage",
        choices=("v4", "m1", "m2", "m3", "m4", "m5", "m31"),
        default="v4",
        help="TC-MSED ablation stage; uses the same stage switch as formal jobs",
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--env-dir", type=Path, default=None)
    parser.add_argument("--base-model", type=Path, default=None)
    parser.add_argument("--meta-path", type=Path, default=None)
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
            "TC_MSED_STAGE": args.tc_msed_stage,
        }
    )
    if args.tc_msed_stage == "m31":
        env.update(
            {
                "RELATION_GATE_MODE": "observe",
                "RELATION_GATE_LOSS_WEIGHT": "0.0",
                "RELATION_SLOT_GATE_LOSS_WEIGHT": "0.5",
                "RELATION_SLOT_OBJECTNESS_LOSS_WEIGHT": "0.5",
                "RELATION_ATTENTION_LOSS_WEIGHT": "0.2",
                "RELATION_BOX_L1_LOSS_WEIGHT": "1.0",
                "RELATION_BOX_GIOU_LOSS_WEIGHT": "1.0",
                "RELATION_COVERAGE_LOSS_WEIGHT": "0.05",
                "RELATION_TASK_HARD_ROUTER": "1",
                "RELATION_TASK_EXPERT_RANK": "8",
                "RELATION_SET_DECODER_LAYERS": "3",
                "RELATION_NUM_SLOTS": "8",
            }
        )
    optional_paths = {
        "ENV_DIR": args.env_dir,
        "BASE_MODEL": args.base_model,
        "MODEL_PATH": args.base_model,
        "META_PATH": args.meta_path,
        "TRAINING_DATA_DIR": args.training_data_dir,
        "OUTPUT_DIR": args.output_dir,
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


def validate_smoke_checkpoint(
    args: argparse.Namespace,
    env: Mapping[str, str],
) -> None:
    """Require the final local-smoke checkpoint to be fully resumable."""

    project_root = args.project_root.expanduser().resolve()
    if args.output_dir is not None:
        output_dir = args.output_dir.expanduser().resolve()
    else:
        output_dir = Path(env["OUTPUT_BASE"]) / env["RUN_NAME"]
    checkpoint = output_dir / f"checkpoint-{args.max_steps}"
    validator = project_root / "scripts" / "locany_ui5_checkpoint.py"
    command = [
        env["ENV_DIR"] + "/bin/python",
        str(validator),
        "validate",
        "--checkpoint",
        str(checkpoint),
        "--mode",
        "resume",
        "--expected-ranks",
        str(args.gpus),
    ]
    print("===== local smoke resume checkpoint audit =====")
    print(f"checkpoint                    : {checkpoint}")
    completed = subprocess.run(
        command,
        cwd=str(project_root),
        env=dict(env),
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Training exited successfully, but the final smoke checkpoint is not "
            f"resumable: {checkpoint}"
        )
    print("LOCAL_SMOKE_CHECKPOINT_STATUS : RESUMABLE")


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
        "TC_MSED_STAGE",
        "MAX_STEPS",
        "SAVE_STEPS",
        "ENABLE_EVAL",
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
    if completed.returncode != 0:
        return int(completed.returncode)
    validate_smoke_checkpoint(args, env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
