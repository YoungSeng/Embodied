#!/usr/bin/env python3
"""Render and submit the single LocateAnything UI5 Merlin job template."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from locany_ui5_common import (
    DEFAULT_CONFIG_PATH,
    PROJECT_ROOT,
    machine_resource_config,
    resolve_runtime_config,
)


TEMPLATE_PATH = PROJECT_ROOT / "jobs" / "locany_ui5_merlin.template.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render and submit a LocateAnything UI5 v4 A800/H20 4/8-GPU job"
    )
    parser.add_argument("--machine", choices=("a800", "h20"), required=True)
    parser.add_argument("--gpus", type=int, choices=(4, 8), required=True)
    parser.add_argument("--cuda-devices", default=None)
    parser.add_argument("--eval-gpu-devices", default=None)
    parser.add_argument("--max-num-tokens", type=int, default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--max-num-tokens-per-sample", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=16000)
    parser.add_argument("--save-steps", type=int, default=4000)
    parser.add_argument("--eval-interval-steps", type=int, default=1000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--learning-rate", default="2e-5")
    parser.add_argument("--version", default="v4")
    parser.add_argument("--data-version", default="v3")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--scorer-root", default=None)
    parser.add_argument(
        "--eval-checkpoint",
        default=None,
        help="Submit an evaluation-only job for this checkpoint instead of training",
    )
    parser.add_argument(
        "--eval-step",
        type=int,
        default=None,
        help="Step represented by --eval-checkpoint",
    )
    parser.add_argument(
        "--eval-skip-patch",
        action="store_true",
        help="Evaluation-only mode: do not patch the checkpoint (used for base-model step 0)",
    )
    parser.add_argument("--eval-fail-policy", choices=("stop", "warn"), default="stop")
    eval_group = parser.add_mutually_exclusive_group()
    eval_group.add_argument("--enable-eval", dest="enable_eval", action="store_true")
    eval_group.add_argument("--disable-eval", dest="enable_eval", action="store_false")
    parser.set_defaults(enable_eval=True)
    start_group = parser.add_mutually_exclusive_group()
    start_group.add_argument("--eval-at-start", dest="eval_at_start", action="store_true")
    start_group.add_argument("--no-eval-at-start", dest="eval_at_start", action="store_false")
    parser.set_defaults(eval_at_start=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--template", type=Path, default=TEMPLATE_PATH)
    parser.add_argument("--output-yaml", type=Path, default=None)
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--mlx-bin", default="mlx")
    return parser.parse_args()


def yaml_scalar(value: Any) -> str:
    text = str(value)
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def render_template(template: str, replacements: dict[str, str]) -> str:
    rendered = template
    for key, value in replacements.items():
        rendered = rendered.replace(f"@@{key}@@", value)
    unresolved = sorted(set(re.findall(r"@@[A-Z0-9_]+@@", rendered)))
    if unresolved:
        raise ValueError(f"Unresolved template placeholders: {unresolved}")
    return rendered


def build_submission_environment(args: argparse.Namespace) -> dict[str, str]:
    env = dict(os.environ)
    cuda_devices = args.cuda_devices or ",".join(
        str(index) for index in range(args.gpus)
    )
    eval_gpu_devices = args.eval_gpu_devices or ",".join(
        part.strip() for part in cuda_devices.split(",")[: min(4, args.gpus)]
    )
    explicit = {
        "MACHINE_TYPE": args.machine,
        "GPU_COUNT": str(args.gpus),
        "CUDA_DEVICES": cuda_devices,
        "EVAL_GPU_DEVICES": eval_gpu_devices,
        "MAX_STEPS": str(args.max_steps),
        "SAVE_STEPS": str(args.save_steps),
        "EVAL_INTERVAL_STEPS": str(args.eval_interval_steps),
        "WARMUP_STEPS": str(args.warmup_steps),
        "LEARNING_RATE": str(args.learning_rate),
        "VERSION": args.version,
        "DATA_VERSION": args.data_version,
        "ENABLE_EVAL": "1" if args.enable_eval else "0",
        "EVAL_AT_START": "1" if args.eval_at_start else "0",
        "EVAL_FAIL_POLICY": args.eval_fail_policy,
    }
    optional = {
        "MAX_NUM_TOKENS": args.max_num_tokens,
        "MAX_SEQ_LENGTH": args.max_seq_length,
        "MAX_NUM_TOKENS_PER_SAMPLE": args.max_num_tokens_per_sample,
        "RUN_NAME": args.run_name,
        "SCORER_ROOT": args.scorer_root,
    }
    env.update(explicit)
    env.update({key: str(value) for key, value in optional.items() if value is not None})
    if (args.eval_checkpoint is None) != (args.eval_step is None):
        raise ValueError("--eval-checkpoint and --eval-step must be provided together")
    if args.eval_checkpoint is not None:
        env.update(
            {
                "PIPELINE_MODE": "eval",
                "EVAL_CHECKPOINT": str(args.eval_checkpoint),
                "EVAL_STEP": str(args.eval_step),
                "EVAL_SKIP_PATCH": "1" if args.eval_skip_patch else "0",
            }
        )
    return env


def render_job(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    submission_env = build_submission_environment(args)
    runtime = resolve_runtime_config(submission_env, config_path=args.config)
    if re.fullmatch(r"[A-Za-z0-9._-]+", str(runtime["VERSION"])) is None:
        raise ValueError("VERSION may contain only letters, digits, '.', '_', and '-'")
    for key in ("EVAL_CHECKPOINT", "EVAL_STEP", "EVAL_SKIP_PATCH"):
        if key in submission_env:
            runtime[key] = submission_env[key]
    resource = machine_resource_config(args.machine, config_path=args.config)
    env_keys = (
        "MACHINE_TYPE",
        "GPU_COUNT",
        "CUDA_DEVICES",
        "EVAL_GPU_DEVICES",
        "DATA_VERSION",
        "VERSION",
        "MAX_STEPS",
        "WARMUP_STEPS",
        "LEARNING_RATE",
        "MAX_SEQ_LENGTH",
        "MAX_NUM_TOKENS_PER_SAMPLE",
        "MAX_NUM_TOKENS",
        "SAVE_STEPS",
        "ENABLE_EVAL",
        "EVAL_AT_START",
        "EVAL_INTERVAL_STEPS",
        "EVAL_FAIL_POLICY",
        "RUN_NAME",
        "PIPELINE_MODE",
    )
    if args.scorer_root:
        env_keys = (*env_keys, "SCORER_ROOT")
    if args.eval_checkpoint is not None:
        env_keys = (*env_keys, "EVAL_CHECKPOINT", "EVAL_STEP", "EVAL_SKIP_PATCH")
    envs_list = "\n".join(
        f"    {key}: {yaml_scalar(runtime[key])}" for key in env_keys
    )
    queue_name = resource.get("queue_name", "")
    queue_line = (
        f"          queueName: {queue_name}" if queue_name else ""
    )
    job_name = f"locany-ui5-{runtime['VERSION']}-{args.machine}x{args.gpus}"
    if args.eval_checkpoint is not None:
        job_name += f"-eval{args.eval_step}"
    replacements = {
        "CAPTION": f"LocateAnything UI5 {runtime['VERSION']} - {args.machine.upper()} x {args.gpus}",
        "PROJECT_ROOT": resource["project_root"],
        "ENV_DIR": resource["env_dir"],
        "IMAGE_URL": resource["image_url"],
        "IMAGE_VID": resource["image_vid"],
        "JOB_NAME": job_name,
        "CLUSTER_ID": str(resource["cluster_id"]),
        "GROUP_ID": str(resource["group_id"]),
        "BYTENAS_NAME": resource["bytenas_name"],
        "CPU": str(resource["cpu"]),
        "GPU_COUNT": str(args.gpus),
        "GPUV": resource["gpuv"],
        "MEMORY": str(resource["memory"]),
        "QUEUE_NAME_LINE": queue_line,
        "VOLUME_ID": str(resource["volume_id"]),
        "MOUNT_PATH": resource["mount_path"],
        "ENVS_LIST": envs_list,
    }
    template = args.template.read_text(encoding="utf-8")
    return render_template(template, replacements), runtime


def main() -> int:
    args = parse_args()
    rendered, runtime = render_job(args)
    if args.output_yaml is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_yaml = (
            PROJECT_ROOT
            / "jobs"
            / "rendered"
            / f"locany_ui5_{args.machine}x{args.gpus}_{runtime['VERSION']}_{stamp}.yaml"
        )
    else:
        output_yaml = args.output_yaml.expanduser().resolve()
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    output_yaml.write_text(rendered, encoding="utf-8")

    print("===== LocateAnything UI5 submission =====")
    for key in (
        "MACHINE_TYPE",
        "GPU_COUNT",
        "CUDA_DEVICES",
        "ATTN_IMPLEMENTATION",
        "MAX_SEQ_LENGTH",
        "MAX_NUM_TOKENS_PER_SAMPLE",
        "MAX_NUM_TOKENS",
        "MAX_NUM_TOKENS_SCOPE",
        "MAX_STEPS",
        "SAVE_STEPS",
        "ENABLE_EVAL",
        "EVAL_AT_START",
        "EVAL_INTERVAL_STEPS",
    ):
        print(f"{key:28s}: {runtime[key]}")
    print(f"rendered_yaml               : {output_yaml}")

    if args.render_only:
        print("[RENDER ONLY] mlx was not invoked")
        return 0
    command = [args.mlx_bin, "job", "submitv2", "--path", str(output_yaml)]
    print("submit_command              :", " ".join(command))
    try:
        completed = subprocess.run(command, check=False)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"Cannot find {args.mlx_bin!r}. Run this command on a host with mlx installed, "
            f"or use --render-only. Rendered YAML: {output_yaml}"
        ) from exc
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
