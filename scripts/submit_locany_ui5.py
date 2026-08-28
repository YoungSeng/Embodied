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
    DEFAULT_UI5_FULL_TEST_UNIQUE_IMAGES,
    DEFAULT_CONFIG_PATH,
    PROJECT_ROOT,
    assert_gpu_mode_consistency,
    machine_resource_config,
    resolve_runtime_config,
)


TEMPLATE_PATH = PROJECT_ROOT / "jobs" / "locany_ui5_merlin.template.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render and submit a LocateAnything UI5 v4 A800/H20 4/8-GPU job"
    )
    parser.add_argument("--machine", choices=("a800", "h20"), required=True)
    parser.add_argument(
        "--resource-group",
        default="default",
        help=(
            "Merlin resource profile from configs/locany_ui5_machines.json; "
            "A800 supports default and aiai_locate"
        ),
    )
    parser.add_argument(
        "--gpus", "--gpu", dest="gpus", type=int, choices=(4, 8), required=True
    )
    parser.add_argument("--cuda-devices", default=None)
    parser.add_argument("--eval-gpu-devices", default=None)
    parser.add_argument("--max-num-tokens", type=int, default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--max-num-tokens-per-sample", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=16000)
    parser.add_argument("--save-steps", type=int, default=4000)
    parser.add_argument("--eval-interval-steps", type=int, default=1000)
    parser.add_argument(
        "--eval-max-images-per-task",
        type=int,
        default=None,
        help="Smoke evaluation limit per UI5 task; omitted/0 keeps the full set",
    )
    parser.add_argument(
        "--eval-inference-crop-mode",
        choices=("full_image", "lossless_tiling", "detector_scan"),
        default="detector_scan",
    )
    parser.add_argument("--eval-parser-root", default=None)
    parser.add_argument("--eval-detector-cache", default=None)
    parser.add_argument(
        "--eval-detector-cache-mode", choices=("build", "readonly"), default="readonly"
    )
    parser.add_argument(
        "--eval-scan-name", default="horizontal_scan_v5_raw_detector_edge_aligned"
    )
    parser.add_argument(
        "--require-cache-scope",
        choices=("preview", "full_test"),
        default="full_test",
    )
    parser.add_argument(
        "--require-strict-nonoverlap",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--require-raw-detector-edge-alignment",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--require-detector-unique-containment",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--eval-expected-unique-images",
        type=int,
        default=DEFAULT_UI5_FULL_TEST_UNIQUE_IMAGES,
    )
    parser.add_argument("--eval-text-python", default=None)
    parser.add_argument("--eval-icon-python", default=None)
    parser.add_argument("--eval-text-model-dir", default=None)
    parser.add_argument("--eval-icon-model", default=None)
    parser.add_argument("--eval-detector-workers-per-gpu", type=int, choices=(1, 2), default=1)
    parser.add_argument("--eval-tile-max-count", type=int, default=10)
    parser.add_argument("--eval-tile-target-long-side", type=int, default=1600)
    parser.add_argument("--eval-tile-overlap-ratio", type=float, default=0.10)
    parser.add_argument("--eval-tile-nms-iou", type=float, default=0.50)
    parser.add_argument("--eval-scan-target-height", type=int, default=960)
    parser.add_argument("--eval-scan-target-guard-ratio", type=float, default=0.0)
    parser.add_argument("--eval-scan-target-guard-min-pixels", type=int, default=0)
    parser.add_argument("--eval-scan-target-guard-max-pixels", type=int, default=0)
    parser.add_argument("--eval-scan-vertical-link-ratio", type=float, default=0.025)
    parser.add_argument("--eval-scan-context-ratio", type=float, default=0.20)
    parser.add_argument("--eval-scan-min-context-image-ratio", type=float, default=0.015)
    parser.add_argument("--eval-scan-dense-band-ratio", type=float, default=0.80)
    parser.add_argument("--eval-scan-visualization-samples", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--learning-rate", default="2e-5")
    parser.add_argument("--version", default="v4")
    parser.add_argument("--data-version", default="v3")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--scorer-root", default=None)
    parser.add_argument(
        "--training-data-source-dir",
        default=None,
        help=(
            "Existing UI5 training-data directory used to bootstrap a new project when its "
            "META_PATH is missing"
        ),
    )
    parser.add_argument(
        "--training-data-dir",
        default=None,
        help="Training-data destination; defaults to PROJECT_ROOT/data/ui_defect_locany_<version>",
    )
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
    parser.add_argument(
        "--relation-gate-mode", choices=("observe", "hard"), default="observe"
    )
    parser.add_argument("--relation-gate-threshold", type=float, default=None)
    parser.add_argument(
        "--use-detection-crops",
        action="store_true",
        help="Train from the audited full+crop recipe (requires --crop-audit-dir)",
    )
    parser.add_argument("--crop-audit-dir", default=None)
    parser.add_argument(
        "--crop-train-mode",
        choices=("full_only", "full_plus_crop"),
        default=None,
    )
    parser.add_argument("--crop-meta-path", default=None)
    runtime_deps_group = parser.add_mutually_exclusive_group()
    runtime_deps_group.add_argument(
        "--install-system-runtime-deps",
        dest="install_system_runtime_deps",
        action="store_true",
        help="Install libgl1/libglib2.0-0 inside the task container when cv2 needs them",
    )
    runtime_deps_group.add_argument(
        "--no-install-system-runtime-deps",
        dest="install_system_runtime_deps",
        action="store_false",
        help="Fail preflight instead of installing missing task-container libraries",
    )
    parser.set_defaults(install_system_runtime_deps=True)
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
    return parser.parse_args(argv)


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
    resource_group = str(getattr(args, "resource_group", "default"))
    use_detection_crops = bool(getattr(args, "use_detection_crops", False))
    crop_train_mode = getattr(args, "crop_train_mode", None) or (
        "full_plus_crop" if use_detection_crops else "full_only"
    )
    cuda_devices = args.cuda_devices or ",".join(
        str(index) for index in range(args.gpus)
    )
    eval_gpu_devices = args.eval_gpu_devices or ",".join(
        part.strip() for part in cuda_devices.split(",")[: min(4, args.gpus)]
    )
    explicit = {
        "MACHINE_TYPE": args.machine,
        "RESOURCE_GROUP": resource_group,
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
        "EVAL_INFERENCE_CROP_MODE": getattr(
            args, "eval_inference_crop_mode", "detector_scan"
        ),
        "EVAL_DETECTOR_WORKERS_PER_GPU": str(
            getattr(args, "eval_detector_workers_per_gpu", 1)
        ),
        "EVAL_DETECTOR_CACHE_MODE": str(
            getattr(args, "eval_detector_cache_mode", "readonly")
        ),
        "EVAL_SCAN_NAME": str(
            getattr(
                args,
                "eval_scan_name",
                "horizontal_scan_v5_raw_detector_edge_aligned",
            )
        ),
        "EVAL_REQUIRE_CACHE_SCOPE": str(
            getattr(args, "require_cache_scope", "full_test")
        ),
        "EVAL_REQUIRE_STRICT_NONOVERLAP": (
            "1" if getattr(args, "require_strict_nonoverlap", True) else "0"
        ),
        "EVAL_REQUIRE_RAW_DETECTOR_EDGE_ALIGNMENT": (
            "1"
            if getattr(args, "require_raw_detector_edge_alignment", True)
            else "0"
        ),
        "EVAL_REQUIRE_DETECTOR_UNIQUE_CONTAINMENT": (
            "1"
            if getattr(args, "require_detector_unique_containment", True)
            else "0"
        ),
        "EVAL_EXPECTED_UNIQUE_IMAGES": str(
            getattr(
                args,
                "eval_expected_unique_images",
                DEFAULT_UI5_FULL_TEST_UNIQUE_IMAGES,
            )
        ),
        "EVAL_TILE_MAX_COUNT": str(getattr(args, "eval_tile_max_count", 10)),
        "EVAL_TILE_TARGET_LONG_SIDE": str(
            getattr(args, "eval_tile_target_long_side", 1600)
        ),
        "EVAL_TILE_OVERLAP_RATIO": str(
            getattr(args, "eval_tile_overlap_ratio", 0.10)
        ),
        "EVAL_TILE_NMS_IOU": str(getattr(args, "eval_tile_nms_iou", 0.50)),
        "EVAL_SCAN_TARGET_HEIGHT": str(getattr(args, "eval_scan_target_height", 960)),
        "EVAL_SCAN_TARGET_GUARD_RATIO": str(
            getattr(args, "eval_scan_target_guard_ratio", 0.0)
        ),
        "EVAL_SCAN_TARGET_GUARD_MIN_PIXELS": str(
            getattr(args, "eval_scan_target_guard_min_pixels", 0)
        ),
        "EVAL_SCAN_TARGET_GUARD_MAX_PIXELS": str(
            getattr(args, "eval_scan_target_guard_max_pixels", 0)
        ),
        "EVAL_SCAN_VERTICAL_LINK_RATIO": str(
            getattr(args, "eval_scan_vertical_link_ratio", 0.025)
        ),
        "EVAL_SCAN_CONTEXT_RATIO": str(
            getattr(args, "eval_scan_context_ratio", 0.20)
        ),
        "EVAL_SCAN_MIN_CONTEXT_IMAGE_RATIO": str(
            getattr(args, "eval_scan_min_context_image_ratio", 0.015)
        ),
        "EVAL_SCAN_DENSE_BAND_RATIO": str(
            getattr(args, "eval_scan_dense_band_ratio", 0.80)
        ),
        "EVAL_SCAN_VISUALIZATION_SAMPLES": str(
            getattr(args, "eval_scan_visualization_samples", 20)
        ),
        "RELATION_GATE_MODE": getattr(args, "relation_gate_mode", "observe"),
        "INSTALL_SYSTEM_RUNTIME_DEPS": (
            "1" if getattr(args, "install_system_runtime_deps", True) else "0"
        ),
        "UI5_USE_DETECTION_CROPS": "1" if use_detection_crops else "0",
        "UI5_CROP_TRAIN_MODE": crop_train_mode,
    }
    optional = {
        "MAX_NUM_TOKENS": args.max_num_tokens,
        "MAX_SEQ_LENGTH": args.max_seq_length,
        "MAX_NUM_TOKENS_PER_SAMPLE": args.max_num_tokens_per_sample,
        "EVAL_MAX_IMAGES_PER_TASK": getattr(
            args, "eval_max_images_per_task", None
        ),
        "RUN_NAME": args.run_name,
        "SCORER_ROOT": args.scorer_root,
        "RELATION_GATE_THRESHOLD": getattr(args, "relation_gate_threshold", None),
        "TRAINING_DATA_SOURCE_DIR": args.training_data_source_dir,
        "TRAINING_DATA_DIR": args.training_data_dir,
        "UI5_CROP_AUDIT_DIR": getattr(args, "crop_audit_dir", None),
        "UI5_CROP_META_PATH": getattr(args, "crop_meta_path", None),
        "EVAL_PARSER_ROOT": getattr(args, "eval_parser_root", None),
        "EVAL_DETECTOR_CACHE": getattr(args, "eval_detector_cache", None),
        "EVAL_TEXT_PYTHON": getattr(args, "eval_text_python", None),
        "EVAL_ICON_PYTHON": getattr(args, "eval_icon_python", None),
        "EVAL_TEXT_MODEL_DIR": getattr(args, "eval_text_model_dir", None),
        "EVAL_ICON_MODEL": getattr(args, "eval_icon_model", None),
    }
    env.update(explicit)
    eval_max_images = optional["EVAL_MAX_IMAGES_PER_TASK"]
    if eval_max_images is not None and int(eval_max_images) < 0:
        raise ValueError("--eval-max-images-per-task cannot be negative")
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
    parity_base = dict(submission_env)
    for key in (
        "GPU_COUNT",
        "CUDA_DEVICES",
        "MAX_NUM_TOKENS",
        "RUN_NAME",
        "OUTPUT_DIR",
    ):
        parity_base.pop(key, None)
    four_env = {
        **parity_base,
        "GPU_COUNT": "4",
        "CUDA_DEVICES": "0,1,2,3",
        "EVAL_GPU_DEVICES": "0,1,2,3",
    }
    eight_env = {
        **parity_base,
        "GPU_COUNT": "8",
        "CUDA_DEVICES": "0,1,2,3,4,5,6,7",
        "EVAL_GPU_DEVICES": "0,1,2,3",
    }
    assert_gpu_mode_consistency(
        resolve_runtime_config(four_env, config_path=args.config),
        resolve_runtime_config(eight_env, config_path=args.config),
    )
    runtime = resolve_runtime_config(submission_env, config_path=args.config)
    if re.fullmatch(r"[A-Za-z0-9._-]+", str(runtime["VERSION"])) is None:
        raise ValueError("VERSION may contain only letters, digits, '.', '_', and '-'")
    for key in ("EVAL_CHECKPOINT", "EVAL_STEP", "EVAL_SKIP_PATCH"):
        if key in submission_env:
            runtime[key] = submission_env[key]
    resource = machine_resource_config(
        args.machine,
        resource_group=str(runtime["RESOURCE_GROUP"]),
        config_path=args.config,
    )
    runtime.update(
        {
            "RESOURCE_GROUP_ID": int(resource["group_id"]),
            "RESOURCE_QUEUE_NAME": str(resource.get("queue_name", "")),
            "RESOURCE_DISPLAY_NAME": str(
                resource.get("display_name", runtime["RESOURCE_GROUP"])
            ),
        }
    )
    env_keys = (
        "MACHINE_TYPE",
        "RESOURCE_GROUP",
        "GPU_COUNT",
        "CUDA_DEVICES",
        "EVAL_GPU_DEVICES",
        "DATA_VERSION",
        "VERSION",
        "MAX_STEPS",
        "WARMUP_STEPS",
        "LEARNING_RATE",
        "GRADIENT_ACCUMULATION_STEPS",
        "RELATION_GATE_LOSS_WEIGHT",
        "RELATION_SLOT_GATE_LOSS_WEIGHT",
        "RELATION_ATTENTION_LOSS_WEIGHT",
        "RELATION_GATE_THRESHOLD",
        "RELATION_GATE_MODE",
        "RELATION_FOCAL_BETA",
        "RELATION_FOCAL_GAMMA",
        "RELATION_NUM_SLOTS",
        "MAX_SEQ_LENGTH",
        "MAX_NUM_TOKENS_PER_SAMPLE",
        "MAX_NUM_TOKENS",
        "SAVE_STEPS",
        "ENABLE_EVAL",
        "EVAL_AT_START",
        "EVAL_INTERVAL_STEPS",
        "EVAL_MAX_IMAGES_PER_TASK",
        "EVAL_FAIL_POLICY",
        "EVAL_INFERENCE_CROP_MODE",
        "EVAL_PARSER_ROOT",
        "EVAL_DETECTOR_CACHE",
        "EVAL_DETECTOR_CACHE_MODE",
        "EVAL_SCAN_NAME",
        "EVAL_REQUIRE_CACHE_SCOPE",
        "EVAL_REQUIRE_STRICT_NONOVERLAP",
        "EVAL_REQUIRE_RAW_DETECTOR_EDGE_ALIGNMENT",
        "EVAL_REQUIRE_DETECTOR_UNIQUE_CONTAINMENT",
        "EVAL_EXPECTED_UNIQUE_IMAGES",
        "EVAL_TEXT_PYTHON",
        "EVAL_ICON_PYTHON",
        "EVAL_TEXT_MODEL_DIR",
        "EVAL_ICON_MODEL",
        "EVAL_DETECTOR_WORKERS_PER_GPU",
        "EVAL_TILE_MAX_COUNT",
        "EVAL_TILE_TARGET_LONG_SIDE",
        "EVAL_TILE_OVERLAP_RATIO",
        "EVAL_TILE_NMS_IOU",
        "EVAL_SCAN_TARGET_HEIGHT",
        "EVAL_SCAN_TARGET_GUARD_RATIO",
        "EVAL_SCAN_TARGET_GUARD_MIN_PIXELS",
        "EVAL_SCAN_TARGET_GUARD_MAX_PIXELS",
        "EVAL_SCAN_VERTICAL_LINK_RATIO",
        "EVAL_SCAN_CONTEXT_RATIO",
        "EVAL_SCAN_MIN_CONTEXT_IMAGE_RATIO",
        "EVAL_SCAN_DENSE_BAND_RATIO",
        "EVAL_SCAN_VISUALIZATION_SAMPLES",
        "INSTALL_SYSTEM_RUNTIME_DEPS",
        "UI5_USE_DETECTION_CROPS",
        "UI5_CROP_AUDIT_DIR",
        "UI5_CROP_TRAIN_MODE",
        "UI5_CROP_META_PATH",
        "RUN_NAME",
        "PIPELINE_MODE",
    )
    if args.scorer_root:
        env_keys = (*env_keys, "SCORER_ROOT")
    if args.training_data_source_dir:
        env_keys = (*env_keys, "TRAINING_DATA_SOURCE_DIR")
    if args.training_data_dir:
        env_keys = (*env_keys, "TRAINING_DATA_DIR")
    if args.eval_checkpoint is not None:
        env_keys = (*env_keys, "EVAL_CHECKPOINT", "EVAL_STEP", "EVAL_SKIP_PATCH")
    envs_list = "\n".join(
        f"    {key}: {yaml_scalar(runtime[key])}" for key in env_keys
    )
    queue_name = resource.get("queue_name", "")
    queue_line = (
        f"          queueName: {queue_name}" if queue_name else ""
    )
    resource_group = str(runtime["RESOURCE_GROUP"])
    resource_job_label = re.sub(r"[^A-Za-z0-9.-]+", "-", resource_group)
    resource_suffix = "" if resource_group == "default" else f"-{resource_job_label}"
    job_name = (
        f"locany-ui5-{runtime['VERSION']}-{args.machine}x{args.gpus}"
        f"{resource_suffix}"
    )
    if args.eval_checkpoint is not None:
        job_name += f"-eval{args.eval_step}"
    replacements = {
        "CAPTION": (
            f"LocateAnything UI5 {runtime['VERSION']} - "
            f"{args.machine.upper()} x {args.gpus} [{resource_group}]"
        ),
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
        "RESOURCE_GROUP",
        "GPU_COUNT",
        "CUDA_DEVICES",
        "ATTN_IMPLEMENTATION",
        "MAX_SEQ_LENGTH",
        "MAX_NUM_TOKENS_PER_SAMPLE",
        "MAX_NUM_TOKENS",
        "MAX_NUM_TOKENS_SCOPE",
        "GRADIENT_ACCUMULATION_STEPS",
        "RELATION_GATE_LOSS_WEIGHT",
        "RELATION_ATTENTION_LOSS_WEIGHT",
        "RELATION_GATE_THRESHOLD",
        "RELATION_FOCAL_BETA",
        "RELATION_FOCAL_GAMMA",
        "RELATION_NUM_SLOTS",
        "MAX_STEPS",
        "SAVE_STEPS",
        "ENABLE_EVAL",
        "EVAL_AT_START",
        "EVAL_INTERVAL_STEPS",
        "EVAL_MAX_IMAGES_PER_TASK",
        "EVAL_INFERENCE_CROP_MODE",
        "EVAL_PARSER_ROOT",
        "EVAL_DETECTOR_CACHE",
        "EVAL_DETECTOR_CACHE_MODE",
        "EVAL_SCAN_NAME",
        "EVAL_REQUIRE_CACHE_SCOPE",
        "EVAL_REQUIRE_STRICT_NONOVERLAP",
        "EVAL_REQUIRE_RAW_DETECTOR_EDGE_ALIGNMENT",
        "EVAL_REQUIRE_DETECTOR_UNIQUE_CONTAINMENT",
        "EVAL_EXPECTED_UNIQUE_IMAGES",
        "EVAL_TEXT_PYTHON",
        "EVAL_ICON_PYTHON",
        "EVAL_ICON_MODEL",
        "EVAL_DETECTOR_WORKERS_PER_GPU",
        "EVAL_TILE_MAX_COUNT",
        "EVAL_TILE_TARGET_LONG_SIDE",
        "EVAL_TILE_OVERLAP_RATIO",
        "EVAL_TILE_NMS_IOU",
        "EVAL_SCAN_TARGET_HEIGHT",
        "EVAL_SCAN_TARGET_GUARD_RATIO",
        "EVAL_SCAN_TARGET_GUARD_MIN_PIXELS",
        "EVAL_SCAN_TARGET_GUARD_MAX_PIXELS",
        "EVAL_SCAN_VERTICAL_LINK_RATIO",
        "EVAL_SCAN_CONTEXT_RATIO",
        "EVAL_SCAN_MIN_CONTEXT_IMAGE_RATIO",
        "EVAL_SCAN_DENSE_BAND_RATIO",
        "INSTALL_SYSTEM_RUNTIME_DEPS",
        "UI5_USE_DETECTION_CROPS",
        "UI5_CROP_AUDIT_DIR",
        "UI5_CROP_TRAIN_MODE",
        "UI5_CROP_META_PATH",
        "META_PATH",
    ):
        print(f"{key:28s}: {runtime[key]}")
    print(f"rendered_yaml               : {output_yaml}")
    print(f"resource_group_id           : {runtime['RESOURCE_GROUP_ID']}")
    print(f"resource_display_name       : {runtime['RESOURCE_DISPLAY_NAME']}")
    print(f"resource_queue_name         : {runtime['RESOURCE_QUEUE_NAME'] or '<default>'}")

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
