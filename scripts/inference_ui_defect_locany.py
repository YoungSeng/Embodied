#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch inference for five UI-defect tasks with a full LocateAnything checkpoint.

The parsed per-image JSON keeps the same top-level structure as the previous
YOLO script: a list of detections. LocateAnything does not expose a calibrated
per-box confidence, so ``confidence`` is ``null`` unless
``--compat-confidence`` is explicitly supplied.

Example:

python inference_ui_defect_locany.py \
  --checkpoint /path/to/checkpoint-25000 \
  --input-dir /path/to/data \
  --output-dir /path/to/locany_results \
  --cuda-visible-devices 0 \
  --skip-figma \
  --tag-filename \
  --save-raw-answer \
  --save-visualization

``--cuda-visible-devices`` is applied before PyTorch is imported. When needed,
the script transparently re-executes itself once so that physical GPU IDs are
actually isolated. After isolation, ``--device cuda:0`` means the first GPU in
the supplied visible-device list.
"""

from __future__ import annotations

import argparse
import os
import sys


def _bootstrap_cuda_visible_devices() -> None:
    """Apply the CLI CUDA mask before importing torch.

    Setting CUDA_VISIBLE_DEVICES after torch has initialized CUDA is unreliable.
    A small pre-parser and one exec solve that while keeping GPU selection a
    normal command-line option.
    """

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--cuda-visible-devices",
        "--cuda_visible_devices",
        dest="cuda_visible_devices",
        default=None,
    )
    known, _ = parser.parse_known_args()
    requested = known.cuda_visible_devices
    if requested is None:
        return

    requested = str(requested).strip()
    current = os.environ.get("CUDA_VISIBLE_DEVICES")
    bootstrapped = os.environ.get("_LOCANY_CUDA_BOOTSTRAPPED") == "1"

    if current != requested and not bootstrapped:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = requested
        env["_LOCANY_CUDA_BOOTSTRAPPED"] = "1"
        os.execvpe(sys.executable, [sys.executable, *sys.argv], env)

    # This branch is normally reached after exec, or when the requested value
    # was already present in the environment.
    os.environ["CUDA_VISIBLE_DEVICES"] = requested
    os.environ["_LOCANY_CUDA_BOOTSTRAPPED"] = "1"


_bootstrap_cuda_visible_devices()


import hashlib
import json
import random
import re
import time
import traceback
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer

from eaglevl.model.locany.relation_modules import UI_RELATION_PROMPT_SPECS
from locany_ui5_common import aggregate_tiled_gate_diagnostics
from ui5_lossless_tiling import (
    assert_lossless_coverage,
    generate_lossless_tiles,
    merge_tile_predictions,
)


PROMPT_TEMPLATE = (
    "Locate all the instances that match the following description: {label}."
)
RELATION_SPEC_BY_TASK = {
    spec.task_name: spec for spec in UI_RELATION_PROMPT_SPECS
}


@dataclass(frozen=True)
class TaskConfig:
    task_name: str
    jsonl_name: str
    class_id: int
    output_label: str
    prompt_label: str

    @property
    def prompt(self) -> str:
        return PROMPT_TEMPLATE.format(label=self.prompt_label)


# Order, JSONL names, class IDs, and Chinese output labels follow the YOLO script.
# prompt_label follows prepare_ui_defect_locany.py exactly when --label-style en.
TASK_CONFIGS = [
    TaskConfig(
        task_name="text_overflow",
        jsonl_name="test_ui_text_overflow_wcnt_no_figma.jsonl",
        class_id=2,
        output_label="文字溢出",
        prompt_label=RELATION_SPEC_BY_TASK["text_overflow"].prompt_label,
    ),
    TaskConfig(
        task_name="text_ellipsis",
        jsonl_name="test_ui_text_ellipsis_wcnt_no_figma.jsonl",
        class_id=3,
        output_label="文本省略",
        prompt_label=RELATION_SPEC_BY_TASK["text_ellipsis"].prompt_label,
    ),
    TaskConfig(
        task_name="occlusion",
        jsonl_name="test_ui_occlusion_wcnt_no_figma.jsonl",
        class_id=0,
        output_label="元素遮挡",
        prompt_label=RELATION_SPEC_BY_TASK["occlusion"].prompt_label,
    ),
    TaskConfig(
        task_name="cropping",
        jsonl_name="test_ui_cropping_wcnt_no_figma.jsonl",
        class_id=1,
        output_label="元素裁切",
        prompt_label=RELATION_SPEC_BY_TASK["cropping"].prompt_label,
    ),
    TaskConfig(
        task_name="content_missing",
        jsonl_name="test_ui_content_missing_wcnt_no_figma.jsonl",
        class_id=4,
        output_label="内容缺失",
        prompt_label=RELATION_SPEC_BY_TASK["content_missing"].prompt_label,
    ),
]

TASK_BY_NAME = {task.task_name: task for task in TASK_CONFIGS}


@dataclass
class TaskWork:
    config: TaskConfig
    jsonl_path: Path
    output_dir: Path
    image_paths: list[str]
    output_stems: dict[str, str]
    pending_paths: list[str]
    skipped_existing: int


@dataclass
class ParsedAnswer:
    status: str
    normalized_boxes: list[list[int]]
    refs: list[str]
    has_none_token: bool
    warnings: list[str]


def parse_optional_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


def parse_args() -> argparse.Namespace:
    task_choices = ["all", *[task.task_name for task in TASK_CONFIGS]]

    parser = argparse.ArgumentParser(
        description=(
            "使用 LocateAnything 全参数 checkpoint 对五类移动端 UI 缺陷批量推理"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        "--model-path",
        "--model_path",
        dest="checkpoint",
        required=True,
        help="LocateAnything 全参数微调 checkpoint 目录",
    )
    parser.add_argument(
        "--processor-path",
        "--processor_path",
        default=None,
        help=(
            "Tokenizer/processor 来源；默认与 checkpoint 相同。仅当 checkpoint "
            "未保存 processor 文件时，显式指向基础 LocateAnything 模型目录"
        ),
    )
    parser.add_argument(
        "--input-dir",
        "--input_dir",
        dest="input_dir",
        required=True,
        help="存放五个 test_ui_*_no_figma.jsonl 的目录",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        required=True,
        help="总输出目录；每类任务写入独立子目录",
    )
    parser.add_argument(
        "--summary-path",
        "--summary_path",
        dest="summary_path",
        default=None,
        help=(
            "推理汇总 JSON 路径；默认写入 OUTPUT_DIR/_summary.json。并行运行单任务时"
            "应为每个 worker 指定不同路径，避免并发覆盖"
        ),
    )
    parser.add_argument(
        "--cuda-visible-devices",
        "--cuda_visible_devices",
        dest="cuda_visible_devices",
        default=None,
        help=(
            "物理 GPU ID，例如 0、3 或 2,3；在导入 PyTorch 前设置。单模型默认仅使用"
            "隔离后的 cuda:0"
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="CUDA 隔离后的逻辑设备；单卡推理通常保持 cuda:0",
    )
    parser.add_argument(
        "--dtype",
        choices=("bf16", "fp16", "fp32"),
        default="bf16",
        help="模型和图像张量精度",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("auto", "sdpa", "flash_attention_2", "eager", "magi"),
        default="auto",
        help="auto 时沿用 checkpoint/模型默认设置",
    )
    parser.add_argument(
        "--vision-attn-implementation",
        choices=("sdpa", "flash_attention_2", "eager"),
        default="flash_attention_2",
        help="MoonViT attention 后端；A800/H20 默认均使用普通 FlashAttention 2",
    )
    parser.add_argument(
        "--generation-mode",
        "--generation_mode",
        choices=("fast", "slow", "hybrid"),
        default="hybrid",
        help="LocateAnything 生成模式",
    )
    parser.add_argument(
        "--max-new-tokens",
        "--max_new_tokens",
        type=int,
        default=4096,
        help="单图最大生成 token 数；极密集目标可提高到 8192",
    )
    parser.add_argument(
        "--n-future-tokens",
        "--n_future_tokens",
        type=int,
        default=6,
        help="fast/hybrid 模式的 PBD block 长度，与训练 --block_size 6 一致",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", "--top_p", dest="top_p", type=float, default=0.9)
    parser.add_argument("--top-k", "--top_k", dest="top_k", type=int, default=0)
    parser.add_argument(
        "--repetition-penalty",
        "--repetition_penalty",
        dest="repetition_penalty",
        type=float,
        default=1.1,
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="关闭采样。默认使用官方 worker 的采样参数，并按图片设置稳定随机种子",
    )
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=task_choices,
        default=["all"],
        help="只运行指定任务；默认运行存在 JSONL 的全部任务",
    )
    parser.add_argument(
        "--skip-figma",
        "--skip_figma",
        dest="skip_figma",
        action="store_true",
        help="跳过文件名中含 ':' 的 Figma 导出图片",
    )
    parser.add_argument(
        "--tag-filename",
        "--tag_filename",
        dest="tag_filename",
        action="store_true",
        help="结果名增加 _defect 或 _ok；解析失败始终增加 _parse_error",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖所处理图片的旧结果；默认按图片断点续推",
    )
    parser.add_argument(
        "--save-raw-answer",
        "--save_raw_answer",
        dest="save_raw_answer",
        action="store_true",
        help="在各任务 raw/ 目录保存原始回答、提示词和解析信息",
    )
    parser.add_argument(
        "--save-visualization",
        "--save_visualization",
        dest="save_visualization",
        action="store_true",
        help="在各任务 visualizations/ 目录保存画框图片",
    )
    parser.add_argument(
        "--print-raw-answer",
        "--print_raw_answer",
        dest="print_raw_answer",
        action="store_true",
        help="逐图在终端打印 LocateAnything 原始回答",
    )
    parser.add_argument(
        "--compat-confidence",
        "--compat_confidence",
        dest="compat_confidence",
        type=float,
        default=None,
        help=(
            "向 YOLO 兼容 JSON 写入固定数值 confidence。默认 null；该值只是旧评测器"
            "兼容占位，不是模型置信度"
        ),
    )
    parser.add_argument(
        "--max-images-per-task",
        "--max_images_per_task",
        dest="max_images_per_task",
        type=int,
        default=0,
        help="每个任务最多处理多少张；0 表示全部，可用于 smoke test",
    )
    parser.add_argument(
        "--inference-crop-mode",
        choices=("full_image", "lossless_tiling", "detector_scan"),
        default="full_image",
        help=(
            "推理图像模式；lossless_tiling 使用不依赖 GT 的重叠矩形切图，"
            "detector_scan 读取预先落盘的 OCR/icon 横向连通扫描 crops；"
            "两者都先把局部预测回写原图坐标再跨 tile 去重"
        ),
    )
    parser.add_argument(
        "--detector-crop-manifest",
        default=None,
        help="detector_scan 模式必需的 detector_scan_crops.jsonl；不得包含 GT repair",
    )
    parser.add_argument("--tile-max-count", type=int, default=10)
    parser.add_argument("--tile-target-long-side", type=int, default=1600)
    parser.add_argument("--tile-overlap-ratio", type=float, default=0.10)
    parser.add_argument("--tile-nms-iou", type=float, default=0.50)
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="只从本地 checkpoint/cache 加载，不访问 Hugging Face 网络",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="LocateAnything 自定义模型代码需要开启",
    )
    parser.add_argument(
        "--enable-ui-relation",
        nargs="?",
        const=True,
        type=parse_optional_bool,
        default=None,
        help=(
            "覆盖 checkpoint 中的 enable_ui_relation；未指定时读取模型配置。"
            "使用 --enable-ui-relation false 可保留原始无 Gate 生成路径。"
        ),
    )
    parser.add_argument(
        "--relation-gate-mode",
        choices=("observe", "hard"),
        default="observe",
        help="observe 始终生成并记录 Gate；hard 才允许阈值提前返回 none",
    )
    parser.add_argument(
        "--relation-gate-threshold",
        type=float,
        default=None,
        help="仅覆盖本次推理阈值，不写回 checkpoint config",
    )
    parser.add_argument(
        "--no-enable-ui-relation",
        dest="enable_ui_relation",
        action="store_false",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--use-fast-processor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="传给 AutoProcessor/AutoTokenizer 的 use_fast",
    )
    parser.add_argument(
        "--verbose-generation",
        "--verbose_generation",
        dest="verbose_generation",
        action="store_true",
        help="开启 LocateAnything generate 内部统计输出",
    )
    parser.add_argument(
        "--fail-fast",
        "--fail_fast",
        dest="fail_fast",
        action="store_true",
        help="单图失败后立即退出；默认记录错误并继续",
    )
    parser.add_argument(
        "--dry-run",
        "--dry_run",
        dest="dry_run",
        action="store_true",
        help="只检查数据、断点状态和参数，不加载模型",
    )
    parser.add_argument(
        "--load-only",
        "--load_only",
        dest="load_only",
        action="store_true",
        help=(
            "只校验并加载 tokenizer、processor 和模型后退出；用于并行推理前预热 "
            "trust_remote_code 缓存并尽早暴露模型加载错误"
        ),
    )
    parser.add_argument(
        "--preflight-forward",
        "--preflight_forward",
        dest="preflight_forward",
        action="store_true",
        help="与 --load-only 一起使用：加载后用所选任务的第一张图片执行一次真实 generation",
    )

    args = parser.parse_args()

    if "all" in args.tasks and len(args.tasks) != 1:
        parser.error("--tasks all 不能与具体任务名同时使用")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens 必须大于 0")
    if args.n_future_tokens <= 0:
        parser.error("--n-future-tokens 必须大于 0")
    if args.temperature < 0:
        parser.error("--temperature 不能小于 0")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p 必须位于 (0, 1]")
    if args.top_k < 0:
        parser.error("--top-k 不能小于 0")
    if args.repetition_penalty <= 0:
        parser.error("--repetition-penalty 必须大于 0")
    if args.max_images_per_task < 0:
        parser.error("--max-images-per-task 不能小于 0")
    if not 1 <= args.tile_max_count <= 10:
        parser.error("--tile-max-count 必须位于 [1, 10]")
    if args.tile_target_long_side <= 0:
        parser.error("--tile-target-long-side 必须大于 0")
    if not 0 < args.tile_overlap_ratio < 1:
        parser.error("--tile-overlap-ratio 必须位于 (0, 1)")
    if not 0 <= args.tile_nms_iou <= 1:
        parser.error("--tile-nms-iou 必须位于 [0, 1]")
    if args.compat_confidence is not None and not 0 <= args.compat_confidence <= 1:
        parser.error("--compat-confidence 必须位于 [0, 1]")
    if args.relation_gate_threshold is not None and not 0 <= args.relation_gate_threshold <= 1:
        parser.error("--relation-gate-threshold 必须位于 [0, 1]")
    if args.preflight_forward and not args.load_only:
        parser.error("--preflight-forward 必须与 --load-only 一起使用")

    return args


def normalize_local_or_hub_path(value: str) -> str:
    candidate = Path(value).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    return value


def validate_local_checkpoint(checkpoint: str) -> None:
    path = Path(checkpoint)
    if not path.exists():
        # A Hub model ID remains supported when --no-local-files-only is used.
        return
    if not path.is_dir():
        raise NotADirectoryError(f"checkpoint 不是目录：{path}")
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"checkpoint 缺少 config.json：{path}")

    weight_patterns = (
        "model.safetensors",
        "model.safetensors.index.json",
        "model-*.safetensors",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
        "pytorch_model-*.bin",
    )
    if not any(list(path.glob(pattern)) for pattern in weight_patterns):
        raise FileNotFoundError(
            "checkpoint 中没有找到完整模型权重（model*.safetensors 或 "
            f"pytorch_model*.bin）：{path}"
        )


def extract_image_paths_from_sample(sample: dict[str, Any], jsonl_dir: Path) -> list[str]:
    """Extract image paths using the same images/image conventions as YOLO."""

    images = sample.get("images", sample.get("image"))
    if images is None:
        return []
    if isinstance(images, (str, dict)):
        images = [images]
    if not isinstance(images, list):
        return []

    image_paths: list[str] = []
    for image in images:
        if isinstance(image, str):
            raw_path = image
        elif isinstance(image, dict):
            raw_path = image.get("path")
        else:
            continue

        if not raw_path:
            continue
        path = Path(str(raw_path)).expanduser()
        if not path.is_absolute():
            path = jsonl_dir / path
        image_paths.append(str(path.resolve(strict=False)))
    return image_paths


def get_image_paths(jsonl_path: Path, skip_figma: bool = False) -> list[str]:
    """Read a task JSONL and return valid, de-duplicated images in source order."""

    jsonl_path = jsonl_path.expanduser().resolve()
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"JSONL 文件不存在：{jsonl_path}")

    image_paths: list[str] = []
    invalid_json = 0
    missing_field = 0
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                invalid_json += 1
                print(
                    f"[WARN] {jsonl_path.name}:{line_number} JSON 解析失败，已跳过：{exc}",
                    file=sys.stderr,
                )
                continue

            paths = extract_image_paths_from_sample(sample, jsonl_path.parent)
            if not paths:
                missing_field += 1
                print(
                    f"[WARN] {jsonl_path.name}:{line_number} 没有有效 images/image 路径",
                    file=sys.stderr,
                )
                continue
            image_paths.extend(paths)

    valid_paths: list[str] = []
    missing_images = 0
    skipped_figma = 0
    for image_path in image_paths:
        if not Path(image_path).is_file():
            missing_images += 1
            continue
        if skip_figma and ":" in Path(image_path).name:
            skipped_figma += 1
            continue
        valid_paths.append(image_path)

    final_paths = list(dict.fromkeys(valid_paths))
    duplicate_count = len(valid_paths) - len(final_paths)

    if invalid_json:
        print(f"[WARN] {jsonl_path.name}: 跳过 {invalid_json} 条非法 JSON")
    if missing_field:
        print(f"[WARN] {jsonl_path.name}: 跳过 {missing_field} 条无图片路径记录")
    if missing_images:
        print(f"[WARN] {jsonl_path.name}: {missing_images} 个图片路径不存在")
    if skipped_figma:
        print(f"[INFO] {jsonl_path.name}: 跳过 {skipped_figma} 张 Figma 图片")
    if duplicate_count:
        print(f"[INFO] {jsonl_path.name}: 去除 {duplicate_count} 个重复图片路径")

    return final_paths


def legacy_output_stem(image_path: str) -> str:
    """Match the old YOLO filename rule."""

    return Path(image_path).stem.replace(":", "_")


def build_output_stems(image_paths: Sequence[str]) -> dict[str, str]:
    """Keep YOLO names, adding a hash only when different paths would collide."""

    grouped: dict[str, list[str]] = defaultdict(list)
    for image_path in image_paths:
        grouped[legacy_output_stem(image_path)].append(image_path)

    result: dict[str, str] = {}
    for base, paths in grouped.items():
        if len(paths) == 1:
            result[paths[0]] = base
            continue

        print(
            f"[WARN] 输出文件名 {base!r} 对应 {len(paths)} 张不同图片，自动追加路径哈希"
        )
        for image_path in paths:
            digest = hashlib.blake2b(
                image_path.encode("utf-8"), digest_size=5
            ).hexdigest()
            result[image_path] = f"{base}__{digest}"
    return result


def result_candidates(task_output_dir: Path, stem: str) -> list[Path]:
    return [
        task_output_dir / f"{stem}.json",
        task_output_dir / f"{stem}_defect.json",
        task_output_dir / f"{stem}_ok.json",
        task_output_dir / f"{stem}_parse_error.json",
    ]


def result_already_exists(task_output_dir: Path, stem: str) -> bool:
    return any(path.is_file() for path in result_candidates(task_output_dir, stem))


def status_suffix(status: str, tag_filename: bool) -> str:
    if status == "parse_error":
        # Never silently turn a malformed answer into an ordinary empty result.
        return "_parse_error"
    return f"_{status}" if tag_filename else ""


def get_output_path(
    task_output_dir: Path,
    stem: str,
    status: str,
    tag_filename: bool,
) -> Path:
    return task_output_dir / f"{stem}{status_suffix(status, tag_filename)}.json"


def remove_old_artifacts(task_output_dir: Path, stem: str) -> None:
    for path in result_candidates(task_output_dir, stem):
        path.unlink(missing_ok=True)

    suffixes = ("", "_defect", "_ok", "_parse_error")
    for suffix in suffixes:
        (task_output_dir / "raw" / f"{stem}{suffix}.json").unlink(missing_ok=True)
        (task_output_dir / "gate" / f"{stem}{suffix}.json").unlink(missing_ok=True)
        (task_output_dir / "visualizations" / f"{stem}{suffix}.jpg").unlink(
            missing_ok=True
        )
    (task_output_dir / "errors" / f"{stem}_error.json").unlink(missing_ok=True)


def prepare_work(args: argparse.Namespace) -> list[TaskWork]:
    selected_names = (
        {task.task_name for task in TASK_CONFIGS}
        if args.tasks == ["all"]
        else set(args.tasks)
    )

    works: list[TaskWork] = []
    for config in TASK_CONFIGS:
        if config.task_name not in selected_names:
            continue

        jsonl_path = args.input_dir / config.jsonl_name
        if not jsonl_path.is_file():
            print(f"[WARN] JSONL 不存在，跳过：{jsonl_path}")
            continue

        image_paths = get_image_paths(jsonl_path, skip_figma=args.skip_figma)
        if args.max_images_per_task:
            image_paths = image_paths[: args.max_images_per_task]

        output_dir = args.output_dir / config.task_name
        output_dir.mkdir(parents=True, exist_ok=True)
        stems = build_output_stems(image_paths)

        if args.overwrite:
            pending_paths = list(image_paths)
            skipped_existing = 0
        else:
            pending_paths = [
                image_path
                for image_path in image_paths
                if not result_already_exists(output_dir, stems[image_path])
            ]
            skipped_existing = len(image_paths) - len(pending_paths)

        works.append(
            TaskWork(
                config=config,
                jsonl_path=jsonl_path,
                output_dir=output_dir,
                image_paths=image_paths,
                output_stems=stems,
                pending_paths=pending_paths,
                skipped_existing=skipped_existing,
            )
        )

    return works


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def validate_device(device: str) -> None:
    if not device.startswith("cuda"):
        return
    if not torch.cuda.is_available():
        raise RuntimeError(
            "--device 指定了 CUDA，但 torch.cuda.is_available() 为 False。"
            "请检查 --cuda-visible-devices、驱动和 CUDA 环境。"
        )

    match = re.fullmatch(r"cuda(?::(\d+))?", device)
    if not match:
        raise ValueError(f"不支持的 CUDA 设备写法：{device}")
    index = int(match.group(1) or 0)
    if index >= torch.cuda.device_count():
        raise RuntimeError(
            f"逻辑设备 {device} 不存在；当前可见 GPU 数为 {torch.cuda.device_count()}"
        )


def configure_attention_backend(
    config: Any,
    requested: str,
    vision_requested: str,
    device: str,
) -> None:
    """Apply the requested text backend to both levels of a composite config.

    LocateAnything stores an attention choice on the outer config and its nested
    Qwen config. Passing ``attn_implementation=`` to ``from_pretrained`` only updates
    the outer value in some Transformers/custom-code combinations; the old nested
    ``magi`` value then wins during Qwen layer construction.
    """
    if requested == "magi" and device.startswith("cuda"):
        logical_index = int(device.split(":", 1)[1]) if ":" in device else 0
        capability = torch.cuda.get_device_capability(logical_index)
        if capability != (9, 0):
            raise RuntimeError(
                "ATTN_IMPLEMENTATION=magi requires an sm90 GPU, but "
                f"{torch.cuda.get_device_name(logical_index)} reports sm{capability[0]}{capability[1]}. "
                "Use sdpa on A800 (sm80); reserve magi for H20/Hopper (sm90)."
            )

    if requested != "auto":
        targets = [("model", config)]
        text_config = getattr(config, "text_config", None)
        if text_config is not None:
            targets.append(("text", text_config))
        for _, target in targets:
            # Transformers uses the property backed by _attn_implementation_internal;
            # older LocateAnything code also reads the public field directly.
            target._attn_implementation = requested
            target._attn_implementation_internal = requested
            target._attn_implementation_autoset = False
            target.attn_implementation = requested

    # Composite-config propagation may copy the outer text choice to every
    # sub-config. Apply the MoonViT choice last so text=sdpa does not turn the
    # quadratic vision path into SDPA for high-resolution UI screenshots.
    vision_config = getattr(config, "vision_config", None)
    if vision_config is not None:
        vision_config._attn_implementation = vision_requested
        vision_config._attn_implementation_internal = vision_requested
        vision_config._attn_implementation_autoset = False
        vision_config.attn_implementation = vision_requested


def enforce_vision_runtime_backend(model: Any, requested: str) -> int:
    """Set the already-instantiated MoonViT blocks to the requested backend."""
    vision_model = getattr(model, "vision_model", None)
    vision_config = getattr(getattr(model, "config", None), "vision_config", None)
    if vision_config is not None:
        vision_config._attn_implementation = requested
        vision_config._attn_implementation_internal = requested
        vision_config._attn_implementation_autoset = False
        vision_config.attn_implementation = requested
    blocks = getattr(getattr(vision_model, "encoder", None), "blocks", ())
    changed = 0
    for block in blocks:
        if hasattr(block, "attn_implementation"):
            block.attn_implementation = requested
            changed += 1
    return changed


def attention_backend_report(model: Any) -> dict[str, str]:
    model_config = getattr(model, "config", None)
    text_config = getattr(model_config, "text_config", None)
    vision_config = getattr(model_config, "vision_config", None)
    top_backend = str(getattr(model_config, "_attn_implementation", None))
    text_backend = str(getattr(text_config, "_attn_implementation", None))
    attention_class = "<unavailable>"
    vision_backend = str(getattr(vision_config, "_attn_implementation", None))
    vision_layer_backend = "<unavailable>"
    try:
        attention_class = type(model.language_model.model.layers[0].self_attn).__name__
    except (AttributeError, IndexError, TypeError):
        pass
    try:
        vision_layer_backend = str(
            model.vision_model.encoder.blocks[0].attn_implementation
        )
    except (AttributeError, IndexError, TypeError):
        pass
    return {
        "top_config": top_backend,
        "text_config": text_backend,
        "first_layer_class": attention_class,
        "vision_config": vision_backend,
        "vision_first_layer": vision_layer_backend,
    }


def apply_chat_template(processor: Any, messages: list[dict[str, Any]]) -> str:
    """Prefer LocateAnything's custom Python template, with official fallbacks."""

    if hasattr(processor, "py_apply_chat_template"):
        return processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    if hasattr(processor, "apply_chat_template"):
        return processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    raise AttributeError("processor 不提供可用的 chat template API")


def decode_generation_output(
    raw_output: Any,
    input_ids: torch.Tensor,
    processor: Any,
    tokenizer: Any,
) -> str:
    """Support both LocateAnything string output and standard token tensors."""

    if isinstance(raw_output, tuple):
        raw_output = raw_output[0]
    if isinstance(raw_output, str):
        return raw_output
    if isinstance(raw_output, list):
        if not raw_output:
            return ""
        if isinstance(raw_output[0], str):
            return raw_output[0]

    if torch.is_tensor(raw_output):
        generated_ids = raw_output
        if generated_ids.ndim == 1:
            generated_ids = generated_ids.unsqueeze(0)

        if generated_ids.ndim == 2 and generated_ids.shape[1] >= input_ids.shape[1]:
            prefix = generated_ids[:, : input_ids.shape[1]]
            if prefix.shape == input_ids.shape and torch.equal(
                prefix.detach().cpu(), input_ids.detach().cpu()
            ):
                generated_ids = generated_ids[:, input_ids.shape[1] :]

        generated_ids = generated_ids.detach().cpu()
        if hasattr(processor, "post_process_image_text_to_text"):
            decoded = processor.post_process_image_text_to_text(
                generated_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if isinstance(decoded, list):
                return str(decoded[0]) if decoded else ""
            return str(decoded)

        decoded = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        return str(decoded[0]) if isinstance(decoded, list) else str(decoded)

    return str(raw_output)


class LocateAnythingInferencer:
    def __init__(self, args: argparse.Namespace):
        validate_device(args.device)
        self.args = args
        self.device = args.device
        self.dtype = resolve_dtype(args.dtype)

        processor_source = args.processor_path or args.checkpoint
        common_kwargs = {
            "trust_remote_code": args.trust_remote_code,
            "local_files_only": args.local_files_only,
        }

        print("\n===== 加载 LocateAnything =====")
        print(f"checkpoint              : {args.checkpoint}")
        print(f"processor/tokenizer     : {processor_source}")
        print(f"CUDA_VISIBLE_DEVICES    : {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        print(f"logical device          : {self.device}")
        print(f"dtype                   : {args.dtype}")
        print(f"generation mode         : {args.generation_mode}")
        print(f"relation gate mode      : {args.relation_gate_mode}")
        print(f"relation gate override  : {args.relation_gate_threshold}")

        if self.device.startswith("cuda"):
            logical_index = int(self.device.split(":", 1)[1]) if ":" in self.device else 0
            print(f"GPU                     : {torch.cuda.get_device_name(logical_index)}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                processor_source,
                use_fast=args.use_fast_processor,
                **common_kwargs,
            )
            self.processor = AutoProcessor.from_pretrained(
                processor_source,
                use_fast=args.use_fast_processor,
                **common_kwargs,
            )
        except Exception as exc:
            dependency_hint = (
                "检测到缺少 libGL.so.1；请在当前任务容器安装 "
                "libgl1 libglib2.0-0。"
                if "libGL.so.1" in str(exc)
                else "若是配置文件缺失再检查 --processor-path；若是 ImportError/OSError，"
                "请先检查当前容器运行时依赖。"
            )
            raise RuntimeError(
                "加载 tokenizer/processor 失败。\n"
                f"processor_path={processor_source}\n"
                f"原始异常={type(exc).__name__}: {exc}\n"
                f"{dependency_hint}"
            ) from exc

        if hasattr(self.processor, "tokenizer"):
            try:
                self.processor.tokenizer.padding_side = "left"
            except Exception:
                pass
        try:
            self.tokenizer.padding_side = "left"
        except Exception:
            pass

        model_kwargs: dict[str, Any] = {
            "torch_dtype": self.dtype,
            "trust_remote_code": args.trust_remote_code,
            "local_files_only": args.local_files_only,
            "low_cpu_mem_usage": True,
        }

        try:
            model_config = AutoConfig.from_pretrained(
                args.checkpoint,
                trust_remote_code=args.trust_remote_code,
                local_files_only=args.local_files_only,
            )
            configure_attention_backend(
                model_config,
                args.attn_implementation,
                args.vision_attn_implementation,
                args.device,
            )
            if args.enable_ui_relation is not None:
                model_config.enable_ui_relation = bool(args.enable_ui_relation)
            model_kwargs["config"] = model_config
            model_kwargs["output_loading_info"] = True
            loaded = AutoModel.from_pretrained(args.checkpoint, **model_kwargs)
            if isinstance(loaded, tuple) and len(loaded) == 2:
                self.model, loading_info = loaded
            else:
                # Keep compatibility with older custom Transformers loaders,
                # although current LocateAnything/Transformers returns a pair.
                self.model, loading_info = loaded, {}
            relation_missing_keys = [
                key
                for key in loading_info.get("missing_keys", ())
                if "relation_pyramid" in key or "relation_pbd" in key
            ]
            relation_unexpected_keys = [
                key
                for key in loading_info.get("unexpected_keys", ())
                if "relation_pyramid" in key or "relation_pbd" in key
            ]
            legacy_slot_gate = bool(
                getattr(
                    model_config,
                    "ui_relation_legacy_slot_gate_as_image_gate",
                    False,
                )
            )
            nonlegacy_missing = [
                key for key in relation_missing_keys if "image_gate_heads" not in key
            ]
            if (
                bool(getattr(model_config, "enable_ui_relation", False))
                and (
                    nonlegacy_missing
                    or relation_unexpected_keys
                    or (relation_missing_keys and not legacy_slot_gate)
                )
            ):
                raise RuntimeError(
                    "Checkpoint Relation/Gate/PBD weights are incompatible; "
                    f"missing_keys={relation_missing_keys}, "
                    f"unexpected_keys={relation_unexpected_keys}"
                )
        except Exception as exc:
            raise RuntimeError(
                "直接加载全参数 checkpoint 失败。请确认 checkpoint 含 config.json、完整"
                " safetensors 权重及 LocateAnything 自定义代码，且当前在 Embodied 环境中运行。"
            ) from exc

        changed_vision_blocks = enforce_vision_runtime_backend(
            self.model,
            args.vision_attn_implementation,
        )
        self.model = self.model.to(self.device).eval()
        backend_report = attention_backend_report(self.model)
        print(f"attention top config    : {backend_report['top_config']}")
        print(f"attention text config   : {backend_report['text_config']}")
        print(f"attention layer class   : {backend_report['first_layer_class']}")
        print(f"vision attention config : {backend_report['vision_config']}")
        print(f"vision layer backend    : {backend_report['vision_first_layer']}")
        print(f"vision blocks configured: {changed_vision_blocks}")
        requested_backend = args.attn_implementation
        if (
            requested_backend != "auto"
            and backend_report["text_config"] != requested_backend
        ):
            raise RuntimeError(
                "Requested attention backend was not applied to the nested Qwen config: "
                f"requested={requested_backend}, actual={backend_report['text_config']}, "
                f"layer={backend_report['first_layer_class']}"
            )
        if requested_backend == "sdpa" and "magi" in backend_report[
            "first_layer_class"
        ].lower():
            raise RuntimeError(
                "Requested sdpa, but LocateAnything instantiated a Magi attention layer: "
                f"{backend_report['first_layer_class']}"
            )
        if backend_report["vision_first_layer"] != args.vision_attn_implementation:
            raise RuntimeError(
                "Requested MoonViT attention backend was not applied: "
                f"requested={args.vision_attn_implementation}, "
                f"actual={backend_report['vision_first_layer']}"
            )
        parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
        print(f"parameters              : {parameter_count:,}")
        print("===== 模型加载完成 =====\n")
        self.last_ui_diagnostics: dict[str, Any] = {
            "available": False,
            "enable_ui_relation": bool(
                getattr(self.model.config, "enable_ui_relation", False)
            ),
        }

    @staticmethod
    def _scalar(value: Any) -> Any:
        if torch.is_tensor(value):
            if value.numel() == 1:
                return value.detach().float().cpu().item()
            return value.detach().float().cpu().tolist()
        return value

    def _capture_ui_diagnostics(self) -> None:
        getter = getattr(self.model, "get_last_ui_defect_interface", None)
        interface = getter() if callable(getter) else None
        if not isinstance(interface, dict):
            self.last_ui_diagnostics = {
                "available": False,
                "enable_ui_relation": bool(
                    getattr(self.model.config, "enable_ui_relation", False)
                ),
            }
            return
        self.last_ui_diagnostics = {
            "available": True,
            "enable_ui_relation": True,
            "relation_family": self._scalar(interface.get("relation_family")),
            "p_defect": self._scalar(interface.get("p_defect")),
            "gate_source": interface.get("gate_source", "image_gate"),
            "gate_threshold": self._scalar(interface.get("gate_threshold")),
            "gate_mode": interface.get("gate_mode", self.args.relation_gate_mode),
            "threshold": self._scalar(interface.get("gate_threshold")),
            "would_pass": bool(interface.get("would_pass", interface.get("gate_passed"))),
            "gate_passed": bool(interface.get("gate_passed")),
            "gate_filtered": bool(interface.get("gate_filtered")),
            "final_has_bbox": bool(interface.get("final_has_bbox")),
            "pbd_delta_norm": self._scalar(interface.get("pbd_delta_norm")),
            "pbd_active_positions": self._scalar(
                interface.get("pbd_active_positions")
            ),
        }

    @torch.inference_mode()
    def predict(self, image: Image.Image, question: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]

        text = apply_chat_template(self.processor, messages)
        if not hasattr(self.processor, "process_vision_info"):
            raise AttributeError("LocateAnything processor 缺少 process_vision_info")
        image_inputs, video_inputs = self.processor.process_vision_info(messages)
        processor_inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        )

        input_ids = processor_inputs["input_ids"].to(self.device)
        attention_mask = processor_inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        pixel_values = processor_inputs.get("pixel_values")
        if pixel_values is None:
            raise KeyError("processor 输出中缺少 pixel_values")
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
        image_grid_hws = processor_inputs.get("image_grid_hws")
        if torch.is_tensor(image_grid_hws):
            image_grid_hws = image_grid_hws.to(self.device)

        generate_kwargs: dict[str, Any] = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_grid_hws": image_grid_hws,
            "tokenizer": self.tokenizer,
            "max_new_tokens": self.args.max_new_tokens,
            "use_cache": True,
            "generation_mode": self.args.generation_mode,
            "relation_gate_mode": self.args.relation_gate_mode,
            "relation_gate_threshold": self.args.relation_gate_threshold,
            "repetition_penalty": self.args.repetition_penalty,
            "verbose": self.args.verbose_generation,
        }

        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            generate_kwargs["eos_token_id"] = eos_token_id

        do_sample = not self.args.greedy and self.args.temperature > 0
        generate_kwargs["do_sample"] = do_sample
        if do_sample:
            generate_kwargs["temperature"] = self.args.temperature
            generate_kwargs["top_p"] = self.args.top_p
            if self.args.top_k > 0:
                generate_kwargs["top_k"] = self.args.top_k

        if self.args.generation_mode in ("fast", "hybrid"):
            generate_kwargs["n_future_tokens"] = self.args.n_future_tokens

        # Do not forward None for optional tensors to custom versions that reject it.
        generate_kwargs = {
            key: value for key, value in generate_kwargs.items() if value is not None
        }
        raw_output = self.model.generate(**generate_kwargs)
        self._capture_ui_diagnostics()
        return decode_generation_output(
            raw_output=raw_output,
            input_ids=input_ids,
            processor=self.processor,
            tokenizer=self.tokenizer,
        )


BOX_PATTERN = re.compile(
    r"<box>\s*<\s*(-?\d+)\s*>\s*<\s*(-?\d+)\s*>\s*"
    r"<\s*(-?\d+)\s*>\s*<\s*(-?\d+)\s*>\s*</box>",
    flags=re.IGNORECASE,
)
NONE_PATTERN = re.compile(r"<box>\s*none\s*</box>", flags=re.IGNORECASE)
REF_PATTERN = re.compile(r"<ref>(.*?)</ref>", flags=re.IGNORECASE | re.DOTALL)


def parse_locateanything_answer(answer: str) -> ParsedAnswer:
    """Parse the exact training box grammar, while tolerating harmless spaces."""

    warnings: list[str] = []
    boxes: list[list[int]] = []

    for match_index, match in enumerate(BOX_PATTERN.finditer(answer), start=1):
        original = [int(value) for value in match.groups()]
        clipped = [min(1000, max(0, value)) for value in original]
        if clipped != original:
            warnings.append(
                f"box {match_index} 坐标超出 [0,1000]，已裁剪：{original} -> {clipped}"
            )

        x1, y1, x2, y2 = clipped
        repaired = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
        if repaired != clipped:
            warnings.append(
                f"box {match_index} 角点顺序反向，已修复：{clipped} -> {repaired}"
            )

        x1, y1, x2, y2 = repaired
        if x2 <= x1 or y2 <= y1:
            warnings.append(f"box {match_index} 为零面积框，已丢弃：{repaired}")
            continue
        boxes.append(repaired)

    # Exact duplicate removal mirrors the training conversion script.
    boxes = list(dict.fromkeys(tuple(box) for box in boxes))
    normalized_boxes = [list(box) for box in boxes]
    refs = [match.strip() for match in REF_PATTERN.findall(answer)]
    has_none = bool(NONE_PATTERN.search(answer))

    if normalized_boxes:
        status = "defect"
        if has_none:
            warnings.append("回答同时包含有效框和 <box>none</box>，按 defect 处理")
    elif has_none:
        status = "ok"
    else:
        status = "parse_error"
        if not answer.strip():
            warnings.append("模型返回空回答")
        elif "<box" in answer.lower():
            warnings.append("回答包含 <box>，但不符合训练格式")
        else:
            warnings.append("回答既无有效框，也无 <box>none</box>")

    return ParsedAnswer(
        status=status,
        normalized_boxes=normalized_boxes,
        refs=refs,
        has_none_token=has_none,
        warnings=warnings,
    )


def normalized_box_to_pixels(
    box: Sequence[int], width: int, height: int
) -> list[int] | None:
    x1, y1, x2, y2 = box
    px1 = min(width, max(0, int(round(x1 / 1000.0 * width))))
    py1 = min(height, max(0, int(round(y1 / 1000.0 * height))))
    px2 = min(width, max(0, int(round(x2 / 1000.0 * width))))
    py2 = min(height, max(0, int(round(y2 / 1000.0 * height))))

    # Preserve a non-zero normalized box after rounding on very small images.
    if px2 <= px1:
        if px1 < width:
            px2 = px1 + 1
        elif px1 > 0:
            px1 -= 1
    if py2 <= py1:
        if py1 < height:
            py2 = py1 + 1
        elif py1 > 0:
            py1 -= 1

    if px2 <= px1 or py2 <= py1:
        return None
    return [px1, py1, px2, py2]


def build_yolo_compatible_detections(
    parsed: ParsedAnswer,
    task: TaskConfig,
    width: int,
    height: int,
    compat_confidence: float | None,
) -> tuple[list[dict[str, Any]], list[list[int]]]:
    detections: list[dict[str, Any]] = []
    pixel_boxes: list[list[int]] = []

    for normalized_box in parsed.normalized_boxes:
        pixel_box = normalized_box_to_pixels(normalized_box, width, height)
        if pixel_box is None:
            continue
        pixel_boxes.append(pixel_box)
        detections.append(
            {
                "bbox_2d": pixel_box,
                "label": task.output_label,
                "class_id": task.class_id,
                "confidence": compat_confidence,
            }
        )

    return detections, pixel_boxes


def atomic_write_json(path: Path, value: Any, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=indent)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def stable_sample_seed(base_seed: int, task_name: str, image_path: str) -> int:
    payload = f"{base_seed}\0{task_name}\0{image_path}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") % (
        2**31
    )


def set_sample_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


VISUALIZATION_COLORS = {
    0: (220, 20, 60),
    1: (255, 140, 0),
    2: (0, 128, 255),
    3: (148, 0, 211),
    4: (34, 139, 34),
}


def save_visualization(
    image: Image.Image,
    pixel_boxes: Sequence[Sequence[int]],
    task: TaskConfig,
    status: str,
    output_path: Path,
) -> None:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    color = VISUALIZATION_COLORS.get(task.class_id, (255, 0, 0))
    line_width = max(2, round(min(canvas.size) / 500))

    for index, box in enumerate(pixel_boxes, start=1):
        x1, y1, x2, y2 = box
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        text = f"{task.task_name} #{index}"
        text_bbox = draw.textbbox((x1, y1), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_y = max(0, y1 - text_height - 4)
        draw.rectangle(
            (x1, text_y, min(canvas.width, x1 + text_width + 6), text_y + text_height + 4),
            fill=color,
        )
        draw.text((x1 + 3, text_y + 2), text, fill=(255, 255, 255), font=font)

    if not pixel_boxes:
        banner = "NO DEFECT" if status == "ok" else "PARSE ERROR"
        banner_color = (32, 160, 80) if status == "ok" else (200, 40, 40)
        banner_bbox = draw.textbbox((0, 0), banner, font=font)
        banner_width = banner_bbox[2] - banner_bbox[0]
        banner_height = banner_bbox[3] - banner_bbox[1]
        draw.rectangle((0, 0, banner_width + 12, banner_height + 8), fill=banner_color)
        draw.text((6, 4), banner, fill=(255, 255, 255), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    try:
        canvas.save(temporary, format="JPEG", quality=95, subsampling=0)
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)


def seconds_to_text(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60}分 {seconds % 60}秒"


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "backend": "LocateAnything",
        "checkpoint": args.checkpoint,
        "processor_path": args.processor_path or args.checkpoint,
        "prompt_template": PROMPT_TEMPLATE,
        "tasks": [asdict(task) | {"prompt": task.prompt} for task in TASK_CONFIGS],
        "generation": {
            "mode": args.generation_mode,
            "max_new_tokens": args.max_new_tokens,
            "n_future_tokens": args.n_future_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "greedy": args.greedy,
            "seed": args.seed,
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
            "enable_ui_relation": args.enable_ui_relation,
            "relation_gate_mode": args.relation_gate_mode,
            "relation_gate_threshold": args.relation_gate_threshold,
        },
        "inference_crop": {
            "mode": args.inference_crop_mode,
            "max_tiles": args.tile_max_count,
            "target_long_side": args.tile_target_long_side,
            "overlap_ratio": args.tile_overlap_ratio,
            "nms_iou": args.tile_nms_iou,
            "detector_crop_manifest": args.detector_crop_manifest,
            "detector_crop_manifest_digest": getattr(
                args, "detector_crop_manifest_digest", None
            ),
            "gt_repair_allowed": False,
        },
        "output": {
            "tag_filename": args.tag_filename,
            "compat_confidence": args.compat_confidence,
            "save_raw_answer": args.save_raw_answer,
            "save_visualization": args.save_visualization,
        },
        "created_or_updated_at": datetime.now(timezone.utc).isoformat(),
    }


def manifest_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        key: manifest.get(key)
        for key in (
            "schema_version",
            "backend",
            "checkpoint",
            "processor_path",
            "prompt_template",
            "tasks",
            "generation",
            "inference_crop",
            "output",
        )
    }


def collect_existing_task_artifacts(output_dir: Path) -> list[Path]:
    artifacts: list[Path] = []
    for task in TASK_CONFIGS:
        task_dir = output_dir / task.task_name
        if not task_dir.is_dir():
            continue
        artifacts.extend(path for path in task_dir.glob("*.json") if path.is_file())
        artifacts.extend(
            path for path in (task_dir / "raw").glob("*.json") if path.is_file()
        )
        artifacts.extend(
            path for path in (task_dir / "gate").glob("*.json") if path.is_file()
        )
        artifacts.extend(
            path
            for path in (task_dir / "visualizations").glob("*.jpg")
            if path.is_file()
        )
        artifacts.extend(
            path for path in (task_dir / "errors").glob("*.json") if path.is_file()
        )
    return artifacts


def clear_existing_task_artifacts(output_dir: Path) -> int:
    artifacts = collect_existing_task_artifacts(output_dir)
    for path in artifacts:
        path.unlink(missing_ok=True)
    (output_dir / "_summary.json").unlink(missing_ok=True)
    return len(artifacts)


def check_and_write_manifest(args: argparse.Namespace) -> None:
    path = args.output_dir / "_run_manifest.json"
    current = build_manifest(args)
    existing_results = collect_existing_task_artifacts(args.output_dir)
    if path.is_file():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = None

        identity_changed = previous is None or (
            manifest_identity(previous) != manifest_identity(current)
        )
        if identity_changed and existing_results:
            if not args.overwrite:
                raise RuntimeError(
                    f"输出目录已有不同 checkpoint/生成参数的结果：{path}\n"
                    "为避免混合模型结果，请更换 --output-dir，或确认后使用 --overwrite。"
                )
            removed = clear_existing_task_artifacts(args.output_dir)
            print(
                "[WARN] --overwrite：运行身份发生变化，已先清理旧任务产物 "
                f"{removed} 个，防止中断续跑时混合模型结果"
            )
    else:
        if existing_results and not args.overwrite:
            raise RuntimeError(
                "输出目录已有单图 JSON，但缺少 _run_manifest.json，无法确认是否来自"
                "同一模型。请更换 --output-dir，或确认后使用 --overwrite。"
            )
        if existing_results and args.overwrite:
            removed = clear_existing_task_artifacts(args.output_dir)
            print(
                "[WARN] --overwrite：旧结果缺少运行清单，已先清理任务产物 "
                f"{removed} 个"
            )

    atomic_write_json(path, current)


def build_raw_record(
    args: argparse.Namespace,
    task: TaskConfig,
    image_path: str,
    width: int,
    height: int,
    answer: str,
    parsed: ParsedAnswer,
    pixel_boxes: list[list[int]],
    elapsed_seconds: float,
    sample_seed: int,
    gate_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "task_name": task.task_name,
        "class_id": task.class_id,
        "output_label": task.output_label,
        "prompt_label": task.prompt_label,
        "prompt": task.prompt,
        "image_path": image_path,
        "image_size": {"width": width, "height": height},
        "checkpoint": args.checkpoint,
        "processor_path": args.processor_path or args.checkpoint,
        "generation": {
            "mode": args.generation_mode,
            "max_new_tokens": args.max_new_tokens,
            "n_future_tokens": args.n_future_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "greedy": args.greedy,
            "sample_seed": sample_seed,
        },
        "raw_answer": answer,
        "parse": {
            "status": parsed.status,
            "refs": parsed.refs,
            "has_none_token": parsed.has_none_token,
            "normalized_boxes_1000": parsed.normalized_boxes,
            "pixel_boxes_xyxy": pixel_boxes,
            "warnings": parsed.warnings,
        },
        "gate": gate_diagnostics,
        "elapsed_seconds": round(elapsed_seconds, 6),
    }


def _pixels_to_normalized_1000(
    bbox: Sequence[int], width: int, height: int
) -> list[int]:
    return [
        round(int(bbox[0]) / width * 1000),
        round(int(bbox[1]) / height * 1000),
        round(int(bbox[2]) / width * 1000),
        round(int(bbox[3]) / height * 1000),
    ]


def predict_with_lossless_tiles(
    *,
    args: argparse.Namespace,
    inferencer: "LocateAnythingInferencer",
    image: Image.Image,
    task: TaskConfig,
    sample_seed: int,
    tiles_override: Sequence[Sequence[int]] | None = None,
    crop_mode: str = "lossless_tiling",
) -> tuple[str, ParsedAnswer, list[dict[str, Any]], list[list[int]], dict[str, Any], list[dict[str, Any]]]:
    """Run GT-free tiled inference and merge only after global coordinate mapping."""

    width, height = image.size
    tiling_task = f"ui_{task.task_name}"
    if tiles_override is None:
        tiles = generate_lossless_tiles(
            width,
            height,
            task=tiling_task,
            max_tiles=args.tile_max_count,
            target_long_side=args.tile_target_long_side,
            overlap_ratio=args.tile_overlap_ratio,
        )
    else:
        tiles = [list(map(int, tile)) for tile in tiles_override]
        assert_lossless_coverage(width, height, tiles)
    pending_predictions: list[dict[str, Any]] = []
    tile_records: list[dict[str, Any]] = []
    warnings: list[str] = []
    any_parse_error = False
    for tile_index, tile in enumerate(tiles):
        tile_image = image.crop(tuple(tile))
        try:
            set_sample_seed(sample_seed + tile_index)
            answer = inferencer.predict(image=tile_image, question=task.prompt)
            gate = dict(inferencer.last_ui_diagnostics)
            parsed = parse_locateanything_answer(answer)
            local_detections, local_boxes = build_yolo_compatible_detections(
                parsed=parsed,
                task=task,
                width=tile[2] - tile[0],
                height=tile[3] - tile[1],
                compat_confidence=args.compat_confidence,
            )
            any_parse_error = any_parse_error or parsed.status == "parse_error"
            warnings.extend(f"tile {tile_index}: {item}" for item in parsed.warnings)
            for detection in local_detections:
                pending_predictions.append(
                    {
                        "bbox": detection["bbox_2d"],
                        "tile_bbox": tile,
                        "label": detection["label"],
                        "class_id": detection["class_id"],
                        "confidence": detection["confidence"],
                        "score": detection["confidence"]
                        if detection["confidence"] is not None
                        else 1.0,
                        "source_tile_index": tile_index,
                    }
                )
            tile_records.append(
                {
                    "tile_index": tile_index,
                    "tile_bbox": tile,
                    "answer": answer,
                    "status": parsed.status,
                    "local_pixel_boxes": local_boxes,
                    "gate": gate,
                }
            )
        finally:
            tile_image.close()

    merged = merge_tile_predictions(
        pending_predictions,
        image_size=(width, height),
        iou_threshold=args.tile_nms_iou,
    )
    detections: list[dict[str, Any]] = []
    pixel_boxes: list[list[int]] = []
    for row in merged:
        bbox = [int(round(value)) for value in row["bbox"]]
        bbox[0] = max(0, min(width, bbox[0]))
        bbox[1] = max(0, min(height, bbox[1]))
        bbox[2] = max(0, min(width, bbox[2]))
        bbox[3] = max(0, min(height, bbox[3]))
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        pixel_boxes.append(bbox)
        detections.append(
            {
                "bbox_2d": bbox,
                "label": row["label"],
                "class_id": row["class_id"],
                "confidence": row.get("confidence"),
                "source_tile_index": row["source_tile_index"],
                "source_tile_bbox": row["source_tile_bbox"],
            }
        )
    status = "defect" if detections else ("parse_error" if any_parse_error else "ok")
    parsed = ParsedAnswer(
        status=status,
        normalized_boxes=[
            _pixels_to_normalized_1000(box, width, height) for box in pixel_boxes
        ],
        refs=[],
        has_none_token=not detections,
        warnings=warnings,
    )
    tile_gates = [row["gate"] for row in tile_records]
    gate_diagnostics = aggregate_tiled_gate_diagnostics(
        tile_gates,
        crop_mode=crop_mode,
    )
    return (
        json.dumps(tile_records, ensure_ascii=False),
        parsed,
        detections,
        pixel_boxes,
        gate_diagnostics,
        tile_records,
    )


def load_detector_scan_index(path: Path) -> tuple[dict[str, dict[str, Any]], str]:
    """Load a detector-only scan manifest and index every source-path alias."""

    if not path.is_file():
        raise FileNotFoundError(f"detector crop manifest does not exist: {path}")
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    index: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid detector crop JSON at {path}:{line_no}: {exc}") from exc
            if row.get("mode") != "detector_scan" or row.get("gt_used") is not False:
                raise ValueError(
                    f"detector crop manifest row {line_no} is not GT-free detector_scan"
                )
            width, height = int(row["width"]), int(row["height"])
            tiles = row.get("tiles") or []
            assert_lossless_coverage(width, height, tiles)
            if int(row.get("detector_boundary_cut_count", -1)) != 0:
                raise ValueError(f"detector crop row {line_no} cuts detector boxes")
            aliases = list(row.get("image_paths") or [])
            aliases.append(row["image_path"])
            for alias in aliases:
                key = str(Path(alias).expanduser().resolve(strict=False))
                previous = index.get(key)
                if previous is not None and previous["image_id"] != row["image_id"]:
                    raise ValueError(f"detector crop path alias collision: {key}")
                index[key] = row
    if not index:
        raise ValueError(f"detector crop manifest is empty: {path}")
    return index, digest


def save_error_record(
    work: TaskWork,
    stem: str,
    image_path: str,
    error: BaseException,
) -> Path:
    error_path = work.output_dir / "errors" / f"{stem}_error.json"
    record = {
        "task_name": work.config.task_name,
        "image_path": image_path,
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
        "time": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(error_path, record)
    return error_path


def run_one_task(
    args: argparse.Namespace,
    inferencer: LocateAnythingInferencer,
    work: TaskWork,
) -> dict[str, Any]:
    config = work.config
    print("\n" + "=" * 88)
    print(f"任务                 : {config.task_name}")
    print(f"输入 JSONL           : {work.jsonl_path}")
    print(f"输出目录             : {work.output_dir}")
    print(f"训练一致提示词       : {config.prompt}")
    print(f"测试集有效图片       : {len(work.image_paths)}")
    print(f"断点跳过             : {work.skipped_existing}")
    print(f"本次待推理           : {len(work.pending_paths)}")
    print("=" * 88)

    counts = {
        "task_name": config.task_name,
        "dataset_images": len(work.image_paths),
        "skipped_existing": work.skipped_existing,
        "processed": 0,
        "defect": 0,
        "ok": 0,
        "parse_error": 0,
        "inference_error": 0,
        "boxes": 0,
        "gate_available": 0,
        "gate_positive": 0,
        "gate_filtered": 0,
        "elapsed_seconds": 0.0,
    }
    if not work.pending_paths:
        print(f"[DONE] {config.task_name} 已全部处理，无需继续推理")
        return counts

    total = len(work.pending_paths)
    task_start = time.time()

    for index, image_path in enumerate(work.pending_paths, start=1):
        if index == 1:
            eta = "计算中..."
        else:
            elapsed = time.time() - task_start
            eta = seconds_to_text(elapsed / (index - 1) * (total - index + 1))

        stem = work.output_stems[image_path]
        print(
            f"\n[{config.task_name}] [{index}/{total}] {image_path} | 预计还需：{eta}"
        )

        if args.overwrite:
            remove_old_artifacts(work.output_dir, stem)

        try:
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
            width, height = image.size

            sample_seed = stable_sample_seed(args.seed, config.task_name, image_path)
            set_sample_seed(sample_seed)

            inference_start = time.time()
            if args.inference_crop_mode in {"lossless_tiling", "detector_scan"}:
                tiles_override = None
                if args.inference_crop_mode == "detector_scan":
                    if config.task_name == "content_missing":
                        # The global task keeps one complete view; it still uses
                        # no GT and shares the same detector cache validation.
                        tiles_override = [[0, 0, width, height]]
                    else:
                        key = str(Path(image_path).expanduser().resolve(strict=False))
                        scan = args.detector_scan_index.get(key)
                        if scan is None:
                            raise KeyError(
                                f"test image is absent from detector crop manifest: {key}"
                            )
                        if (int(scan["width"]), int(scan["height"])) != (width, height):
                            raise ValueError(
                                f"detector crop dimensions changed for {key}: "
                                f"manifest={scan['width']}x{scan['height']}, image={width}x{height}"
                            )
                        tiles_override = scan["tiles"]
                (
                    answer,
                    parsed,
                    detections,
                    pixel_boxes,
                    gate_diagnostics,
                    tile_records,
                ) = predict_with_lossless_tiles(
                    args=args,
                    inferencer=inferencer,
                    image=image,
                    task=config,
                    sample_seed=sample_seed,
                    tiles_override=tiles_override,
                    crop_mode=args.inference_crop_mode,
                )
            else:
                answer = inferencer.predict(image=image, question=config.prompt)
                gate_diagnostics = dict(inferencer.last_ui_diagnostics)
                parsed = parse_locateanything_answer(answer)
                detections, pixel_boxes = build_yolo_compatible_detections(
                    parsed=parsed,
                    task=config,
                    width=width,
                    height=height,
                    compat_confidence=args.compat_confidence,
                )
                tile_records = []
            inference_elapsed = time.time() - inference_start

            if args.print_raw_answer:
                print(f"[RAW] {answer}")

            # A normalized box can only disappear here on a degenerate tiny image.
            if parsed.status == "defect" and not detections:
                parsed.status = "parse_error"
                parsed.warnings.append("有效归一化框无法转换为有效像素框")

            suffix = status_suffix(parsed.status, args.tag_filename)
            output_path = work.output_dir / f"{stem}{suffix}.json"

            # Parsed JSON is written last and acts as the resume completion marker.
            if args.save_raw_answer:
                raw_path = work.output_dir / "raw" / f"{stem}{suffix}.json"
                atomic_write_json(
                    raw_path,
                    build_raw_record(
                        args=args,
                        task=config,
                        image_path=image_path,
                        width=width,
                        height=height,
                        answer=answer,
                        parsed=parsed,
                        pixel_boxes=pixel_boxes,
                        elapsed_seconds=inference_elapsed,
                        sample_seed=sample_seed,
                        gate_diagnostics=gate_diagnostics,
                    ),
                )
                if tile_records:
                    raw_payload = json.loads(raw_path.read_text(encoding="utf-8"))
                    raw_payload["inference_crop"] = {
                        "mode": args.inference_crop_mode,
                        "tiles": tile_records,
                        "global_merge_iou": args.tile_nms_iou,
                        "detector_crop_manifest": args.detector_crop_manifest,
                        "detector_crop_manifest_digest": getattr(
                            args, "detector_crop_manifest_digest", None
                        ),
                        "gt_repair_used": False,
                    }
                    atomic_write_json(raw_path, raw_payload)

            gate_path = work.output_dir / "gate" / f"{stem}{suffix}.json"
            gate_diagnostics.setdefault("p_defect", None)
            gate_diagnostics.setdefault("gate_mode", args.relation_gate_mode)
            gate_diagnostics.setdefault("threshold", args.relation_gate_threshold)
            gate_diagnostics.setdefault("would_pass", None)
            gate_diagnostics.setdefault("gate_filtered", False)
            gate_diagnostics["final_has_bbox"] = bool(detections)
            atomic_write_json(
                gate_path,
                {
                    "task_name": config.task_name,
                    "image_path": image_path,
                    "prediction_status": parsed.status,
                    "prediction_boxes": len(detections),
                    **gate_diagnostics,
                },
            )

            if args.save_visualization:
                visualization_path = (
                    work.output_dir / "visualizations" / f"{stem}{suffix}.jpg"
                )
                try:
                    save_visualization(
                        image=image,
                        pixel_boxes=pixel_boxes,
                        task=config,
                        status=parsed.status,
                        output_path=visualization_path,
                    )
                except Exception as visualization_error:
                    print(
                        f"[WARN] 可视化保存失败，但保留推理结果：{visualization_error}",
                        file=sys.stderr,
                    )

            atomic_write_json(output_path, detections)
            (work.output_dir / "errors" / f"{stem}_error.json").unlink(missing_ok=True)

            counts["processed"] += 1
            counts[parsed.status] += 1
            counts["boxes"] += len(detections)
            if gate_diagnostics.get("available"):
                counts["gate_available"] += 1
                counts["gate_positive"] += int(
                    bool(gate_diagnostics.get("would_pass"))
                )
                counts["gate_filtered"] += int(
                    bool(gate_diagnostics.get("gate_filtered"))
                )
            warning_text = (
                f" | warnings={len(parsed.warnings)}" if parsed.warnings else ""
            )
            print(
                f"[OK] status={parsed.status}, boxes={len(detections)}, "
                f"time={inference_elapsed:.2f}s{warning_text}"
            )
            print(f"[SAVE] {output_path}")

        except KeyboardInterrupt:
            print("\n[STOP] 收到键盘中断；已完成图片可在下次自动跳过")
            raise
        except Exception as exc:
            counts["inference_error"] += 1
            error_path = save_error_record(work, stem, image_path, exc)
            print(
                f"[ERROR] {config.task_name} 推理失败：{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            print(f"[ERROR] 详情：{error_path}", file=sys.stderr)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if args.fail_fast:
                raise

    counts["elapsed_seconds"] = round(time.time() - task_start, 6)
    print(
        f"\n[DONE] {config.task_name}: processed={counts['processed']}, "
        f"defect={counts['defect']}, ok={counts['ok']}, "
        f"parse_error={counts['parse_error']}, inference_error={counts['inference_error']}, "
        f"boxes={counts['boxes']}, time={seconds_to_text(counts['elapsed_seconds'])}"
    )
    return counts


def print_preflight(args: argparse.Namespace, works: Sequence[TaskWork]) -> None:
    print("\n===== 推理预检查 =====")
    print(f"checkpoint              : {args.checkpoint}")
    print(f"input_dir               : {args.input_dir}")
    print(f"output_dir              : {args.output_dir}")
    print(f"CUDA_VISIBLE_DEVICES    : {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"device                  : {args.device}")
    print(f"generation_mode         : {args.generation_mode}")
    print(f"inference_crop_mode     : {args.inference_crop_mode}")
    if args.inference_crop_mode in {"lossless_tiling", "detector_scan"}:
        print(
            f"{args.inference_crop_mode:24s}: "
            f"max={args.tile_max_count}, long_side={args.tile_target_long_side}, "
            f"overlap={args.tile_overlap_ratio}, nms={args.tile_nms_iou}, GT repair=disabled"
        )
    if args.inference_crop_mode == "detector_scan":
        print(f"detector_crop_manifest : {args.detector_crop_manifest}")
        print(f"detector_manifest_sha  : {args.detector_crop_manifest_digest}")
    print(f"save_raw_answer         : {args.save_raw_answer}")
    print(f"save_visualization      : {args.save_visualization}")
    print(f"tag_filename            : {args.tag_filename}")
    print(f"overwrite               : {args.overwrite}")

    if not works:
        print("没有找到所选任务的任何测试 JSONL")
        return

    for work in works:
        print(
            f"{work.config.task_name:16s}: total={len(work.image_paths):6d}, "
            f"skip={work.skipped_existing:6d}, pending={len(work.pending_paths):6d}, "
            f"prompt={work.config.prompt!r}"
        )
    print(
        f"总待推理张次           : {sum(len(work.pending_paths) for work in works)}"
    )
    print("========================\n")


def main() -> int:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    args.checkpoint = normalize_local_or_hub_path(args.checkpoint)
    if args.processor_path is not None:
        args.processor_path = normalize_local_or_hub_path(args.processor_path)
    args.input_dir = Path(args.input_dir).expanduser().resolve()
    args.output_dir = Path(args.output_dir).expanduser().resolve()
    args.summary_path = (
        Path(args.summary_path).expanduser().resolve()
        if args.summary_path
        else args.output_dir / "_summary.json"
    )
    args.detector_scan_index = {}
    args.detector_crop_manifest_digest = None
    if args.inference_crop_mode == "detector_scan":
        if not args.detector_crop_manifest:
            raise ValueError("detector_scan requires --detector-crop-manifest")
        detector_manifest = Path(args.detector_crop_manifest).expanduser().resolve(strict=True)
        args.detector_scan_index, args.detector_crop_manifest_digest = load_detector_scan_index(
            detector_manifest
        )
        args.detector_crop_manifest = str(detector_manifest)
    elif args.detector_crop_manifest:
        raise ValueError("--detector-crop-manifest is only valid with detector_scan")

    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在：{args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validate_local_checkpoint(args.checkpoint)

    if args.load_only:
        print("[MODEL LOAD PREFLIGHT] 开始单进程模型加载检查", flush=True)
        preflight_works = prepare_work(args) if args.preflight_forward else []
        inferencer = LocateAnythingInferencer(args)
        if args.preflight_forward:
            if not preflight_works or not preflight_works[0].image_paths:
                raise RuntimeError("forward preflight 找不到可用的任务图片")
            work = preflight_works[0]
            image_path = work.image_paths[0]
            print(
                f"[MODEL FORWARD PREFLIGHT] task={work.config.task_name} image={image_path}",
                flush=True,
            )
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
            set_sample_seed(
                stable_sample_seed(args.seed, work.config.task_name, image_path)
            )
            answer = inferencer.predict(image=image, question=work.config.prompt)
            print(
                f"[MODEL FORWARD PREFLIGHT] generation passed, answer_chars={len(answer)}",
                flush=True,
            )
        print("[MODEL LOAD PREFLIGHT] 模型加载检查通过", flush=True)
        return 0

    works = prepare_work(args)
    print_preflight(args, works)
    if not works:
        return 1
    if args.dry_run:
        print("[DRY RUN] 检查完成，未加载模型、未运行推理")
        return 0

    check_and_write_manifest(args)
    total_pending = sum(len(work.pending_paths) for work in works)
    if total_pending == 0:
        print("[DONE] 所选任务均已有结果，未加载模型")
        return 0

    inferencer = LocateAnythingInferencer(args)
    all_stats: list[dict[str, Any]] = []
    wall_start = time.time()
    for work in works:
        all_stats.append(run_one_task(args, inferencer, work))

    summary = {
        "checkpoint": args.checkpoint,
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "tasks": all_stats,
        "totals": {
            key: sum(int(stats[key]) for stats in all_stats)
            for key in (
                "dataset_images",
                "skipped_existing",
                "processed",
                "defect",
                "ok",
                "parse_error",
                "inference_error",
                "boxes",
                "gate_available",
                "gate_positive",
                "gate_filtered",
            )
        },
        "wall_elapsed_seconds": round(time.time() - wall_start, 6),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(args.summary_path, summary)

    totals = summary["totals"]
    print("\n" + "=" * 88)
    print("五类任务推理结束")
    print(f"本次成功推理         : {totals['processed']} 张次")
    print(f"缺陷 / 正常          : {totals['defect']} / {totals['ok']}")
    print(f"解析失败             : {totals['parse_error']}")
    print(f"推理异常             : {totals['inference_error']}")
    print(f"预测框总数           : {totals['boxes']}")
    print(f"输出目录             : {args.output_dir}")
    print(f"汇总                 : {args.summary_path}")
    print("=" * 88)

    return 2 if totals["inference_error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
