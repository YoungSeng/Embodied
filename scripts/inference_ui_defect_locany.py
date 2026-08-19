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
from transformers import AutoModel, AutoProcessor, AutoTokenizer


PROMPT_TEMPLATE = (
    "Locate all the instances that match the following description: {label}."
)


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
        prompt_label="text overflow",
    ),
    TaskConfig(
        task_name="text_ellipsis",
        jsonl_name="test_ui_text_ellipsis_wcnt_no_figma.jsonl",
        class_id=3,
        output_label="文本省略",
        prompt_label="abnormal text ellipsis",
    ),
    TaskConfig(
        task_name="occlusion",
        jsonl_name="test_ui_occlusion_wcnt_no_figma.jsonl",
        class_id=0,
        output_label="元素遮挡",
        prompt_label="overlapping elements",
    ),
    TaskConfig(
        task_name="cropping",
        jsonl_name="test_ui_cropping_wcnt_no_figma.jsonl",
        class_id=1,
        output_label="元素裁切",
        prompt_label="cropped element",
    ),
    TaskConfig(
        task_name="content_missing",
        jsonl_name="test_ui_content_missing_wcnt_no_figma.jsonl",
        class_id=4,
        output_label="内容缺失",
        prompt_label="missing content",
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
    if args.compat_confidence is not None and not 0 <= args.compat_confidence <= 1:
        parser.error("--compat-confidence 必须位于 [0, 1]")

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
            raise RuntimeError(
                "加载 tokenizer/processor 失败。若训练 checkpoint 未保存这些文件，请用 "
                "--processor-path 显式指向本地 nvidia/LocateAnything-3B 目录。"
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
        if args.attn_implementation != "auto":
            model_kwargs["attn_implementation"] = args.attn_implementation

        try:
            self.model = AutoModel.from_pretrained(args.checkpoint, **model_kwargs)
        except Exception as exc:
            raise RuntimeError(
                "直接加载全参数 checkpoint 失败。请确认 checkpoint 含 config.json、完整"
                " safetensors 权重及 LocateAnything 自定义代码，且当前在 Embodied 环境中运行。"
            ) from exc

        self.model = self.model.to(self.device).eval()
        parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
        print(f"parameters              : {parameter_count:,}")
        print("===== 模型加载完成 =====\n")

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
        "elapsed_seconds": round(elapsed_seconds, 6),
    }


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
            answer = inferencer.predict(image=image, question=config.prompt)
            inference_elapsed = time.time() - inference_start

            if args.print_raw_answer:
                print(f"[RAW] {answer}")

            parsed = parse_locateanything_answer(answer)
            detections, pixel_boxes = build_yolo_compatible_detections(
                parsed=parsed,
                task=config,
                width=width,
                height=height,
                compat_confidence=args.compat_confidence,
            )

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
                    ),
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

    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在：{args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validate_local_checkpoint(args.checkpoint)

    if args.load_only:
        print("[MODEL LOAD PREFLIGHT] 开始单进程模型加载检查", flush=True)
        LocateAnythingInferencer(args)
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
