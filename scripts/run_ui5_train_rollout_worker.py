#!/usr/bin/env python3
"""One persistent UI5 train-rollout worker or a CPU progress snapshot.

Run mode loads exactly one checkpoint once, then processes every portable
``image_id+task`` sample.  Crop mode performs in-memory base-tile crops and
reuses the tiled-eval branch's global mapping plus class-aware greedy NMS.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 3
BASE_COMMITS = {"m31": "5d7a313", "crop": "945ce39"}
MODEL_IDS = ("m31", "crop")
TASKS = ("occlusion", "cropping", "text_overflow", "text_ellipsis", "content_missing")
FORMAL_SEEDS = {0: 20260903, 1: 20260917, 2: 20260931, 3: 20260947}
MAX_SEQ_LENGTH = 7268
MAX_NUM_TOKENS_PER_SAMPLE = 7268
TRAINING_MAX_NUM_TOKENS = 12800
PROCESSOR_IN_TOKEN_LIMIT = 25600
ROLLOUT_MAX_NEW_TOKENS = 512


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("run", "progress-snapshot"), default="run"
    )
    parser.add_argument("--model-id", choices=MODEL_IDS)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--processor-path", type=Path)
    parser.add_argument("--bundle-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--inference-script", type=Path, default=None)
    parser.add_argument("--rollout-ids")
    parser.add_argument("--seeds")
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument("--gpu-model-processes", type=int, default=2)
    parser.add_argument("--part-size", type=int, default=10000)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--progress-seconds", type=float, default=60.0)
    parser.add_argument("--iou-threshold", type=float, default=0.1)
    parser.add_argument("--tile-nms-iou", type=float, default=0.5)
    parser.add_argument("--expected-workers", type=int, default=8)
    parser.add_argument("--dtype", choices=("bf16",), default="bf16")
    parser.add_argument("--attn-implementation", choices=("sdpa",), default="sdpa")
    parser.add_argument("--vision-attn-implementation", choices=("sdpa",), default="sdpa")
    parser.add_argument("--generation-mode", choices=("hybrid",), default="hybrid")
    parser.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument(
        "--max-num-tokens-per-sample", type=int, default=MAX_NUM_TOKENS_PER_SAMPLE
    )
    parser.add_argument(
        "--training-max-num-tokens", type=int, default=TRAINING_MAX_NUM_TOKENS
    )
    parser.add_argument(
        "--processor-in-token-limit", type=int, default=PROCESSOR_IN_TOKEN_LIMIT
    )
    parser.add_argument("--max-new-tokens", type=int, default=ROLLOUT_MAX_NEW_TOKENS)
    parser.add_argument("--n-future-tokens", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    return parser.parse_args(argv)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
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


def stable_seed(base_seed: int, *parts: str) -> int:
    payload = "\0".join((str(base_seed), *map(str, parts))).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") % (2**31)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_no}")
            rows.append(value)
    return rows


def fixed_interleaved_samples(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Round-robin the fixed task/polarity buckets without any randomness."""
    bucket_order = [(task, polarity) for task in TASKS for polarity in ("positive", "negative")]
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {
        key: [] for key in bucket_order
    }
    for raw in rows:
        row = dict(raw)
        task = str(row.get("task"))
        if task not in TASKS:
            raise ValueError(f"unknown task in rollout bundle: {task}")
        positive = bool(row.get("positive", bool(row.get("gt_global"))))
        buckets[(task, "positive" if positive else "negative")].append(row)
    for bucket in buckets.values():
        bucket.sort(key=lambda row: (str(row["record_id"]), str(row.get("sample_id", ""))))
    ordered: list[dict[str, Any]] = []
    next_index = {key: 0 for key in bucket_order}
    while len(ordered) < len(rows):
        added = 0
        for key in bucket_order:
            index = next_index[key]
            if index >= len(buckets[key]):
                continue
            ordered.append(buckets[key][index])
            next_index[key] = index + 1
            added += 1
        if not added:
            raise RuntimeError("fixed interleaving stalled before all samples were emitted")
    if len({str(row["record_id"]) for row in ordered}) != len(ordered):
        raise RuntimeError("fixed interleaved sample order contains duplicate record_id")
    return ordered


def load_module(path: Path, name: str):
    path = path.resolve(strict=True)
    sys.path.insert(0, str(path.parent))
    sys.path.insert(0, str(path.parents[1]))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def git_output(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def verify_code_identity(repo: Path, model_id: str) -> dict[str, Any]:
    baseline = BASE_COMMITS[model_id]
    head = git_output(repo, "rev-parse", "HEAD")
    subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", baseline, "HEAD"],
        check=True,
        capture_output=True,
    )
    protected = [
        "scripts/inference_ui_defect_locany.py",
        "eaglevl/model",
        "eaglevl/utils/locany",
    ]
    if model_id == "crop":
        protected.extend(
            ["scripts/ui5_lossless_tiling.py", "scripts/locany_ui5_common.py"]
        )
    changed = subprocess.run(
        ["git", "-C", str(repo), "diff", "--name-only", baseline, "--", *protected],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    if changed:
        raise RuntimeError(
            f"protected inference/model files differ from baseline {baseline}: {changed}"
        )
    return {"head": head, "baseline": baseline, "protected_paths_unchanged": True}


def make_generation_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        checkpoint=str(args.checkpoint.resolve(strict=True)),
        processor_path=str(args.processor_path.resolve(strict=True)),
        device="cuda:0",
        dtype=args.dtype,
        trust_remote_code=True,
        local_files_only=True,
        use_fast_processor=True,
        attn_implementation=args.attn_implementation,
        vision_attn_implementation=args.vision_attn_implementation,
        generation_mode=args.generation_mode,
        relation_gate_mode="observe",
        relation_gate_threshold=None,
        max_new_tokens=args.max_new_tokens,
        n_future_tokens=args.n_future_tokens,
        repetition_penalty=args.repetition_penalty,
        verbose_generation=False,
        greedy=False,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        enable_ui_relation=None,
        enable_pbd=True,
    )


def generation_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "vision_attn_implementation": args.vision_attn_implementation,
        "generation_mode": args.generation_mode,
        "max_seq_length": args.max_seq_length,
        "max_num_tokens_per_sample": args.max_num_tokens_per_sample,
        "training_max_num_tokens_record_only": args.training_max_num_tokens,
        "processor_in_token_limit": args.processor_in_token_limit,
        "max_new_tokens": args.max_new_tokens,
        "effective_max_new_tokens_rule": (
            "min(max_new_tokens, max_seq_length - input_tokens)"
        ),
        "n_future_tokens": args.n_future_tokens,
        "do_sample": True,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "relation_gate_mode": "observe",
        "tile_nms_iou": args.tile_nms_iou if args.model_id == "crop" else None,
    }


def install_generation_token_budget(
    inferencer: Any, args: argparse.Namespace
) -> dict[str, Any]:
    """Bound every old-policy generate call without editing either inference baseline."""
    processor = inferencer.processor
    previous_in_token_limit = getattr(processor, "in_token_limit", None)
    processor.in_token_limit = int(args.processor_in_token_limit)
    original_generate = inferencer.model.generate
    inferencer.last_rollout_token_usage = None

    def bounded_generate(*positional: Any, **inputs: Any) -> Any:
        if "input_ids" not in inputs:
            raise KeyError("rollout generation requires keyword input_ids for token budgeting")
        input_tokens = int(inputs["input_ids"].shape[-1])
        effective_max_new_tokens = min(
            int(args.max_new_tokens), int(args.max_seq_length) - input_tokens
        )
        if effective_max_new_tokens <= 0:
            raise RuntimeError(
                "rollout input exceeds MAX_SEQ_LENGTH: "
                f"input_tokens={input_tokens} max_seq_length={args.max_seq_length}"
            )
        inputs["max_new_tokens"] = effective_max_new_tokens
        inferencer.last_rollout_token_usage = {
            "input_tokens": input_tokens,
            "configured_max_new_tokens": int(args.max_new_tokens),
            "effective_max_new_tokens": effective_max_new_tokens,
            "max_seq_length": int(args.max_seq_length),
            "input_plus_generation_limit": input_tokens + effective_max_new_tokens,
        }
        return original_generate(*positional, **inputs)

    inferencer.model.generate = bounded_generate
    return {
        "previous_in_token_limit": previous_in_token_limit,
        "active_in_token_limit": getattr(processor, "in_token_limit", None),
    }


def area(box: Sequence[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(
        0.0, float(box[3]) - float(box[1])
    )


def score_prediction(
    scorer: Any,
    gt_boxes: Sequence[Sequence[float]],
    pred_boxes: Sequence[Sequence[float]],
    parse_status: str,
    threshold: float,
    image_size: tuple[int, int],
) -> dict[str, Any]:
    gt = [list(map(float, box)) for box in gt_boxes]
    pred = [list(map(float, box)) for box in pred_boxes]
    gt_count, pred_count = len(gt), len(pred)
    if parse_status == "parse_error":
        image_confusion = "FN" if gt_count else "FP"
        return {
            "matched_pairs": [],
            "TP_box": 0,
            "FP_box": 0 if gt_count else 1,
            "FN_box": gt_count,
            "image_confusion": image_confusion,
            "error_type": "PARSE_ERROR",
            "exact_correct": False,
        }
    if gt_count and pred_count:
        matrix = scorer.np.zeros((gt_count, pred_count), dtype=scorer.np.float64)
        for gt_index, gt_box in enumerate(gt):
            for pred_index, pred_box in enumerate(pred):
                matrix[gt_index, pred_index] = scorer.calculate_iou(gt_box, pred_box)
        gt_indices, pred_indices = scorer.linear_sum_assignment(-matrix)
    else:
        matrix = None
        gt_indices, pred_indices = [], []
    diagonal = max(1e-12, math.hypot(*image_size))
    matched_pairs: list[dict[str, Any]] = []
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    for gt_index, pred_index in zip(gt_indices, pred_indices):
        iou = float(matrix[gt_index, pred_index])
        is_tp = iou >= threshold
        gt_box, pred_box = gt[int(gt_index)], pred[int(pred_index)]
        gt_center = ((gt_box[0] + gt_box[2]) / 2, (gt_box[1] + gt_box[3]) / 2)
        pred_center = (
            (pred_box[0] + pred_box[2]) / 2,
            (pred_box[1] + pred_box[3]) / 2,
        )
        center_distance = math.hypot(
            pred_center[0] - gt_center[0], pred_center[1] - gt_center[1]
        )
        pair = {
            "gt_index": int(gt_index),
            "pred_index": int(pred_index),
            "gt_bbox": gt_box,
            "pred_bbox": pred_box,
            "iou": iou,
            "is_tp": is_tp,
            "center_distance_px": center_distance,
            "center_distance_normalized": center_distance / diagonal,
            "pred_gt_area_ratio": area(pred_box) / max(1e-12, area(gt_box)),
        }
        matched_pairs.append(pair)
        if is_tp:
            matched_gt.add(int(gt_index))
            matched_pred.add(int(pred_index))
    tp = len(matched_gt)
    fp = pred_count - tp
    fn = gt_count - tp
    if gt_count and pred_count:
        image_confusion = "TP"
    elif gt_count:
        image_confusion = "FN"
    elif pred_count:
        image_confusion = "FP"
    else:
        image_confusion = "TN"
    if not gt_count and not pred_count:
        error_type = "TN"
    elif not gt_count:
        error_type = "FP_ONLY"
    elif not pred_count:
        error_type = "FN_NO_PRED"
    elif tp == gt_count and fp == 0:
        error_type = "EXACT_TP"
    elif tp == 0:
        error_type = "LOC_WRONG"
    elif fn > 0 and fp == 0:
        error_type = "PARTIAL_MISS"
    elif fn == 0 and fp > 0:
        error_type = "PARTIAL_EXTRA"
    else:
        error_type = "PARTIAL_BOTH"
    exact = bool((not gt_count and not pred_count) or (gt_count and tp == gt_count and fp == 0))
    return {
        "matched_pairs": matched_pairs,
        "TP_box": tp,
        "FP_box": fp,
        "FN_box": fn,
        "image_confusion": image_confusion,
        "error_type": error_type,
        "exact_correct": exact,
    }


class PartWriter:
    def __init__(self, directory: Path, part_size: int):
        self.directory = directory
        self.part_size = part_size
        self.directory.mkdir(parents=True, exist_ok=True)
        stale = sorted(self.directory.glob("part-*.jsonl"))
        if stale:
            raise RuntimeError(
                f"raw rollout directory is not empty; refusing to mix runs: {self.directory}"
            )
        self.part_index = -1
        self.part_rows = 0
        self.handle = None

    def _rotate(self) -> None:
        self.close()
        self.part_index += 1
        self.part_rows = 0
        path = self.directory / f"part-{self.part_index:05d}.jsonl"
        self.handle = path.open("x", encoding="utf-8")

    def write(self, row: Mapping[str, Any]) -> None:
        if self.handle is None or self.part_rows >= self.part_size:
            self._rotate()
        assert self.handle is not None
        self.handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.part_rows += 1

    def flush(self) -> None:
        if self.handle is not None:
            self.handle.flush()
            os.fsync(self.handle.fileno())

    def close(self) -> None:
        if self.handle is not None:
            self.flush()
            self.handle.close()
            self.handle = None


class ProgressWriter:
    def __init__(self, path: Path, model_id: str, rollout_id: int, seed: int, total: int):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size:
            raise RuntimeError(f"progress file already exists: {path}")
        self.handle = path.open("x", encoding="utf-8")
        self.model_id = model_id
        self.rollout_id = rollout_id
        self.seed = seed
        self.total = total
        self.started = time.monotonic()
        self.inference_started: float | None = None
        self.last_emit = self.started

    def should_emit(self, attempted: int, every: int, seconds: float) -> bool:
        now = time.monotonic()
        return bool(attempted % every == 0 or now - self.last_emit >= seconds)

    def emit(
        self,
        counters: Mapping[str, int],
        status: str,
        *,
        force: bool = False,
        memory: Mapping[str, Any] | None = None,
    ) -> None:
        now = time.monotonic()
        attempted = int(counters["attempted"])
        if not force and not self.should_emit(attempted, 100, 60.0):
            return
        if status == "running" and self.inference_started is None:
            self.inference_started = now
        elapsed = max(0.0, now - self.started)
        inference_elapsed = max(
            0.0,
            now - (self.inference_started if self.inference_started is not None else now),
        )
        inference_success = int(counters["inference_success"])
        attempted_throughput = (
            attempted / inference_elapsed if attempted and inference_elapsed else 0.0
        )
        success_throughput = (
            inference_success / inference_elapsed
            if inference_success and inference_elapsed
            else 0.0
        )
        remaining_samples = max(0, self.total - attempted)
        remaining = (
            remaining_samples / success_throughput
            if success_throughput
            else (0.0 if remaining_samples == 0 else None)
        )
        eta = (
            datetime.fromtimestamp(time.time() + remaining, timezone.utc).isoformat()
            if remaining is not None
            else None
        )
        row = {
            "timestamp": utc_now(),
            "model_id": self.model_id,
            "rollout_id": self.rollout_id,
            "seed": self.seed,
            "status": status,
            "attempted": attempted,
            "completed": attempted,
            "inference_success": inference_success,
            "runtime_error": int(counters["runtime_error"]),
            "parse_error": int(counters["parse_error"]),
            "total": self.total,
            "throughput_attempted_per_second": attempted_throughput,
            "throughput_inference_success_per_second": success_throughput,
            "throughput_samples_per_second": success_throughput,
            "elapsed_seconds": elapsed,
            "inference_elapsed_seconds": inference_elapsed,
            "eta_basis": "inference_success_throughput_excluding_model_load",
            "remaining_seconds": remaining,
            "estimated_completion": eta,
            "gpu_memory": dict(memory or {}),
        }
        self.handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())
        print(
            "[PROGRESS] "
            f"model={self.model_id} rollout={self.rollout_id} "
            f"attempted={attempted}/{self.total} "
            f"inference_success={inference_success} "
            f"runtime_error={counters['runtime_error']} "
            f"parse_error={counters['parse_error']} "
            f"throughput_attempted={attempted_throughput:.6f} "
            f"throughput_inference_success={success_throughput:.6f} "
            f"elapsed={elapsed:.1f} inference_elapsed={inference_elapsed:.1f} "
            f"eta_seconds={remaining} estimated_completion={eta} "
            f"gpu_allocated_gib={(memory or {}).get('allocated_gib')} "
            f"gpu_reserved_gib={(memory or {}).get('reserved_gib')}",
            flush=True,
        )
        self.last_emit = now

    def close(self) -> None:
        self.handle.close()


def task_config(module: Any, task: str):
    if task not in module.TASK_BY_NAME:
        raise KeyError(f"inference entrypoint has no task config for {task}")
    return module.TASK_BY_NAME[task]


def full_image_prediction(
    *, module: Any, inferencer: Any, scorer: Any, row: Mapping[str, Any], image: Any,
    args: argparse.Namespace, config: Any
) -> dict[str, Any]:
    sample_seed = stable_seed(int(args.seed), str(row["record_id"]))
    module.set_sample_seed(sample_seed)
    started = time.monotonic()
    answer = inferencer.predict(image=image, question=str(row["prompt"]))
    latency = time.monotonic() - started
    parsed = module.parse_locateanything_answer(answer)
    _, boxes = module.build_yolo_compatible_detections(
        parsed, config, image.width, image.height, None
    )
    score = score_prediction(
        scorer, row["gt_global"], boxes, parsed.status, args.iou_threshold, image.size
    )
    return {
        "sample_seed": sample_seed,
        "token_usage": dict(inferencer.last_rollout_token_usage or {}),
        "raw_output": answer,
        "parse_status": parsed.status,
        "parse_warnings": parsed.warnings,
        "pred_local": boxes,
        "pred_global": boxes,
        "gt_local": row["gt_global"],
        "gate_diagnostics": dict(getattr(inferencer, "last_ui_diagnostics", {})),
        "crop_outputs": [],
        "latency_seconds": latency,
        **score,
    }


def crop_prediction(
    *, module: Any, tiling: Any, inferencer: Any, scorer: Any,
    row: Mapping[str, Any], crop_rows: Sequence[Mapping[str, Any]], image: Any,
    args: argparse.Namespace, config: Any
) -> dict[str, Any]:
    pending: list[dict[str, Any]] = []
    native: list[dict[str, Any]] = []
    local_by_crop: dict[str, list[list[int]]] = {}
    total_latency = 0.0
    any_parse_error = False
    for crop in sorted(crop_rows, key=lambda item: int(item["crop_index"])):
        crop_box = [int(value) for value in crop["crop_xyxy"]]
        crop_id = str(crop["crop_id"])
        tile = image.crop(tuple(crop_box))
        crop_seed = stable_seed(int(args.seed), str(row["record_id"]), crop_id)
        try:
            module.set_sample_seed(crop_seed)
            started = time.monotonic()
            answer = inferencer.predict(image=tile, question=str(row["prompt"]))
            latency = time.monotonic() - started
            total_latency += latency
            parsed = module.parse_locateanything_answer(answer)
            detections, local_boxes = module.build_yolo_compatible_detections(
                parsed, config, tile.width, tile.height, None
            )
            any_parse_error = any_parse_error or parsed.status == "parse_error"
            local_score = score_prediction(
                scorer,
                crop.get("gt_local", []),
                local_boxes,
                parsed.status,
                args.iou_threshold,
                tile.size,
            )
            for detection in detections:
                pending.append(
                    {
                        "bbox": detection["bbox_2d"],
                        "tile_bbox": crop_box,
                        "label": detection["label"],
                        "class_id": detection["class_id"],
                        "confidence": detection.get("confidence"),
                        "score": 1.0,
                        "source_tile_index": int(crop["crop_index"]),
                        "crop_id": crop_id,
                    }
                )
            local_by_crop[crop_id] = local_boxes
            native.append(
                {
                    "crop_id": crop_id,
                    "crop_index": int(crop["crop_index"]),
                    "crop_xyxy": crop_box,
                    "sample_seed": crop_seed,
                    "token_usage": dict(inferencer.last_rollout_token_usage or {}),
                    "prompt": row["prompt"],
                    "gt_local": crop.get("gt_local", []),
                    "gt_global": crop.get("gt_global", []),
                    "coordinate_transforms": crop.get("coordinate_transforms", []),
                    "raw_output": answer,
                    "parse_status": parsed.status,
                    "parse_warnings": parsed.warnings,
                    "pred_local": local_boxes,
                    "gate_diagnostics": dict(
                        getattr(inferencer, "last_ui_diagnostics", {})
                    ),
                    "latency_seconds": latency,
                    **local_score,
                }
            )
        finally:
            tile.close()
    merged = tiling.merge_tile_predictions(
        pending, image_size=image.size, iou_threshold=args.tile_nms_iou
    )
    global_boxes: list[list[int]] = []
    for item in merged:
        box = [int(round(value)) for value in item["bbox"]]
        box = [
            max(0, min(image.width, box[0])),
            max(0, min(image.height, box[1])),
            max(0, min(image.width, box[2])),
            max(0, min(image.height, box[3])),
        ]
        if box[2] > box[0] and box[3] > box[1]:
            global_boxes.append(box)
    parse_status = "defect" if global_boxes else ("parse_error" if any_parse_error else "ok")
    score = score_prediction(
        scorer,
        row["gt_global"],
        global_boxes,
        parse_status,
        args.iou_threshold,
        image.size,
    )
    return {
        "sample_seed": stable_seed(int(args.seed), str(row["record_id"])),
        "token_usage": {
            "crop_calls": len(native),
            "input_tokens_total": sum(
                int(item["token_usage"].get("input_tokens", 0)) for item in native
            ),
            "effective_max_new_tokens_total": sum(
                int(item["token_usage"].get("effective_max_new_tokens", 0))
                for item in native
            ),
            "per_crop": [item["token_usage"] for item in native],
        },
        "raw_output": native,
        "parse_status": parse_status,
        "contains_crop_parse_error": any_parse_error,
        "parse_warnings": [
            warning
            for item in native
            for warning in item.get("parse_warnings", [])
        ],
        "pred_local": local_by_crop,
        "pred_global": global_boxes,
        "gt_local": {str(item["crop_id"]): item.get("gt_local", []) for item in crop_rows},
        "gate_diagnostics": {
            "mode": "base_scan_plans.base_tiles",
            "crop_gate_diagnostics": [item["gate_diagnostics"] for item in native],
        },
        "crop_outputs": native,
        "latency_seconds": total_latency,
        **score,
    }


def worker_record(
    args: argparse.Namespace,
    code: Mapping[str, Any],
    row: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    width, height = int(row["width"]), int(row["height"])
    return {
        "schema_version": SCHEMA_VERSION,
        "model_id": args.model_id,
        "checkpoint": str(args.checkpoint),
        "git_commit": code["head"],
        "baseline_git_commit": code["baseline"],
        "rollout_id": int(args.rollout_id),
        "seed": int(args.seed),
        "generation_config": generation_config(args),
        "record_id": row["record_id"],
        "sample_id": row["sample_id"],
        "source_image_id": row["source_image_id"],
        "image_id": row["source_image_id"],
        "image_relpath": row["image_relpath"],
        "image_size": {"width": width, "height": height},
        "task": row["task"],
        "crop_id": "full_image" if args.model_id == "m31" else "merged_base_tiles",
        "crop_xyxy": [0, 0, width, height],
        "source_records": row.get("source_records", []),
        "original_training_record": row.get("original_training_record"),
        "prompt": row["prompt"],
        "gt_local": result["gt_local"],
        "gt_global": row["gt_global"],
        "raw_output": result["raw_output"],
        "parse_status": result["parse_status"],
        "parse_warnings": result.get("parse_warnings", []),
        "pred_local": result["pred_local"],
        "pred_global": result["pred_global"],
        "matched_pairs": result["matched_pairs"],
        "TP_box": result["TP_box"],
        "FP_box": result["FP_box"],
        "FN_box": result["FN_box"],
        "image_confusion": result["image_confusion"],
        "error_type": result["error_type"],
        "exact_correct": result["exact_correct"],
        "iou_threshold": args.iou_threshold,
        "gate_diagnostics": result["gate_diagnostics"],
        "crop_outputs": result.get("crop_outputs", []),
        "contains_crop_parse_error": result.get("contains_crop_parse_error", False),
        "pipeline_coverage_failure": bool(row.get("pipeline_coverage_failure")),
        "coverage_failure_type": row.get("coverage_failure_type"),
        "annotation_anomaly": bool(row.get("annotation_anomaly")),
        "coordinate_transform_anomaly": bool(row.get("coordinate_transform_anomaly")),
        "grpo_eligible": bool(row.get("grpo_eligible")),
        "sample_seed": result["sample_seed"],
        "token_usage": result.get("token_usage"),
        "latency_seconds": result["latency_seconds"],
        "finished_at": utc_now(),
        "inference_success": True,
        "runtime_error": None,
    }


def error_record(
    args: argparse.Namespace,
    code: Mapping[str, Any],
    row: Mapping[str, Any],
    exc: Exception,
    latency_seconds: float,
) -> dict[str, Any]:
    width, height = int(row["width"]), int(row["height"])
    return {
        "schema_version": SCHEMA_VERSION,
        "model_id": args.model_id,
        "checkpoint": str(args.checkpoint),
        "git_commit": code["head"],
        "baseline_git_commit": code["baseline"],
        "rollout_id": int(args.rollout_id),
        "seed": int(args.seed),
        "generation_config": generation_config(args),
        "record_id": row["record_id"],
        "sample_id": row["sample_id"],
        "source_image_id": row["source_image_id"],
        "image_id": row["source_image_id"],
        "image_relpath": row["image_relpath"],
        "image_size": {"width": width, "height": height},
        "task": row["task"],
        "crop_id": "full_image" if args.model_id == "m31" else "merged_base_tiles",
        "crop_xyxy": [0, 0, width, height],
        "source_records": row.get("source_records", []),
        "original_training_record": row.get("original_training_record"),
        "prompt": row["prompt"],
        "gt_local": row.get("gt_global", []),
        "gt_global": row.get("gt_global", []),
        "raw_output": None,
        "parse_status": "not_attempted",
        "parse_warnings": [],
        "pred_local": None,
        "pred_global": None,
        "matched_pairs": [],
        "TP_box": None,
        "FP_box": None,
        "FN_box": None,
        "image_confusion": None,
        "error_type": "RUNTIME_ERROR",
        "exact_correct": None,
        "iou_threshold": args.iou_threshold,
        "gate_diagnostics": {},
        "crop_outputs": [],
        "pipeline_coverage_failure": bool(row.get("pipeline_coverage_failure")),
        "coverage_failure_type": row.get("coverage_failure_type"),
        "annotation_anomaly": bool(row.get("annotation_anomaly")),
        "coordinate_transform_anomaly": bool(row.get("coordinate_transform_anomaly")),
        "grpo_eligible": False,
        "token_usage": None,
        "runtime_error": {
            "type": classify_runtime_error(exc),
            "python_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(limit=50),
        },
        "inference_success": False,
        "sample_seed": stable_seed(int(args.seed), str(row["record_id"])),
        "latency_seconds": latency_seconds,
        "finished_at": utc_now(),
    }


def parse_int_csv(value: str | None, name: str) -> list[int]:
    if value is None:
        raise ValueError(f"--{name} is required")
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"--{name} must be a comma-separated integer list") from exc
    if not values:
        raise ValueError(f"--{name} cannot be empty")
    return values


def classify_runtime_error(exc: BaseException) -> str:
    message = f"{type(exc).__name__}: {exc}".lower()
    if "out of memory" in message or "cuda_oom" in message:
        return "CUDA_OOM"
    if "cuda" in message:
        return "CUDA_ERROR"
    return type(exc).__name__


def cuda_memory(torch: Any) -> dict[str, Any]:
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        divisor = float(1024**3)
        return {
            "allocated_gib": round(torch.cuda.memory_allocated() / divisor, 3),
            "reserved_gib": round(torch.cuda.memory_reserved() / divisor, 3),
            "device_free_gib": round(free_bytes / divisor, 3),
            "device_total_gib": round(total_bytes / divisor, 3),
        }
    except Exception as exc:
        return {"memory_query_error": f"{type(exc).__name__}: {exc}"}


def rollout_args(
    args: argparse.Namespace, rollout_id: int, seed: int
) -> argparse.Namespace:
    values = dict(vars(args))
    values.update({"rollout_id": rollout_id, "seed": seed})
    return argparse.Namespace(**values)


def validate_run_args(args: argparse.Namespace) -> None:
    required = {
        "model-id": args.model_id,
        "checkpoint": args.checkpoint,
        "processor-path": args.processor_path,
        "bundle-root": args.bundle_root,
        "rollout-ids": args.rollout_ids,
        "seeds": args.seeds,
        "physical-gpu": args.physical_gpu,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError("run mode missing arguments: " + ", ".join(missing))
    rollout_ids = parse_int_csv(args.rollout_ids, "rollout-ids")
    seeds = parse_int_csv(args.seeds, "seeds")
    if len(rollout_ids) != 2 or len(set(rollout_ids)) != 2:
        raise ValueError("--rollout-ids must contain exactly two distinct rollout IDs")
    if any(rollout_id not in range(4) for rollout_id in rollout_ids):
        raise ValueError("--rollout-ids values must be in 0,1,2,3")
    if len(seeds) != len(rollout_ids):
        raise ValueError("--seeds must have one seed per rollout ID")
    if tuple(rollout_ids) not in ((0, 1), (2, 3)):
        raise ValueError("formal workers must own rollout IDs 0,1 or 2,3")
    if any(FORMAL_SEEDS[rollout_id] != seed for rollout_id, seed in zip(rollout_ids, seeds)):
        raise ValueError(
            f"formal rollout seeds must match {FORMAL_SEEDS}; got {dict(zip(rollout_ids, seeds))}"
        )
    if args.gpu_model_processes != 2:
        raise ValueError("formal H20x2 execution requires two model processes per GPU")
    token_values = {
        "max_seq_length": args.max_seq_length,
        "max_num_tokens_per_sample": args.max_num_tokens_per_sample,
        "training_max_num_tokens": args.training_max_num_tokens,
        "processor_in_token_limit": args.processor_in_token_limit,
        "max_new_tokens": args.max_new_tokens,
        "n_future_tokens": args.n_future_tokens,
    }
    expected_tokens = {
        "max_seq_length": MAX_SEQ_LENGTH,
        "max_num_tokens_per_sample": MAX_NUM_TOKENS_PER_SAMPLE,
        "training_max_num_tokens": TRAINING_MAX_NUM_TOKENS,
        "processor_in_token_limit": PROCESSOR_IN_TOKEN_LIMIT,
        "max_new_tokens": ROLLOUT_MAX_NEW_TOKENS,
        "n_future_tokens": 6,
    }
    if token_values != expected_tokens:
        raise ValueError(
            f"formal rollout token configuration mismatch: actual={token_values} "
            f"expected={expected_tokens}"
        )
    if args.part_size <= 0:
        raise ValueError("--part-size must be positive")
    if args.progress_every <= 0 or args.progress_seconds <= 0:
        raise ValueError("progress intervals must be positive")
    if not 0 <= args.iou_threshold <= 1 or not 0 <= args.tile_nms_iou <= 1:
        raise ValueError("IoU thresholds must be in [0,1]")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise RuntimeError(
            "each worker must be isolated to exactly one physical GPU via CUDA_VISIBLE_DEVICES"
        )


def run_worker(args: argparse.Namespace) -> int:
    validate_run_args(args)
    repo = args.repo_root.expanduser().resolve(strict=True)
    code = verify_code_identity(repo, str(args.model_id))
    bundle = args.bundle_root.expanduser().resolve(strict=True)
    samples = fixed_interleaved_samples(
        read_jsonl(bundle / "manifest" / "task_samples.jsonl")
    )
    sample_order_digest = hashlib.sha256(
        "\n".join(str(row["record_id"]) for row in samples).encode("utf-8")
    ).hexdigest()
    crop_index: dict[str, list[dict[str, Any]]] = {}
    if args.model_id == "crop":
        for crop in read_jsonl(bundle / "manifest" / "crop_samples.jsonl"):
            crop_index.setdefault(str(crop["record_id"]), []).append(crop)
        if set(crop_index) != {str(row["record_id"]) for row in samples}:
            raise RuntimeError("crop manifest/sample IDs are not a complete 1:many mapping")
        base_plans = json.loads((bundle / "base_scan_plans.json").read_text(encoding="utf-8"))
        if not isinstance(base_plans, dict):
            raise ValueError("base_scan_plans.json must be indexed by source_image_id")
        forbidden_geometry_keys = {"final_tiles", "removed_gt_crossing_seams", "manual_gt_repair"}
        for row in samples:
            record_id = str(row["record_id"])
            metadata = sorted(crop_index[record_id], key=lambda item: int(item["crop_index"]))
            if str(row["task"]) == "content_missing":
                geometry = [[0, 0, int(row["width"]), int(row["height"])]]
            else:
                plan = base_plans.get(str(row["source_image_id"]))
                if not isinstance(plan, dict) or plan.get("gt_used") is not False:
                    raise ValueError(
                        f"missing/invalid GT-free base scan plan: {row['source_image_id']}"
                    )
                if forbidden_geometry_keys & set(plan):
                    raise ValueError(
                        f"forbidden training-only geometry in base plan: {row['source_image_id']}"
                    )
                geometry = plan.get("base_tiles")
                if not isinstance(geometry, list) or not geometry:
                    raise ValueError(
                        f"base plan has no base_tiles: {row['source_image_id']}"
                    )
            if len(metadata) != len(geometry):
                raise ValueError(f"crop metadata/base geometry count mismatch: {record_id}")
            # Geometry is overwritten from the GT-free base plan.  crop_samples
            # supplies labels/transforms only and cannot change inference crops.
            crop_index[record_id] = [
                {**item, "crop_xyxy": [int(value) for value in geometry[index]]}
                for index, item in enumerate(metadata)
            ]

    rollout_ids = parse_int_csv(args.rollout_ids, "rollout-ids")
    seeds = parse_int_csv(args.seeds, "seeds")
    assigned = [
        rollout_args(args, rollout_id, seed)
        for rollout_id, seed in zip(rollout_ids, seeds)
    ]
    output_root = args.output_root.expanduser().resolve(strict=False)
    contexts = []
    for assigned_args in assigned:
        raw_dir = (
            output_root
            / "raw"
            / str(args.model_id)
            / f"rollout_{assigned_args.rollout_id}"
        )
        progress_path = (
            output_root
            / "progress"
            / str(args.model_id)
            / f"rollout_{assigned_args.rollout_id}.jsonl"
        )
        contexts.append(
            {
                "args": assigned_args,
                "writer": PartWriter(raw_dir, args.part_size),
                "progress": ProgressWriter(
                    progress_path,
                    str(args.model_id),
                    int(assigned_args.rollout_id),
                    int(assigned_args.seed),
                    len(samples),
                ),
                "counters": {
                    "attempted": 0,
                    "inference_success": 0,
                    "runtime_error": 0,
                    "parse_error": 0,
                },
            }
        )
    rollout_text = ",".join(map(str, rollout_ids))
    worker_manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "model_id": args.model_id,
        "physical_gpu": args.physical_gpu,
        "pid": os.getpid(),
        "rollout_ids": rollout_ids,
        "seeds": seeds,
        "checkpoint": str(args.checkpoint),
        "processor_path": str(args.processor_path),
        "generation_config": generation_config(args),
        "sample_order": {
            "policy": "fixed_round_robin_task_then_positive_negative",
            "tasks": list(TASKS),
            "polarity_order": ["positive", "negative"],
            "records": len(samples),
            "record_id_sha256": sample_order_digest,
        },
        "gpu_model_processes": args.gpu_model_processes,
    }
    atomic_json(
        output_root
        / "manifests"
        / f"{args.model_id}_rollouts_{rollout_text.replace(',', '_')}.json",
        worker_manifest,
    )
    print(
        "[MODEL_PROCESS_START] "
        f"model={args.model_id} gpu={args.physical_gpu} pid={os.getpid()} "
        f"rollouts={rollout_text} gpu_model_processes={args.gpu_model_processes} "
        f"checkpoint={args.checkpoint} processor={args.processor_path}",
        flush=True,
    )
    print(
        "[TOKEN_CONFIG] "
        f"model={args.model_id} gpu={args.physical_gpu} pid={os.getpid()} "
        f"rollouts={rollout_text} MAX_SEQ_LENGTH={args.max_seq_length} "
        f"MAX_NUM_TOKENS_PER_SAMPLE={args.max_num_tokens_per_sample} "
        f"MAX_NUM_TOKENS={args.training_max_num_tokens} "
        f"processor_in_token_limit={args.processor_in_token_limit} "
        f"max_new_tokens={args.max_new_tokens} n_future_tokens={args.n_future_tokens} "
        "effective_rule=min(512,7268-input_tokens) "
        f"sample_order_sha256={sample_order_digest}",
        flush=True,
    )
    inference_path = (
        args.inference_script.expanduser().resolve(strict=True)
        if args.inference_script is not None
        else repo / "scripts" / "inference_ui_defect_locany.py"
    )
    load_started = time.monotonic()
    memory_before: dict[str, Any] = {}
    model_loaded = False
    try:
        module = load_module(
            inference_path, f"ui5_rollout_inference_{args.model_id}_{os.getpid()}"
        )
        scorer = load_module(
            repo / "qwen3vl_merge_and_score_fixed_5tasks.py",
            f"ui5_formal_scorer_{args.model_id}_{os.getpid()}",
        )
        tiling = None
        if args.model_id == "crop":
            tiling = load_module(
                repo / "scripts" / "ui5_lossless_tiling.py",
                f"ui5_lossless_tiling_worker_{os.getpid()}",
            )
        import torch
        from PIL import Image

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError(
                "worker expected exactly one visible CUDA device; "
                f"available={torch.cuda.is_available()} count={torch.cuda.device_count()}"
            )
        memory_before = cuda_memory(torch)
        for context in contexts:
            context["progress"].emit(
                context["counters"], "loading_model", force=True, memory=memory_before
            )
        model_args = make_generation_args(assigned[0])
        inferencer = module.LocateAnythingInferencer(model_args)
        processor_limit = install_generation_token_budget(inferencer, assigned[0])
        torch.cuda.synchronize()
        model_loaded = True
        load_seconds = time.monotonic() - load_started
        memory_after = cuda_memory(torch)
        print(
            "[MODEL_LOAD_OK] "
            f"model={args.model_id} gpu={args.physical_gpu} pid={os.getpid()} "
            f"rollouts={rollout_text} checkpoint={args.checkpoint} "
            f"processor={args.processor_path} load_seconds={load_seconds:.3f} "
            f"token_config={json.dumps(generation_config(args), sort_keys=True)} "
            f"processor_limit={json.dumps(processor_limit, sort_keys=True)} "
            f"memory_before={json.dumps(memory_before, sort_keys=True)} "
            f"memory_after={json.dumps(memory_after, sort_keys=True)}",
            flush=True,
        )
        for context in contexts:
            context["progress"].emit(
                context["counters"], "running", force=True, memory=memory_after
            )
        for row in samples:
            image = None
            try:
                image_path = bundle / str(row["image_relpath"])
                with Image.open(image_path) as opened:
                    image = opened.convert("RGB")
            except Exception as exc:
                for context in contexts:
                    context["counters"]["attempted"] += 1
                    context["counters"]["runtime_error"] += 1
                    context["writer"].write(
                        error_record(context["args"], code, row, exc, 0.0)
                    )
            else:
                for context in contexts:
                    inference_started = time.monotonic()
                    context["counters"]["attempted"] += 1
                    try:
                        assigned_args = context["args"]
                        assert image is not None
                        config = task_config(module, str(row["task"]))
                        if args.model_id == "m31":
                            result = full_image_prediction(
                                module=module,
                                inferencer=inferencer,
                                scorer=scorer,
                                row=row,
                                image=image,
                                args=assigned_args,
                                config=config,
                            )
                        else:
                            assert tiling is not None
                            result = crop_prediction(
                                module=module,
                                tiling=tiling,
                                inferencer=inferencer,
                                scorer=scorer,
                                row=row,
                                crop_rows=crop_index[str(row["record_id"])],
                                image=image,
                                args=assigned_args,
                                config=config,
                            )
                        record = worker_record(assigned_args, code, row, result)
                        context["counters"]["inference_success"] += 1
                        context["counters"]["parse_error"] += int(
                            result["parse_status"] == "parse_error"
                            or result.get("contains_crop_parse_error", False)
                        )
                    except Exception as exc:
                        context["counters"]["runtime_error"] += 1
                        record = error_record(
                            context["args"],
                            code,
                            row,
                            exc,
                            time.monotonic() - inference_started,
                        )
                    context["writer"].write(record)
            finally:
                if image is not None:
                    image.close()
            due_contexts = [
                context
                for context in contexts
                if context["progress"].should_emit(
                    context["counters"]["attempted"],
                    args.progress_every,
                    args.progress_seconds,
                )
            ]
            if due_contexts:
                memory = cuda_memory(torch)
            for context in due_contexts:
                progress = context["progress"]
                counters = context["counters"]
                context["writer"].flush()
                progress.emit(counters, "running", force=True, memory=memory)
        final_memory = cuda_memory(torch)
        for context in contexts:
            context["writer"].flush()
            status = (
                "completed"
                if context["counters"]["runtime_error"] == 0
                else "completed_with_runtime_errors"
            )
            context["progress"].emit(
                context["counters"], status, force=True, memory=final_memory
            )
        return 0
    except BaseException as exc:
        load_seconds = time.monotonic() - load_started
        error_type = classify_runtime_error(exc)
        try:
            import torch

            memory_after_failure = cuda_memory(torch)
        except Exception:
            memory_after_failure = {}
        event = "MODEL_LOAD_FAIL" if not model_loaded else "WORKER_FATAL"
        print(
            f"[{event}] "
            f"model={args.model_id} gpu={args.physical_gpu} pid={os.getpid()} "
            f"rollouts={rollout_text} checkpoint={args.checkpoint} "
            f"processor={args.processor_path} error={error_type} "
            f"message={str(exc)!r} load_seconds={load_seconds:.3f} "
            f"memory_before={json.dumps(memory_before, sort_keys=True)} "
            f"memory_after={json.dumps(memory_after_failure, sort_keys=True)}",
            flush=True,
        )
        for context in contexts:
            context["progress"].emit(
                context["counters"],
                "model_load_failed" if not model_loaded else "failed",
                force=True,
                memory=memory_after_failure,
            )
        traceback.print_exc()
        return 2
    finally:
        for context in contexts:
            context["writer"].close()
            context["progress"].close()


def last_jsonl_row(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    last = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last = json.loads(line)
    return last


def progress_snapshot(args: argparse.Namespace) -> int:
    root = args.output_root.expanduser().resolve(strict=False)
    latest = []
    for model_id in MODEL_IDS:
        for rollout_id in range(4):
            path = root / "progress" / model_id / f"rollout_{rollout_id}.jsonl"
            row = last_jsonl_row(path)
            if row is not None:
                latest.append(row)
    remaining_values = [
        float(row["remaining_seconds"])
        for row in latest
        if isinstance(row.get("remaining_seconds"), (int, float))
    ]
    total_remaining = max(remaining_values, default=None)
    snapshot = {
        "timestamp": utc_now(),
        "workers_seen": len(latest),
        "workers_expected": args.expected_workers,
        "workers": latest,
        "total_remaining_seconds": total_remaining,
        "total_estimated_completion": (
            datetime.fromtimestamp(time.time() + total_remaining, timezone.utc).isoformat()
            if total_remaining is not None
            else None
        ),
    }
    path = root / "progress" / "total_eta.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps(snapshot, ensure_ascii=False))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "progress-snapshot":
        return progress_snapshot(args)
    return run_worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
