#!/usr/bin/env python3
"""One persistent UI5 train-rollout worker or a CPU progress snapshot.

Run mode loads exactly one checkpoint once, then processes every portable
``image_id+task`` sample for its one assigned rollout in the common fixed order.
Crop mode performs in-memory base-tile crops and reuses the tiled-eval branch's
global mapping plus class-aware greedy NMS.
"""
from __future__ import annotations

import argparse
import gc
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

from ui5_metric_matching import threshold_aware_linear_sum_assignment


SCHEMA_VERSION = 6
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
    parser.add_argument("--gpu-model-processes", type=int, default=1)
    parser.add_argument("--part-size", type=int, default=10000)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--progress-seconds", type=float, default=60.0)
    parser.add_argument("--iou-threshold", type=float, default=0.1)
    parser.add_argument("--tile-nms-iou", type=float, default=0.5)
    parser.add_argument("--expected-workers", type=int, default=8)
    parser.add_argument(
        "--physical-worker",
        action="append",
        default=[],
        help="progress mode: model,gpu,pid,single_rollout_id",
    )
    parser.add_argument("--dtype", choices=("bf16",), default="bf16")
    parser.add_argument("--attn-implementation", choices=("sdpa",), default="sdpa")
    parser.add_argument(
        "--vision-attn-implementation",
        choices=("flash_attention_2",),
        default="flash_attention_2",
    )
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
    inferencer.active_rollout_context = None

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
        active_context = getattr(inferencer, "active_rollout_context", None)
        if isinstance(active_context, dict):
            active_context["input_tokens"] = input_tokens
            active_context["effective_max_new_tokens"] = effective_max_new_tokens
        return original_generate(*positional, **inputs)

    inferencer.model.generate = bounded_generate
    return {
        "previous_in_token_limit": previous_in_token_limit,
        "active_in_token_limit": getattr(processor, "in_token_limit", None),
    }


def verify_loaded_attention_backends(
    inferencer: Any, args: argparse.Namespace
) -> dict[str, Any]:
    """Fail closed unless the instantiated text/Vision backends match training."""
    model = inferencer.model
    model_config = getattr(model, "config", None)
    text_config = getattr(model_config, "text_config", None)
    vision_config = getattr(model_config, "vision_config", None)
    text_backend = str(getattr(text_config, "_attn_implementation", None))
    vision_backend = str(getattr(vision_config, "_attn_implementation", None))
    blocks = list(getattr(getattr(model.vision_model, "encoder", None), "blocks", ()))
    block_backends = [str(getattr(block, "attn_implementation", None)) for block in blocks]
    matching_blocks = sum(
        backend == args.vision_attn_implementation for backend in block_backends
    )
    report = {
        "text_config": text_backend,
        "vision_config": vision_backend,
        "vision_first_layer": block_backends[0] if block_backends else "<missing>",
        "vision_blocks_matching": matching_blocks,
        "vision_blocks_total": len(blocks),
        "vision_blocks": f"{matching_blocks}/{len(blocks)}",
        "vision_block_backends": block_backends,
    }
    failures: list[str] = []
    if text_backend != args.attn_implementation:
        failures.append(
            f"text_config={text_backend!r}, expected={args.attn_implementation!r}"
        )
    if vision_backend != args.vision_attn_implementation:
        failures.append(
            f"vision_config={vision_backend!r}, "
            f"expected={args.vision_attn_implementation!r}"
        )
    if len(blocks) != 27 or matching_blocks != 27:
        failures.append(
            "vision_blocks="
            f"{matching_blocks}/{len(blocks)}, expected=27/27 "
            f"backends={block_backends}"
        )
    if failures:
        raise RuntimeError("loaded attention backend audit failed: " + "; ".join(failures))
    return report


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
        gt_indices, pred_indices = threshold_aware_linear_sum_assignment(
            matrix, threshold
        )
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


def _resume_recovery_path(
    directory: Path,
    *,
    model_id: str,
    rollout_id: int,
    kind: str,
) -> Path:
    safe_kind = "".join(character if character.isalnum() else "_" for character in kind)
    return directory / (
        f"{model_id}_rollout_{rollout_id}_{safe_kind}_"
        f"{time.time_ns()}_{os.getpid()}.json"
    )


def _write_resume_recovery(
    directory: Path | None,
    *,
    path: Path,
    model_id: str,
    rollout_id: int,
    kind: str,
    action: str,
    original_size: int,
    new_size: int,
    fragment: bytes,
) -> None:
    recovery_directory = directory or (path.parent / "resume_recovery")
    event = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": utc_now(),
        "action": action,
        "kind": kind,
        "model_id": model_id,
        "rollout_id": int(rollout_id),
        "path": str(path.resolve(strict=False)),
        "original_size": int(original_size),
        "new_size": int(new_size),
        "removed_bytes": max(0, int(original_size) - int(new_size)),
        "fragment_bytes": len(fragment),
        "fragment_sha256": hashlib.sha256(fragment).hexdigest(),
        "fragment_preview": fragment[:512].decode("utf-8", errors="replace"),
    }
    diagnostic_path = _resume_recovery_path(
        recovery_directory,
        model_id=model_id,
        rollout_id=rollout_id,
        kind=kind,
    )
    atomic_json(diagnostic_path, event)
    print(
        "[RESUME_RECOVERY] "
        f"model={model_id} rollout={rollout_id} kind={kind} action={action} "
        f"path={path} original_size={original_size} new_size={new_size} "
        f"fragment_sha256={event['fragment_sha256']} "
        f"diagnostic={diagnostic_path}",
        flush=True,
    )


def _safe_truncate_eof_fragment(
    path: Path,
    *,
    offset: int,
    expected_size: int,
    fragment: bytes,
    recovery_diagnostics_dir: Path | None,
    model_id: str,
    rollout_id: int,
    kind: str,
) -> None:
    current_size = path.stat().st_size
    if current_size != expected_size or not 0 <= offset < expected_size:
        raise RuntimeError(
            "refusing unsafe resume truncation after concurrent file change: "
            f"path={path} expected_size={expected_size} current_size={current_size} "
            f"offset={offset}"
        )
    with path.open("r+b") as handle:
        handle.truncate(offset)
        handle.flush()
        os.fsync(handle.fileno())
    _write_resume_recovery(
        recovery_diagnostics_dir,
        path=path,
        model_id=model_id,
        rollout_id=rollout_id,
        kind=kind,
        action="truncate_incomplete_unterminated_eof_fragment",
        original_size=expected_size,
        new_size=offset,
        fragment=fragment,
    )


def _safe_append_missing_newline(
    path: Path,
    *,
    expected_size: int,
    recovery_diagnostics_dir: Path | None,
    model_id: str,
    rollout_id: int,
    kind: str,
) -> None:
    current_size = path.stat().st_size
    if current_size != expected_size or expected_size <= 0:
        raise RuntimeError(
            "refusing unsafe resume newline repair after concurrent file change: "
            f"path={path} expected_size={expected_size} current_size={current_size}"
        )
    with path.open("ab") as handle:
        handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    _write_resume_recovery(
        recovery_diagnostics_dir,
        path=path,
        model_id=model_id,
        rollout_id=rollout_id,
        kind=kind,
        action="append_missing_newline_after_valid_eof_record",
        original_size=expected_size,
        new_size=expected_size + 1,
        fragment=b"",
    )


def _iter_jsonl_with_tail_recovery(
    path: Path,
    *,
    allow_incomplete_tail_recovery: bool,
    normalize_valid_eof_newline: bool,
    recovery_diagnostics_dir: Path | None,
    model_id: str,
    rollout_id: int,
    kind: str,
) -> Iterable[tuple[int, dict[str, Any]]]:
    """Read JSONL, repairing only an unterminated malformed EOF fragment."""
    file_size = path.stat().st_size
    repair: tuple[int, bytes] | None = None
    append_newline = False
    with path.open("rb") as handle:
        line_no = 0
        while True:
            start = handle.tell()
            raw_line = handle.readline()
            if not raw_line:
                break
            line_no += 1
            at_eof = handle.tell() == file_size
            terminated = raw_line.endswith(b"\n")
            if not raw_line.strip():
                if at_eof and not terminated:
                    if not allow_incomplete_tail_recovery:
                        raise RuntimeError(
                            f"unterminated blank JSONL tail is not recoverable at "
                            f"{path}:{line_no}"
                        )
                    repair = (start, raw_line)
                    break
                continue
            try:
                decoded = raw_line.decode("utf-8")
                row = json.loads(decoded)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                if at_eof and not terminated and allow_incomplete_tail_recovery:
                    repair = (start, raw_line)
                    break
                raise RuntimeError(
                    f"invalid JSONL at {path}:{line_no}: {type(exc).__name__}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"JSONL row is not an object at {path}:{line_no}")
            yield line_no, row
            if at_eof and not terminated and normalize_valid_eof_newline:
                append_newline = True
    if repair is not None:
        _safe_truncate_eof_fragment(
            path,
            offset=repair[0],
            expected_size=file_size,
            fragment=repair[1],
            recovery_diagnostics_dir=recovery_diagnostics_dir,
            model_id=model_id,
            rollout_id=rollout_id,
            kind=kind,
        )
    elif append_newline:
        _safe_append_missing_newline(
            path,
            expected_size=file_size,
            recovery_diagnostics_dir=recovery_diagnostics_dir,
            model_id=model_id,
            rollout_id=rollout_id,
            kind=kind,
        )


def resume_route_state(
    raw_dir: Path,
    *,
    model_id: str,
    rollout_id: int,
    seed: int,
    checkpoint: Path,
    processor_path: Path,
    generation: Mapping[str, Any],
    git_commit: str,
    baseline_git_commit: str,
    worker_git_commit: str,
    samples_by_record: Mapping[str, Mapping[str, Any]],
    recovery_diagnostics_dir: Path | None = None,
) -> tuple[set[str], dict[str, int]]:
    """Validate prior JSONL and reconstruct cumulative route counters.

    Presence of one valid raw record means that route/sample attempt is
    complete, including a persisted technical error.  Resume therefore never
    silently retries or duplicates a record whose outcome is already auditable.
    """
    completed: set[str] = set()
    counters = {
        "attempted": 0,
        "inference_success": 0,
        "runtime_error": 0,
        "parse_error": 0,
        "oom_exception_count": 0,
        "oom_recovered_samples": 0,
        "oom_final_failed_samples": 0,
    }
    if not raw_dir.exists():
        return completed, counters
    indexed_paths: list[tuple[int, Path]] = []
    for path in raw_dir.glob("part-*.jsonl"):
        suffix = path.stem.removeprefix("part-")
        if not suffix.isdigit():
            raise RuntimeError(f"invalid rollout part filename: {path}")
        indexed_paths.append((int(suffix), path))
    indexed_paths.sort()
    indices = [index for index, _ in indexed_paths]
    if len(indices) != len(set(indices)):
        raise RuntimeError(f"duplicate rollout part index in {raw_dir}")
    highest_index = max(indices, default=None)
    for part_index, path in indexed_paths:
        rows = _iter_jsonl_with_tail_recovery(
            path,
            allow_incomplete_tail_recovery=part_index == highest_index,
            normalize_valid_eof_newline=True,
            recovery_diagnostics_dir=recovery_diagnostics_dir,
            model_id=model_id,
            rollout_id=rollout_id,
            kind=f"raw_part_{part_index:05d}",
        )
        for line_no, row in rows:
            record_id = str(row.get("record_id", ""))
            expected = samples_by_record.get(record_id)
            if expected is None:
                raise RuntimeError(
                    f"unknown resumed record_id at {path}:{line_no}: {record_id!r}"
                )
            if record_id in completed:
                raise RuntimeError(
                    f"duplicate resumed record_id at {path}:{line_no}: {record_id}"
                )
            expected_sample_id = str(expected.get("sample_id", ""))
            prior_checkpoint = Path(str(row.get("checkpoint", ""))).expanduser()
            prior_processor = Path(str(row.get("processor_path", ""))).expanduser()
            if (
                int(row.get("schema_version", -1)) != SCHEMA_VERSION
                or str(row.get("model_id")) != model_id
                or int(row.get("rollout_id", -1)) != rollout_id
                or int(row.get("seed", -1)) != seed
                or str(row.get("sample_id", "")) != expected_sample_id
                or prior_checkpoint.resolve(strict=False)
                != checkpoint.expanduser().resolve(strict=False)
                or prior_processor.resolve(strict=False)
                != processor_path.expanduser().resolve(strict=False)
                or row.get("generation_config") != dict(generation)
                or str(row.get("git_commit")) != git_commit
                or str(row.get("baseline_git_commit")) != baseline_git_commit
                or str(row.get("worker_git_commit")) != worker_git_commit
                or str(row.get("task")) != str(expected.get("task"))
                or str(row.get("source_image_id"))
                != str(expected.get("source_image_id"))
                or str(row.get("image_relpath"))
                != str(expected.get("image_relpath"))
            ):
                raise RuntimeError(
                    "resumed route identity mismatch at "
                    f"{path}:{line_no}: schema={row.get('schema_version')} "
                    f"model={row.get('model_id')} "
                    f"rollout={row.get('rollout_id')} seed={row.get('seed')} "
                    f"record_id={record_id} sample_id={row.get('sample_id')} "
                    f"checkpoint={row.get('checkpoint')} "
                    f"processor_path={row.get('processor_path')} "
                    f"git_commit={row.get('git_commit')} "
                    f"baseline_git_commit={row.get('baseline_git_commit')} "
                    f"worker_git_commit={row.get('worker_git_commit')} "
                    f"task={row.get('task')} "
                    f"source_image_id={row.get('source_image_id')}"
                )
            completed.add(record_id)
            counters["attempted"] += 1
            inference_success = bool(row.get("inference_success"))
            counters["inference_success"] += int(inference_success)
            counters["runtime_error"] += int(
                not inference_success or row.get("runtime_error") is not None
            )
            counters["parse_error"] += int(
                inference_success
                and (
                    row.get("parse_status") == "parse_error"
                    or bool(row.get("contains_crop_parse_error"))
                )
            )
            counters["oom_exception_count"] += int(row.get("oom_events", 0))
            counters["oom_recovered_samples"] += int(bool(row.get("oom_recovered")))
            counters["oom_final_failed_samples"] += int(
                bool(row.get("oom_final_failure"))
            )
    return completed, counters


class PartWriter:
    def __init__(self, directory: Path, part_size: int):
        self.directory = directory
        self.part_size = part_size
        self.directory.mkdir(parents=True, exist_ok=True)
        existing_indices: list[int] = []
        for path in sorted(self.directory.glob("part-*.jsonl")):
            suffix = path.stem.removeprefix("part-")
            if not suffix.isdigit():
                raise RuntimeError(f"invalid rollout part filename: {path}")
            existing_indices.append(int(suffix))
        if len(existing_indices) != len(set(existing_indices)):
            raise RuntimeError(f"duplicate rollout part index in {self.directory}")
        # A resumed worker never appends to an existing part.  The completed
        # JSONL is validated before this writer is constructed, and new rows
        # start in the next part so a killed process cannot corrupt old data.
        self.part_index = max(existing_indices, default=-1)
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
    def __init__(
        self,
        path: Path,
        model_id: str,
        rollout_id: int,
        seed: int,
        total: int,
        resume_attempted: int = 0,
        recovery_diagnostics_dir: Path | None = None,
    ):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        prior_elapsed = 0.0
        prior_inference_elapsed = 0.0
        prior_attempted = 0
        if path.exists():
            rows = _iter_jsonl_with_tail_recovery(
                path,
                allow_incomplete_tail_recovery=True,
                normalize_valid_eof_newline=True,
                recovery_diagnostics_dir=recovery_diagnostics_dir,
                model_id=model_id,
                rollout_id=rollout_id,
                kind="progress",
            )
            for line_no, row in rows:
                if (
                    row.get("model_id") != model_id
                    or int(row.get("rollout_id", -1)) != rollout_id
                    or int(row.get("seed", -1)) != seed
                    or int(row.get("total", -1)) != total
                ):
                    raise RuntimeError(
                        f"progress route mismatch at {path}:{line_no}: {row}"
                    )
                attempted = int(row.get("attempted", -1))
                if attempted < prior_attempted or not 0 <= attempted <= total:
                    raise RuntimeError(
                        f"invalid/non-monotonic progress at {path}:{line_no}: "
                        f"attempted={attempted} previous={prior_attempted} total={total}"
                    )
                prior_attempted = attempted
                prior_elapsed = max(prior_elapsed, float(row.get("elapsed_seconds", 0)))
                prior_inference_elapsed = max(
                    prior_inference_elapsed,
                    float(row.get("inference_elapsed_seconds", 0)),
                )
        if prior_attempted > resume_attempted:
            raise RuntimeError(
                f"progress is ahead of durable raw JSONL for {model_id} rollout "
                f"{rollout_id}: progress={prior_attempted} raw={resume_attempted}"
            )
        self.handle = path.open("a", encoding="utf-8")
        self.model_id = model_id
        self.rollout_id = rollout_id
        self.seed = seed
        self.total = total
        self.started = time.monotonic()
        self.inference_started: float | None = None
        self.last_emit = self.started
        self.prior_elapsed = prior_elapsed
        self.prior_inference_elapsed = prior_inference_elapsed

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
        elapsed = self.prior_elapsed + max(0.0, now - self.started)
        inference_elapsed = self.prior_inference_elapsed + max(
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
            remaining_samples / attempted_throughput
            if attempted_throughput
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
            "oom_exception_count": int(counters.get("oom_exception_count", 0)),
            "oom_recovered_samples": int(counters.get("oom_recovered_samples", 0)),
            "oom_final_failed_samples": int(
                counters.get("oom_final_failed_samples", 0)
            ),
            "total": self.total,
            "throughput_attempted_per_second": attempted_throughput,
            "throughput_inference_success_per_second": success_throughput,
            "throughput_samples_per_second": attempted_throughput,
            "elapsed_seconds": elapsed,
            "inference_elapsed_seconds": inference_elapsed,
            "eta_basis": "attempted_throughput_excluding_model_load",
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
            f"oom_exceptions={counters.get('oom_exception_count', 0)} "
            f"oom_recovered={counters.get('oom_recovered_samples', 0)} "
            f"oom_final_failed={counters.get('oom_final_failed_samples', 0)} "
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


def set_inference_context(
    inferencer: Any,
    context: dict[str, Any],
    torch: Any,
    *,
    stage: str,
    crop_id: str,
    crop_index: int | None,
    crop_xyxy: Sequence[int],
    tile_count: int,
    tile_size: tuple[int, int],
) -> None:
    context.clear()
    context.update(
        {
            "stage": stage,
            "crop_id": crop_id,
            "crop_index": crop_index,
            "crop_xyxy": list(map(int, crop_xyxy)),
            "tile_count": int(tile_count),
            "tile_size": {"width": int(tile_size[0]), "height": int(tile_size[1])},
            "input_tokens": None,
            "memory_before_oom": cuda_memory(torch),
        }
    )
    inferencer.active_rollout_context = context


def full_image_prediction(
    *, module: Any, inferencer: Any, scorer: Any, row: Mapping[str, Any], image: Any,
    args: argparse.Namespace, config: Any, torch: Any,
    inference_context: dict[str, Any],
) -> dict[str, Any]:
    sample_seed = stable_seed(int(args.seed), str(row["record_id"]))
    module.set_sample_seed(sample_seed)
    set_inference_context(
        inferencer,
        inference_context,
        torch,
        stage="full_image_generate",
        crop_id="full_image",
        crop_index=None,
        crop_xyxy=[0, 0, image.width, image.height],
        tile_count=1,
        tile_size=image.size,
    )
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
    args: argparse.Namespace, config: Any, torch: Any,
    inference_context: dict[str, Any],
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
            set_inference_context(
                inferencer,
                inference_context,
                torch,
                stage="crop_generate",
                crop_id=crop_id,
                crop_index=int(crop["crop_index"]),
                crop_xyxy=crop_box,
                tile_count=len(crop_rows),
                tile_size=tile.size,
            )
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
    inference_context.update(
        {
            "stage": "tiled_merge_nms",
            "crop_id": "merged_base_tiles",
            "crop_index": None,
            "crop_xyxy": [0, 0, image.width, image.height],
            "tile_count": len(crop_rows),
            "tile_size": {"width": image.width, "height": image.height},
            "memory_before_oom": cuda_memory(torch),
        }
    )
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
        "processor_path": str(args.processor_path),
        "git_commit": code["head"],
        "baseline_git_commit": code["baseline"],
        "worker_git_commit": args.worker_git_commit,
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
        "oom_recovered": bool(result.get("oom_recovered")),
        "oom_events": int(result.get("oom_events", 0)),
        "oom_retry": result.get("oom_retry"),
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
    failure_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    width, height = int(row["width"]), int(row["height"])
    context = dict(failure_context or {})
    oom_diagnostics = getattr(exc, "oom_diagnostics", None)
    if isinstance(oom_diagnostics, Mapping):
        retry_context = oom_diagnostics.get("retry_attempt", {}).get("context", {})
        if isinstance(retry_context, Mapping):
            context = {**context, **retry_context}
    runtime_error = exception_payload(exc, traceback.format_exc(limit=50))
    runtime_error.update(
        {
            "stage": context.get("stage", "sample_inference"),
            "crop_id": context.get("crop_id"),
            "crop_index": context.get("crop_index"),
            "crop_xyxy": context.get("crop_xyxy"),
            "input_tokens": context.get("input_tokens"),
            "tile_count": context.get("tile_count"),
            "tile_size": context.get("tile_size"),
            "memory_before_oom": context.get("memory_before_oom"),
            "oom": oom_diagnostics,
        }
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "model_id": args.model_id,
        "checkpoint": str(args.checkpoint),
        "processor_path": str(args.processor_path),
        "git_commit": code["head"],
        "baseline_git_commit": code["baseline"],
        "worker_git_commit": args.worker_git_commit,
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
        "crop_id": context.get(
            "crop_id", "full_image" if args.model_id == "m31" else "merged_base_tiles"
        ),
        "crop_index": context.get("crop_index"),
        "crop_xyxy": context.get("crop_xyxy", [0, 0, width, height]),
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
        "oom_recovered": False,
        "oom_events": int(
            oom_diagnostics.get("oom_events", 0)
            if isinstance(oom_diagnostics, Mapping)
            else 0
        ),
        "oom_final_failure": bool(
            oom_diagnostics.get("oom_final_failure")
            if isinstance(oom_diagnostics, Mapping)
            else False
        ),
        "oom_retry": oom_diagnostics,
        "runtime_error": runtime_error,
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
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(f"{type(current).__name__}: {current}")
        current = current.__cause__ or current.__context__
    message = " | ".join(parts).lower()
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


def exception_payload(exc: BaseException, traceback_text: str | None = None) -> dict[str, Any]:
    return {
        "type": classify_runtime_error(exc),
        "python_type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback_text or "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
    }


def exception_chain(exc: BaseException) -> list[dict[str, str]]:
    chain: list[dict[str, str]] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(
            {
                "python_type": type(current).__name__,
                "message": str(current),
            }
        )
        current = current.__cause__ or current.__context__
    return chain


def release_after_oom(torch: Any, inferencer: Any) -> dict[str, Any]:
    inferencer.active_rollout_context = None
    inferencer.last_rollout_token_usage = None
    collected = gc.collect()
    torch.cuda.empty_cache()
    return {
        "gc_collected": int(collected),
        "memory_after_cleanup": cuda_memory(torch),
    }


class OOMRetryFailure(RuntimeError):
    def __init__(self, diagnostics: Mapping[str, Any]) -> None:
        retry_exception = diagnostics.get("retry_attempt", {}).get("exception", {})
        super().__init__(
            "CUDA OOM retry failed: "
            f"{retry_exception.get('python_type')}: {retry_exception.get('message')}"
        )
        self.oom_diagnostics = dict(diagnostics)


def prediction_with_oom_retry(
    operation: Any,
    *,
    torch: Any,
    inferencer: Any,
) -> dict[str, Any]:
    first_context: dict[str, Any] = {}
    try:
        result = operation(first_context)
    except torch.cuda.OutOfMemoryError as exc:
        first_failure = {
            "status": "oom",
            "exception": exception_payload(exc),
            "context": dict(first_context),
            "memory_after_oom": cuda_memory(torch),
        }
    else:
        result["oom_recovered"] = False
        result["oom_events"] = 0
        result["oom_retry"] = None
        return result

    cleanup = release_after_oom(torch, inferencer)
    retry_context: dict[str, Any] = {}
    retry_failure: dict[str, Any] | None = None
    try:
        result = operation(retry_context)
    except Exception as retry_exc:
        retry_is_oom = isinstance(retry_exc, torch.cuda.OutOfMemoryError)
        retry_failure = {
            "oom_recovered": False,
            "oom_final_failure": True,
            "oom_events": 2 if retry_is_oom else 1,
            "first_attempt": first_failure,
            "cleanup": cleanup,
            "retry_attempt": {
                "status": "oom" if retry_is_oom else "runtime_error",
                "exception": exception_payload(retry_exc),
                "context": dict(retry_context),
                "memory_after_failure": cuda_memory(torch),
            },
        }
    if retry_failure is not None:
        retry_failure["final_cleanup"] = release_after_oom(torch, inferencer)
        raise OOMRetryFailure(retry_failure)
    result["oom_recovered"] = True
    result["oom_events"] = 1
    result["oom_retry"] = {
        "oom_recovered": True,
        "oom_final_failure": False,
        "oom_events": 1,
        "first_attempt": first_failure,
        "cleanup": cleanup,
        "retry_attempt": {
            "status": "success",
            "context": dict(retry_context),
            "memory_after_success": cuda_memory(torch),
        },
    }
    return result


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
    if len(rollout_ids) != 1:
        raise ValueError("--rollout-ids must contain exactly one rollout ID")
    if any(rollout_id not in range(4) for rollout_id in rollout_ids):
        raise ValueError("--rollout-ids values must be in 0,1,2,3")
    if len(seeds) != len(rollout_ids):
        raise ValueError("--seeds must have one seed per rollout ID")
    if any(FORMAL_SEEDS[rollout_id] != seed for rollout_id, seed in zip(rollout_ids, seeds)):
        raise ValueError(
            f"formal rollout seeds must match {FORMAL_SEEDS}; got {dict(zip(rollout_ids, seeds))}"
        )
    if args.gpu_model_processes != 4:
        raise ValueError("formal H20x2 execution requires four model processes per GPU")
    expected_physical_gpu = {"m31": 0, "crop": 1}[str(args.model_id)]
    if args.physical_gpu != expected_physical_gpu:
        raise ValueError(
            f"formal {args.model_id} worker must use physical GPU "
            f"{expected_physical_gpu}, got {args.physical_gpu}"
        )
    if args.attn_implementation != "sdpa":
        raise ValueError("formal rollout text attention must be sdpa")
    if args.vision_attn_implementation != "flash_attention_2":
        raise ValueError(
            "formal rollout vision attention must be flash_attention_2"
        )
    generation_values = {
        "dtype": args.dtype,
        "generation_mode": args.generation_mode,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
    }
    expected_generation = {
        "dtype": "bf16",
        "generation_mode": "hybrid",
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 0,
        "repetition_penalty": 1.1,
    }
    if generation_values != expected_generation:
        raise ValueError(
            "formal rollout generation configuration mismatch: "
            f"actual={generation_values} expected={expected_generation}"
        )
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


def wait_for_model_load_barrier(
    output_root: Path,
    *,
    model_id: str,
    physical_gpu: int,
    rollout_ids: Sequence[int],
) -> dict[str, Any]:
    """Wait until the launcher validates all eight live model processes.

    The launcher removes this marker before spawning workers and publishes it
    atomically only after all eight MODEL_LOAD_OK records pass their PID and
    attention-backend checks.  In particular, no worker starts inference while
    any same-GPU peer is still allocating model weights.
    """
    marker = output_root / "diagnostics" / "_MODEL_LOADS_OK"
    rollout_text = ",".join(str(value) for value in rollout_ids)
    started = time.monotonic()
    print(
        "[MODEL_LOAD_BARRIER_WAIT] "
        f"model={model_id} gpu={physical_gpu} pid={os.getpid()} "
        f"rollouts={rollout_text} marker={marker}",
        flush=True,
    )
    while not marker.is_file():
        time.sleep(1.0)
    waited = max(0.0, time.monotonic() - started)
    print(
        "[MODEL_LOAD_BARRIER_RELEASE] "
        f"model={model_id} gpu={physical_gpu} pid={os.getpid()} "
        f"rollouts={rollout_text} marker={marker} waited_seconds={waited:.3f}",
        flush=True,
    )
    return {"marker": str(marker), "waited_seconds": waited}


def run_worker(args: argparse.Namespace) -> int:
    validate_run_args(args)
    repo = args.repo_root.expanduser().resolve(strict=True)
    code = verify_code_identity(repo, str(args.model_id))
    worker_repo = Path(__file__).resolve().parents[1]
    args.worker_git_commit = git_output(worker_repo, "rev-parse", "HEAD")
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
    recovery_diagnostics_dir = output_root / "diagnostics" / "resume_recovery"
    samples_by_record = {str(row["record_id"]): row for row in samples}
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
        completed_record_ids, counters = resume_route_state(
            raw_dir,
            model_id=str(args.model_id),
            rollout_id=int(assigned_args.rollout_id),
            seed=int(assigned_args.seed),
            checkpoint=args.checkpoint,
            processor_path=args.processor_path,
            generation=generation_config(assigned_args),
            git_commit=str(code["head"]),
            baseline_git_commit=str(code["baseline"]),
            worker_git_commit=str(args.worker_git_commit),
            samples_by_record=samples_by_record,
            recovery_diagnostics_dir=recovery_diagnostics_dir,
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
                    resume_attempted=int(counters["attempted"]),
                    recovery_diagnostics_dir=recovery_diagnostics_dir,
                ),
                "counters": counters,
                "completed_record_ids": completed_record_ids,
                "logical_status": "pending",
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
            "policy": "sample_major_fixed_round_robin_task_then_positive_negative",
            "tasks": list(TASKS),
            "polarity_order": ["positive", "negative"],
            "records": len(samples),
            "record_id_sha256": sample_order_digest,
        },
        "gpu_model_processes": args.gpu_model_processes,
        "physical_process_topology": {
            "physical_processes_total": 8,
            "physical_processes_per_gpu": 4,
            "logical_rollouts_per_process": 1,
            "ownership": "one_model_one_rollout",
        },
        "worker_git_commit": args.worker_git_commit,
        "resume": {
            str(context["args"].rollout_id): {
                "completed_records": len(context["completed_record_ids"]),
                "remaining_records": len(samples) - len(context["completed_record_ids"]),
            }
            for context in contexts
        },
        "runtime_environment": {
            "HF_MODULES_CACHE": os.environ.get("HF_MODULES_CACHE"),
            "PYTHONPYCACHEPREFIX": os.environ.get("PYTHONPYCACHEPREFIX"),
            "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        },
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
        "[RESUME_STATE] "
        f"model={args.model_id} gpu={args.physical_gpu} pid={os.getpid()} "
        f"rollouts={rollout_text} routes="
        + json.dumps(worker_manifest["resume"], sort_keys=True),
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
    load_stage = "load_inference_module"
    try:
        module = load_module(
            inference_path, f"ui5_rollout_inference_{args.model_id}_{os.getpid()}"
        )
        load_stage = "load_formal_scorer"
        scorer = load_module(
            repo / "qwen3vl_merge_and_score_fixed_5tasks.py",
            f"ui5_formal_scorer_{args.model_id}_{os.getpid()}",
        )
        tiling = None
        if args.model_id == "crop":
            load_stage = "load_tiling_module"
            tiling = load_module(
                repo / "scripts" / "ui5_lossless_tiling.py",
                f"ui5_lossless_tiling_worker_{os.getpid()}",
            )
        load_stage = "import_runtime_dependencies"
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
        load_stage = "checkpoint_constructor"
        inferencer = module.LocateAnythingInferencer(model_args)
        load_stage = "install_token_budget"
        processor_limit = install_generation_token_budget(inferencer, assigned[0])
        load_stage = "verify_attention_backends"
        attention_report = verify_loaded_attention_backends(inferencer, args)
        load_stage = "cuda_synchronize"
        torch.cuda.synchronize()
        model_loaded = True
        load_stage = "loaded"
        load_seconds = time.monotonic() - load_started
        memory_after = cuda_memory(torch)
        model_load_path = (
            output_root
            / "diagnostics"
            / "model_load"
            / f"{args.model_id}_rollouts_{rollout_text.replace(',', '_')}.json"
        )
        atomic_json(
            model_load_path,
            {
                "schema_version": SCHEMA_VERSION,
                "status": "MODEL_LOAD_OK",
                "created_at": utc_now(),
                "model_id": args.model_id,
                "physical_gpu": args.physical_gpu,
                "pid": os.getpid(),
                "rollout_ids": rollout_ids,
                "checkpoint": str(args.checkpoint),
                "processor_path": str(args.processor_path),
                "load_seconds": load_seconds,
                "memory_before": memory_before,
                "memory_after": memory_after,
                "runtime_environment": worker_manifest["runtime_environment"],
                "generation_config": generation_config(args),
                "attention_backends": attention_report,
            },
        )
        print(
            "[MODEL_LOAD_OK] "
            f"model={args.model_id} gpu={args.physical_gpu} pid={os.getpid()} "
            f"rollouts={rollout_text} checkpoint={args.checkpoint} "
            f"processor={args.processor_path} load_seconds={load_seconds:.3f} "
            f"text_config={attention_report['text_config']} "
            f"vision_config={attention_report['vision_config']} "
            f"vision_first_layer={attention_report['vision_first_layer']} "
            f"vision_blocks={attention_report['vision_blocks']} "
            f"token_config={json.dumps(generation_config(args), sort_keys=True)} "
            f"processor_limit={json.dumps(processor_limit, sort_keys=True)} "
            f"memory_before={json.dumps(memory_before, sort_keys=True)} "
            f"memory_after={json.dumps(memory_after, sort_keys=True)}",
            flush=True,
        )
        wait_for_model_load_barrier(
            output_root,
            model_id=str(args.model_id),
            physical_gpu=int(args.physical_gpu),
            rollout_ids=rollout_ids,
        )
        for context in contexts:
            already_complete = len(context["completed_record_ids"]) == len(samples)
            status = "completed" if already_complete else "running"
            if already_complete and context["counters"]["runtime_error"]:
                status = "completed_with_runtime_errors"
            context["logical_status"] = (
                "completed" if already_complete else "running"
            )
            context["progress"].emit(
                context["counters"], status, force=True, memory=memory_after
            )

        # All eight workers use this same deterministic sample order.  This
        # single-route process finishes its current sample before advancing.
        load_stage = "sample_major_inference"
        for row in samples:
            record_id = str(row["record_id"])
            pending_contexts = [
                context
                for context in contexts
                if record_id not in context["completed_record_ids"]
            ]
            if not pending_contexts:
                continue
            image = None
            image_read_started = time.monotonic()
            image_failure_context: dict[str, Any] = {
                "stage": "image_read",
                "crop_id": None,
                "crop_index": None,
                "crop_xyxy": [0, 0, int(row["width"]), int(row["height"])],
                "tile_count": (
                    len(crop_index.get(record_id, []))
                    if args.model_id == "crop"
                    else 1
                ),
                "tile_size": {
                    "width": int(row["width"]),
                    "height": int(row["height"]),
                },
                "input_tokens": None,
                "memory_before_oom": cuda_memory(torch),
            }
            try:
                image_path = bundle / str(row["image_relpath"])
                with Image.open(image_path) as opened:
                    image = opened.convert("RGB")
                config = task_config(module, str(row["task"]))
            except Exception as image_exc:
                image_latency = time.monotonic() - image_read_started
                for context in pending_contexts:
                    assigned_args = context["args"]
                    context["counters"]["attempted"] += 1
                    context["counters"]["runtime_error"] += 1
                    record = error_record(
                        assigned_args,
                        code,
                        row,
                        image_exc,
                        image_latency,
                        failure_context=image_failure_context,
                    )
                    context["writer"].write(record)
                    context["completed_record_ids"].add(record_id)
                    if context["progress"].should_emit(
                        context["counters"]["attempted"],
                        args.progress_every,
                        args.progress_seconds,
                    ):
                        context["writer"].flush()
                        context["progress"].emit(
                            context["counters"],
                            "running",
                            force=True,
                            memory=cuda_memory(torch),
                        )
                if image is not None:
                    image.close()
                continue

            try:
                for context in pending_contexts:
                    assigned_args = context["args"]
                    inference_started = time.monotonic()
                    failure_context = dict(image_failure_context)
                    context["counters"]["attempted"] += 1
                    try:
                        def predict_once(
                            active_context: dict[str, Any],
                        ) -> dict[str, Any]:
                            inferencer.last_rollout_token_usage = None
                            inferencer.last_ui_diagnostics = {}
                            assert image is not None
                            try:
                                if args.model_id == "m31":
                                    return full_image_prediction(
                                        module=module,
                                        inferencer=inferencer,
                                        scorer=scorer,
                                        row=row,
                                        image=image,
                                        args=assigned_args,
                                        config=config,
                                        torch=torch,
                                        inference_context=active_context,
                                    )
                                assert tiling is not None
                                return crop_prediction(
                                    module=module,
                                    tiling=tiling,
                                    inferencer=inferencer,
                                    scorer=scorer,
                                    row=row,
                                    crop_rows=crop_index[record_id],
                                    image=image,
                                    args=assigned_args,
                                    config=config,
                                    torch=torch,
                                    inference_context=active_context,
                                )
                            finally:
                                failure_context.clear()
                                failure_context.update(active_context)

                        result = prediction_with_oom_retry(
                            predict_once, torch=torch, inferencer=inferencer
                        )
                        record = worker_record(assigned_args, code, row, result)
                        context["counters"]["inference_success"] += 1
                        context["counters"]["parse_error"] += int(
                            result["parse_status"] == "parse_error"
                            or result.get("contains_crop_parse_error", False)
                        )
                        context["counters"]["oom_exception_count"] += int(
                            result.get("oom_events", 0)
                        )
                        context["counters"]["oom_recovered_samples"] += int(
                            bool(result.get("oom_recovered"))
                        )
                    except Exception as exc:
                        oom_diagnostics = getattr(exc, "oom_diagnostics", {})
                        context["counters"]["runtime_error"] += 1
                        if isinstance(oom_diagnostics, Mapping):
                            context["counters"]["oom_exception_count"] += int(
                                oom_diagnostics.get("oom_events", 0)
                            )
                            context["counters"]["oom_final_failed_samples"] += int(
                                bool(oom_diagnostics.get("oom_final_failure"))
                            )
                        record = error_record(
                            assigned_args,
                            code,
                            row,
                            exc,
                            time.monotonic() - inference_started,
                            failure_context=failure_context,
                        )
                    finally:
                        inferencer.active_rollout_context = None
                    context["writer"].write(record)
                    context["completed_record_ids"].add(record_id)
                    if context["progress"].should_emit(
                        context["counters"]["attempted"],
                        args.progress_every,
                        args.progress_seconds,
                    ):
                        context["writer"].flush()
                        context["progress"].emit(
                            context["counters"],
                            "running",
                            force=True,
                            memory=cuda_memory(torch),
                        )
            finally:
                if image is not None:
                    image.close()

        for context in contexts:
            context["writer"].flush()
            context["logical_status"] = "completed"
            raw_status = (
                "completed"
                if context["counters"]["runtime_error"] == 0
                else "completed_with_runtime_errors"
            )
            context["progress"].emit(
                context["counters"], raw_status, force=True, memory=cuda_memory(torch)
            )
        return 0
    except BaseException as exc:
        load_seconds = time.monotonic() - load_started
        error_type = classify_runtime_error(exc)
        traceback_text = traceback.format_exc()
        chain = exception_chain(exc)
        try:
            import torch

            memory_after_failure = cuda_memory(torch)
        except Exception:
            memory_after_failure = {}
        event = "MODEL_LOAD_FAIL" if not model_loaded else "WORKER_FATAL"
        failure_path = (
            output_root
            / "diagnostics"
            / ("model_load" if not model_loaded else "worker_fatal")
            / f"{args.model_id}_rollouts_{rollout_text.replace(',', '_')}.json"
        )
        atomic_json(
            failure_path,
            {
                "schema_version": SCHEMA_VERSION,
                "status": event,
                "created_at": utc_now(),
                "model_id": args.model_id,
                "physical_gpu": args.physical_gpu,
                "pid": os.getpid(),
                "rollout_ids": rollout_ids,
                "checkpoint": str(args.checkpoint),
                "processor_path": str(args.processor_path),
                "failure_stage": load_stage,
                "error_type": error_type,
                "top_exception": chain[0] if chain else None,
                "root_exception": chain[-1] if chain else None,
                "exception_chain": chain,
                "traceback": traceback_text,
                "load_seconds": load_seconds,
                "memory_before": memory_before,
                "memory_after": memory_after_failure,
                "runtime_environment": worker_manifest["runtime_environment"],
            },
        )
        print(
            f"[{event}] "
            f"model={args.model_id} gpu={args.physical_gpu} pid={os.getpid()} "
            f"rollouts={rollout_text} checkpoint={args.checkpoint} "
            f"processor={args.processor_path} error={error_type} "
            f"exception_type={type(exc).__name__} message={str(exc)!r} "
            f"root_exception={json.dumps(chain[-1] if chain else None, sort_keys=True)} "
            f"failure_stage={load_stage} failure_json={failure_path} "
            f"load_seconds={load_seconds:.3f} "
            f"memory_before={json.dumps(memory_before, sort_keys=True)} "
            f"memory_after={json.dumps(memory_after_failure, sort_keys=True)}",
            flush=True,
        )
        for context in contexts:
            if model_loaded and context.get("logical_status") == "completed":
                logical_failure_status = "completed"
            else:
                logical_failure_status = (
                    "model_load_failed" if not model_loaded else "failed"
                )
                context["logical_status"] = "failed"
            context["progress"].emit(
                context["counters"],
                logical_failure_status,
                force=True,
                memory=memory_after_failure,
            )
        print(traceback_text, file=sys.stderr, flush=True)
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


def progress_file_state(path: Path) -> tuple[dict[str, Any] | None, int]:
    last = None
    count = 0
    if not path.is_file():
        return None, count
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last = json.loads(line)
                count += 1
    return last, count


def parse_physical_worker(value: str) -> dict[str, Any]:
    parts = value.split(",", 3)
    if len(parts) != 4:
        raise ValueError(
            "--physical-worker must be model,gpu,pid,single_rollout_id"
        )
    model_id, gpu, pid, rollout_text = parts
    rollout_ids = [int(item) for item in rollout_text.split("|") if item]
    physical_gpu = int(gpu)
    physical_pid = int(pid)
    expected_gpu = {"m31": 0, "crop": 1}
    if (
        model_id not in MODEL_IDS
        or physical_gpu != expected_gpu.get(model_id)
        or len(rollout_ids) != 1
        or rollout_ids[0] not in range(4)
        or physical_pid <= 0
    ):
        raise ValueError(f"invalid --physical-worker mapping: {value}")
    return {
        "model_id": model_id,
        "physical_gpu": physical_gpu,
        "pid": physical_pid,
        "rollout_ids": rollout_ids,
    }


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def progress_snapshot(args: argparse.Namespace) -> int:
    root = args.output_root.expanduser().resolve(strict=False)
    latest: list[dict[str, Any]] = []
    logical_rollouts: list[dict[str, Any]] = []
    history_rows_total = 0
    progress_files_discovered = 0
    for model_id in MODEL_IDS:
        for rollout_id in range(4):
            path = root / "progress" / model_id / f"rollout_{rollout_id}.jsonl"
            row, history_rows = progress_file_state(path)
            history_rows_total += history_rows
            progress_files_discovered += int(path.is_file())
            if row is not None:
                latest.append(row)
            raw_status = str((row or {}).get("status", "missing"))
            if raw_status in {"completed", "completed_with_runtime_errors"}:
                logical_status = "completed"
                phase = "completed"
            elif raw_status in {"failed", "model_load_failed"}:
                logical_status = "failed"
                phase = raw_status
            else:
                logical_status = "running"
                phase = "pending" if raw_status in {"pending", "missing"} else raw_status
            logical_rollouts.append(
                {
                    "model_id": model_id,
                    "rollout_id": rollout_id,
                    "status": logical_status,
                    "phase": phase,
                    "progress_path": str(path),
                    "progress_history_rows": history_rows,
                    "latest": row,
                }
            )
    physical_processes = []
    physical_keys: set[tuple[str, int, tuple[int, ...]]] = set()
    physical_pids: set[int] = set()
    for value in args.physical_worker:
        physical = parse_physical_worker(value)
        physical_key = (
            str(physical["model_id"]),
            int(physical["physical_gpu"]),
            tuple(int(item) for item in physical["rollout_ids"]),
        )
        if physical_key in physical_keys:
            raise ValueError(f"duplicate --physical-worker ownership: {value}")
        physical_pid = int(physical["pid"])
        if physical_pid in physical_pids:
            raise ValueError(
                f"duplicate --physical-worker PID {physical_pid}: all eight "
                "formal workers must be distinct physical processes"
            )
        physical_keys.add(physical_key)
        physical_pids.add(physical_pid)
        owned = [
            row
            for row in logical_rollouts
            if row["model_id"] == physical["model_id"]
            and row["rollout_id"] in physical["rollout_ids"]
        ]
        alive = pid_alive(int(physical["pid"]))
        all_completed = all(row["status"] == "completed" for row in owned)
        has_failed = any(row["status"] == "failed" for row in owned)
        if alive:
            physical_status = "alive"
        elif all_completed:
            physical_status = "completed"
        else:
            physical_status = "failed"
            for logical in owned:
                if logical["status"] != "completed":
                    logical["status"] = "failed"
                    logical["phase"] = "physical_process_failed"
            has_failed = True
        running_rates = [
            float(row["latest"].get("throughput_attempted_per_second", 0.0))
            for row in owned
            if isinstance(row.get("latest"), Mapping)
            and row["phase"] == "running"
        ]
        completed_rates = [
            (
                int(row["rollout_id"]),
                float(row["latest"].get("throughput_attempted_per_second", 0.0)),
            )
            for row in owned
            if isinstance(row.get("latest"), Mapping)
            and row["status"] == "completed"
        ]
        attempted_rate = (
            sum(running_rates)
            if running_rates
            else max(completed_rates, default=(-1, 0.0))[1]
        )
        remaining_attempts = sum(
            max(
                0,
                int(row["latest"].get("total", 0))
                - int(row["latest"].get("attempted", 0)),
            )
            for row in owned
            if isinstance(row.get("latest"), Mapping)
        )
        physical_remaining = (
            remaining_attempts / attempted_rate
            if attempted_rate > 0 and not has_failed and physical_status != "failed"
            else (0.0 if all_completed else None)
        )
        physical_processes.append(
            {
                **physical,
                "status": physical_status,
                "alive": alive,
                "logical_rollout_statuses": {
                    str(row["rollout_id"]): row["status"] for row in owned
                },
                "has_failed_logical_rollout": has_failed,
                "attempted_throughput_per_second": attempted_rate,
                "remaining_attempts": remaining_attempts,
                "remaining_seconds": physical_remaining,
            }
        )
    expected_physical_keys = {
        (model_id, physical_gpu, (rollout_id,))
        for model_id, physical_gpu in (("m31", 0), ("crop", 1))
        for rollout_id in range(4)
    }
    if args.physical_worker and physical_keys != expected_physical_keys:
        raise ValueError(
            "formal progress snapshot requires eight physical workers with "
            f"exact ownership {sorted(expected_physical_keys)}; got "
            f"{sorted(physical_keys)}"
        )
    if args.physical_worker and args.expected_workers != 8:
        raise ValueError("formal progress snapshot requires --expected-workers 8")
    failed_physical = [
        row for row in physical_processes if row["status"] == "failed"
    ]
    failed_logical = [row for row in logical_rollouts if row["status"] == "failed"]
    eta_blocked = bool(failed_physical or failed_logical)
    eta_incomplete = len(physical_processes) != 8 or any(
        not isinstance(row.get("remaining_seconds"), (int, float))
        for row in physical_processes
    )
    physical_remaining_values = [
        float(row["remaining_seconds"])
        for row in physical_processes
        if isinstance(row.get("remaining_seconds"), (int, float))
    ]
    total_remaining = (
        None
        if eta_blocked or eta_incomplete
        else max(physical_remaining_values, default=None)
    )
    snapshot = {
        "timestamp": utc_now(),
        "workers_seen": len(latest),
        "workers_expected": args.expected_workers,
        "workers": latest,
        "physical_processes_expected": 8,
        "physical_processes_seen": len(physical_processes),
        "physical_pids_unique": len(physical_pids) == len(physical_processes),
        "physical_processes": physical_processes,
        "logical_rollouts_expected": args.expected_workers,
        "logical_rollouts": logical_rollouts,
        "progress_files_discovered": progress_files_discovered,
        "progress_history_rows_total": history_rows_total,
        "failed_physical_processes": len(failed_physical),
        "failed_logical_rollouts": len(failed_logical),
        "eta_valid": (
            not eta_blocked and not eta_incomplete and total_remaining is not None
        ),
        "eta_unavailable_reason": (
            "failed physical worker or logical rollout"
            if eta_blocked
            else (
                "not all eight physical workers have an attempted-throughput ETA"
                if eta_incomplete
                else None
            )
        ),
        "eta_basis": (
            "maximum remaining time across eight parallel single-rollout "
            "physical processes using attempted throughput"
        ),
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
