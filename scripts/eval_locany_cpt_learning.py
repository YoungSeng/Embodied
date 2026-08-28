#!/usr/bin/env python3
"""Task-specific Base-vs-checkpoint evaluation for LocateAnything UI CPT.

Held-out evaluation is the default contract. Training-pool/domain-absorption
checks require ``--eval-split train_pool`` and cannot select a best checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import re
import shutil
import sys
import tempfile
import time
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eaglevl.train.cpt_eval_metrics import (  # noqa: E402
    aggregate_scores,
    parse_action_type,
    parse_boxes,
    parse_labeled_boxes,
    parse_points,
    parse_vqa_label,
    micro_primary,
    score_task,
    task_macro_primary,
)
from eaglevl.train.cpt_checkpoint_selection import select_checkpoint  # noqa: E402
from eaglevl.train.cpt_eval_queue import (  # noqa: E402
    exclusive_file_lock,
    fsync_if_supported,
)
from eaglevl.train.cpt_observability import CPT_TASKS  # noqa: E402
from scripts.inference_ui_defect_locany import LocateAnythingInferencer  # noqa: E402


IMAGE_TOKEN_RE = re.compile(r"<image(?:-\d+)?>")
EVALUATOR_PROTOCOL_VERSION = 2


@dataclass(frozen=True)
class Example:
    key: str
    task: str
    record_id: str
    group_id: str
    split: str
    image: str
    prompt: str
    target: str
    source: str
    line: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--processor-path", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-task", type=int, default=100)
    parser.add_argument("--qualitative-per-task", type=int, default=50)
    parser.add_argument(
        "--manual-review-jsonl",
        type=Path,
        default=None,
        help="optional completed referring_kg manual-review JSONL",
    )
    parser.add_argument("--tasks", default="all")
    parser.add_argument(
        "--subset-strategy", choices=("hash", "random", "first"), default="hash"
    )
    parser.add_argument(
        "--eval-split", choices=("heldout", "train_pool"), default="heldout"
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--base-cache-dir", type=Path, default=None)
    parser.add_argument(
        "--metrics-jsonl",
        type=Path,
        default=None,
        help="optional run-level diagnostics/cpt_eval_metrics.jsonl to append",
    )
    parser.add_argument(
        "--train-metrics-jsonl",
        type=Path,
        default=None,
        help="optional cpt_train_metrics.jsonl for train-val CE gaps",
    )
    parser.add_argument(
        "--teacher-forced",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also measure answer-token teacher-forced CE",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument(
        "--attn-implementation",
        choices=("sdpa", "flash_attention_2", "eager", "magi"),
        default="sdpa",
    )
    parser.add_argument(
        "--vision-attn-implementation",
        choices=("sdpa", "flash_attention_2", "eager"),
        default="flash_attention_2",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--fail-fast-inference-errors",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="stop on the first processor/model/OOM error instead of caching partial results",
    )
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    if args.samples_per_task <= 0:
        parser.error("--samples-per-task must be positive")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.qualitative_per_task < 0:
        parser.error("--qualitative-per-task cannot be negative")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(*values: object) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def ensure_checkpoint_remote_code(checkpoint: str, base_model: str) -> None:
    checkpoint_dir = Path(checkpoint).expanduser().resolve()
    base_dir = Path(base_model).expanduser().resolve()
    if not checkpoint_dir.is_dir() or not (checkpoint_dir / "config.json").is_file():
        raise RuntimeError(f"invalid checkpoint directory: {checkpoint_dir}")
    if not base_dir.is_dir():
        raise RuntimeError(f"base model directory does not exist: {base_dir}")
    for source in sorted(base_dir.glob("*.py")):
        destination = checkpoint_dir / source.name
        if not destination.is_file() or destination.stat().st_size == 0:
            shutil.copy2(source, destination)
    required = ("configuration_locateanything.py", "modeling_locateanything.py")
    missing = [name for name in required if not (checkpoint_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"checkpoint remote code is incomplete: missing={missing}")


def _resolve_recipe_path(value: str, recipe: Path, relative: bool) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return ((recipe.parent if relative else Path.cwd()) / path).resolve()


def _first_text_turn(record: dict[str, Any], role: str) -> str | None:
    aliases = {"human", "user"} if role == "human" else {"gpt", "assistant"}
    for turn in record.get("conversations", []):
        if str(turn.get("from", turn.get("role", ""))).lower() not in aliases:
            continue
        value = turn.get("value", turn.get("content"))
        if isinstance(value, str):
            return value
    return None


def _resolve_image(record: dict[str, Any], root: Path | None) -> Path | None:
    value = record.get("image") or record.get("images")
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            return None
        value = value[0]
    if not isinstance(value, str) or not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute() and root is not None:
        path = root / path
    return path.resolve()


def _iter_jsonl(paths: Iterable[Path]) -> Iterable[tuple[Path, int, dict[str, Any]]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: row is not an object")
                yield path, line_number, value


def _select_examples(
    candidates: list[Example], count: int, strategy: str, seed: int, task: str
) -> list[Example]:
    if strategy == "first":
        return candidates[:count]
    if strategy == "hash":
        return sorted(
            candidates,
            key=lambda value: stable_hash(seed, value.group_id, value.record_id),
        )[:count]
    rng = random.Random(stable_hash(seed, task))
    candidates = list(candidates)
    rng.shuffle(candidates)
    return candidates[:count]


def load_examples(
    recipe_path: Path,
    samples_per_task: int,
    selected: set[str] | None,
    *,
    strategy: str,
    seed: int,
    eval_split: str,
) -> list[Example]:
    recipe_path = recipe_path.expanduser().resolve()
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"), object_pairs_hook=OrderedDict)
    expected = "heldout" if eval_split == "heldout" else "train"
    by_task: OrderedDict[str, list[Example]] = OrderedDict()
    for recipe_name, meta in recipe.items():
        recipe_task = str(meta.get("cpt_task") or recipe_name.removeprefix("locany_cpt_"))
        if selected is not None and recipe_task not in selected:
            continue
        declared = str(meta.get("cpt_split", ""))
        if declared and declared != expected:
            raise RuntimeError(
                f"recipe {recipe_name} declares cpt_split={declared!r}, expected {expected!r}"
            )
        relative = bool(meta.get("paths_relative_to_meta", False))
        annotations = meta.get("annotation", [])
        if isinstance(annotations, str):
            annotations = [annotations]
        paths = [_resolve_recipe_path(str(value), recipe_path, relative) for value in annotations]
        root_value = str(meta.get("root", "")).strip()
        root = _resolve_recipe_path(root_value, recipe_path, relative) if root_value else None
        for source, line_number, record in _iter_jsonl(paths):
            task = str(record.get("cpt_task") or recipe_task)
            if selected is not None and task not in selected:
                continue
            actual = str(record.get("cpt_split", ""))
            if actual != expected:
                raise RuntimeError(
                    f"{source}:{line_number}: expected cpt_split={expected!r}, got {actual!r}"
                )
            prompt, target = _first_text_turn(record, "human"), _first_text_turn(record, "gpt")
            image = _resolve_image(record, root)
            if prompt is None or target is None or image is None or not image.is_file():
                raise RuntimeError(f"{source}:{line_number}: unusable prompt/target/image")
            record_id = str(
                record.get("cpt_record_id") or record.get("id") or f"{source.name}:{line_number}"
            )
            group_id = str(record.get("cpt_group_id") or "")
            if not group_id:
                raise RuntimeError(f"{source}:{line_number}: missing cpt_group_id")
            by_task.setdefault(task, []).append(
                Example(
                    key=f"{task}:{record_id}",
                    task=task,
                    record_id=record_id,
                    group_id=group_id,
                    split=eval_split,
                    image=str(image),
                    prompt=IMAGE_TOKEN_RE.sub("", prompt).strip(),
                    target=target,
                    source=str(source),
                    line=line_number,
                )
            )
    examples = []
    for task, candidates in by_task.items():
        examples.extend(_select_examples(candidates, samples_per_task, strategy, seed, task))
    if not examples:
        raise RuntimeError("recipe contains no usable examples")
    return examples


def validate_examples_against_manifest(
    examples: list[Example], manifest_path: Path
) -> None:
    """Stream the full manifest while retaining only selected-record matches."""
    expected_by_id: dict[str, Example] = {}
    for example in examples:
        if example.record_id in expected_by_id:
            raise RuntimeError(
                f"selected recipe contains duplicate record_id={example.record_id!r}"
            )
        expected_by_id[example.record_id] = example
    found: set[str] = set()
    with manifest_path.expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{manifest_path}:{line_number}: row is not an object")
            record_id = str(row.get("record_id") or "")
            example = expected_by_id.get(record_id)
            if example is None:
                continue
            if record_id in found:
                raise RuntimeError(
                    f"{manifest_path}:{line_number}: duplicate selected record_id={record_id!r}"
                )
            expected_split = "heldout" if example.split == "heldout" else "train"
            mismatches = {}
            for key, expected in (
                ("split", expected_split),
                ("task", example.task),
                ("group_id", example.group_id),
            ):
                actual = str(row.get(key) or "")
                if actual != expected:
                    mismatches[key] = {"manifest": actual, "recipe": expected}
            if mismatches:
                raise RuntimeError(
                    f"{manifest_path}:{line_number}: selected record {record_id!r} "
                    f"does not match manifest: {mismatches}"
                )
            found.add(record_id)
    missing = sorted(set(expected_by_id).difference(found))
    if missing:
        raise RuntimeError(
            f"manifest does not contain {len(missing)} selected records; first={missing[:5]}"
        )


def inference_namespace(args: argparse.Namespace, model_path: str) -> SimpleNamespace:
    return SimpleNamespace(
        checkpoint=model_path,
        processor_path=args.processor_path or args.base_model,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        vision_attn_implementation=args.vision_attn_implementation,
        generation_mode="slow",
        max_new_tokens=args.max_new_tokens,
        n_future_tokens=6,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        repetition_penalty=1.0,
        greedy=True,
        verbose_generation=False,
        relation_gate_mode="observe",
        relation_gate_threshold=None,
        trust_remote_code=True,
        local_files_only=not args.allow_download,
        use_fast_processor=True,
        enable_ui_relation=False,
    )


def _full_conversation_text(processor: Any, messages: list[dict[str, Any]]) -> str:
    for owner in (processor, getattr(processor, "tokenizer", None)):
        if owner is None:
            continue
        method = getattr(owner, "py_apply_chat_template", None) or getattr(
            owner, "apply_chat_template", None
        )
        if method is not None:
            return method(messages, tokenize=False, add_generation_prompt=False)
    raise AttributeError("processor does not expose a chat-template method")


@torch.inference_mode()
def teacher_forced_main_ce(
    inferencer: LocateAnythingInferencer,
    image: Image.Image,
    example: Example,
) -> dict[str, float | int]:
    """Return exact answer-token CE sum/count without a second model load."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": example.prompt},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": example.target}]},
    ]
    text = _full_conversation_text(inferencer.processor, messages)
    image_inputs, video_inputs = inferencer.processor.process_vision_info(messages)
    inputs = inferencer.processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=False,
        truncation=False,
    )
    input_ids = inputs["input_ids"]
    labels = torch.full_like(input_ids, -100)
    tokenizer = inferencer.processor.tokenizer
    start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    assistant_id = tokenizer.convert_tokens_to_ids("assistant")
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    starts = torch.where(input_ids[0].eq(start_id))[0]
    assistants = torch.where(input_ids[0].eq(assistant_id))[0]
    ends = torch.where(input_ids[0].eq(end_id))[0]
    valid_assistants = {int(value.item()) + 1 for value in starts}
    for assistant in assistants:
        position = int(assistant.item())
        if position not in valid_assistants:
            continue
        answer_start = position + 2
        answer_end = next((int(value.item()) for value in ends if int(value.item()) >= answer_start), None)
        if answer_end is not None:
            labels[0, answer_start : answer_end + 1] = input_ids[0, answer_start : answer_end + 1]
    token_count = int(labels[..., 1:].ne(-100).sum().item())
    if token_count <= 0:
        raise ValueError(
            f"no answer labels after chat-template masking: task={example.task}, record={example.record_id}"
        )
    model_inputs: dict[str, Any] = {
        "input_ids": input_ids.to(inferencer.device),
        "labels": labels.to(inferencer.device),
        "return_dict": True,
    }
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        model_inputs["attention_mask"] = attention_mask.to(inferencer.device)
    pixel_values = inputs.get("pixel_values")
    if pixel_values is None:
        raise KeyError("processor output has no pixel_values")
    model_inputs["pixel_values"] = pixel_values.to(
        device=inferencer.device, dtype=inferencer.dtype
    )
    grid = inputs.get("image_grid_hws")
    if grid is not None:
        if not torch.is_tensor(grid):
            grid = torch.as_tensor(grid, dtype=torch.long)
        grid = grid.to(device=inferencer.device, dtype=torch.long)
        model_inputs["image_grid_hws"] = grid
        model_inputs["image_flags"] = torch.tensor(
            [len(grid)], dtype=torch.long, device=inferencer.device
        )
    outputs = inferencer.model(**model_inputs)
    loss = getattr(outputs, "lm_loss", None)
    if loss is None:
        loss = getattr(outputs, "loss", None)
    if loss is None:
        raise RuntimeError("model returned no teacher-forced LM loss")
    ce = float(loss.detach().float().item())
    return {
        "teacher_forced_main_loss_sum": ce * token_count,
        "teacher_forced_main_tokens": token_count,
        "teacher_forced_main_token_ce": ce,
    }


def score_result(example: Example, prediction: str, error: str | None) -> dict[str, Any]:
    metrics = score_task(example.task, prediction, example.target)
    if error:
        metrics["evaluation_error"] = 1.0
        metrics["primary_metric"] = 0.0
        primary_name = metrics.get("primary_name")
        if isinstance(primary_name, str):
            metrics[primary_name] = 0.0
    return metrics


def run_model(
    label: str, model_path: str, examples: list[Example], args: argparse.Namespace
) -> dict[str, dict[str, Any]]:
    print(f"\n===== {label}: {model_path} =====", flush=True)
    started = time.time()
    inferencer = LocateAnythingInferencer(inference_namespace(args, model_path))
    # Checkpoints may persist the training-only diagnostics flag. Evaluation
    # needs only the scalar fused CE and must not allocate the token-loss buffer.
    inferencer.model._cpt_observability_enabled = False
    results: dict[str, dict[str, Any]] = {}
    try:
        for index, example in enumerate(examples):
            torch.manual_seed(args.seed + index)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed + index)
            try:
                with Image.open(example.image) as opened:
                    image = opened.convert("RGB")
                with torch.inference_mode():
                    prediction = inferencer.predict(image=image, question=example.prompt)
                    teacher_forced = (
                        teacher_forced_main_ce(inferencer, image, example)
                        if args.teacher_forced
                        else {}
                    )
                results[example.key] = {
                    "prediction": prediction,
                    "error": None,
                    "metrics": score_result(example, prediction, None),
                    **teacher_forced,
                }
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                results[example.key] = {
                    "prediction": "",
                    "error": error,
                    "metrics": score_result(example, "", error),
                    "teacher_forced_main_loss_sum": None,
                    "teacher_forced_main_tokens": 0,
                    "teacher_forced_main_token_ce": None,
                }
                print(f"[ERROR] {example.key}: {exc}", file=sys.stderr, flush=True)
                # CUDA OOM exceptions retain traceback frames containing large
                # tensors. Drop the traceback before emptying the allocator so
                # a single bad sample cannot poison every following example.
                try:
                    exc.__traceback__ = None
                except Exception:
                    pass
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if getattr(args, "fail_fast_inference_errors", False):
                    raise RuntimeError(
                        "evaluation stopped on the first inference error: "
                        f"example={example.key}; {error}"
                    ) from None
    finally:
        del inferencer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"{label} wall_time_seconds={time.time() - started:.3f}")
    return results


def summarize(
    examples: list[Example], results: dict[str, dict[str, Any]], *, split: str
) -> dict[str, Any]:
    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    errors = Counter()
    for example in examples:
        result = results[example.key]
        if result["error"]:
            errors[example.task] += 1
        grouped.setdefault(example.task, []).append(result["metrics"])
    per_task = {task: aggregate_scores(task, scores) for task, scores in grouped.items()}
    total_loss_sum = 0.0
    total_loss_tokens = 0
    for task in per_task:
        task_results = [
            results[example.key]
            for example in examples
            if example.task == task and results[example.key].get("error") is None
        ]
        loss_sum = sum(
            float(result.get("teacher_forced_main_loss_sum") or 0.0)
            for result in task_results
        )
        token_count = sum(
            int(result.get("teacher_forced_main_tokens") or 0)
            for result in task_results
        )
        per_task[task]["eval_main_loss_sum"] = loss_sum if token_count else None
        per_task[task]["eval_main_loss_tokens"] = token_count
        per_task[task]["eval_main_token_ce"] = loss_sum / token_count if token_count else None
        total_loss_sum += loss_sum
        total_loss_tokens += token_count
    macro = task_macro_primary(per_task)
    task_ces = [
        float(value["eval_main_token_ce"])
        for value in per_task.values()
        if value.get("eval_main_token_ce") is not None
    ]
    return {
        "split": split,
        "examples": len(examples),
        "successful": len(examples) - sum(errors.values()),
        "errors": dict(sorted(errors.items())),
        "micro_primary": micro_primary(per_task),
        "eval_main_token_ce": total_loss_sum / total_loss_tokens if total_loss_tokens else None,
        "eval_main_loss_tokens": total_loss_tokens,
        "task_macro_eval_main_token_ce": sum(task_ces) / len(task_ces) if task_ces else None,
        "heldout_task_macro_primary": macro if split == "heldout" else None,
        "train_pool_task_macro_primary": macro if split == "train_pool" else None,
        "per_task": per_task,
    }


def parsed_value(task: str, text: str) -> Any:
    if task == "vqa":
        return parse_vqa_label(text)
    if task == "agent_action":
        return {"action_type": parse_action_type(text), "points": parse_points(text)}
    if task == "agent_grounding":
        return {"points": parse_points(text), "boxes": parse_boxes(text)}
    if task in {"ui_defect", "all_ui_elements", "single_grounding", "ocr"}:
        return parse_labeled_boxes(text)
    return text


def cache_key(args: argparse.Namespace, examples: list[Example], manifest_id: str) -> str:
    return stable_hash(
        EVALUATOR_PROTOCOL_VERSION,
        Path(args.base_model).expanduser().resolve(),
        Path(args.processor_path or args.base_model).expanduser().resolve(),
        manifest_id,
        args.eval_split,
        args.subset_strategy,
        args.seed,
        args.max_new_tokens,
        args.teacher_forced,
        args.dtype,
        args.attn_implementation,
        args.vision_attn_implementation,
        *(example.key for example in examples),
    )


def load_or_run_base(
    args: argparse.Namespace, examples: list[Example], manifest_id: str
) -> tuple[dict[str, dict[str, Any]], Path, bool]:
    cache_dir = args.base_cache_dir or (args.output_dir.parent / "base_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"base-{cache_key(args, examples, manifest_id)}.json"
    with exclusive_file_lock(path.with_suffix(path.suffix + ".lock")):
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (
                payload.get("evaluator_protocol_version")
                == EVALUATOR_PROTOCOL_VERSION
                and payload.get("example_keys")
                == [example.key for example in examples]
            ):
                results = payload["results"]
                for example in examples:
                    result = results[example.key]
                    result["metrics"] = score_result(
                        example,
                        str(result.get("prediction", "")),
                        result.get("error"),
                    )
                return results, path, True
        results = run_model("base", args.base_model, examples, args)
        temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
        temporary.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "evaluator_protocol_version": EVALUATOR_PROTOCOL_VERSION,
                    "base_model": str(Path(args.base_model).expanduser().resolve()),
                    "manifest_id": manifest_id,
                    "example_keys": [example.key for example in examples],
                    "results": results,
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        return results, path, False


def metric_delta(base: dict[str, Any], checkpoint: dict[str, Any]) -> dict[str, Any]:
    per_task = {}
    for task in sorted(set(base.get("per_task", {})) | set(checkpoint.get("per_task", {}))):
        left = base.get("per_task", {}).get(task, {}).get("primary_metric")
        right = checkpoint.get("per_task", {}).get(task, {}).get("primary_metric")
        per_task[task] = None if left is None or right is None else float(right) - float(left)
    macro_key = (
        "heldout_task_macro_primary"
        if checkpoint["split"] == "heldout"
        else "train_pool_task_macro_primary"
    )
    left, right = base.get(macro_key), checkpoint.get(macro_key)
    return {
        "per_task_primary": per_task,
        "task_macro_primary": None if left is None or right is None else float(right) - float(left),
    }


def apply_manual_referring_review(
    summary: dict[str, Any], review_path: Path | None
) -> None:
    task = summary.get("per_task", {}).get("referring_kg")
    if task is None:
        return
    task["manual_semantic_accuracy"] = None
    task["manual_semantic_review_count"] = 0
    task["manual_semantic_review_status"] = "pending"
    if review_path is None:
        return
    values = []
    with review_path.expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            value = row.get("semantic_correct")
            if value is None:
                continue
            if not isinstance(value, bool):
                raise ValueError(
                    f"{review_path}:{line_number}: semantic_correct must be true/false/null"
                )
            values.append(value)
    task["manual_semantic_review_count"] = len(values)
    task["manual_semantic_accuracy"] = (
        sum(values) / len(values) if values else None
    )
    task["manual_semantic_review_status"] = "complete" if len(values) >= 50 else "partial"


def write_outputs(
    args: argparse.Namespace,
    examples: list[Example],
    base_results: dict[str, dict[str, Any]],
    checkpoint_results: dict[str, dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as output, (
        args.output_dir / "errors_by_task.jsonl"
    ).open("w", encoding="utf-8") as error_output:
        for example in examples:
            for model_label, result in (
                ("base", base_results[example.key]),
                ("checkpoint", checkpoint_results[example.key]),
            ):
                row = {
                    **asdict(example),
                    "model": model_label,
                    "checkpoint": args.base_model if model_label == "base" else args.checkpoint,
                    "prediction": result["prediction"],
                    "parsed_target": parsed_value(example.task, example.target),
                    "parsed_prediction": parsed_value(example.task, result["prediction"]),
                    "metrics": result["metrics"],
                    "error": result["error"],
                    "teacher_forced_main_loss_sum": result.get("teacher_forced_main_loss_sum"),
                    "teacher_forced_main_tokens": result.get("teacher_forced_main_tokens"),
                    "teacher_forced_main_token_ce": result.get("teacher_forced_main_token_ce"),
                }
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                primary = result["metrics"].get("primary_metric") if result["metrics"] else None
                if result["error"] or (isinstance(primary, (int, float)) and primary < 0.5):
                    error_output.write(json.dumps(row, ensure_ascii=False) + "\n")

    markdown = [
        "# CPT qualitative samples",
        "",
        f"- split: `{args.eval_split}`",
        f"- subset strategy: `{args.subset_strategy}`",
        "",
    ]
    seen: Counter[str] = Counter()
    for example in examples:
        if seen[example.task] >= args.qualitative_per_task:
            continue
        seen[example.task] += 1
        markdown.extend(
            [
                f"## {example.task} — {example.record_id}",
                "",
                f"- image: `{example.image}`",
                f"- target: {example.target}",
                f"- base: {base_results[example.key]['prediction']}",
                f"- checkpoint: {checkpoint_results[example.key]['prediction']}",
                "",
            ]
        )
    (args.output_dir / "qualitative_samples.md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )
    manual_template = args.output_dir / "manual_review_referring_kg.jsonl"
    if not manual_template.exists():
        with manual_template.open("w", encoding="utf-8") as handle:
            reviewed = 0
            for example in examples:
                if example.task != "referring_kg" or reviewed >= 50:
                    continue
                handle.write(
                    json.dumps(
                        {
                            "split": args.eval_split,
                            "record_id": example.record_id,
                            "group_id": example.group_id,
                            "image": example.image,
                            "target": example.target,
                            "prediction": checkpoint_results[example.key]["prediction"],
                            "semantic_correct": None,
                            "error_category": None,
                            "notes": None,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                reviewed += 1
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def checkpoint_step(path: str) -> int | None:
    matches = re.findall(r"checkpoint-(\d+)", str(path))
    return int(matches[-1]) if matches else None


def _read_jsonl_rows(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            rows.append(value)
    return rows


def _train_ce_before(
    rows: list[dict[str, Any]], task: str, step: int | None
) -> float | None:
    candidates = [
        row
        for row in rows
        if row.get("task") == task
        and isinstance(row.get("train_main_token_ce"), (int, float))
        and (step is None or int(row.get("step") or -1) <= step)
    ]
    if not candidates:
        return None
    return float(max(candidates, key=lambda row: int(row.get("step") or -1))["train_main_token_ce"])


def build_eval_metric_rows(
    args: argparse.Namespace,
    summary: dict[str, Any],
    history: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    step = checkpoint_step(args.checkpoint)
    train_rows = _read_jsonl_rows(args.train_metrics_jsonl)
    base_per_task = summary["base"].get("per_task", {})
    checkpoint_per_task = summary["checkpoint_metrics"].get("per_task", {})
    evaluation_protocol_id = stable_hash(
        summary["manifest_id"],
        args.eval_split,
        args.subset_strategy,
        args.seed,
        args.samples_per_task,
        args.max_new_tokens,
        args.teacher_forced,
        args.dtype,
        args.attn_implementation,
        args.vision_attn_implementation,
        Path(args.processor_path or args.base_model).expanduser().resolve(),
        *sorted(checkpoint_per_task),
    )
    rows = []
    for task, metrics in checkpoint_per_task.items():
        train_ce = _train_ce_before(train_rows, task, step)
        eval_ce = metrics.get("eval_main_token_ce")
        base_primary = base_per_task.get(task, {}).get("primary_metric")
        primary = metrics.get("primary_metric")
        row = {
            "schema_version": 1,
            "evaluation_id": stable_hash(
                evaluation_protocol_id, args.checkpoint, task
            ),
            "evaluation_protocol_id": evaluation_protocol_id,
            "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
            "manifest_id": summary["manifest_id"],
            "subset_strategy": args.subset_strategy,
            "samples_per_task": args.samples_per_task,
            "step": step,
            "split": args.eval_split,
            "task": task,
            "primary_name": metrics.get("primary_name"),
            "primary_metric": primary,
            "base_primary": base_primary,
            "delta_vs_base": (
                float(primary) - float(base_primary)
                if isinstance(primary, (int, float)) and isinstance(base_primary, (int, float))
                else None
            ),
            "train_main_token_ce": train_ce,
            "train_token_ce": train_ce,
            "eval_token_ce": eval_ce,
            "ce_kind": "main",
            "train_val_main_ce_gap": (
                float(eval_ce) - float(train_ce)
                if isinstance(eval_ce, (int, float)) and isinstance(train_ce, (int, float))
                else None
            ),
            "train_val_ce_gap": (
                float(eval_ce) - float(train_ce)
                if isinstance(eval_ce, (int, float)) and isinstance(train_ce, (int, float))
                else None
            ),
            "eval_loss_tokens": metrics.get("eval_main_loss_tokens"),
            "metrics": metrics,
        }
        rows.append(row)

    macro_primary_key = (
        "heldout_task_macro_primary"
        if args.eval_split == "heldout"
        else "train_pool_task_macro_primary"
    )
    macro = {
        "schema_version": 1,
        "evaluation_id": stable_hash(
            evaluation_protocol_id, args.checkpoint, "__task_macro__"
        ),
        "evaluation_protocol_id": evaluation_protocol_id,
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "manifest_id": summary["manifest_id"],
        "subset_strategy": args.subset_strategy,
        "samples_per_task": args.samples_per_task,
        "step": step,
        "split": args.eval_split,
        "task": "__task_macro__",
        "primary_name": macro_primary_key,
        "primary_metric": summary["checkpoint_metrics"].get(macro_primary_key),
        "base_primary": summary["base"].get(macro_primary_key),
        "delta_vs_base": summary["checkpoint_minus_base"].get("task_macro_primary"),
        "eval_token_ce": summary["checkpoint_metrics"].get("task_macro_eval_main_token_ce"),
        "eval_micro_token_ce": summary["checkpoint_metrics"].get("eval_main_token_ce"),
        "eval_loss_tokens": summary["checkpoint_metrics"].get("eval_main_loss_tokens"),
    }
    complete_heldout = (
        args.eval_split == "heldout"
        and set(checkpoint_per_task) == set(CPT_TASKS)
    )
    macro["complete_ten_task_heldout"] = complete_heldout
    current_checkpoint = macro["checkpoint"]
    prior_history = [
        row for row in history if row.get("checkpoint") != current_checkpoint
    ]
    selection = select_checkpoint(macro, checkpoint_per_task, prior_history)
    selection["complete_ten_task_heldout"] = complete_heldout
    if not complete_heldout:
        selection["is_best_overall"] = False
    macro.update(selection)
    rows.append(macro)
    return rows, selection


def write_eval_metric_rows(
    output_dir: Path,
    rows: list[dict[str, Any]],
    append_path: Path | None,
) -> None:
    local = output_dir / "cpt_eval_metrics.jsonl"
    with local.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    if append_path is not None:
        append_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = append_path.with_suffix(append_path.suffix + ".lock")
        with exclusive_file_lock(lock_path):
            temporary: Path | None = None
            try:
                existing = _read_jsonl_rows(append_path)
                replacement_ids = {row["evaluation_id"] for row in rows}
                retained = [
                    row
                    for row in existing
                    if row.get("evaluation_id") not in replacement_ids
                ]
                with tempfile.NamedTemporaryFile(
                    "w",
                    encoding="utf-8",
                    dir=append_path.parent,
                    prefix=append_path.name + ".",
                    delete=False,
                ) as handle:
                    temporary = Path(handle.name)
                    for row in [*retained, *rows]:
                        handle.write(
                            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                        )
                    handle.flush()
                    fsync_if_supported(handle, path=temporary)
                os.replace(temporary, append_path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)


def main() -> int:
    evaluation_started = time.time()
    args = parse_args()
    ensure_checkpoint_remote_code(args.checkpoint, args.base_model)
    selected = None if args.tasks.strip().lower() == "all" else {
        value.strip() for value in args.tasks.split(",") if value.strip()
    }
    manifest = args.manifest
    if manifest is None:
        candidate = (
            args.recipe.expanduser().resolve().parent.parent
            / "diagnostics"
            / "split_manifest.jsonl"
        )
        manifest = candidate if candidate.is_file() else None
    manifest_id = (
        sha256_file(manifest.expanduser().resolve())
        if manifest
        else sha256_file(args.recipe.expanduser().resolve())
    )
    examples = load_examples(
        args.recipe,
        args.samples_per_task,
        selected,
        strategy=args.subset_strategy,
        seed=args.seed,
        eval_split=args.eval_split,
    )
    if manifest is not None:
        validate_examples_against_manifest(examples, manifest)
    counts = Counter(example.task for example in examples)
    print(f"Selected {len(examples)} examples by {args.subset_strategy}: {dict(counts)}")
    base_results, base_cache, base_cache_hit = load_or_run_base(args, examples, manifest_id)
    checkpoint_results = run_model("checkpoint", args.checkpoint, examples, args)
    base_summary = summarize(examples, base_results, split=args.eval_split)
    checkpoint_summary = summarize(examples, checkpoint_results, split=args.eval_split)
    apply_manual_referring_review(checkpoint_summary, args.manual_review_jsonl)
    summary = {
        "schema_version": 2,
        "evaluation_kind": (
            "heldout_generalization"
            if args.eval_split == "heldout"
            else "train_pool_domain_absorption"
        ),
        "eligible_for_best_checkpoint": args.eval_split == "heldout",
        "split": args.eval_split,
        "manifest": str(manifest) if manifest else None,
        "manifest_id": manifest_id,
        "subset_strategy": args.subset_strategy,
        "seed": args.seed,
        "samples_per_task": args.samples_per_task,
        "task_counts": dict(counts),
        "base_model": args.base_model,
        "checkpoint": args.checkpoint,
        "base_cache": str(base_cache),
        "base_cache_hit": base_cache_hit,
        "teacher_forced": args.teacher_forced,
        "base": base_summary,
        "checkpoint_metrics": checkpoint_summary,
        "checkpoint_minus_base": metric_delta(base_summary, checkpoint_summary),
    }
    summary["eval_wall_time_seconds"] = time.time() - evaluation_started
    history = _read_jsonl_rows(args.metrics_jsonl)
    metric_rows, selection = build_eval_metric_rows(args, summary, history)
    summary["evaluation_protocol_id"] = metric_rows[0]["evaluation_protocol_id"]
    for row in metric_rows:
        row["eval_wall_time_seconds"] = summary["eval_wall_time_seconds"]
    summary["checkpoint_selection"] = selection
    write_outputs(args, examples, base_results, checkpoint_results, summary)
    write_eval_metric_rows(args.output_dir, metric_rows, args.metrics_jsonl)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
