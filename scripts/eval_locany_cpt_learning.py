#!/usr/bin/env python3
"""Small base-vs-CPT check on records from a LocateAnything UI recipe.

The script intentionally evaluates only a few deterministic training-format
records.  It prints predictions for subjective inspection and writes simple
objective metrics: normalized exact match, character F1, canonical grounding
format rate, mean best IoU, and box recall at IoU >= 0.5.

These numbers measure fitting/domain absorption on the selected records.  They
are not a held-out generalization score unless ``--recipe`` points to data that
was excluded from training.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import shutil
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.inference_ui_defect_locany import LocateAnythingInferencer  # noqa: E402


GROUNDING_TASKS = {
    "agent_grounding",
    "ui_defect",
    "all_ui_elements",
    "single_grounding",
    "ocr",
    "agent_other",
    "multi_grounding_other",
}
IMAGE_TOKEN_RE = re.compile(r"<image(?:-\d+)?>")
BOX_RE = re.compile(
    r"<box>\s*<\s*(-?\d+)\s*>\s*<\s*(-?\d+)\s*>\s*"
    r"<\s*(-?\d+)\s*>\s*<\s*(-?\d+)\s*>\s*</box>",
    re.IGNORECASE,
)
PAIR_RE = re.compile(
    r"<ref>(.*?)</ref>\s*"
    r"<box>\s*<\s*(-?\d+)\s*>\s*<\s*(-?\d+)\s*>\s*"
    r"<\s*(-?\d+)\s*>\s*<\s*(-?\d+)\s*>\s*</box>",
    re.IGNORECASE | re.DOTALL,
)
SPECIAL_END_RE = re.compile(r"(?:<\|im_end\|>|<\|endoftext\|>)+\s*$")


@dataclass(frozen=True)
class Example:
    key: str
    task: str
    image: str
    prompt: str
    target: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare nvidia/LocateAnything-3B with one UI CPT checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="checkpoint-<step> directory")
    parser.add_argument("--base-model", required=True, help="original LocateAnything model directory")
    parser.add_argument("--recipe", type=Path, required=True, help="prepared CPT recipe JSON")
    parser.add_argument(
        "--processor-path",
        default=None,
        help="processor/tokenizer directory; defaults to --base-model",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-task", type=int, default=1)
    parser.add_argument(
        "--tasks",
        default="all",
        help="comma-separated cpt_task names, or all",
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="allow Hugging Face network access instead of local-files-only",
    )
    args = parser.parse_args()
    if args.samples_per_task <= 0:
        parser.error("--samples-per-task must be positive")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    return args


def ensure_checkpoint_remote_code(checkpoint: str, base_model: str) -> None:
    """
    Hugging Face trust_remote_code requires LocateAnything custom Python files
    to exist beside config.json in the checkpoint directory.

    Training checkpoints may contain weights/config.json but omit those source
    files. Copy missing top-level *.py files from the original base-model
    snapshot before loading the checkpoint.

    IMPORTANT:
      - Never overwrite checkpoint/config.json.
      - Never copy model weights from the base model.
      - Existing checkpoint Python files are preserved.
    """
    checkpoint_dir = Path(checkpoint).expanduser().resolve()
    base_dir = Path(base_model).expanduser().resolve()

    if not checkpoint_dir.is_dir():
        raise RuntimeError(f"checkpoint directory does not exist: {checkpoint_dir}")

    if not base_dir.is_dir():
        raise RuntimeError(f"base model directory does not exist: {base_dir}")

    checkpoint_config = checkpoint_dir / "config.json"
    if not checkpoint_config.is_file():
        raise RuntimeError(
            f"checkpoint is missing config.json: {checkpoint_config}"
        )

    base_python_files = sorted(base_dir.glob("*.py"))
    if not base_python_files:
        raise RuntimeError(
            f"base model contains no Hugging Face remote-code Python files: {base_dir}"
        )

    copied = []
    existing = []

    for source in base_python_files:
        destination = checkpoint_dir / source.name

        if destination.is_file() and destination.stat().st_size > 0:
            existing.append(source.name)
            continue

        shutil.copy2(source, destination)
        copied.append(source.name)

    print("\n===== checkpoint remote-code preparation =====")
    print(f"base model : {base_dir}")
    print(f"checkpoint : {checkpoint_dir}")

    if copied:
        print(f"copied     : {', '.join(copied)}")
    else:
        print("copied     : none")

    print(f"available  : {len(existing) + len(copied)} python files")

    # Explicitly check the two files required by AutoConfig/AutoModel.
    required = [
        "configuration_locateanything.py",
        "modeling_locateanything.py",
    ]

    missing = [
        name
        for name in required
        if not (checkpoint_dir / name).is_file()
    ]

    if missing:
        raise RuntimeError(
            "checkpoint remote code is still incomplete after copying from "
            f"base model; missing={missing}"
        )

    print("remote code: OK")


def _resolve_recipe_path(value: str, recipe: Path, relative_to_meta: bool) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    base = recipe.parent if relative_to_meta else Path.cwd()
    return (base / path).resolve()


def _first_text_turn(record: dict[str, Any], role: str) -> str | None:
    aliases = {"human", "user"} if role == "human" else {"gpt", "assistant"}
    for turn in record.get("conversations", []):
        if str(turn.get("from", turn.get("role", ""))).lower() not in aliases:
            continue
        value = turn.get("value", turn.get("content"))
        if isinstance(value, str):
            return value
    return None


def _resolve_image(record: dict[str, Any], media_root: Path | None) -> Path | None:
    value = record.get("image") or record.get("images")
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            return None
        value = value[0]
    if not isinstance(value, str) or not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute() and media_root is not None:
        path = media_root / path
    return path.resolve()


def _iter_jsonl(paths: Iterable[Path]) -> Iterable[tuple[Path, int, dict[str, Any]]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if isinstance(value, dict):
                    yield path, line_number, value


def load_examples(recipe_path: Path, samples_per_task: int, selected: set[str] | None) -> list[Example]:
    recipe_path = recipe_path.expanduser().resolve()
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    examples: list[Example] = []

    for recipe_name, meta in recipe.items():
        recipe_task = recipe_name.removeprefix("locany_cpt_")
        if selected is not None and recipe_task not in selected:
            continue
        relative = bool(meta.get("paths_relative_to_meta", False))
        annotations = meta.get("annotation", [])
        if isinstance(annotations, str):
            annotations = [annotations]
        annotation_paths = [
            _resolve_recipe_path(value, recipe_path, relative) for value in annotations
        ]
        root_value = str(meta.get("root", "")).strip()
        media_root = (
            _resolve_recipe_path(root_value, recipe_path, relative) if root_value else None
        )

        accepted = 0
        for source, line_number, record in _iter_jsonl(annotation_paths):
            task = str(record.get("cpt_task") or recipe_task)
            if selected is not None and task not in selected:
                continue
            prompt = _first_text_turn(record, "human")
            target = _first_text_turn(record, "gpt")
            image = _resolve_image(record, media_root)
            if prompt is None or target is None or image is None or not image.is_file():
                continue
            prompt = IMAGE_TOKEN_RE.sub("", prompt).strip()
            record_id = record.get("id", f"{source.name}:{line_number}")
            examples.append(
                Example(
                    key=f"{task}:{record_id}",
                    task=task,
                    image=str(image),
                    prompt=prompt,
                    target=target,
                )
            )
            accepted += 1
            if accepted >= samples_per_task:
                break

    if not examples:
        raise RuntimeError("recipe contains no usable single-image examples")
    return examples


def normalize_text(text: str) -> str:
    text = SPECIAL_END_RE.sub("", text.strip())
    return re.sub(r"\s+", " ", text).strip()


def char_f1(prediction: str, target: str) -> float:
    pred = Counter(char for char in normalize_text(prediction) if not char.isspace())
    gold = Counter(char for char in normalize_text(target) if not char.isspace())
    pred_total = sum(pred.values())
    gold_total = sum(gold.values())
    if pred_total == 0 or gold_total == 0:
        return float(pred_total == gold_total)
    common = sum((pred & gold).values())
    precision = common / pred_total
    recall = common / gold_total
    return 2.0 * precision * recall / (precision + recall) if common else 0.0


def parse_grounding(text: str) -> tuple[list[list[int]], list[tuple[str, list[int]]], bool]:
    boxes = [[int(value) for value in match.groups()] for match in BOX_RE.finditer(text)]
    pairs = [
        (match.group(1).strip(), [int(match.group(index)) for index in range(2, 6)])
        for match in PAIR_RE.finditer(text)
    ]
    valid = bool(boxes) and len(boxes) == len(pairs)
    return boxes, pairs, valid


def box_iou(left: list[int], right: list[int]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    left_area = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    right_area = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def grounding_scores(prediction: str, target: str) -> dict[str, Any]:
    pred_boxes, _, pred_format_valid = parse_grounding(prediction)
    gold_boxes, _, _ = parse_grounding(target)
    remaining = set(range(len(pred_boxes)))
    best_ious: list[float] = []
    for gold in gold_boxes:
        candidate = max(
            ((box_iou(gold, pred_boxes[index]), index) for index in remaining),
            default=(0.0, -1),
        )
        best_iou, best_index = candidate
        best_ious.append(best_iou)
        if best_index >= 0:
            remaining.remove(best_index)
    return {
        "grounding_example": bool(gold_boxes),
        "format_valid": pred_format_valid if gold_boxes else None,
        "gold_box_count": len(gold_boxes),
        "pred_box_count": len(pred_boxes),
        "best_iou_sum": sum(best_ious),
        "box_hits_50": sum(value >= 0.5 for value in best_ious),
    }


def score_prediction(example: Example, prediction: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "exact_match": int(normalize_text(prediction) == normalize_text(example.target)),
        "char_f1": char_f1(prediction, example.target),
    }
    if example.task in GROUNDING_TASKS:
        metrics.update(grounding_scores(prediction, example.target))
    return metrics


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


def run_model(
    label: str,
    model_path: str,
    examples: list[Example],
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    print(f"\n===== {label}: {model_path} =====", flush=True)
    inferencer = LocateAnythingInferencer(inference_namespace(args, model_path))
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
                results[example.key] = {
                    "prediction": prediction,
                    "error": None,
                    "metrics": score_prediction(example, prediction),
                }
            except Exception as exc:  # Keep the remaining tasks useful in a smoke test.
                results[example.key] = {
                    "prediction": "",
                    "error": f"{type(exc).__name__}: {exc}",
                    "metrics": {},
                }
                print(f"[ERROR] {example.key}: {exc}", file=sys.stderr, flush=True)
    finally:
        del inferencer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def aggregate(
    examples: list[Example],
    results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    def one_group(group: list[Example]) -> dict[str, Any]:
        valid = [results[item.key] for item in group if not results[item.key]["error"]]
        metrics = [item["metrics"] for item in valid]
        grounding = [item for item in metrics if item.get("grounding_example")]
        gold_boxes = sum(int(item["gold_box_count"]) for item in grounding)
        return {
            "examples": len(group),
            "successful": len(valid),
            "errors": len(group) - len(valid),
            "exact_match": _mean([float(item["exact_match"]) for item in metrics]),
            "char_f1": _mean([float(item["char_f1"]) for item in metrics]),
            "grounding_format_rate": _mean(
                [float(bool(item["format_valid"])) for item in grounding]
            ),
            "grounding_mean_best_iou": (
                sum(float(item["best_iou_sum"]) for item in grounding) / gold_boxes
                if gold_boxes
                else None
            ),
            "grounding_box_recall_50": (
                sum(int(item["box_hits_50"]) for item in grounding) / gold_boxes
                if gold_boxes
                else None
            ),
            "grounding_gold_boxes": gold_boxes,
        }

    tasks: OrderedDict[str, list[Example]] = OrderedDict()
    for example in examples:
        tasks.setdefault(example.task, []).append(example)
    return {
        "overall": one_group(examples),
        "by_task": {task: one_group(group) for task, group in tasks.items()},
    }


def metric_delta(base: dict[str, Any], checkpoint: dict[str, Any]) -> dict[str, float | None]:
    keys = (
        "exact_match",
        "char_f1",
        "grounding_format_rate",
        "grounding_mean_best_iou",
        "grounding_box_recall_50",
    )
    output: dict[str, float | None] = {}
    for key in keys:
        left, right = base.get(key), checkpoint.get(key)
        output[key] = None if left is None or right is None else float(right) - float(left)
    return output


def preview(value: str, limit: int = 600) -> str:
    value = value.strip()
    return value if len(value) <= limit else value[:limit] + " ...[truncated]"


def main() -> int:
    args = parse_args()

    ensure_checkpoint_remote_code(
        checkpoint=args.checkpoint,
        base_model=args.base_model,
    )

    selected = None if args.tasks.strip().lower() == "all" else {
        value.strip() for value in args.tasks.split(",") if value.strip()
    }
    examples = load_examples(args.recipe, args.samples_per_task, selected)
    task_counts = Counter(example.task for example in examples)
    print(f"Selected {len(examples)} examples: {dict(task_counts)}")

    base_results = run_model("base", args.base_model, examples, args)
    checkpoint_results = run_model("checkpoint", args.checkpoint, examples, args)

    rows = []
    for example in examples:
        row = {
            "key": example.key,
            "task": example.task,
            "image": example.image,
            "prompt": example.prompt,
            "target": example.target,
            "base": base_results[example.key],
            "checkpoint": checkpoint_results[example.key],
        }
        rows.append(row)
        print("\n" + "=" * 88)
        print(f"[{example.task}] {example.key}")
        print(f"PROMPT     : {preview(example.prompt)}")
        print(f"TARGET     : {preview(example.target)}")
        print(f"BASE       : {preview(row['base']['prediction'])}")
        print(f"CHECKPOINT : {preview(row['checkpoint']['prediction'])}")

    base_summary = aggregate(examples, base_results)
    checkpoint_summary = aggregate(examples, checkpoint_results)
    summary = {
        "note": (
            "Training-format fitting/domain-absorption smoke test; use a recipe excluded "
            "from training for a generalization estimate."
        ),
        "base_model": args.base_model,
        "checkpoint": args.checkpoint,
        "recipe": str(args.recipe.expanduser().resolve()),
        "samples_per_task": args.samples_per_task,
        "base": base_summary,
        "checkpoint_metrics": checkpoint_summary,
        "checkpoint_minus_base": metric_delta(
            base_summary["overall"], checkpoint_summary["overall"]
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("\n===== objective summary =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"predictions: {predictions_path}")
    print(f"summary    : {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
