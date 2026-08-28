#!/usr/bin/env python3
"""Evaluate one CPT checkpoint on the external five-task UI5 test sets.

Generation is performed once.  The saved predictions are then scored at every
requested IoU threshold and projected into the authoritative CPT eval JSONL so
the normal three-sheet workbook can include held-out and external results.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for import_root in (PROJECT_ROOT, SCRIPTS_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from eaglevl.train.cpt_eval_queue import exclusive_file_lock, fsync_if_supported  # noqa: E402
from locany_ui5_common import TASK_ISSUE_NAMES, TASK_JSONL, TASKS  # noqa: E402
from run_locany_cpt_ui5_checkpoint_sweep import (  # noqa: E402
    atomic_write_json,
    load_metric_json,
    normalized_model_result,
    threshold_tag,
)


DEFAULT_INFERENCE = PROJECT_ROOT / "scripts/inference_ui_defect_locany.py"
DEFAULT_SCORER = PROJECT_ROOT / "qwen3vl_merge_and_score_fixed_5tasks.py"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--processor-path", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--inference-script", type=Path, default=DEFAULT_INFERENCE)
    parser.add_argument("--scorer-script", type=Path, default=DEFAULT_SCORER)
    parser.add_argument("--metrics-jsonl", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--vision-attn-implementation", default="flash_attention_2")
    parser.add_argument("--generation-mode", choices=("fast", "slow", "hybrid"), default="hybrid")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--n-future-tokens", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--max-images-per-task", type=int, default=0)
    parser.add_argument("--iou-thresholds", nargs="+", type=float, default=(0.1,))
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--build-excel", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    if args.checkpoint_step < 0:
        parser.error("--checkpoint-step cannot be negative")
    if args.max_new_tokens <= 0 or args.n_future_tokens <= 0:
        parser.error("token limits must be positive")
    if args.max_images_per_task < 0:
        parser.error("--max-images-per-task cannot be negative")
    args.iou_thresholds = tuple(dict.fromkeys(float(value) for value in args.iou_thresholds))
    if not args.iou_thresholds or any(not 0.0 < value <= 1.0 for value in args.iou_thresholds):
        parser.error("--iou-thresholds values must be in (0, 1]")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def external_manifest(args: argparse.Namespace) -> dict[str, Any]:
    files = {}
    for task in TASKS:
        path = args.input_dir / TASK_JSONL[task]
        if not path.is_file():
            raise FileNotFoundError(f"missing external UI5 {task} JSONL: {path}")
        files[task] = {
            "path": str(path),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    manifest_id = hashlib.sha256(
        json.dumps(files, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {"manifest_id": manifest_id, "files": files}


def run(command: Sequence[str]) -> None:
    print("EXTERNAL_UI5_COMMAND=" + " ".join(str(item) for item in command), flush=True)
    subprocess.run(list(command), cwd=PROJECT_ROOT, check=True)


def inference_command(args: argparse.Namespace, prediction_dir: Path) -> list[str]:
    command = [
        str(args.python),
        str(args.inference_script),
        "--checkpoint", str(args.checkpoint),
        "--processor-path", str(args.processor_path or args.base_model),
        "--input-dir", str(args.input_dir),
        "--output-dir", str(prediction_dir),
        "--summary-path", str(prediction_dir / "_summary.json"),
        "--cuda-visible-devices", "0",
        "--device", args.device,
        "--dtype", args.dtype,
        "--attn-implementation", args.attn_implementation,
        "--vision-attn-implementation", args.vision_attn_implementation,
        "--generation-mode", args.generation_mode,
        "--max-new-tokens", str(args.max_new_tokens),
        "--n-future-tokens", str(args.n_future_tokens),
        "--seed", str(args.seed),
        "--tasks", "all",
        "--skip-figma",
        "--fail-fast",
        "--save-raw-answer",
        "--relation-gate-mode", "observe",
        "--no-enable-ui-relation",
        "--no-enable-pbd",
    ]
    if args.max_images_per_task:
        command.extend(["--max-images-per-task", str(args.max_images_per_task)])
    if args.force:
        command.append("--overwrite")
    return command


def score_command(
    args: argparse.Namespace,
    prediction_dir: Path,
    score_root: Path,
    run_name: str,
    threshold: float,
) -> list[str]:
    return [
        str(args.python),
        str(args.scorer_script),
        "--all_tasks",
        "--input_mode", "yolo_dir",
        "--gt_dir", str(args.input_dir),
        "--pred_root", str(prediction_dir),
        "--output_root", str(score_root),
        "--run_name", run_name,
        "--yolo_bbox_format", "xyxy",
        "--iou_thresh", str(threshold),
    ]


def score_predictions(
    args: argparse.Namespace,
    *,
    prediction_dir: Path,
    output_dir: Path,
    threshold: float,
) -> tuple[Path, dict[str, Any]]:
    tag = threshold_tag(threshold)
    cached = output_dir / "metrics" / f"{tag}.json"
    if cached.is_file() and not args.force:
        return cached, load_metric_json(cached)
    attempt = "attempt-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    score_root = output_dir / "score_runs" / tag
    run(score_command(args, prediction_dir, score_root, attempt, threshold))
    produced = score_root / attempt / "all_tasks_evaluation.json"
    metrics = load_metric_json(produced)
    metrics["evaluation_metadata"] = {
        "checkpoint": str(args.checkpoint),
        "step": args.checkpoint_step,
        "iou_threshold": threshold,
        "prediction_dir": str(prediction_dir),
        "scorer_output": str(produced),
        "created_at": utc_now(),
    }
    atomic_write_json(cached, metrics)
    return cached, metrics


def cpt_metric_shape(
    metrics_path: Path, metrics: Mapping[str, Any], threshold: float
) -> dict[str, Any]:
    normalized = normalized_model_result(
        label="external-ui5",
        kind="external_ui5",
        step=None,
        checkpoint=None,
        prediction_dir=None,
        metrics_path=metrics_path,
        metrics=metrics,
        iou_threshold=threshold,
    )
    per_class = {}
    for task in TASKS:
        values = normalized["tasks"][task]
        image_values = dict(values["image"])
        image_count = sum(
            int(image_values.get(key, 0)) for key in ("tp", "fp", "fn", "tn")
        )
        image_values["images"] = image_count
        bbox_values = dict(values["bbox"])
        bbox_values["images"] = image_count
        per_class[task] = {
            "display_label": TASK_ISSUE_NAMES[task],
            "image": image_values,
            "bbox": bbox_values,
        }
    total_images = sum(
        int(values["image"]["images"]) for values in per_class.values()
    )
    image_macro = dict(normalized["macro"]["image"])
    bbox_macro = dict(normalized["macro"]["bbox"])
    image_micro = dict(normalized["micro"]["image"])
    bbox_micro = dict(normalized["micro"]["bbox"])
    for values in (image_macro, bbox_macro, image_micro, bbox_micro):
        values["images"] = total_images
    return {
        "iou_threshold": threshold,
        "per_class": per_class,
        "image_macro": image_macro,
        "bbox_macro": bbox_macro,
        "image_micro": image_micro,
        "bbox_micro": bbox_micro,
    }


def print_metrics(step: int, metrics: Mapping[str, Any]) -> None:
    threshold = float(metrics["iou_threshold"])
    print(f"\n===== EXTERNAL UI5 step={step} IoU={threshold:g} =====", flush=True)
    print("class | image P/R/F1 | bbox P/R/F1", flush=True)
    for task in TASKS:
        class_metrics = metrics["per_class"][task]
        image = class_metrics["image"]
        bbox = class_metrics["bbox"]
        print(
            f"{task} | {float(image.get('precision', 0)):.4f}/"
            f"{float(image.get('recall', 0)):.4f}/{float(image.get('f1', 0)):.4f} | "
            f"{float(bbox.get('precision', 0)):.4f}/"
            f"{float(bbox.get('recall', 0)):.4f}/{float(bbox.get('f1', 0)):.4f}",
            flush=True,
        )
    for aggregate in ("macro", "micro"):
        image = metrics[f"image_{aggregate}"]
        bbox = metrics[f"bbox_{aggregate}"]
        print(
            f"{aggregate} | image_f1={float(image.get('f1', 0)):.4f} "
            f"bbox_f1={float(bbox.get('f1', 0)):.4f}",
            flush=True,
        )


def read_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"external Base summary is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("metrics"), dict):
        raise ValueError(f"invalid external UI5 summary: {path}")
    return value


def write_eval_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with exclusive_file_lock(lock_path):
        existing: list[dict[str, Any]] = []
        if path.is_file():
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(f"{path}:{line_number}: expected JSON object")
                    existing.append(value)
        replacements = {str(row["evaluation_id"]): dict(row) for row in rows}
        allowed_external_thresholds = {
            float(row["iou_threshold"])
            for row in replacements.values()
            if row.get("split") == "external_ui5"
            and isinstance(row.get("iou_threshold"), (int, float))
        }
        merged = []
        for row in existing:
            if str(row.get("evaluation_id")) in replacements:
                continue
            if (
                allowed_external_thresholds
                and row.get("split") == "external_ui5"
                and row.get("task") == "ui_defect_external"
                and isinstance(row.get("iou_threshold"), (int, float))
                and float(row["iou_threshold"]) not in allowed_external_thresholds
            ):
                continue
            merged.append(row)
        merged.extend(replacements.values())
        merged.sort(key=lambda row: (int(row.get("step") or 0), str(row.get("task")), float(row.get("iou_threshold") or 0)))
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", delete=False
        ) as handle:
            temporary = Path(handle.name)
            for row in merged:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            fsync_if_supported(handle, path=temporary)
        os.replace(temporary, path)


def evaluation_identity(
    *, manifest_id: str, checkpoint: Path, step: int, threshold: float
) -> str:
    payload = {
        "protocol": "cpt-external-ui5-v1",
        "manifest_id": manifest_id,
        "checkpoint": str(checkpoint),
        "step": step,
        "iou_threshold": threshold,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def base_bootstrap_command(args: argparse.Namespace, *, force: bool) -> list[str]:
    command = [
        str(args.python),
        str(Path(__file__).resolve()),
        "--checkpoint", str(args.base_model),
        "--checkpoint-step", "0",
        "--base-model", str(args.base_model),
        "--processor-path", str(args.processor_path or args.base_model),
        "--run-dir", str(args.run_dir),
        "--input-dir", str(args.input_dir),
        "--python", str(args.python),
        "--inference-script", str(args.inference_script),
        "--scorer-script", str(args.scorer_script),
        "--metrics-jsonl", str(args.metrics_jsonl),
        "--device", args.device,
        "--dtype", args.dtype,
        "--attn-implementation", args.attn_implementation,
        "--vision-attn-implementation", args.vision_attn_implementation,
        "--generation-mode", args.generation_mode,
        "--max-new-tokens", str(args.max_new_tokens),
        "--n-future-tokens", str(args.n_future_tokens),
        "--seed", str(args.seed),
        "--max-images-per-task", str(args.max_images_per_task),
        "--iou-thresholds",
        *(str(value) for value in args.iou_thresholds),
        "--no-build-excel",
    ]
    if force:
        command.append("--force")
    return command


def main() -> int:
    args = parse_args()
    for name in (
        "checkpoint", "base_model", "run_dir", "input_dir", "inference_script", "scorer_script"
    ):
        value = getattr(args, name)
        setattr(args, name, value.expanduser().resolve())
    if args.processor_path is not None:
        args.processor_path = args.processor_path.expanduser().resolve()
    args.metrics_jsonl = (
        args.metrics_jsonl.expanduser().resolve()
        if args.metrics_jsonl is not None
        else args.run_dir / "diagnostics/cpt_eval_metrics.jsonl"
    )
    for path, label, kind in (
        (args.checkpoint, "checkpoint", "dir"),
        (args.base_model, "Base model", "dir"),
        (args.input_dir, "external UI5 data", "dir"),
        (args.inference_script, "inference script", "file"),
        (args.scorer_script, "canonical scorer", "file"),
    ):
        exists = path.is_dir() if kind == "dir" else path.is_file()
        if not exists:
            raise FileNotFoundError(f"missing {label}: {path}")

    manifest = external_manifest(args)
    if args.checkpoint_step > 0:
        base_summary_path = args.run_dir / "eval_external_ui5/checkpoint-0/summary.json"
        bootstrap_reason = None
        if not base_summary_path.is_file():
            bootstrap_reason = "missing"
        else:
            try:
                existing_base = read_summary(base_summary_path)
                expected_tags = {
                    threshold_tag(value) for value in args.iou_thresholds
                }
                if (
                    existing_base.get("manifest_id") != manifest["manifest_id"]
                    or set(existing_base.get("metrics", {})) != expected_tags
                ):
                    bootstrap_reason = "identity_mismatch"
            except (OSError, ValueError, json.JSONDecodeError):
                bootstrap_reason = "invalid"
        if bootstrap_reason is not None:
            print(
                "EXTERNAL_UI5_BASE_CACHE=BOOTSTRAP "
                f"reason={bootstrap_reason} path={base_summary_path}",
                flush=True,
            )
            run(
                base_bootstrap_command(
                    args, force=bootstrap_reason != "missing"
                )
            )

    output_dir = args.run_dir / "eval_external_ui5" / f"checkpoint-{args.checkpoint_step}"
    prediction_dir = output_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"EXTERNAL_UI5_EVAL=START step={args.checkpoint_step} "
        f"checkpoint={args.checkpoint} manifest_id={manifest['manifest_id']}",
        flush=True,
    )
    run(inference_command(args, prediction_dir))

    current_metrics: dict[str, dict[str, Any]] = {}
    metric_paths: dict[str, str] = {}
    for threshold in args.iou_thresholds:
        path, raw = score_predictions(
            args,
            prediction_dir=prediction_dir,
            output_dir=output_dir,
            threshold=threshold,
        )
        shaped = cpt_metric_shape(path, raw, threshold)
        tag = threshold_tag(threshold)
        current_metrics[tag] = shaped
        metric_paths[tag] = str(path)
        print_metrics(args.checkpoint_step, shaped)

    if args.checkpoint_step == 0:
        base_metrics = current_metrics
    else:
        base_summary_path = args.run_dir / "eval_external_ui5/checkpoint-0/summary.json"
        base_summary = read_summary(base_summary_path)
        if base_summary.get("manifest_id") != manifest["manifest_id"]:
            raise RuntimeError(
                "external Base cache manifest differs from current external test set; "
                "rerun step-0 evaluation with --force"
            )
        base_metrics = base_summary["metrics"]

    summary = {
        "schema_version": 1,
        "evaluation_kind": "external_ui5_generalization",
        "split": "external_ui5",
        "step": args.checkpoint_step,
        "checkpoint": str(args.checkpoint),
        "base_model": str(args.base_model),
        "manifest_id": manifest["manifest_id"],
        "manifest": manifest,
        "task_counts": {
            task: current_metrics[threshold_tag(args.iou_thresholds[0])][
                "per_class"
            ][task]["image"]["images"]
            for task in TASKS
        },
        "iou_thresholds": list(args.iou_thresholds),
        "metrics": current_metrics,
        "base_metrics": base_metrics,
        "metric_paths": metric_paths,
        "prediction_dir": str(prediction_dir),
        "completed_at": utc_now(),
    }
    summary_path = output_dir / "summary.json"
    atomic_write_json(summary_path, summary)

    rows = []
    for threshold in args.iou_thresholds:
        tag = threshold_tag(threshold)
        metrics = current_metrics[tag]
        base = base_metrics.get(tag)
        if not isinstance(base, Mapping):
            raise RuntimeError(f"external Base cache has no threshold {threshold:g}")
        primary = metrics["bbox_macro"].get("f1")
        base_primary = base["bbox_macro"].get("f1")
        evaluation_id = evaluation_identity(
            manifest_id=manifest["manifest_id"],
            checkpoint=args.checkpoint,
            step=args.checkpoint_step,
            threshold=threshold,
        )
        rows.append(
            {
                "schema_version": 1,
                "evaluation_id": evaluation_id,
                "evaluation_protocol_id": f"cpt-external-ui5-v1-{manifest['manifest_id'][:16]}-iou-{threshold:g}",
                "evaluation_kind": "external_ui5_generalization",
                "eligible_for_best_checkpoint": False,
                "checkpoint": str(args.checkpoint),
                "step": args.checkpoint_step,
                "split": "external_ui5",
                "task": "ui_defect_external",
                "manifest_id": manifest["manifest_id"],
                "subset_strategy": "full" if not args.max_images_per_task else "first",
                "samples_per_task": None if not args.max_images_per_task else args.max_images_per_task,
                "task_counts": summary["task_counts"],
                "iou_threshold": threshold,
                "primary_name": f"external_bbox_macro_f1@{threshold:g}",
                "primary_metric": primary,
                "base_primary": base_primary,
                "delta_vs_base": (
                    float(primary) - float(base_primary)
                    if isinstance(primary, (int, float)) and isinstance(base_primary, (int, float))
                    else None
                ),
                "metrics": metrics,
                "base_metrics": base,
                "summary": str(summary_path),
            }
        )
    write_eval_rows(args.metrics_jsonl, rows)

    workbook_exit_code = None
    if args.build_excel:
        workbook = subprocess.run(
            [
                str(args.python),
                str(PROJECT_ROOT / "scripts/build_locany_cpt_excel.py"),
                "--diagnostics-dir", str(args.run_dir / "diagnostics"),
            ],
            cwd=PROJECT_ROOT,
            check=False,
        )
        workbook_exit_code = int(workbook.returncode)
    print(
        f"EXTERNAL_UI5_EVAL=COMPLETED step={args.checkpoint_step} "
        f"summary={summary_path} metrics_jsonl={args.metrics_jsonl} "
        f"workbook_exit_code={workbook_exit_code}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
