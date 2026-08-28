#!/usr/bin/env python3
"""Claim CPT checkpoints and run held-out plus optional external evaluation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eaglevl.train.cpt_eval_queue import claim_next_eval, finish_eval
from eaglevl.train.cpt_eval_metrics import UI_DEFECT_CLASSES
from eaglevl.train.cpt_observability import CPT_TASKS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--eval-recipe-name", default="locany_cpt_val_fast.json")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--processor-path", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--evaluator", type=Path, default=PROJECT_ROOT / "scripts/eval_locany_cpt_learning.py"
    )
    parser.add_argument(
        "--external-evaluator",
        type=Path,
        default=PROJECT_ROOT / "scripts/run_locany_cpt_external_ui5_eval.py",
    )
    parser.add_argument("--external-ui5-data-dir", type=Path, default=None)
    parser.add_argument(
        "--external-ui5-eval",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--external-max-new-tokens", type=int, default=4096)
    parser.add_argument("--external-max-images-per-task", type=int, default=0)
    parser.add_argument(
        "--external-iou-thresholds", nargs="+", type=float, default=(0.1,)
    )
    parser.add_argument("--samples-per-task", type=int, default=10)
    parser.add_argument("--max-pending", type=int, default=1)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--vision-attn-implementation", default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument(
        "--require-zero-inference-errors",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    if args.samples_per_task <= 0 or args.max_pending <= 0:
        parser.error("--samples-per-task and --max-pending must be positive")
    if args.external_max_new_tokens <= 0 or args.external_max_images_per_task < 0:
        parser.error("invalid external UI5 token/image limit")
    return args


def validate_eval_summary(
    summary: Mapping[str, Any],
    *,
    samples_per_task: int,
    require_zero_errors: bool,
) -> None:
    if summary.get("split") != "heldout":
        raise RuntimeError(f"eval summary is not held-out: split={summary.get('split')!r}")
    if summary.get("teacher_forced") is not True:
        raise RuntimeError("eval summary did not run teacher-forced CE")
    task_counts = {str(key): int(value) for key, value in summary.get("task_counts", {}).items()}
    if set(task_counts) != set(CPT_TASKS):
        raise RuntimeError(
            "held-out eval task set mismatch: "
            f"missing={sorted(set(CPT_TASKS) - set(task_counts))}, "
            f"unexpected={sorted(set(task_counts) - set(CPT_TASKS))}"
        )
    short = {task: count for task, count in task_counts.items() if count < samples_per_task}
    if short:
        raise RuntimeError(f"held-out eval has fewer than {samples_per_task} rows: {short}")
    for label in ("base", "checkpoint_metrics"):
        aggregate = summary.get(label, {})
        per_task = aggregate.get("per_task", {})
        if set(per_task) != set(CPT_TASKS):
            raise RuntimeError(f"{label} summary does not contain all ten CPT tasks")
        if aggregate.get("heldout_task_macro_primary") is None:
            raise RuntimeError(f"{label} summary has no heldout_task_macro_primary")
        for task in CPT_TASKS:
            metrics = per_task[task]
            if metrics.get("eval_main_token_ce") is None:
                raise RuntimeError(f"{label} task={task} has no eval_main_token_ce")
            if require_zero_errors and int(metrics.get("inference_error_count") or 0):
                raise RuntimeError(
                    f"{label} task={task} inference_error_count="
                    f"{metrics.get('inference_error_count')}"
                )
        defect = per_task["ui_defect"]
        defect_classes = defect.get("per_class", {})
        missing_defect_classes = set(UI_DEFECT_CLASSES).difference(defect_classes)
        if missing_defect_classes:
            raise RuntimeError(
                f"{label} ui_defect is missing five-class metrics: "
                f"{sorted(missing_defect_classes)}"
            )
        for metric in (
            "defect_image_macro_f1",
            "defect_image_micro_f1",
            "defect_bbox_macro_f1_50",
            "defect_bbox_micro_f1_50",
        ):
            if defect.get(metric) is None:
                raise RuntimeError(f"{label} ui_defect has no {metric}")


def evaluator_command(args: argparse.Namespace, row: Mapping[str, Any], output_dir: Path) -> list[str]:
    run_dir = args.run_dir.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    return [
        str(args.python),
        str(args.evaluator.expanduser().resolve()),
        "--checkpoint", str(Path(str(row["checkpoint"])).expanduser().resolve()),
        "--base-model", str(Path(args.base_model).expanduser().resolve()),
        "--processor-path", str(Path(args.processor_path or args.base_model).expanduser().resolve()),
        "--recipe", str(data_dir / "recipe" / args.eval_recipe_name),
        "--manifest", str(data_dir / "diagnostics/split_manifest.jsonl"),
        "--eval-split", "heldout",
        "--subset-strategy", "hash",
        "--samples-per-task", str(args.samples_per_task),
        "--base-cache-dir", str(run_dir / "eval/base_cache"),
        "--train-metrics-jsonl", str(run_dir / "diagnostics/cpt_train_metrics.jsonl"),
        "--metrics-jsonl", str(run_dir / "diagnostics/cpt_eval_metrics.jsonl"),
        "--output-dir", str(output_dir),
        "--device", args.device,
        "--dtype", args.dtype,
        "--attn-implementation", args.attn_implementation,
        "--vision-attn-implementation", args.vision_attn_implementation,
        "--max-new-tokens", str(args.max_new_tokens),
        "--seed", str(args.seed),
        "--teacher-forced",
        "--fail-fast-inference-errors",
    ]


def run_claim(args: argparse.Namespace, row: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint = Path(str(row["checkpoint"])).expanduser().resolve()
    marker = checkpoint / "checkpoint_complete.json"
    if not marker.is_file():
        raise RuntimeError(f"checkpoint is not complete: {marker}")
    step = int(row["step"])
    output_dir = args.run_dir.expanduser().resolve() / "eval" / f"checkpoint-{step}"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    heldout_cache_hit = False
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            validate_eval_summary(
                summary,
                samples_per_task=args.samples_per_task,
                require_zero_errors=args.require_zero_inference_errors,
            )
            if int(summary.get("step") or -1) != step:
                raise RuntimeError(
                    f"held-out cache step mismatch: {summary.get('step')!r} != {step}"
                )
            if Path(str(summary.get("checkpoint"))).expanduser().resolve() != checkpoint:
                raise RuntimeError(
                    "held-out cache checkpoint mismatch: "
                    f"{summary.get('checkpoint')!r} != {checkpoint}"
                )
            heldout_cache_hit = True
            print(f"HELDOUT_EVAL_CACHE=HIT step={step} summary={summary_path}")
        except (OSError, ValueError, json.JSONDecodeError, RuntimeError):
            heldout_cache_hit = False
    if not heldout_cache_hit:
        command = evaluator_command(args, row, output_dir)
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        if not summary_path.is_file():
            raise RuntimeError(f"evaluator produced no summary: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        validate_eval_summary(
            summary,
            samples_per_task=args.samples_per_task,
            require_zero_errors=args.require_zero_inference_errors,
        )
    external_summary_path = None
    if args.external_ui5_eval:
        if args.external_ui5_data_dir is None:
            raise RuntimeError(
                "external UI5 evaluation is enabled but --external-ui5-data-dir is missing"
            )
        external_command = [
            str(args.python),
            str(args.external_evaluator.expanduser().resolve()),
            "--checkpoint", str(checkpoint),
            "--checkpoint-step", str(step),
            "--base-model", str(Path(args.base_model).expanduser().resolve()),
            "--processor-path", str(
                Path(args.processor_path or args.base_model).expanduser().resolve()
            ),
            "--run-dir", str(args.run_dir.expanduser().resolve()),
            "--input-dir", str(args.external_ui5_data_dir.expanduser().resolve()),
            "--python", str(args.python),
            "--device", args.device,
            "--dtype", args.dtype,
            "--attn-implementation", args.attn_implementation,
            "--vision-attn-implementation", args.vision_attn_implementation,
            "--max-new-tokens", str(args.external_max_new_tokens),
            "--seed", str(args.seed),
            "--max-images-per-task", str(args.external_max_images_per_task),
            "--iou-thresholds",
            *(str(value) for value in args.external_iou_thresholds),
            "--no-build-excel",
        ]
        subprocess.run(external_command, cwd=PROJECT_ROOT, check=True)
        external_summary_path = (
            args.run_dir.expanduser().resolve()
            / "eval_external_ui5"
            / f"checkpoint-{step}"
            / "summary.json"
        )
        if not external_summary_path.is_file():
            raise RuntimeError(
                f"external UI5 evaluator produced no summary: {external_summary_path}"
            )
    # Workbook projection is optional and must not invalidate good JSON/JSONL.
    workbook_command = [
        str(args.python),
        str(PROJECT_ROOT / "scripts/build_locany_cpt_excel.py"),
        "--diagnostics-dir", str(args.run_dir.expanduser().resolve() / "diagnostics"),
    ]
    workbook = subprocess.run(workbook_command, cwd=PROJECT_ROOT, check=False)
    return {
        "step": step,
        "summary": str(summary_path),
        "external_ui5_summary": (
            str(external_summary_path) if external_summary_path is not None else None
        ),
        "output_dir": str(output_dir),
        "heldout_cache_hit": heldout_cache_hit,
        "workbook_exit_code": int(workbook.returncode),
    }


def main() -> int:
    args = parse_args()
    processed = []
    for _ in range(args.max_pending):
        row = claim_next_eval(args.queue, retry_failed=args.retry_failed)
        if row is None:
            break
        try:
            details = run_claim(args, row)
            finish_eval(
                args.queue,
                row["queue_id"],
                status="completed",
                details={"result": details},
            )
            processed.append(details)
        except BaseException as exc:
            finish_eval(
                args.queue,
                row["queue_id"],
                status="failed",
                details={"error": f"{type(exc).__name__}: {exc}"},
            )
            traceback.print_exc()
            return 1
    print(json.dumps({"processed": processed, "count": len(processed)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
