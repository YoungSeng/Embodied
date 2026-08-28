#!/usr/bin/env python3
"""Run and validate CPT step-0 held-out plus external UI5 evaluation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_locany_cpt_eval_queue import validate_eval_summary  # noqa: E402


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--eval-recipe-name", default="locany_cpt_val_fast.json")
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--evaluator",
        type=Path,
        default=PROJECT_ROOT / "scripts/eval_locany_cpt_learning.py",
    )
    parser.add_argument(
        "--external-evaluator",
        type=Path,
        default=PROJECT_ROOT / "scripts/run_locany_cpt_external_ui5_eval.py",
    )
    parser.add_argument("--external-ui5-data-dir", type=Path, required=True)
    parser.add_argument("--external-max-new-tokens", type=int, default=4096)
    parser.add_argument("--external-max-images-per-task", type=int, default=0)
    parser.add_argument(
        "--external-iou-thresholds", nargs="+", type=float, default=(0.1,)
    )
    parser.add_argument("--samples-per-task", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--vision-attn-implementation", default="flash_attention_2")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--iou-threshold", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument(
        "--require-zero-inference-errors",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.samples_per_task <= 0 or args.max_new_tokens <= 0:
        parser.error("--samples-per-task and --max-new-tokens must be positive")
    if args.external_max_new_tokens <= 0 or args.external_max_images_per_task < 0:
        parser.error("invalid external UI5 token/image limit")
    if not 0.0 < args.iou_threshold <= 1.0:
        parser.error("--iou-threshold must be in (0, 1]")
    return args


def expected_identity(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "step": 0,
        "split": "heldout",
        "base_model": str(args.base_model),
        "data_dir": str(args.data_dir),
        "eval_recipe_name": args.eval_recipe_name,
        "samples_per_task": int(args.samples_per_task),
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "vision_attn_implementation": args.vision_attn_implementation,
        "max_new_tokens": int(args.max_new_tokens),
        "heldout_iou_threshold": float(args.iou_threshold),
        "seed": int(args.seed),
        "external_ui5_data_dir": str(args.external_ui5_data_dir),
        "external_max_new_tokens": int(args.external_max_new_tokens),
        "external_max_images_per_task": int(args.external_max_images_per_task),
        "external_iou_thresholds": [float(value) for value in args.external_iou_thresholds],
    }


def validate_summary(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"initial evaluator produced no summary: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    validate_eval_summary(
        summary,
        samples_per_task=args.samples_per_task,
        require_zero_errors=args.require_zero_inference_errors,
        iou_threshold=args.iou_threshold,
    )
    if summary.get("step") != 0:
        raise RuntimeError(f"initial evaluation summary step is not 0: {summary.get('step')!r}")
    return summary


def validate_external_summary(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"initial external evaluator produced no summary: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    expected_tags = {
        f"iou-{float(value):g}".replace(".", "p")
        for value in args.external_iou_thresholds
    }
    if summary.get("split") != "external_ui5" or summary.get("step") != 0:
        raise RuntimeError(
            "initial external summary has wrong split/step: "
            f"split={summary.get('split')!r}, step={summary.get('step')!r}"
        )
    if set(summary.get("metrics", {})) != expected_tags:
        raise RuntimeError(
            "initial external summary threshold mismatch: "
            f"actual={sorted(summary.get('metrics', {}))}, expected={sorted(expected_tags)}"
        )
    return summary


def main() -> int:
    args = parse_args()
    args.run_dir = args.run_dir.expanduser().resolve()
    args.data_dir = args.data_dir.expanduser().resolve()
    args.base_model = args.base_model.expanduser().resolve()
    args.evaluator = args.evaluator.expanduser().resolve()
    args.external_evaluator = args.external_evaluator.expanduser().resolve()
    args.external_ui5_data_dir = args.external_ui5_data_dir.expanduser().resolve()
    for path, label in (
        (args.data_dir, "CPT split data"),
        (args.base_model, "Base model"),
        (args.external_ui5_data_dir, "external UI5 data"),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if not args.evaluator.is_file():
        raise FileNotFoundError(f"evaluator does not exist: {args.evaluator}")
    if not args.external_evaluator.is_file():
        raise FileNotFoundError(
            f"external evaluator does not exist: {args.external_evaluator}"
        )

    output_dir = args.run_dir / "eval/checkpoint-0"
    summary_path = output_dir / "summary.json"
    marker_path = output_dir / "initial_eval_complete.json"
    identity = expected_identity(args)
    if marker_path.is_file() and not args.force:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        previous_identity = marker.get("identity")
        legacy_identity = isinstance(previous_identity, dict) and all(
            identity.get(key) == value
            for key, value in previous_identity.items()
        )
        if previous_identity != identity and not legacy_identity:
            raise RuntimeError(
                "completed initial evaluation uses different immutable settings; "
                f"existing={previous_identity}, requested={identity}"
            )
        if previous_identity == identity:
            validate_summary(summary_path, args)
            external_summary_path = (
                args.run_dir / "eval_external_ui5/checkpoint-0/summary.json"
            )
            validate_external_summary(external_summary_path, args)
            print(f"INITIAL_CPT_EVAL=SKIPPED_VALID_COMPLETION summary={summary_path}")
            return 0
        print(
            "INITIAL_CPT_EVAL=UPGRADE_LEGACY_COMPLETION "
            "reason=evaluation_protocol_added_or_changed"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(args.python),
        str(args.evaluator),
        "--checkpoint",
        str(args.base_model),
        "--checkpoint-step",
        "0",
        "--base-model",
        str(args.base_model),
        "--processor-path",
        str(args.base_model),
        "--recipe",
        str(args.data_dir / "recipe" / args.eval_recipe_name),
        "--manifest",
        str(args.data_dir / "diagnostics/split_manifest.jsonl"),
        "--eval-split",
        "heldout",
        "--subset-strategy",
        "hash",
        "--samples-per-task",
        str(args.samples_per_task),
        "--base-cache-dir",
        str(args.run_dir / "eval/base_cache"),
        "--train-metrics-jsonl",
        str(args.run_dir / "diagnostics/cpt_train_metrics.jsonl"),
        "--metrics-jsonl",
        str(args.run_dir / "diagnostics/cpt_eval_metrics.jsonl"),
        "--output-dir",
        str(output_dir),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--attn-implementation",
        args.attn_implementation,
        "--vision-attn-implementation",
        args.vision_attn_implementation,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--iou-threshold",
        str(args.iou_threshold),
        "--seed",
        str(args.seed),
        "--teacher-forced",
        "--fail-fast-inference-errors",
    ]
    started = time.time()
    print("INITIAL_CPT_EVAL=START")
    print("INITIAL_CPT_EVAL_COMMAND=" + " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    summary = validate_summary(summary_path, args)
    external_command = [
        str(args.python),
        str(args.external_evaluator),
        "--checkpoint", str(args.base_model),
        "--checkpoint-step", "0",
        "--base-model", str(args.base_model),
        "--processor-path", str(args.base_model),
        "--run-dir", str(args.run_dir),
        "--input-dir", str(args.external_ui5_data_dir),
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
    if args.force:
        external_command.append("--force")
    print("INITIAL_EXTERNAL_UI5_COMMAND=" + " ".join(external_command))
    subprocess.run(external_command, cwd=PROJECT_ROOT, check=True)
    external_summary_path = args.run_dir / "eval_external_ui5/checkpoint-0/summary.json"
    validate_external_summary(external_summary_path, args)
    marker = {
        "schema_version": 1,
        "identity": identity,
        "summary": str(summary_path),
        "manifest_id": summary.get("manifest_id"),
        "evaluation_protocol_id": summary.get("evaluation_protocol_id"),
        "heldout_task_macro_primary": summary.get("checkpoint_metrics", {}).get(
            "heldout_task_macro_primary"
        ),
        "external_ui5_summary": str(external_summary_path),
        "elapsed_seconds": time.time() - started,
        "completed_at_unix": time.time(),
    }
    atomic_write_json(marker_path, marker)

    # Excel is optional and must not invalidate a good JSON/JSONL evaluation.
    workbook = subprocess.run(
        [
            str(args.python),
            str(PROJECT_ROOT / "scripts/build_locany_cpt_excel.py"),
            "--diagnostics-dir",
            str(args.run_dir / "diagnostics"),
        ],
        cwd=PROJECT_ROOT,
        check=False,
    )
    print(
        f"INITIAL_CPT_EVAL=COMPLETED summary={summary_path} "
        f"workbook_exit_code={workbook.returncode}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
