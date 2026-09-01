#!/usr/bin/env python3
"""Run/repair held-out and external evaluations between CPT train segments."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for root in (PROJECT_ROOT, SCRIPTS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from run_ui5_parallel_inference import atomic_write_json  # noqa: E402
from run_locany_cpt_external_ui5_eval import write_eval_rows  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--processor-path", type=Path, default=None)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--external-input-dir", type=Path, required=True)
    parser.add_argument("--gpu-devices", required=True)
    parser.add_argument("--samples-per-task", type=int, default=200)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--vision-attn-implementation", default="flash_attention_2")
    parser.add_argument("--heldout-max-new-tokens", type=int, default=1024)
    parser.add_argument("--external-max-new-tokens", type=int, default=4096)
    parser.add_argument("--iou-thresholds", nargs="+", type=float, default=(0.1, 0.5))
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--external-ui5", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--repair-prior", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fail-policy", choices=("warn", "stop"), default="warn")
    return parser.parse_args()


def run(command: Sequence[str], log_path: Path) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print("SEGMENTED_EVAL_COMMAND=" + " ".join(command), flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n===== {datetime.now(timezone.utc).isoformat()} =====\n")
        log.write(" ".join(command) + "\n")
        log.flush()
        result = subprocess.run(
            list(command), cwd=PROJECT_ROOT, stdout=log,
            stderr=subprocess.STDOUT, check=False,
        )
    return int(result.returncode), time.time() - started


def status_id(checkpoint: Path, step: int, component: str) -> str:
    return hashlib.sha256(f"{checkpoint}|{step}|{component}".encode()).hexdigest()


def component_status_row(
    *, checkpoint: Path, step: int, component: str, log: Path, return_code: int
) -> dict[str, Any]:
    split = "heldout" if component == "heldout" else "external_ui5"
    return {
        "schema_version": 1,
        "evaluation_id": status_id(checkpoint, step, component),
        "evaluation_protocol_id": f"cpt-segmented-{component}-status-v1",
        "evaluation_kind": f"{component}_evaluation_status",
        "eligible_for_best_checkpoint": False,
        "checkpoint": str(checkpoint),
        "step": step,
        "split": split,
        "task": f"__{component}_status__",
        "evaluation_status": "complete" if return_code == 0 else "failed",
        "inference_error_count": None,
        "return_code": return_code,
        "log_path": str(log),
        "primary_metric": None,
    }


def evaluation_command(args: argparse.Namespace, checkpoint: Path, step: int) -> list[str]:
    command = [
        str(args.python), str(Path(__file__).resolve()),
        "--checkpoint", str(checkpoint), "--checkpoint-step", str(step),
        "--base-model", str(args.base_model),
        "--processor-path", str(args.processor_path or args.base_model),
        "--recipe", str(args.recipe), "--manifest", str(args.manifest),
        "--run-dir", str(args.run_dir),
        "--external-input-dir", str(args.external_input_dir),
        "--gpu-devices", args.gpu_devices,
        "--samples-per-task", str(args.samples_per_task),
        "--python", str(args.python), "--dtype", args.dtype,
        "--attn-implementation", args.attn_implementation,
        "--vision-attn-implementation", args.vision_attn_implementation,
        "--heldout-max-new-tokens", str(args.heldout_max_new_tokens),
        "--external-max-new-tokens", str(args.external_max_new_tokens),
        "--iou-thresholds", *(str(value) for value in args.iou_thresholds),
        "--seed", str(args.seed), "--fail-policy", args.fail_policy,
        "--no-repair-prior",
    ]
    command.append("--external-ui5" if args.external_ui5 else "--no-external-ui5")
    return command


def repair_prior(args: argparse.Namespace) -> list[dict[str, Any]]:
    repaired = []
    eval_root = args.run_dir / "eval"
    for status_path in sorted(eval_root.glob("checkpoint-*/segmented_eval_status.json")):
        try:
            value = json.loads(status_path.read_text(encoding="utf-8"))
            step = int(value["step"])
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
        if step >= args.checkpoint_step or value.get("success"):
            continue
        checkpoint = args.base_model if step == 0 else args.run_dir / f"checkpoint-{step}"
        if not checkpoint.is_dir():
            repaired.append({"step": step, "status": "missing_checkpoint"})
            continue
        print(f"SEGMENTED_EVAL_REPAIR_PRIOR=START step={step}", flush=True)
        code = subprocess.run(
            evaluation_command(args, checkpoint, step), cwd=PROJECT_ROOT, check=False
        ).returncode
        repaired.append({"step": step, "return_code": int(code)})
    return repaired


def main() -> int:
    args = parse_args()
    if args.checkpoint_step < 0 or args.samples_per_task <= 0:
        raise ValueError("invalid checkpoint step or samples-per-task")
    for name in (
        "checkpoint", "base_model", "recipe", "manifest", "run_dir", "external_input_dir"
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.processor_path is not None:
        args.processor_path = args.processor_path.expanduser().resolve()
    metrics_jsonl = args.run_dir / "diagnostics/cpt_eval_metrics.jsonl"
    step_dir = args.run_dir / "eval" / f"checkpoint-{args.checkpoint_step}"
    step_dir.mkdir(parents=True, exist_ok=True)
    repaired = repair_prior(args) if args.repair_prior else []

    heldout_log = step_dir / "heldout_parallel.log"
    heldout_command = [
        str(args.python), str(PROJECT_ROOT / "scripts/run_locany_cpt_parallel_eval.py"),
        "--checkpoint", str(args.checkpoint),
        "--checkpoint-step", str(args.checkpoint_step),
        "--base-model", str(args.base_model),
        "--processor-path", str(args.processor_path or args.base_model),
        "--recipe", str(args.recipe), "--manifest", str(args.manifest),
        "--run-dir", str(args.run_dir), "--output-dir", str(step_dir),
        "--gpu-devices", args.gpu_devices,
        "--samples-per-task", str(args.samples_per_task),
        "--python", str(args.python), "--metrics-jsonl", str(metrics_jsonl),
        "--train-metrics-jsonl", str(args.run_dir / "diagnostics/cpt_train_metrics.jsonl"),
        "--dtype", args.dtype, "--attn-implementation", args.attn_implementation,
        "--vision-attn-implementation", args.vision_attn_implementation,
        "--max-new-tokens", str(args.heldout_max_new_tokens),
        "--iou-threshold", "0.1", "--seed", str(args.seed),
        "--task-retries", "1",
    ]
    heldout_code, heldout_seconds = run(heldout_command, heldout_log)

    external_code = 0
    external_seconds = 0.0
    external_log = step_dir / "external_ui5_parallel.log"
    if args.external_ui5:
        external_command = [
            str(args.python), str(PROJECT_ROOT / "scripts/run_locany_cpt_external_ui5_eval.py"),
            "--checkpoint", str(args.checkpoint),
            "--checkpoint-step", str(args.checkpoint_step),
            "--base-model", str(args.base_model),
            "--processor-path", str(args.processor_path or args.base_model),
            "--run-dir", str(args.run_dir), "--input-dir", str(args.external_input_dir),
            "--gpu-devices", args.gpu_devices, "--python", str(args.python),
            "--metrics-jsonl", str(metrics_jsonl), "--dtype", args.dtype,
            "--attn-implementation", args.attn_implementation,
            "--vision-attn-implementation", args.vision_attn_implementation,
            "--generation-mode", "hybrid",
            "--max-new-tokens", str(args.external_max_new_tokens),
            "--seed", str(args.seed), "--iou-thresholds",
            *(str(value) for value in args.iou_thresholds), "--no-build-excel",
        ]
        external_code, external_seconds = run(external_command, external_log)

    failures = []
    component_rows = [
        component_status_row(
            checkpoint=args.checkpoint, step=args.checkpoint_step,
            component="heldout", log=heldout_log, return_code=heldout_code,
        )
    ]
    if heldout_code:
        failures.append("heldout")
    if args.external_ui5:
        component_rows.append(
            component_status_row(
                checkpoint=args.checkpoint, step=args.checkpoint_step,
                component="external_ui5", log=external_log, return_code=external_code,
            )
        )
    if external_code:
        failures.append("external_ui5")
    write_eval_rows(metrics_jsonl, component_rows)

    excel_log = step_dir / "excel.log"
    excel_command = [
        str(args.python), str(PROJECT_ROOT / "scripts/build_locany_cpt_excel.py"),
        "--diagnostics-dir", str(args.run_dir / "diagnostics"),
    ]
    excel_code, excel_seconds = run(excel_command, excel_log)
    # Excel is intentionally optional and never changes evaluation completeness.
    status = {
        "schema_version": 1,
        "step": args.checkpoint_step,
        "checkpoint": str(args.checkpoint),
        "gpu_devices": args.gpu_devices,
        "heldout": {
            "return_code": heldout_code, "wall_time_seconds": heldout_seconds,
            "log_path": str(heldout_log),
        },
        "external_ui5": {
            "enabled": args.external_ui5, "return_code": external_code,
            "wall_time_seconds": external_seconds, "log_path": str(external_log),
        },
        "excel": {
            "return_code": excel_code, "wall_time_seconds": excel_seconds,
            "log_path": str(excel_log),
        },
        "repaired_prior": repaired,
        "failed_components": failures,
        "success": not failures,
        "fail_policy": args.fail_policy,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(step_dir / "segmented_eval_status.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    return 1 if failures and args.fail_policy == "stop" else 0


if __name__ == "__main__":
    raise SystemExit(main())
