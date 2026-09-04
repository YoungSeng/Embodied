#!/usr/bin/env python3
"""Run one fail-closed UI5 curriculum evaluation node.

The parent process launches five independent, single-visible-GPU workers at
once.  Placement is fixed: occlusion/cropping run on the first GPU and the
remaining three tasks run on the second GPU.  Every worker uses the same
candidate, processor, generation, crop, NMS, and evaluator identity.  It also
re-evaluates the task's frozen 0/4 hard groups with four deterministic rollout
seeds without loading the model a second time.

The worker implementation is ``inference_ui_defect_locany.py``.  A different
worker/scorer script can be supplied for hermetic scheduling tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from ui5_lossless_tiling import assert_lossless_coverage


TASKS = (
    "occlusion",
    "cropping",
    "text_overflow",
    "text_ellipsis",
    "content_missing",
)
EXTERNAL_TASK_DIR = {task: f"ui_{task}" for task in TASKS}
TASK_GPU_SLOT = {
    "occlusion": 0,
    "cropping": 0,
    "text_overflow": 1,
    "text_ellipsis": 1,
    "content_missing": 1,
}
TASK_GT_FILE = {
    "occlusion": "test_ui_occlusion_wcnt_no_figma.jsonl",
    "cropping": "test_ui_cropping_wcnt_no_figma.jsonl",
    "text_overflow": "test_ui_text_overflow_wcnt_no_figma.jsonl",
    "text_ellipsis": "test_ui_text_ellipsis_wcnt_no_figma.jsonl",
    "content_missing": "test_ui_content_missing_wcnt_no_figma.jsonl",
}
CURRICULUM_PHASES = (
    (0.60, 0.25, 0.15, 1.0e-6),
    (0.45, 0.35, 0.20, 7.0e-7),
    (0.30, 0.30, 0.40, 5.0e-7),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", "--candidate-checkpoint", dest="checkpoint", type=Path, required=True
    )
    parser.add_argument("--processor-path", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument(
        "--total-steps", type=int, default=env_int("TOTAL_STEPS", 1200)
    )
    parser.add_argument("--hard-groups-jsonl", type=Path, required=True)
    parser.add_argument("--rollout-bundle-root", type=Path, required=True)
    parser.add_argument(
        "--expected-hard-groups",
        type=int,
        default=env_int("EXPECTED_HARD_GROUPS", 72),
    )
    parser.add_argument("--seed", type=int, default=env_int("SEED", 42))
    parser.add_argument("--gpu-devices", default="0,1")
    parser.add_argument(
        "--gpu0-workers", type=int, default=env_int("UI5_GPU0_WORKERS", 2)
    )
    parser.add_argument(
        "--gpu1-workers", type=int, default=env_int("UI5_GPU1_WORKERS", 3)
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--worker-script",
        type=Path,
        default=project_root / "scripts" / "inference_ui_defect_locany.py",
    )
    parser.add_argument(
        "--scorer-script",
        type=Path,
        default=project_root / "qwen3vl_merge_and_score_fixed_5tasks.py",
    )
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument(
        "--attn-implementation",
        choices=("auto", "sdpa", "flash_attention_2", "eager", "magi"),
        default="sdpa",
    )
    parser.add_argument(
        "--vision-attn-implementation",
        choices=("sdpa", "flash_attention_2", "eager"),
        default="flash_attention_2",
    )
    parser.add_argument(
        "--generation-mode", choices=("fast", "slow", "hybrid"), default="hybrid"
    )
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--n-future-tokens", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument(
        "--relation-gate-mode", choices=("observe", "hard"), default="observe"
    )
    parser.add_argument("--relation-gate-threshold", type=float, default=None)
    parser.add_argument(
        "--inference-crop-mode",
        choices=("full_image", "lossless_tiling", "detector_scan"),
        default="detector_scan",
    )
    parser.add_argument("--detector-crop-manifest", type=Path, default=None)
    parser.add_argument("--tile-max-count", type=int, default=10)
    parser.add_argument("--tile-target-long-side", type=int, default=1600)
    parser.add_argument("--tile-overlap-ratio", type=float, default=0.10)
    parser.add_argument("--tile-nms-iou", type=float, default=0.50)
    parser.add_argument("--evaluator-iou-threshold", type=float, default=0.10)
    parser.add_argument("--max-images-per-task", type=int, default=0)
    parser.add_argument("--score-run-name", default="ui5_score")
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=env_float("UI5_EVAL_HEARTBEAT_SECONDS", 30.0),
        help="worker barrier progress heartbeat interval (default: 30 seconds)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--fake-worker",
        action="store_true",
        help="Relax worker artifact validation for a hermetic fake worker test only",
    )
    return parser.parse_args(argv)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_inventory(path: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for item in sorted((candidate for candidate in path.rglob("*") if candidate.is_file())):
        stat = item.stat()
        entries.append(
            {
                "path": item.relative_to(path).as_posix(),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "path": str(path),
        "file_count": len(entries),
        "total_bytes": sum(row["size"] for row in entries),
        "inventory_sha256": hashlib.sha256(payload).hexdigest(),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            value["_source_line"] = line_number
            rows.append(value)
    return rows


def normalize_task(value: Any) -> str:
    task = str(value or "").strip()
    if task.startswith("ui_"):
        task = task[3:]
    if task not in TASKS:
        raise ValueError(f"unsupported hard-group task {value!r}")
    return task


def _safe_bundle_image(bundle_root: Path, image_relpath: Any) -> Path:
    relative = Path(str(image_relpath or ""))
    if not str(relative) or relative.is_absolute():
        raise ValueError(f"hard group image_relpath must be relative, got {image_relpath!r}")
    if not relative.parts or relative.parts[0] != "images":
        raise ValueError(
            f"hard group image_relpath must be under bundle images/: {image_relpath!r}"
        )
    resolved = (bundle_root / relative).resolve(strict=True)
    try:
        resolved.relative_to(bundle_root)
    except ValueError as exc:
        raise ValueError(f"hard group image escapes rollout bundle: {image_relpath}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"hard group image is not a file: {resolved}")
    return resolved


def _bundle_sample_source_ids(bundle_root: Path) -> dict[str, str]:
    path = bundle_root / "manifest" / "task_samples.jsonl"
    if not path.is_file():
        return {}
    result: dict[str, str] = {}
    for row in read_jsonl(path):
        source_id = row.get("source_image_id", row.get("image_id"))
        if source_id is None:
            continue
        for key in (row.get("record_id"), row.get("sample_id")):
            if key is not None:
                result[str(key)] = str(source_id)
    return result


def resolve_hard_groups(
    hard_path: Path,
    bundle_root: Path,
    expected_count: int,
) -> tuple[list[dict[str, Any]], dict[str, int], Path]:
    rows = read_jsonl(hard_path)
    if len(rows) != expected_count:
        raise ValueError(
            f"hard group count mismatch: expected exactly {expected_count}, got {len(rows)}"
        )
    plan_path = bundle_root / "base_scan_plans.json"
    if not plan_path.is_file():
        raise FileNotFoundError(f"rollout base scan plans do not exist: {plan_path}")
    plans = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(plans, dict):
        raise ValueError("base_scan_plans.json must be an object indexed by source_image_id")
    sample_source_ids = _bundle_sample_source_ids(bundle_root)
    seen: set[str] = set()
    counts = {task: 0 for task in TASKS}
    resolved_rows: list[dict[str, Any]] = []
    for row in rows:
        line = row.pop("_source_line")
        record_id = str(row.get("record_id") or "").strip()
        sample_id = str(row.get("sample_id") or "").strip()
        if not record_id or not sample_id:
            raise ValueError(f"hard group line {line} lacks record_id/sample_id")
        if record_id in seen:
            raise ValueError(f"duplicate hard group record_id={record_id!r}")
        seen.add(record_id)
        task = normalize_task(row.get("task"))
        if str(row.get("task")) != task:
            raise ValueError(
                f"hard group {record_id} task must use the unprefixed canonical name {task!r}"
            )
        if row.get("crop_complete4") is not True:
            raise ValueError(f"hard group {record_id} is not crop_complete4=true")
        if int(row.get("crop_correct_count", -1)) != 0:
            raise ValueError(f"hard group {record_id} is not a crop 0/4 group")
        prompt = str(row.get("prompt") or "").strip()
        if not prompt:
            raise ValueError(f"hard group {record_id} has no prompt")
        gt_global = row.get("gt_global")
        if not isinstance(gt_global, list) or any(
            not isinstance(box, list) or len(box) != 4 for box in gt_global
        ):
            raise ValueError(f"hard group {record_id} has invalid gt_global")
        if bool(row.get("positive")) != bool(gt_global):
            raise ValueError(f"hard group {record_id} positive flag disagrees with gt_global")
        image_path = _safe_bundle_image(bundle_root, row.get("image_relpath"))
        source_image_id = row.get("source_image_id", row.get("image_id"))
        if source_image_id is None:
            source_image_id = sample_source_ids.get(record_id) or sample_source_ids.get(sample_id)
        base_tiles: list[list[int]] | None = None
        plan_digest: str | None = None
        plan_width: int | None = None
        plan_height: int | None = None
        if task != "content_missing":
            if source_image_id is None or str(source_image_id) not in plans:
                raise ValueError(
                    f"hard group {record_id} has no source_image_id/base scan plan"
                )
            plan = plans[str(source_image_id)]
            if not isinstance(plan, dict) or plan.get("gt_used") is not False:
                raise ValueError(f"hard group {record_id} base scan plan is not GT-free")
            candidate_tiles = plan.get("base_tiles", plan.get("tiles"))
            if not isinstance(candidate_tiles, list) or not candidate_tiles:
                raise ValueError(f"hard group {record_id} base scan plan has no tiles")
            base_tiles = []
            for tile in candidate_tiles:
                if not isinstance(tile, list) or len(tile) != 4:
                    raise ValueError(f"hard group {record_id} has malformed base tile")
                normalized = [int(value) for value in tile]
                if normalized[2] <= normalized[0] or normalized[3] <= normalized[1]:
                    raise ValueError(f"hard group {record_id} has empty base tile {tile}")
                base_tiles.append(normalized)
            plan_width = int(plan["width"])
            plan_height = int(plan["height"])
            assert_lossless_coverage(plan_width, plan_height, base_tiles)
            plan_digest = str(plan.get("geometry_digest") or "")
        counts[task] += 1
        resolved_rows.append(
            {
                **row,
                "record_id": record_id,
                "sample_id": sample_id,
                "task": task,
                "source_image_id": source_image_id,
                "_resolved_image_path": str(image_path),
                "_base_tiles": base_tiles,
                "_base_plan_width": plan_width,
                "_base_plan_height": plan_height,
                "_base_plan_digest": plan_digest,
            }
        )
    resolved_rows.sort(key=lambda row: (TASKS.index(str(row["task"])), str(row["record_id"])))
    return resolved_rows, counts, plan_path


def parse_gpu_devices(value: str) -> tuple[str, str]:
    devices = tuple(item.strip() for item in value.split(",") if item.strip())
    if len(devices) != 2 or len(set(devices)) != 2:
        raise ValueError(
            f"--gpu-devices must contain exactly two distinct physical IDs, got {value!r}"
        )
    return devices[0], devices[1]


def _input_image_paths(path: Path, limit: int) -> list[str]:
    images: list[str] = []
    missing: list[str] = []
    for row in read_jsonl(path):
        raw = row.get("images", row.get("image"))
        candidates = raw if isinstance(raw, list) else [raw]
        for candidate in candidates:
            raw_path = (
                candidate
                if isinstance(candidate, str)
                else candidate.get("path") if isinstance(candidate, dict) else None
            )
            if not isinstance(raw_path, str) or not raw_path:
                raise ValueError(f"UI5 row in {path} has no valid image path")
            image_path = Path(raw_path).expanduser()
            if not image_path.is_absolute():
                image_path = path.parent / image_path
            image_path = image_path.resolve(strict=False)
            if not image_path.is_file():
                missing.append(str(image_path))
                continue
            if ":" not in image_path.name:
                images.append(str(image_path))
    if missing:
        preview = ", ".join(missing[:3])
        raise FileNotFoundError(
            f"UI5 input {path} references {len(missing)} missing images; first: {preview}"
        )
    unique = list(dict.fromkeys(images))
    return unique[:limit] if limit else unique


def _input_image_count(path: Path, limit: int) -> int:
    return len(_input_image_paths(path, limit))


def _expected_output_stems(image_paths: Sequence[str]) -> set[str]:
    grouped: dict[str, list[str]] = {}
    for image_path in image_paths:
        base = Path(image_path).stem.replace(":", "_")
        grouped.setdefault(base, []).append(image_path)
    stems: set[str] = set()
    for base, paths in grouped.items():
        if len(paths) == 1:
            stems.add(base)
            continue
        for image_path in paths:
            digest = hashlib.blake2b(
                image_path.encode("utf-8"), digest_size=5
            ).hexdigest()
            stems.add(f"{base}__{digest}")
    return stems


def clean_owned_outputs(output_dir: Path, score_run_name: str = "ui5_score") -> None:
    for name in (
        *EXTERNAL_TASK_DIR.values(),
        "_worker_logs",
        "_worker_summaries",
        "_runtime_cache",
        score_run_name,
    ):
        path = output_dir / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    for name in (
        "evaluation_manifest.json",
        "evaluation_status.json",
        "resolved_hard_groups.jsonl",
        "ui5_metrics.json",
        "hard_transition.jsonl",
        "hard_rollout4_summary.json",
        "_run_manifest.json",
    ):
        (output_dir / name).unlink(missing_ok=True)


@dataclass(frozen=True)
class WorkerSpec:
    task: str
    physical_gpu: str
    output_dir: Path
    summary_path: Path
    log_path: Path
    command: tuple[str, ...]


def build_worker_specs(
    args: argparse.Namespace,
    devices: tuple[str, str],
    hard_counts: Mapping[str, int],
    resolved_hard_path: Path,
    identity_path: Path,
) -> list[WorkerSpec]:
    specs: list[WorkerSpec] = []
    for task in TASKS:
        gpu = devices[TASK_GPU_SLOT[task]]
        task_output = args.output_dir / EXTERNAL_TASK_DIR[task]
        summary = args.output_dir / "_worker_summaries" / f"{task}.json"
        log = args.output_dir / "_worker_logs" / f"{task}.log"
        command = [
            args.python,
            str(args.worker_script),
            "--checkpoint",
            str(args.checkpoint),
            "--processor-path",
            str(args.processor_path),
            "--input-dir",
            str(args.input_dir),
            "--output-dir",
            str(args.output_dir),
            "--single-task-output-dir",
            str(task_output),
            "--summary-path",
            str(summary),
            "--cuda-visible-devices",
            gpu,
            "--device",
            "cuda:0",
            "--dtype",
            args.dtype,
            "--attn-implementation",
            args.attn_implementation,
            "--vision-attn-implementation",
            args.vision_attn_implementation,
            "--generation-mode",
            args.generation_mode,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--n-future-tokens",
            str(args.n_future_tokens),
            "--temperature",
            str(args.temperature),
            "--top-p",
            str(args.top_p),
            "--top-k",
            str(args.top_k),
            "--repetition-penalty",
            str(args.repetition_penalty),
            "--seed",
            str(args.seed),
            "--tasks",
            task,
            "--skip-figma",
            "--fail-fast",
            "--local-files-only",
            "--trust-remote-code",
            "--relation-gate-mode",
            args.relation_gate_mode,
            "--inference-crop-mode",
            args.inference_crop_mode,
            "--tile-max-count",
            str(args.tile_max_count),
            "--tile-target-long-side",
            str(args.tile_target_long_side),
            "--tile-overlap-ratio",
            str(args.tile_overlap_ratio),
            "--tile-nms-iou",
            str(args.tile_nms_iou),
            "--hard-groups-jsonl",
            str(resolved_hard_path),
            "--rollout-bundle-root",
            str(args.rollout_bundle_root),
            "--hard-rollout-seeds",
            *(str(args.seed + index) for index in range(4)),
            "--hard-rollout-output-dir",
            str(task_output / "rollout4"),
            "--expected-hard-task-count",
            str(hard_counts[task]),
            "--rollout-scorer-script",
            str(args.scorer_script),
            "--hard-rollout-iou-threshold",
            str(args.evaluator_iou_threshold),
            "--evaluation-identity-file",
            str(identity_path),
        ]
        if args.greedy:
            command.append("--greedy")
        if args.relation_gate_threshold is not None:
            command.extend(["--relation-gate-threshold", str(args.relation_gate_threshold)])
        if args.inference_crop_mode == "detector_scan":
            command.extend(
                ["--detector-crop-manifest", str(args.detector_crop_manifest), "--save-raw-answer"]
            )
        if args.max_images_per_task:
            command.extend(["--max-images-per-task", str(args.max_images_per_task)])
        if args.overwrite:
            command.append("--overwrite")
        specs.append(
            WorkerSpec(
                task=task,
                physical_gpu=gpu,
                output_dir=task_output,
                summary_path=summary,
                log_path=log,
                command=tuple(command),
            )
        )
    return specs


def launch_workers(
    specs: Sequence[WorkerSpec],
    *,
    project_root: Path,
    dry_run: bool = False,
    heartbeat_seconds: float = 30.0,
    popen_factory: Any = subprocess.Popen,
) -> list[dict[str, Any]]:
    if dry_run:
        return [
            {
                "task": spec.task,
                "physical_gpu": spec.physical_gpu,
                "logical_device": "cuda:0",
                "pid": None,
                "return_code": 0,
                "command": list(spec.command),
                "log_path": str(spec.log_path),
            }
            for spec in specs
        ]
    running: list[tuple[WorkerSpec, Any, Any, float]] = []
    try:
        # Do not wait in this loop: all five model processes must coexist.
        for spec in specs:
            spec.log_path.parent.mkdir(parents=True, exist_ok=True)
            spec.summary_path.parent.mkdir(parents=True, exist_ok=True)
            cache_root = spec.log_path.parents[1] / "_runtime_cache" / spec.task
            hf_cache = cache_root / "hf_modules"
            pycache = cache_root / "pycache"
            hf_cache.mkdir(parents=True, exist_ok=True)
            pycache.mkdir(parents=True, exist_ok=True)
            child_env = dict(os.environ)
            child_env.update(
                {
                    "CUDA_VISIBLE_DEVICES": spec.physical_gpu,
                    "HF_MODULES_CACHE": str(hf_cache),
                    "PYTHONPYCACHEPREFIX": str(pycache),
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                    "PYTHONUNBUFFERED": "1",
                    "TOKENIZERS_PARALLELISM": "false",
                }
            )
            log_handle = spec.log_path.open("a", encoding="utf-8")
            log_handle.write(
                f"\n===== {utc_now()} task={spec.task} physical_gpu={spec.physical_gpu} =====\n"
            )
            log_handle.write(" ".join(spec.command) + "\n")
            log_handle.flush()
            process = popen_factory(
                list(spec.command),
                cwd=str(project_root),
                env=child_env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            running.append((spec, process, log_handle, time.monotonic()))
            print(
                f"[UI5 START] task={spec.task} gpu={spec.physical_gpu} "
                f"logical=cuda:0 pid={getattr(process, 'pid', None)}",
                flush=True,
            )
    except BaseException:
        for _, process, handle, _ in running:
            try:
                process.terminate()
            except Exception:
                pass
            handle.close()
        for _, process, _, _ in running:
            try:
                process.wait(timeout=30)
            except Exception:
                pass
        raise

    results_by_task: dict[str, dict[str, Any]] = {}
    try:
        # Poll all workers together.  This preserves the all-started barrier and
        # avoids hiding progress behind a long blocking wait on the first task.
        pending = {spec.task for spec, _, _, _ in running}
        next_heartbeat = time.monotonic() + heartbeat_seconds
        while pending:
            now = time.monotonic()
            for spec, process, _, started in running:
                if spec.task not in pending:
                    continue
                return_code = process.poll()
                if return_code is None:
                    continue
                return_code = int(process.wait())
                elapsed = time.monotonic() - started
                results_by_task[spec.task] = {
                    "task": spec.task,
                    "physical_gpu": spec.physical_gpu,
                    "logical_device": "cuda:0",
                    "pid": getattr(process, "pid", None),
                    "return_code": return_code,
                    "elapsed_seconds": round(elapsed, 6),
                    "command": list(spec.command),
                    "log_path": str(spec.log_path),
                    "summary_path": str(spec.summary_path),
                }
                pending.remove(spec.task)
                print(
                    f"[UI5 DONE] task={spec.task} gpu={spec.physical_gpu} "
                    f"pid={getattr(process, 'pid', None)} rc={return_code} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )
            now = time.monotonic()
            if pending and now >= next_heartbeat:
                workers = []
                for spec, process, _, started in running:
                    result = results_by_task.get(spec.task)
                    workers.append(
                        {
                            "task": spec.task,
                            "gpu": spec.physical_gpu,
                            "pid": getattr(process, "pid", None),
                            "state": "done" if result is not None else "running",
                            "rc": result["return_code"] if result is not None else None,
                            "elapsed_seconds": round(now - started, 1),
                            "summary": (
                                "present" if spec.summary_path.is_file() else "pending"
                            ),
                        }
                    )
                heartbeat = {
                    "event": "ui5_worker_heartbeat",
                    "elapsed_seconds": round(
                        now - min(started for _, _, _, started in running), 1
                    ),
                    "workers": workers,
                }
                print(
                    "[UI5 HEARTBEAT] "
                    + json.dumps(heartbeat, ensure_ascii=False, separators=(",", ":")),
                    flush=True,
                )
                next_heartbeat = now + heartbeat_seconds
            if pending:
                time.sleep(min(0.25, max(0.01, next_heartbeat - time.monotonic())))
    except BaseException:
        for _, process, _, _ in running:
            try:
                process.terminate()
            except Exception:
                pass
        for _, process, _, _ in running:
            try:
                process.wait(timeout=30)
            except Exception:
                pass
        raise
    finally:
        for _, _, log_handle, _ in running:
            log_handle.close()
    return [results_by_task[spec.task] for spec in specs]


def tail(path: Path, lines: int = 80) -> str:
    if not path.is_file():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def validate_worker_results(
    specs: Sequence[WorkerSpec],
    results: Sequence[Mapping[str, Any]],
    *,
    expected_images: Mapping[str, int],
    expected_stems: Mapping[str, set[str]],
    expected_hard: Mapping[str, int],
    expected_hard_ids: Mapping[str, set[str]],
    rollout_seeds: Sequence[int],
    identity_digest: str,
    fake_worker: bool,
) -> None:
    result_by_task = {str(row["task"]): row for row in results}
    if set(result_by_task) != set(TASKS):
        raise RuntimeError(f"worker barrier is incomplete: {sorted(result_by_task)}")
    failures = [row for row in results if int(row["return_code"]) != 0]
    if failures:
        details = []
        for row in failures:
            log_path = Path(str(row["log_path"]))
            details.append(
                f"task={row['task']} gpu={row['physical_gpu']} rc={row['return_code']} "
                f"log={log_path}\n{tail(log_path)}"
            )
        raise RuntimeError("one or more UI5 workers failed; scoring is blocked:\n" + "\n".join(details))
    for spec in specs:
        if not spec.summary_path.is_file():
            raise RuntimeError(f"worker summary is missing: {spec.summary_path}")
        summary = json.loads(spec.summary_path.read_text(encoding="utf-8"))
        if fake_worker:
            continue
        if summary.get("evaluation_identity_digest") != identity_digest:
            raise RuntimeError(f"worker {spec.task} used a different evaluation identity")
        task_rows = summary.get("tasks") or []
        reported_task = (
            task_rows[0].get("task_name", task_rows[0].get("task"))
            if len(task_rows) == 1
            else None
        )
        if len(task_rows) != 1 or reported_task != spec.task:
            raise RuntimeError(f"worker {spec.task} summary has wrong task ownership")
        stats = task_rows[0]
        if int(stats.get("dataset_images", -1)) != expected_images[spec.task]:
            raise RuntimeError(f"worker {spec.task} dataset image count changed")
        if int(stats.get("processed", 0)) + int(stats.get("skipped_existing", 0)) != expected_images[spec.task]:
            raise RuntimeError(f"worker {spec.task} did not cover every evaluation image")
        if int(stats.get("inference_error", -1)) != 0:
            raise RuntimeError(f"worker {spec.task} reported inference errors")
        actual_prediction_files = {
            path.name: path for path in spec.output_dir.glob("*.json")
        }
        matched_prediction_files: set[str] = set()
        for stem in expected_stems[spec.task]:
            candidates = [
                actual_prediction_files[name]
                for name in (f"{stem}.json", f"{stem}_parse_error.json")
                if name in actual_prediction_files
            ]
            if len(candidates) != 1:
                raise RuntimeError(
                    f"worker {spec.task} has {len(candidates)} predictions for {stem!r}"
                )
            prediction_path = candidates[0]
            try:
                prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"worker {spec.task} produced invalid JSON: {prediction_path}"
                ) from exc
            if prediction_path.name.endswith("_parse_error.json"):
                if prediction is not None:
                    raise RuntimeError(
                        f"worker {spec.task} parse-error output must be JSON null: "
                        f"{prediction_path}"
                    )
            elif not isinstance(prediction, list):
                raise RuntimeError(
                    f"worker {spec.task} detection output must be a JSON list: "
                    f"{prediction_path}"
                )
            matched_prediction_files.add(prediction_path.name)
        if set(actual_prediction_files) != matched_prediction_files:
            raise RuntimeError(
                f"worker {spec.task} prediction file set mismatch: "
                f"unexpected={sorted(set(actual_prediction_files) - matched_prediction_files)}"
            )
        hard = summary.get("hard_rollout") or {}
        if int(hard.get("group_count", -1)) != expected_hard[spec.task]:
            raise RuntimeError(f"worker {spec.task} hard group count changed")
        if int(hard.get("attempt_count", -1)) != expected_hard[spec.task] * 4:
            raise RuntimeError(f"worker {spec.task} did not finish four hard rollouts")
        runtime_error_count = int(hard.get("runtime_error_count", -1))
        error_count = int(hard.get("error_count", -1))
        if runtime_error_count != 0 or error_count != runtime_error_count:
            raise RuntimeError(
                f"worker {spec.task} hard rollout reported runtime errors"
            )
        parse_error_count = int(hard.get("parse_error_count", -1))
        if not 0 <= parse_error_count <= expected_hard[spec.task] * 4:
            raise RuntimeError(
                f"worker {spec.task} hard rollout parse-error count is invalid"
            )
        rollout_root = spec.output_dir / "rollout4"
        observed_parse_errors = 0
        for rollout_id in range(4):
            path = rollout_root / f"rollout_{rollout_id}.jsonl"
            if not path.is_file():
                raise RuntimeError(f"worker {spec.task} incomplete hard rollout file: {path}")
            rows = read_jsonl(path)
            record_ids = [str(row.get("record_id") or "") for row in rows]
            if (
                len(rows) != expected_hard[spec.task]
                or len(record_ids) != len(set(record_ids))
                or set(record_ids) != expected_hard_ids[spec.task]
            ):
                raise RuntimeError(
                    f"worker {spec.task} hard rollout record set differs: {path}"
                )
            for row in rows:
                if (
                    row.get("task") != spec.task
                    or int(row.get("rollout_id", -1)) != rollout_id
                    or int(row.get("rollout_seed", -1)) != int(rollout_seeds[rollout_id])
                    or row.get("evaluation_identity_digest") != identity_digest
                    or row.get("runtime_error") is not None
                    or not isinstance(row.get("exact_correct"), bool)
                ):
                    raise RuntimeError(
                        f"worker {spec.task} hard rollout row identity is invalid: {path}"
                    )
                observed_parse_errors += int(row.get("parse_status") == "parse_error")
        if observed_parse_errors != parse_error_count:
            raise RuntimeError(
                f"worker {spec.task} hard rollout parse-error count differs"
            )


def validate_identity_unchanged(args: argparse.Namespace, identity: Mapping[str, Any]) -> None:
    if directory_inventory(args.checkpoint) != identity["candidate"]:
        raise RuntimeError("candidate checkpoint changed while UI5 workers were running")
    if directory_inventory(args.processor_path) != identity["processor"]:
        raise RuntimeError("processor/tokenizer changed while UI5 workers were running")
    immutable_files = (
        (args.worker_script, identity["worker_script"]["sha256"], "worker script"),
        (
            Path(identity["crop_and_merge"]["implementation_path"]),
            identity["crop_and_merge"]["implementation_sha256"],
            "crop/remap/NMS implementation",
        ),
        (args.scorer_script, identity["evaluator"]["sha256"], "evaluator"),
        (
            args.hard_groups_jsonl,
            identity["hard_rollout"]["source_sha256"],
            "hard group source",
        ),
        (
            Path(identity["hard_rollout"]["resolved_source"]),
            identity["hard_rollout"]["resolved_source_sha256"],
            "resolved hard group source",
        ),
        (
            args.rollout_bundle_root / "base_scan_plans.json",
            identity["hard_rollout"]["base_scan_plans_sha256"],
            "hard rollout base plans",
        ),
    )
    for path, expected_digest, label in immutable_files:
        if file_sha256(Path(path)) != expected_digest:
            raise RuntimeError(f"{label} changed while UI5 workers were running")
    detector = args.detector_crop_manifest
    if detector is not None and file_sha256(detector) != identity["crop_and_merge"][
        "detector_crop_manifest_sha256"
    ]:
        raise RuntimeError("detector crop manifest changed while UI5 workers were running")


def merge_hard_rollout_outputs(
    output_dir: Path,
    expected_hard: Mapping[str, int],
    expected_hard_ids: Mapping[str, set[str]],
    identity_digest: str,
) -> dict[str, Any]:
    merged_groups: list[dict[str, Any]] = []
    task_summaries: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    for task in TASKS:
        rollout_root = output_dir / EXTERNAL_TASK_DIR[task] / "rollout4"
        summary_path = rollout_root / "rollout4_summary.json"
        groups_path = rollout_root / "groups.jsonl"
        if not summary_path.is_file() or not groups_path.is_file():
            raise RuntimeError(f"hard rollout aggregate inputs are missing for {task}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        groups = read_jsonl(groups_path)
        if summary.get("evaluation_identity_digest") != identity_digest:
            raise RuntimeError(f"hard rollout summary identity differs for {task}")
        if int(summary.get("group_count", -1)) != expected_hard[task]:
            raise RuntimeError(f"hard rollout summary count differs for {task}")
        if len(groups) != expected_hard[task]:
            raise RuntimeError(f"hard rollout groups.jsonl count differs for {task}")
        group_ids = {str(row.get("record_id") or "") for row in groups}
        if group_ids != expected_hard_ids[task]:
            raise RuntimeError(f"hard rollout authoritative group set differs for {task}")
        for row in groups:
            row.pop("_source_line", None)
            record_id = str(row.get("record_id") or "")
            if row.get("task") != task or not record_id or record_id in seen:
                raise RuntimeError(f"invalid/duplicate merged hard group {record_id!r}")
            seen.add(record_id)
            merged_groups.append(row)
        task_summaries[task] = summary
    if len(merged_groups) != sum(expected_hard.values()):
        raise RuntimeError("merged hard rollout group count is incomplete")
    merged_groups.sort(
        key=lambda row: (TASKS.index(str(row["task"])), str(row["record_id"]))
    )
    transition_path = output_dir / "hard_transition.jsonl"
    atomic_write_jsonl(transition_path, merged_groups)
    summary = {
        "schema_version": 1,
        "evaluation_identity_digest": identity_digest,
        "group_count": len(merged_groups),
        "attempt_count": len(merged_groups) * 4,
        "error_count": sum(
            int(row.get("error_count", row.get("runtime_error_count", 0)))
            for row in task_summaries.values()
        ),
        "runtime_error_count": sum(
            int(row.get("runtime_error_count", row.get("error_count", 0)))
            for row in task_summaries.values()
        ),
        "parse_error_count": sum(
            int(row.get("parse_error_count", 0))
            for row in task_summaries.values()
        ),
        "groups_improved": sum(row.get("transition") == "improved" for row in merged_groups),
        "groups_still_hard": sum(
            row.get("transition") == "still_hard" for row in merged_groups
        ),
        "correct_count_distribution": {
            str(count): sum(
                int(row.get("candidate_correct_count", -1)) == count
                for row in merged_groups
            )
            for count in range(5)
        },
        "by_task": task_summaries,
        "groups_path": str(transition_path),
        "finished_at": utc_now(),
    }
    atomic_write_json(output_dir / "hard_rollout4_summary.json", summary)
    return summary


@contextmanager
def scorer_input_view(output_dir: Path) -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix=".ui5-scorer-input-", dir=output_dir) as temporary:
        root = Path(temporary)
        for task in TASKS:
            source = output_dir / EXTERNAL_TASK_DIR[task]
            destination = root / task
            destination.mkdir()
            for item in source.glob("*.json"):
                target = destination / item.name
                try:
                    os.link(item, target)
                except OSError:
                    shutil.copy2(item, target)
        yield root


def _safe_prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def enrich_metrics(raw: Mapping[str, Any]) -> dict[str, Any]:
    tasks = raw.get("tasks")
    if not isinstance(tasks, dict) or set(tasks) != set(TASKS):
        raise RuntimeError("canonical evaluator did not return all five task summaries")
    image_counts = {
        key: sum(int(tasks[task]["image"][key]) for task in TASKS)
        for key in ("tp", "fp", "fn", "tn")
    }
    bbox_counts = {
        key: sum(int(tasks[task]["bbox"][key]) for task in TASKS)
        for key in ("tp", "fp", "fn")
    }
    image_micro = {**_safe_prf(image_counts["tp"], image_counts["fp"], image_counts["fn"]), **image_counts}
    image_total = sum(image_counts.values())
    image_micro["accuracy"] = (
        (image_counts["tp"] + image_counts["tn"]) / image_total if image_total else 0.0
    )
    bbox_micro = {**_safe_prf(bbox_counts["tp"], bbox_counts["fp"], bbox_counts["fn"]), **bbox_counts}
    task_image_counts = {
        task: sum(int(tasks[task]["image"][key]) for key in ("tp", "fp", "fn", "tn"))
        for task in TASKS
    }
    bbox_accuracy_terms = [
        (float(tasks[task]["bbox"]["count_accuracy"]), task_image_counts[task])
        for task in TASKS
        if tasks[task]["bbox"].get("count_accuracy") is not None
    ]
    bbox_accuracy_weight = sum(weight for _, weight in bbox_accuracy_terms)
    bbox_micro["count_accuracy"] = (
        sum(value * weight for value, weight in bbox_accuracy_terms)
        / bbox_accuracy_weight
        if bbox_accuracy_weight
        else None
    )
    macro = {
        granularity: dict((raw.get("macro") or {}).get(granularity) or {})
        for granularity in ("image", "bbox")
    }
    macro["image"]["accuracy"] = sum(
        float(tasks[task]["image"]["accuracy"]) for task in TASKS
    ) / len(TASKS)
    macro["bbox"]["count_accuracy"] = sum(
        float(tasks[task]["bbox"]["count_accuracy"])
        for task in TASKS
        if tasks[task]["bbox"].get("count_accuracy") is not None
    ) / max(
        1,
        sum(
            tasks[task]["bbox"].get("count_accuracy") is not None for task in TASKS
        ),
    )
    image_macro_f1 = float(macro.get("image", {}).get("f1"))
    bbox_macro_f1 = float(macro.get("bbox", {}).get("f1"))
    return {
        **dict(raw),
        "schema_version": 2,
        "macro": macro,
        "micro": {"image": image_micro, "bbox": bbox_micro},
        "overall": {
            "image_macro_f1": image_macro_f1,
            "bbox_macro_f1": bbox_macro_f1,
            "joint_score": (image_macro_f1 + bbox_macro_f1) / 2.0,
        },
    }


def evaluation_status_payload(
    *, args: argparse.Namespace, metrics: Mapping[str, Any], evaluation_seconds: float
) -> dict[str, Any]:
    step = args.step
    if step == 0:
        phase: int | str | None = "baseline"
        profile = CURRICULUM_PHASES[0]
    elif step is None:
        phase = None
        profile = None
    else:
        width = args.total_steps // len(CURRICULUM_PHASES)
        phase_index = min((max(1, step) - 1) // width, len(CURRICULUM_PHASES) - 1)
        phase = phase_index + 1
        profile = CURRICULUM_PHASES[phase_index]
    curriculum_target = (
        {
            "hard_ratio": profile[0],
            "anchor_ratio": profile[1],
            "global_replay_ratio": profile[2],
            "llm_lr": profile[3],
        }
        if profile is not None
        else None
    )
    return {
        "event": "ui5_evaluation_complete",
        "step": step,
        "phase": phase,
        "curriculum_target": curriculum_target,
        "tasks": {
            f"ui_{task}": {
                "image_f1": float(metrics["tasks"][task]["image"]["f1"]),
                "bbox_f1": float(metrics["tasks"][task]["bbox"]["f1"]),
            }
            for task in TASKS
        },
        "macro": {
            "image_f1": float(metrics["macro"]["image"]["f1"]),
            "bbox_f1": float(metrics["macro"]["bbox"]["f1"]),
        },
        "micro": {
            "image_f1": float(metrics["micro"]["image"]["f1"]),
            "bbox_f1": float(metrics["micro"]["bbox"]["f1"]),
        },
        "joint_score": float(metrics["overall"]["joint_score"]),
        "evaluation_seconds": float(evaluation_seconds),
        "next_action": "register_metrics_update_excel_and_best_checkpoint",
    }


def print_evaluation_status(payload: Mapping[str, Any]) -> None:
    for task, values in payload["tasks"].items():
        print(
            f"[UI5 TASK METRICS] step={payload['step']} task={task} "
            f"image_f1={values['image_f1']:.8f} bbox_f1={values['bbox_f1']:.8f}",
            flush=True,
        )
    print(
        f"[UI5 AGGREGATE] step={payload['step']} "
        f"image_macro_f1={payload['macro']['image_f1']:.8f} "
        f"bbox_macro_f1={payload['macro']['bbox_f1']:.8f} "
        f"image_micro_f1={payload['micro']['image_f1']:.8f} "
        f"bbox_micro_f1={payload['micro']['bbox_f1']:.8f} "
        f"joint_score={payload['joint_score']:.8f} "
        f"evaluation_seconds={payload['evaluation_seconds']:.1f} "
        f"next={payload['next_action']}",
        flush=True,
    )
    print(
        "[CURRICULUM STATUS] "
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        flush=True,
    )


def run_scorer(args: argparse.Namespace) -> tuple[Path, dict[str, Any], list[str]]:
    score_root = args.output_dir / args.score_run_name
    if score_root.exists():
        raise FileExistsError(f"score output already exists: {score_root}")
    log_path = args.output_dir / "_worker_logs" / "merge_and_score.log"
    with scorer_input_view(args.output_dir) as pred_root:
        command = [
            args.python,
            str(args.scorer_script),
            "--all_tasks",
            "--input_mode",
            "yolo_dir",
            "--gt_dir",
            str(args.input_dir),
            "--pred_root",
            str(pred_root),
            "--output_root",
            str(args.output_dir),
            "--run_name",
            args.score_run_name,
            "--yolo_bbox_format",
            "xyxy",
            "--iou_thresh",
            str(args.evaluator_iou_threshold),
        ]
        with log_path.open("a", encoding="utf-8") as log_handle:
            completed = subprocess.run(
                command,
                cwd=str(args.scorer_script.parent),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"canonical UI5 merge/scorer failed rc={completed.returncode}: {log_path}\n{tail(log_path)}"
            )
    metrics_path = score_root / "all_tasks_evaluation.json"
    if not metrics_path.is_file():
        raise RuntimeError(f"canonical evaluator metrics are missing: {metrics_path}")
    raw = json.loads(metrics_path.read_text(encoding="utf-8"))
    enriched = enrich_metrics(raw)
    final_path = args.output_dir / "ui5_metrics.json"
    atomic_write_json(final_path, enriched)
    return final_path, enriched, command


def run(args: argparse.Namespace) -> int:
    started = time.monotonic()
    args.project_root = args.project_root.expanduser().resolve(strict=True)
    args.checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    args.processor_path = args.processor_path.expanduser().resolve(strict=True)
    args.input_dir = args.input_dir.expanduser().resolve(strict=True)
    args.hard_groups_jsonl = args.hard_groups_jsonl.expanduser().resolve(strict=True)
    args.rollout_bundle_root = args.rollout_bundle_root.expanduser().resolve(strict=True)
    args.worker_script = args.worker_script.expanduser().resolve(strict=True)
    args.scorer_script = args.scorer_script.expanduser().resolve(strict=True)
    args.output_dir = args.output_dir.expanduser().resolve(strict=False)
    output_step_match = re.fullmatch(r"step-(\d+)", args.output_dir.name)
    output_step = int(output_step_match.group(1)) if output_step_match else None
    if args.step is None:
        args.step = output_step
    elif output_step is not None and args.step != output_step:
        raise ValueError(
            f"--step={args.step} disagrees with output directory {args.output_dir.name}"
        )
    if args.step is not None and args.step < 0:
        raise ValueError("--step cannot be negative")
    if args.total_steps <= 0 or args.total_steps % len(CURRICULUM_PHASES):
        raise ValueError("--total-steps must be positive and divisible by three")
    if args.step is not None and args.step > args.total_steps:
        raise ValueError("--step cannot exceed --total-steps")
    if not args.checkpoint.is_dir() or not args.processor_path.is_dir() or not args.input_dir.is_dir():
        raise ValueError("checkpoint, processor path, and input directory must be directories")
    if not args.rollout_bundle_root.is_dir():
        raise ValueError("--rollout-bundle-root must be a directory")
    if args.expected_hard_groups <= 0:
        raise ValueError("--expected-hard-groups must be positive")
    if not args.heartbeat_seconds > 0:
        raise ValueError("--heartbeat-seconds must be positive")
    if (
        not args.score_run_name
        or Path(args.score_run_name).name != args.score_run_name
        or args.score_run_name in {".", ".."}
    ):
        raise ValueError("--score-run-name must be one safe directory name")
    if (args.gpu0_workers, args.gpu1_workers) != (2, 3):
        raise ValueError(
            "formal UI5 placement requires UI5_GPU0_WORKERS=2 and UI5_GPU1_WORKERS=3"
        )
    devices = parse_gpu_devices(args.gpu_devices)
    if args.inference_crop_mode == "detector_scan":
        if args.detector_crop_manifest is None:
            raise ValueError("detector_scan requires --detector-crop-manifest")
        args.detector_crop_manifest = args.detector_crop_manifest.expanduser().resolve(strict=True)
    elif args.detector_crop_manifest is not None:
        raise ValueError("--detector-crop-manifest is only valid with detector_scan")

    if args.output_dir.exists() and args.overwrite:
        clean_owned_outputs(args.output_dir, args.score_run_name)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    owned = [args.output_dir / name for name in EXTERNAL_TASK_DIR.values()]
    if any(path.exists() for path in owned):
        raise FileExistsError(
            "one or more task output directories already exist; use a new step directory "
            "or --overwrite"
        )

    resolved_rows, hard_counts, base_plan_path = resolve_hard_groups(
        args.hard_groups_jsonl, args.rollout_bundle_root, args.expected_hard_groups
    )
    resolved_hard_path = args.output_dir / "resolved_hard_groups.jsonl"
    atomic_write_jsonl(resolved_hard_path, resolved_rows)
    expected_images: dict[str, int] = {}
    expected_stems: dict[str, set[str]] = {}
    for task in TASKS:
        input_path = args.input_dir / TASK_GT_FILE[task]
        if not input_path.is_file():
            raise FileNotFoundError(f"UI5 input is missing: {input_path}")
        image_paths = _input_image_paths(input_path, args.max_images_per_task)
        expected_images[task] = len(image_paths)
        expected_stems[task] = _expected_output_stems(image_paths)
        if expected_images[task] <= 0:
            raise ValueError(f"UI5 task {task} has no valid evaluation images")

    rollout_seeds = [args.seed + index for index in range(4)]
    expected_hard_ids = {
        task: {
            str(row["record_id"])
            for row in resolved_rows
            if row["task"] == task
        }
        for task in TASKS
    }
    crop_implementation = Path(__file__).with_name("ui5_lossless_tiling.py").resolve(
        strict=True
    )
    identity = {
        "schema_version": 1,
        "step": args.step,
        "candidate": directory_inventory(args.checkpoint),
        "processor": directory_inventory(args.processor_path),
        "worker_script": {
            "path": str(args.worker_script),
            "sha256": file_sha256(args.worker_script),
        },
        "generation": {
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
            "vision_attn_implementation": args.vision_attn_implementation,
            "mode": args.generation_mode,
            "max_new_tokens": args.max_new_tokens,
            "n_future_tokens": args.n_future_tokens,
            "greedy": args.greedy,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "seed": args.seed,
        },
        "crop_and_merge": {
            "mode": args.inference_crop_mode,
            "implementation_path": str(crop_implementation),
            "implementation_sha256": file_sha256(crop_implementation),
            "detector_crop_manifest": (
                str(args.detector_crop_manifest) if args.detector_crop_manifest else None
            ),
            "detector_crop_manifest_sha256": (
                file_sha256(args.detector_crop_manifest) if args.detector_crop_manifest else None
            ),
            "tile_max_count": args.tile_max_count,
            "tile_target_long_side": args.tile_target_long_side,
            "tile_overlap_ratio": args.tile_overlap_ratio,
            "tile_nms_iou": args.tile_nms_iou,
            "coordinate_mapping": "inference_ui_defect_locany.predict_with_lossless_tiles",
            "cross_crop_merge": "ui5_lossless_tiling.merge_tile_predictions",
            "content_missing": (
                "inference_ui_defect_locany.predict_with_direct_full_image:"
                "predict_parse_build_detections_no_crop_no_remap_no_nms"
            ),
        },
        "evaluator": {
            "path": str(args.scorer_script),
            "sha256": file_sha256(args.scorer_script),
            "iou_threshold": args.evaluator_iou_threshold,
            "bbox_format": "xyxy",
            "parse_error_policy": (
                "invalid_model_prediction_scored_incorrect_not_runtime_failure"
            ),
        },
        "hard_rollout": {
            "source": str(args.hard_groups_jsonl),
            "source_sha256": file_sha256(args.hard_groups_jsonl),
            "resolved_source": str(resolved_hard_path),
            "resolved_source_sha256": file_sha256(resolved_hard_path),
            "bundle_root": str(args.rollout_bundle_root),
            "base_scan_plans": str(base_plan_path),
            "base_scan_plans_sha256": file_sha256(base_plan_path),
            "expected_groups": args.expected_hard_groups,
            "groups_by_task": hard_counts,
            "rollout_seeds": rollout_seeds,
            "seed_derivation": "base_seed + rollout_id",
            "sample_seed_derivation": (
                "inference_ui_defect_locany.stable_sample_seed("
                "rollout_seed, task, record_id)"
            ),
        },
        "task_placement": {
            task: {
                "external_output_dir": EXTERNAL_TASK_DIR[task],
                "physical_gpu": devices[TASK_GPU_SLOT[task]],
                "logical_device": "cuda:0",
            }
            for task in TASKS
        },
        "worker_counts": {devices[0]: 2, devices[1]: 3},
        "expected_ui5_images": expected_images,
        "created_at": utc_now(),
    }
    identity_path = args.output_dir / "evaluation_manifest.json"
    atomic_write_json(identity_path, identity)
    identity_digest = file_sha256(identity_path)
    specs = build_worker_specs(
        args, devices, hard_counts, resolved_hard_path, identity_path
    )
    status_path = args.output_dir / "evaluation_status.json"
    status: dict[str, Any] = {
        "schema_version": 1,
        "status": "dry_run" if args.dry_run else "running",
        "success": False,
        "evaluation_identity": str(identity_path),
        "evaluation_identity_digest": identity_digest,
        "started_at": utc_now(),
        "step": args.step,
        "workers": [
            {
                **asdict(spec),
                "output_dir": str(spec.output_dir),
                "summary_path": str(spec.summary_path),
                "log_path": str(spec.log_path),
                "command": list(spec.command),
            }
            for spec in specs
        ],
    }
    atomic_write_json(status_path, status)
    results = launch_workers(
        specs,
        project_root=args.project_root,
        dry_run=args.dry_run,
        heartbeat_seconds=args.heartbeat_seconds,
    )
    status["workers"] = results
    if args.dry_run:
        status.update(
            {
                "status": "dry_run",
                "success": True,
                "finished_at": utc_now(),
                "evaluation_seconds": round(time.monotonic() - started, 6),
            }
        )
        atomic_write_json(status_path, status)
        for spec in specs:
            print(
                f"[DRY RUN] task={spec.task} gpu={spec.physical_gpu} "
                + " ".join(spec.command)
            )
        return 0

    status["status"] = "workers_completed"
    atomic_write_json(status_path, status)
    if file_sha256(identity_path) != identity_digest:
        raise RuntimeError("evaluation identity manifest changed while workers were running")
    validate_identity_unchanged(args, identity)
    validate_worker_results(
        specs,
        results,
        expected_images=expected_images,
        expected_stems=expected_stems,
        expected_hard=hard_counts,
        expected_hard_ids=expected_hard_ids,
        rollout_seeds=rollout_seeds,
        identity_digest=identity_digest,
        fake_worker=args.fake_worker,
    )
    hard_summary = merge_hard_rollout_outputs(
        args.output_dir, hard_counts, expected_hard_ids, identity_digest
    )
    status["status"] = "hard_rollouts_merged"
    status["hard_rollout"] = hard_summary
    atomic_write_json(status_path, status)
    metrics_path, metrics, score_command = run_scorer(args)
    evaluation_seconds = round(time.monotonic() - started, 6)
    evaluation_status = evaluation_status_payload(
        args=args,
        metrics=metrics,
        evaluation_seconds=evaluation_seconds,
    )
    status.update(
        {
            "status": "completed",
            "success": True,
            "score_command": score_command,
            "metrics_path": str(metrics_path),
            "overall": metrics["overall"],
            "hard_rollout": hard_summary,
            "finished_at": utc_now(),
            "evaluation_seconds": evaluation_seconds,
            "curriculum_status": evaluation_status,
        }
    )
    atomic_write_json(status_path, status)
    print_evaluation_status(evaluation_status)
    print(
        "[UI5 EVALUATION COMPLETE] "
        f"image_macro_f1={metrics['overall']['image_macro_f1']:.8f} "
        f"bbox_macro_f1={metrics['overall']['bbox_macro_f1']:.8f} "
        f"joint_score={metrics['overall']['joint_score']:.8f} "
        f"metrics={metrics_path}",
        flush=True,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args: argparse.Namespace | None = None
    try:
        args = parse_args(argv)
        return run(args)
    except KeyboardInterrupt:
        print("[UI5 EVALUATION INTERRUPTED]", file=sys.stderr, flush=True)
        return 130
    except Exception as exc:
        print(f"[UI5 EVALUATION FAILED] {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        if args is not None:
            try:
                output_dir = args.output_dir.expanduser().resolve(strict=False)
                output_dir.mkdir(parents=True, exist_ok=True)
                status_path = output_dir / "evaluation_status.json"
                previous: dict[str, Any] = {}
                if status_path.is_file():
                    try:
                        previous = json.loads(status_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        previous = {}
                previous.update(
                    {
                        "status": "failed",
                        "success": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                        "finished_at": utc_now(),
                    }
                )
                atomic_write_json(status_path, previous)
            except Exception:
                pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
