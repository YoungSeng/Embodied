#!/usr/bin/env python3
"""Reuse or run UI-Lens predictions, then score seen-app and unseen-app splits.

The split is based only on ``extra_info.original_infos.app_name``.  Training
application names must be traceable to explicit names or training annotations;
the script never guesses seen/unseen membership from UI-Lens labels or GT.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Iterable

from locany_ui5_common import TASK_JSONL
from subset_ui_lens_eval import app_name


DEFAULT_PROJECT = Path("/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle_LocateUI5_v4/Embodied-m32-cpt-sft-croponly-v1")
DEFAULT_CHECKPOINT = DEFAULT_PROJECT / "work_dirs/locany-ui5-m32-cpt3000-croponly-sourcebalanced-a800x4-v1/checkpoint-9000"
DEFAULT_PROCESSOR = Path("/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/hf_home/hub/models--nvidia--LocateAnything-3B/snapshots/c32291ca5e996f5a7a485845b4f57a233936bba0")
DEFAULT_EVAL = Path("/mnt/bn/intelligent-service-yg/dataset/UI_lens_ui5_eval_clip_v1")
DEFAULT_CACHE = Path("/mnt/bn/intelligent-service-yg/dataset/UI_lens_detector_cache_v1/horizontal_scan_v5_raw_detector_edge_aligned/detector_scan_crops.jsonl")


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield row


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def ui_lens_rows(eval_dir: Path) -> tuple[dict[str, list[dict[str, Any]]], Counter[str]]:
    records, inventory = {}, Counter()
    for task, filename in TASK_JSONL.items():
        path = eval_dir / filename
        task_rows = read_jsonl(path)
        for line_number, row in enumerate(task_rows, 1):
            try:
                name = app_name(row["extra_info"]["original_infos"]["app_name"])
                images = row["images"]
                boxes = row["answer"]["bbox"]
                if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], str):
                    raise ValueError("expected exactly one image path")
                if not isinstance(boxes, list):
                    raise ValueError("answer.bbox must be a list")
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid converted UI-Lens record: {exc}") from exc
            row["_normalized_app_name"] = name
            inventory[name] += 1
        records[task] = task_rows
    return records, inventory


def metadata_app(row: dict[str, Any]) -> str | None:
    candidates = [
        row.get("app_name"),
        (row.get("infos") or {}).get("app_name") if isinstance(row.get("infos"), dict) else None,
        (row.get("metadata") or {}).get("app_name") if isinstance(row.get("metadata"), dict) else None,
    ]
    extra = row.get("extra_info")
    if isinstance(extra, dict):
        candidates.append(extra.get("app_name"))
        original = extra.get("original_infos")
        if isinstance(original, dict):
            candidates.append(original.get("app_name"))
    for value in candidates:
        if value is None:
            continue
        try:
            return app_name(value)
        except ValueError:
            pass
    return None


def training_image(row: dict[str, Any]) -> str | None:
    value = row.get("image")
    if isinstance(value, str):
        return value
    images = row.get("images")
    if isinstance(images, list) and images and isinstance(images[0], str):
        return images[0]
    return None


def path_mentions_app(path: str, ui_apps: Iterable[str]) -> str | None:
    # Exact path-token matching is deliberately conservative and auditable.
    components = [token for token in re.split(r"[\\/]+", path) if token.strip()]
    tokens = {app_name(token) for token in components}
    tokens.update(
        app_name(token)
        for component in components
        for token in re.split(r"[_.:-]+", component)
        if token.strip()
    )
    matches = sorted(set(ui_apps) & tokens)
    return matches[0] if len(matches) == 1 else None


def discover_training_jsonls(training_data_dir: Path | None) -> list[Path]:
    if training_data_dir is None or not training_data_dir.is_dir():
        return []
    return sorted(path for path in training_data_dir.rglob("*.jsonl") if "train" in path.name.casefold())


def load_training_apps(
    explicit: Iterable[str], apps_file: Path | None, training_jsonls: Iterable[Path], ui_apps: Iterable[str]
) -> tuple[set[str], dict[str, Any]]:
    found, evidence = set(), {"explicit": [], "metadata": Counter(), "path_token": Counter(), "files": []}
    for value in explicit:
        normalized = app_name(value)
        found.add(normalized)
        evidence["explicit"].append(normalized)
    if apps_file is not None:
        if not apps_file.is_file():
            raise FileNotFoundError(
                f"Training app list does not exist: {apps_file}. Create it, omit "
                "--train-apps-file to scan --training-data-dir, or repeat --train-app-name."
            )
        raw = apps_file.read_text(encoding="utf-8-sig")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = [line for line in raw.splitlines() if line.strip() and not line.lstrip().startswith("#")]
        if isinstance(parsed, dict):
            parsed = parsed.get("apps") or parsed.get("train_apps")
        if not isinstance(parsed, list):
            raise ValueError("--train-apps-file must be a JSON list, {'apps': [...]}, or one app per line")
        for value in parsed:
            normalized = app_name(value)
            found.add(normalized)
            evidence["explicit"].append(normalized)
    for path in training_jsonls:
        path = path.expanduser().resolve(strict=True)
        file_stats = {"path": str(path), "rows": 0, "metadata_matches": 0, "path_token_matches": 0}
        for row in iter_jsonl(path):
            file_stats["rows"] += 1
            name = metadata_app(row)
            method = "metadata"
            if name is None:
                image = training_image(row)
                name = path_mentions_app(image, ui_apps) if image else None
                method = "path_token"
            if name is not None:
                found.add(name)
                evidence[method][name] += 1
                file_stats[f"{method}_matches"] += 1
        evidence["files"].append(file_stats)
        print(
            f"[TRAIN APP SCAN] {path}: rows={file_stats['rows']} "
            f"metadata={file_stats['metadata_matches']} "
            f"path_token={file_stats['path_token_matches']}",
            flush=True,
        )
    evidence["explicit"] = sorted(set(evidence["explicit"]))
    evidence["metadata"] = dict(sorted(evidence["metadata"].items()))
    evidence["path_token"] = dict(sorted(evidence["path_token"].items()))
    return found, evidence


def write_partitions(records: dict[str, list[dict[str, Any]]], output: Path, train_apps: set[str]) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    groups = {"train_source_apps": train_apps, "other_apps": None}
    stats: dict[str, Any] = {}
    output.mkdir(parents=True)
    try:
        for group, included in groups.items():
            target = output / "subsets" / group
            target.mkdir(parents=True)
            group_apps, unique_images = Counter(), set()
            task_stats = {}
            for task, rows in records.items():
                selected = [
                    row for row in rows
                    if (row["_normalized_app_name"] in included if included is not None
                        else row["_normalized_app_name"] not in train_apps)
                ]
                clean = []
                positives = boxes = 0
                for row in selected:
                    row = dict(row)
                    name = row.pop("_normalized_app_name")
                    clean.append(row)
                    group_apps[name] += 1
                    unique_images.add(row["images"][0])
                    positives += bool(row["answer"]["bbox"])
                    boxes += len(row["answer"]["bbox"])
                (target / TASK_JSONL[task]).write_text(
                    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in clean), encoding="utf-8"
                )
                task_stats[task] = {"records": len(clean), "positive": positives,
                                    "negative": len(clean) - positives, "boxes": boxes}
            stats[group] = {"apps": dict(sorted(group_apps.items())), "tasks": task_stats,
                            "image_task_records": sum(v["records"] for v in task_stats.values()),
                            "unique_image_paths": len(unique_images)}
            if not stats[group]["image_task_records"]:
                raise ValueError(f"{group} is empty; verify the training app list")
    except Exception:
        # Leave an explicit marker instead of presenting a partial directory as valid.
        (output / "INCOMPLETE.txt").write_text("Partition creation did not complete.\n", encoding="utf-8")
        raise
    return stats


def output_stems(rows: list[dict[str, Any]]) -> dict[str, str]:
    grouped: dict[str, list[str]] = {}
    for row in rows:
        image = row["images"][0]
        grouped.setdefault(Path(image).stem.replace(":", "_"), []).append(image)
    result = {}
    for stem, images in grouped.items():
        for image in images:
            result[image] = stem if len(images) == 1 else stem + "__" + hashlib.blake2b(image.encode(), digest_size=5).hexdigest()
    return result


def prediction_audit(prediction_dir: Path, records: dict[str, list[dict[str, Any]]], checkpoint: Path,
                     crop_mode: str) -> dict[str, Any]:
    report: dict[str, Any] = {"prediction_dir": str(prediction_dir), "exists": prediction_dir.is_dir(),
                              "tasks": {}, "manifest": None, "usable": False, "reasons": []}
    if not prediction_dir.is_dir():
        report["reasons"].append("prediction directory does not exist")
        return report
    manifest_path = prediction_dir / "_run_manifest.json"
    if not manifest_path.is_file():
        report["reasons"].append("missing _run_manifest.json; checkpoint identity cannot be certified")
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            report["manifest"] = manifest
            if Path(str(manifest.get("checkpoint", ""))).resolve(strict=False) != checkpoint.resolve(strict=False):
                report["reasons"].append("checkpoint in manifest does not match requested checkpoint")
            actual_mode = (manifest.get("inference_crop") or {}).get("mode")
            if actual_mode != crop_mode:
                report["reasons"].append(f"inference crop mode is {actual_mode!r}, expected {crop_mode!r}")
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            report["reasons"].append(f"invalid run manifest: {exc}")
    for task, rows in records.items():
        task_dir = prediction_dir / task
        missing, invalid, ambiguous = [], [], []
        for image, stem in output_stems(rows).items():
            candidates = [task_dir / f"{stem}.json", task_dir / f"{stem}_defect.json",
                          task_dir / f"{stem}_ok.json", task_dir / f"{stem}_parse_error.json"]
            existing = [path for path in candidates if path.is_file()]
            if not existing:
                missing.append(image)
                continue
            if len(existing) != 1:
                ambiguous.append({"image": image, "files": [str(path) for path in existing]})
                continue
            try:
                value = json.loads(existing[0].read_text(encoding="utf-8"))
                if not isinstance(value, list) or existing[0].stem.endswith("_parse_error"):
                    invalid.append(str(existing[0]))
            except (OSError, json.JSONDecodeError):
                invalid.append(str(existing[0]))
        report["tasks"][task] = {"expected": len(rows), "present": len(rows) - len(missing),
                                  "missing": len(missing), "invalid": len(invalid), "ambiguous": len(ambiguous),
                                  "missing_examples": missing[:5], "invalid_examples": invalid[:5],
                                  "invalid_files": invalid,
                                  "ambiguous_examples": ambiguous[:3]}
        if missing or invalid or ambiguous:
            report["reasons"].append(f"{task}: missing={len(missing)} invalid={len(invalid)} ambiguous={len(ambiguous)}")
    report["usable"] = not report["reasons"]
    return report


def find_reusable_predictions(project: Path, preferred: Path, records: dict[str, list[dict[str, Any]]],
                              checkpoint: Path, crop_mode: str) -> tuple[Path, dict[str, Any]]:
    candidates = [preferred]
    work_dirs = project / "work_dirs"
    if work_dirs.is_dir():
        for manifest in work_dirs.rglob("_run_manifest.json"):
            if manifest.parent not in candidates:
                candidates.append(manifest.parent)
    audits = [prediction_audit(path, records, checkpoint, crop_mode) for path in candidates]
    for path, audit in zip(candidates, audits):
        if audit["usable"]:
            return path, {"selected": audit, "checked": audits}
    return preferred, {"selected": audits[0], "checked": audits}


def run_command(command: list[str], cwd: Path) -> None:
    print("RUN:", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def run_inference(args: argparse.Namespace, prediction_dir: Path) -> None:
    for path, label in ((args.checkpoint, "checkpoint"), (args.processor_path, "processor"),
                        (args.detector_crop_manifest, "detector crop manifest")):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    command = [sys.executable, str(args.project_root / "scripts/run_ui5_parallel_inference.py"),
               "--checkpoint", str(args.checkpoint), "--processor-path", str(args.processor_path),
               "--input-dir", str(args.eval_dir), "--output-dir", str(prediction_dir),
               "--gpu-devices", args.gpu_devices, "--attn-implementation", args.attn_implementation,
               "--inference-script", str(args.project_root / "scripts/inference_ui_defect_locany.py"),
               "--relation-gate-mode", "observe", "--enable-pbd", "--inference-crop-mode", args.crop_mode,
               "--save-raw-answer"]
    if args.crop_mode == "detector_scan":
        command += ["--detector-crop-manifest", str(args.detector_crop_manifest)]
    run_command(command, args.project_root)


def resumable_audit(audit: dict[str, Any]) -> bool:
    """A matching, valid manifest with only absent predictions can be resumed."""
    manifest = audit.get("manifest")
    if not isinstance(manifest, dict):
        return not audit.get("exists")
    if any(task.get("invalid") or task.get("ambiguous") for task in audit.get("tasks", {}).values()):
        return False
    non_task_reasons = [reason for reason in audit.get("reasons", []) if not re.match(r"^[a-z_]+: missing=", reason)]
    return not non_task_reasons


def repairable_audit(audit: dict[str, Any]) -> bool:
    """Return true when only missing/invalid per-image outputs prevent reuse."""
    if not isinstance(audit.get("manifest"), dict):
        return False
    if any(task.get("ambiguous") for task in audit.get("tasks", {}).values()):
        return False
    non_task_reasons = [
        reason for reason in audit.get("reasons", [])
        if not re.match(r"^[a-z_]+: missing=\d+ invalid=\d+ ambiguous=0$", reason)
    ]
    return not non_task_reasons


def quarantine_invalid_predictions(prediction_dir: Path, audit: dict[str, Any]) -> Path | None:
    invalid = [
        (task, Path(filename))
        for task, task_audit in audit.get("tasks", {}).items()
        for filename in task_audit.get("invalid_files", [])
    ]
    if not invalid:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = prediction_dir / "_invalid_before_app_eval" / stamp
    manifest = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
                "prediction_dir": str(prediction_dir), "files": []}
    # Resolve and validate every source before moving the first file.
    checked = []
    for task, source in invalid:
        source = source.resolve(strict=True)
        expected_parent = (prediction_dir / task).resolve(strict=True)
        if source.parent != expected_parent:
            raise ValueError(f"Refusing to quarantine prediction outside its task directory: {source}")
        target = backup / task / source.name
        checked.append((task, source, target))
    moved = []
    try:
        for task, source, target in checked:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
            moved.append((source, target))
            manifest["files"].append({"task": task, "source": str(source), "backup": str(target)})
            print(f"QUARANTINED INVALID PREDICTION: {source} -> {target}", flush=True)
    except Exception:
        for source, target in reversed(moved):
            if target.is_file() and not source.exists():
                shutil.move(str(target), str(source))
        raise
    (backup / "quarantine_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return backup


def score_subset(args: argparse.Namespace, gt_dir: Path, prediction_dir: Path, output_root: Path, run_name: str) -> dict[str, Any]:
    command = [sys.executable, str(args.project_root / "qwen3vl_merge_and_score_fixed_5tasks.py"),
               "--all_tasks", "--input_mode", "yolo_dir", "--gt_dir", str(gt_dir),
               "--pred_root", str(prediction_dir), "--output_root", str(output_root),
               "--run_name", run_name, "--yolo_bbox_format", "xyxy", "--iou_thresh", str(args.iou_threshold)]
    run_command(command, args.project_root)
    return json.loads((output_root / run_name / "all_tasks_evaluation.json").read_text(encoding="utf-8"))


def make_report(summary: dict[str, Any]) -> str:
    lines = ["# UI-Lens 跨应用评测（Ours）", "", f"- checkpoint: `{summary['checkpoint']}`",
             f"- predictions: `{summary['prediction_dir']}`", f"- IoU threshold: {summary['iou_threshold']}", "",
             "| 分组 | Apps | 图片×任务 | Image F1 | Bbox F1 |", "|---|---:|---:|---:|---:|"]
    for group in ("train_source_apps", "other_apps"):
        split, metric = summary["splits"][group], summary["metrics"][group]
        lines.append(f"| {group} | {len(split['apps'])} | {split['image_task_records']} | "
                     f"{metric['macro']['image']['f1']:.4f} | {metric['macro']['bbox']['f1']:.4f} |")
    gap = summary["generalization_gap_train_minus_other"]
    lines += ["", f"训练来源 − 其他应用：Image F1 `{gap['image_f1']:+.4f}`，"
              f"Bbox F1 `{gap['bbox_f1']:+.4f}`。正值表示其他应用更低。"]
    lines += ["", "## 两组的五任务结果", "",
              "| 分组 | 任务 | 样本 | 正样本 | Image F1 | Bbox F1 |",
              "|---|---|---:|---:|---:|---:|"]
    for group in ("train_source_apps", "other_apps"):
        for task, values in summary["metrics"][group]["tasks"].items():
            population = summary["splits"][group]["tasks"][task]
            lines.append(f"| {group} | {values['issue_name']} | {population['records']} | "
                         f"{population['positive']} | {values['image']['f1']:.4f} | {values['bbox']['f1']:.4f} |")
    lines += ["", "## 每个 App", "", "| App | 分组 | 图片×任务 | Image F1 | Bbox F1 |",
              "|---|---|---:|---:|---:|"]
    for name, item in sorted(summary.get("per_app", {}).items()):
        lines.append(f"| {name} | {item['group']} | {item['image_task_records']} | "
                     f"{item['metrics']['macro']['image']['f1']:.4f} | {item['metrics']['macro']['bbox']['f1']:.4f} |")
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--processor-path", type=Path, default=DEFAULT_PROCESSOR)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--prediction-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--detector-crop-manifest", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--training-data-dir", type=Path, default=None)
    parser.add_argument("--training-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--train-app-name", action="append", default=[])
    parser.add_argument("--train-apps-file", type=Path)
    parser.add_argument("--gpu-devices", default="0,1,2,3")
    parser.add_argument("--attn-implementation", default="sdpa", choices=("sdpa", "magi", "flash_attention_2", "eager", "auto"))
    parser.add_argument("--crop-mode", default="detector_scan", choices=("detector_scan", "full_image", "lossless_tiling"))
    parser.add_argument("--iou-threshold", type=float, default=0.1)
    parser.add_argument("--run-if-missing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retry-invalid-predictions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--per-app", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--list-ui-apps", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.project_root = args.project_root.expanduser().resolve(strict=True)
    args.checkpoint = args.checkpoint.expanduser().resolve(strict=False)
    args.processor_path = args.processor_path.expanduser().resolve(strict=False)
    args.eval_dir = args.eval_dir.expanduser().resolve(strict=True)
    args.detector_crop_manifest = args.detector_crop_manifest.expanduser().resolve(strict=False)
    records, inventory = ui_lens_rows(args.eval_dir)
    print("UI-Lens apps (image-task records):", json.dumps(dict(sorted(inventory.items())), ensure_ascii=False))
    if args.list_ui_apps:
        return 0
    preferred = (args.prediction_dir or args.project_root / "work_dirs/ui-lens-checkpoint9000-detectorscan-clip-v1/predictions").expanduser().resolve(strict=False)
    prediction_dir, search = find_reusable_predictions(
        args.project_root, preferred, records, args.checkpoint, args.crop_mode
    )
    if search["selected"]["usable"]:
        print(f"FOUND COMPLETE MATCHING PREDICTIONS: {prediction_dir}")
    else:
        print("NO COMPLETE MATCHING PREDICTIONS:")
        for reason in search["selected"]["reasons"]:
            print(f"  - {reason}")
    training_data_dir = args.training_data_dir or args.project_root / "data"
    training_jsonls = list(args.training_jsonl) or discover_training_jsonls(training_data_dir)
    train_apps, evidence = load_training_apps(args.train_app_name, args.train_apps_file, training_jsonls, inventory)
    train_apps &= set(inventory)
    if not train_apps:
        raise ValueError("No UI-Lens app could be certified as a training-source app. Supply --train-app-name, "
                         "--train-apps-file, or training JSONL files that retain app_name/path tokens.")
    other_apps = set(inventory) - train_apps
    if not other_apps:
        raise ValueError("No other apps remain after applying the training app list")
    print("TRAIN-SOURCE APPS:", json.dumps(sorted(train_apps), ensure_ascii=False))
    print("OTHER APPS:", json.dumps(sorted(other_apps), ensure_ascii=False))
    reused_predictions = bool(search["selected"]["usable"])
    if reused_predictions:
        print(f"REUSE COMPLETE PREDICTIONS: {prediction_dir}")
    elif not args.run_if_missing:
        print(json.dumps(search, ensure_ascii=False, indent=2))
        raise RuntimeError("No complete matching predictions; rerun without --no-run-if-missing on a GPU node")
    else:
        print("No complete matching prediction set; starting/resuming full inference.")
        selected_audit = search["selected"]
        if prediction_dir.exists() and repairable_audit(selected_audit) and args.retry_invalid_predictions:
            backup = quarantine_invalid_predictions(prediction_dir, selected_audit)
            if backup is not None:
                print(f"Only invalid/missing samples will be inferred again; backup={backup}")
        elif prediction_dir.exists() and not resumable_audit(selected_audit):
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            prediction_dir = prediction_dir.with_name(prediction_dir.name + f"-retry-{stamp}")
            print(f"Existing directory is not safely resumable; using fresh prediction directory: {prediction_dir}")
        run_inference(args, prediction_dir)
        audit = prediction_audit(prediction_dir, records, args.checkpoint, args.crop_mode)
        if not audit["usable"]:
            raise RuntimeError("Inference returned but predictions are incomplete: " + "; ".join(audit["reasons"]))
        search["selected"] = audit
    output = (args.output_dir or args.project_root / "work_dirs/ui-lens-checkpoint9000-app-generalization-v1").expanduser().resolve(strict=False)
    split_stats = write_partitions(records, output, train_apps)
    metrics = {}
    for group in ("train_source_apps", "other_apps"):
        metrics[group] = score_subset(args, output / "subsets" / group, prediction_dir, output / "evaluation", group)
    per_app = {}
    if args.per_app:
        for name in sorted(inventory):
            slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_") or "app"
            slug += "_" + hashlib.blake2b(name.encode(), digest_size=4).hexdigest()
            app_dir = output / "per_app_subsets" / slug
            app_dir.mkdir(parents=True)
            count = 0
            for task, rows in records.items():
                selected = []
                for row in rows:
                    if row["_normalized_app_name"] == name:
                        clean = dict(row); clean.pop("_normalized_app_name"); selected.append(clean)
                count += len(selected)
                (app_dir / TASK_JSONL[task]).write_text(
                    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8"
                )
            app_metrics = score_subset(args, app_dir, prediction_dir, output / "evaluation/per_app", slug)
            per_app[name] = {"group": "train_source_apps" if name in train_apps else "other_apps",
                             "image_task_records": count, "metrics": app_metrics,
                             "subset_dir": str(app_dir), "score_run_name": slug}
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint),
        "eval_dir": str(args.eval_dir),
        "prediction_dir": str(prediction_dir),
        "prediction_reused": reused_predictions,
        "prediction_audit": search,
        "crop_mode": args.crop_mode,
        "iou_threshold": args.iou_threshold,
        "training_source_apps": sorted(train_apps),
        "other_apps": sorted(other_apps),
        "training_app_evidence": evidence,
        "ui_lens_app_inventory": dict(sorted(inventory.items())),
        "splits": split_stats,
        "metrics": metrics,
        "generalization_gap_train_minus_other": {
            "image_f1": metrics["train_source_apps"]["macro"]["image"]["f1"]
            - metrics["other_apps"]["macro"]["image"]["f1"],
            "bbox_f1": metrics["train_source_apps"]["macro"]["bbox"]["f1"]
            - metrics["other_apps"]["macro"]["bbox"]["f1"],
        },
        "per_app": per_app,
    }
    (output / "app_generalization_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = make_report(summary)
    (output / "app_generalization_report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"REPORT: {output / 'app_generalization_report.md'}")
    print(f"JSON: {output / 'app_generalization_summary.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
