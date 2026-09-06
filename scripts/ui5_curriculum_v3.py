#!/usr/bin/env python3
"""Prepare/submit a fresh v3 experiment and run its formal paired decode study."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from scripts import prepare_ui5_curriculum_snapshot as preparation
from scripts import run_ui5_curriculum_evaluation as evaluation
from scripts.ui5_output_validity import audit_evaluation
from eaglevl.train.ui5_checkpoint_utils import validate_checkpoint
from eaglevl.train.ui5_curriculum_profiles import profile_env
from scripts.restart_ui5_after_storage_failure import check_storage
from scripts.ui5_curriculum_text_revision import publish_revision, supervision_audit, verify_revision, sha

ORIGINAL_MODEL_RELATIVE = "gui_models/Embodied-ui5-det-crop/locany-ui5-v5-croponly-sourcebalanced-a800x4-20260830/checkpoint-12000"


def pool_coverage(manifest):
    strata = manifest["bundle_group_selection"]["selected_pool_strata"]
    rows = []
    for pool in ("hard", "matched_anchor", "global_replay"):
        if any(sum(int(n) for n in strata[pool][task].values()) <= 0 for task in evaluation.TASKS):
            raise ValueError(f"immutable pool {pool} lacks a UI5 task")
        if any(sum(int(strata[pool][task][p]) for task in evaluation.TASKS) <= 0
               for p in ("positive", "negative")):
            raise ValueError(f"immutable pool {pool} lacks positive/negative coverage")
        for task in evaluation.TASKS:
            for polarity in ("positive", "negative"):
                count = int(strata[pool][task][polarity])
                if count < 0 or (pool == "global_replay" and count == 0):
                    raise ValueError(f"immutable pool {pool} lacks {task}/{polarity}; cannot fabricate coverage or change frozen IDs")
                rows.append({"pool": pool, "task": "ui_" + task, "polarity": polarity,
                             "group_count": count, "sampling_unit": "sample_group", "scope": "train-only immutable pool"})
    return rows


def resolve_source(directory):
    """Follow only durable linked successors, never a newest-directory guess."""
    directory = directory.resolve(strict=True)
    root, seen = directory.parent, set()
    while True:
        if directory in seen:
            raise ValueError("submission chain cycle")
        seen.add(directory)
        markers = [directory / name for name in ("caption-retry.started", "detail-audit-restart.started",
                                                 "storage-restart.started") if (directory / name).exists()]
        if len(markers) > 1:
            raise ValueError("ambiguous submission successors")
        if not markers:
            return directory / "snapshot-switch.json"
        following = Path(markers[0].read_text(encoding="utf-8").strip()).resolve(strict=True)
        if following.name != "snapshot-switch.json" or following.parent.parent != root:
            raise ValueError("submission pointer leaves the known log root")
        state = json.loads(following.read_text(encoding="utf-8"))
        if Path(state.get("retry_of", "")).resolve() != directory / "snapshot-switch.json":
            raise ValueError("successor does not point back to its source")
        directory = following.parent


def model_view(source: Path, destination: Path):
    """Private metadata/code, hard-linked read-only model weights; no optimizer state."""
    source = source.resolve(strict=True)
    report = validate_checkpoint(source, mode="eval")
    if not report["valid"]:
        raise ValueError(f"model-only checkpoint is unavailable: {source}: {report['errors']}")
    destination.mkdir(exist_ok=False, parents=True)
    copied = []
    for path in sorted(source.iterdir()):
        if not path.is_file():
            continue
        weight = path.suffix == ".safetensors" or path.name.startswith("pytorch_model") and path.suffix == ".bin"
        if not weight and (path.name in {"trainer_state.json", "continuity_state.json", "checkpoint_complete.json"}
                           or path.suffix not in {".json", ".py", ".txt", ".model", ".tiktoken"}):
            continue
        if weight:
            # Same mount is required; no silent full-weight copy that exhausts quota.
            os.link(path.resolve(), destination / path.name)
        else:
            shutil.copy2(path, destination / path.name)
        copied.append({"name": path.name, "bytes": path.stat().st_size, "weights_hardlinked": weight})
    return {"source": str(source), "view": str(destination), "files": copied,
            "optimizer_restored": False, "global_step": 0}


def frozen_eval_subset(source: Path, destination: Path, seed: int, count: int):
    """A task-balanced fixed hash sample; only image identity selects rows, never GT."""
    destination.mkdir(parents=True, exist_ok=False)
    identities = {}
    for task, filename in evaluation.TASK_GT_FILE.items():
        source_file = source / filename
        images = evaluation._input_image_paths(source_file, 0)
        selected = set(sorted(images, key=lambda name: hashlib.sha256(
            f"{seed}:{task}:{name}".encode()).hexdigest())[:count])
        output = []
        for raw in evaluation.read_jsonl(source_file):
            row = dict(raw)
            row.pop("_source_line", None)
            field = "images" if "images" in row else "image"
            items = row[field] if isinstance(row[field], list) else [row[field]]
            paths, normalized = [], []
            for item in items:
                value = item if isinstance(item, str) else item["path"]
                path = Path(value).expanduser()
                path = (path if path.is_absolute() else source_file.parent / path).resolve()
                paths.append(str(path))
                normalized.append(str(path) if isinstance(item, str) else {**item, "path": str(path)})
            if set(paths) & selected:
                if not set(paths) <= selected:
                    raise ValueError("multi-image row crosses the fixed comparison selection")
                row[field] = normalized if isinstance(row[field], list) else normalized[0]
                output.append(row)
        evaluation.atomic_write_jsonl(destination / filename, output)
        if set(evaluation._input_image_paths(destination / filename, 0)) != selected:
            raise ValueError("comparison subset image identity changed")
        identities[task] = {"images": sorted(selected), "source_sha256": evaluation.file_sha256(source_file),
                            "subset_sha256": evaluation.file_sha256(destination / filename)}
    return {"seed": seed, "selection": "sha256(seed, task, absolute image identity); no GT selection",
            "per_task_limit": count, "tasks": identities}


def prepare(args):
    old_path = resolve_source(args.previous_submission_dir)
    reservation = old_path.parent / "curriculum-v3-text-v3-1.started"
    if args.submit and reservation.exists():
        raise RuntimeError(f"v3 submission already reserved: {reservation}; inspect its target, do not repeat")
    old = json.loads(old_path.read_text(encoding="utf-8"))
    old_job_path = Path(old["job_yaml"]).resolve(strict=True)
    if old_job_path.parent != old_path.parent:
        raise ValueError("source YAML is outside the selected submission directory")
    template = preparation.yaml.safe_load(old_job_path.read_text(encoding="utf-8"))
    env = dict(template["jobRunParams"]["envsList"])
    if any(env.get(k) != v for k, v in old["runtime"].items()):
        raise ValueError("source runtime/YAML differs")
    if Path(env["PROJECT_ROOT"]).resolve() != ROOT:
        raise ValueError("use the saved project checkout to prepare the v3 job")
    if Path(old["snapshot"]).name != "hour_021_20260905T060754Z":
        raise ValueError("v3 must reuse the existing hour021 selection, not refreeze another snapshot")
    curriculum = preparation.verify_prepared_curriculum(env)
    coverage = pool_coverage(curriculum)
    previous_output = Path(env["OUTPUT_DIR"]).resolve(strict=True)
    # Read the actual raw/worker summaries before preparing any new experiment.
    audits = {str(step): audit_evaluation(previous_output / "evaluation" / f"step-{step:06d}")
              for step in (0, 200)}
    workspace = Path(env.get("WORKSPACE", preparation.WORKSPACE))
    original = workspace / ORIGINAL_MODEL_RELATIVE
    storage = check_storage(previous_output.parent, old_path.parent.parent,
                            previous_output / "resume/latest", original)
    text_source = env["CURRICULUM_DATA_DIR"]
    text_directory, curriculum = publish_revision(text_source)
    supervision = supervision_audit(text_directory)
    env.update({"CURRICULUM_DATA_DIR": str(text_directory),
                "META_PATH": str(text_directory / "ui5_crop_rollout4_curriculum.json"),
                "UI5_TRAIN_TEXT_IDENTITY": curriculum["identity_digest"],
                "UI5_TRAIN_RECIPE_SHA256": curriculum["training_text"]["recipe_sha256"]})
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    name = "ui5-crop-curriculum-v3-text-v3-1-h20x2-" + stamp
    output = previous_output.parent / name
    submission = old_path.parent.parent / name
    output.mkdir(exist_ok=False)
    submission.mkdir(exist_ok=False)
    for key in ("RESUME_FROM_CHECKPOINT", "CURRICULUM_START_STEP", "LOCANY_STOP_AFTER_STEP", "LOCANY_SEGMENT_MODE"):
        env.pop(key, None)
    source_view = model_view(original, output / "initial_model")
    degraded = args.degraded_checkpoint
    if degraded is None:
        identity = json.loads((previous_output / "evaluation/step-000200/evaluation_manifest.json").read_text())
        candidate = Path(identity["candidate"]["path"])
        state = candidate / "trainer_state.json"
        if state.is_file() and json.loads(state.read_text()).get("global_step") == 200:
            degraded = candidate
    degraded_view = None
    if degraded is not None:
        degraded_view = model_view(degraded, output / "comparison_model")
    env.update({**preparation.FORMAL_ENV, **profile_env("global_replay_v3"), "RUN_NAME": name, "OUTPUT_DIR": str(output),
                "MODEL_PATH": str(output / "initial_model"), "UI5_V3_ORIGINAL_MODEL": str(original),
                "UI5_V3_OLD_OUTPUT": str(previous_output),
                "UI5_V3_DEGRADED_MODEL": str(output / "comparison_model") if degraded_view else "",
                "UI5_V3_COMPARISON_INPUT": str(output / "diagnostics/decoder_comparison_inputs"),
                "GRADIENT_ACCUMULATION_STEPS": env.get("GRADIENT_ACCUMULATION_STEPS", "4"),
                "PYTHON_BIN": env.get("PYTHON_BIN", str(workspace / "conda_envs/LocateAnything/bin/python")),
                "CODE_REVISION": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()})
    subset = frozen_eval_subset(Path(env["EVAL_INPUT_DIR"]), Path(env["UI5_V3_COMPARISON_INPUT"]), 42, 32)
    report = {"schema_version": 1, "code_sha": env["CODE_REVISION"], "old_run": str(previous_output),
              "raw_audits": audits, "curriculum_identity": curriculum["identity_digest"],
              "hard_groups": curriculum["hard_groups"], "anchor_groups": curriculum["matched_anchor_groups"],
              "source_model_view": source_view, "degraded_model_view": degraded_view,
              "storage_check": storage,
              "pool_coverage": coverage,
              "training_supervision_audit": supervision,
              "training_text": {"source_directory": text_source, "directory": str(text_directory),
                                "recipe_path": env["META_PATH"], "identity_digest": curriculum["identity_digest"],
                                "manifest_sha256": sha(text_directory / "curriculum_manifest.json"),
                                **curriculum["training_text"]},
              "supervision_format": json.loads((text_directory / "supervision_format.json").read_text(encoding="utf-8")),
              "batch_profile": {"per_device_train_batch_size": 1,
                                "gradient_accumulation_steps": env["GRADIENT_ACCUMULATION_STEPS"]},
              "degraded_status": "available" if degraded_view else "step-200 checkpoint unavailable; NOT replaced by later weights",
              "fixed_comparison_inputs": subset, "decoder_policy_changed": True,
              "decoder_policy": "boundary_v3",
              "causal_conclusion": "pending paired inference; category counts alone are not a causal finding",
              "no_rollout_or_png_generation": True}
    preparation.write_state(output / "diagnostics/v3_preparation.json", report)
    published_template = preparation.yaml.safe_load((ROOT / "jobs/ui5_crop_curriculum_v3_h20x2.yaml").read_text(encoding="utf-8"))
    # Preserve the resource/image settings that actually ran on this user's H20 job.
    for field in ("resource", "imageMeta", "volumes"):
        if field in template["jobDefVersion"]:
            published_template["jobDefVersion"][field] = copy.deepcopy(template["jobDefVersion"][field])
    published_template["namespace"] = template["namespace"]
    job = preparation.render_job(published_template, env, "ui5-curriculum-v3-" + stamp)
    job["jobRunParams"]["entrypointFullScript"] = job["jobRunParams"]["entrypointFullScript"].replace(
        "shell/run_locany_ui5_crop_rollout4_curriculum_h20x2.sh", "shell/run_ui5_crop_curriculum_v3_h20x2.sh")
    job_path = submission / "formal.yaml"
    with job_path.open("x", encoding="utf-8") as handle:
        preparation.yaml.safe_dump(job, handle, sort_keys=False)
    state = {"status": "prepared_v3", "source_submission": str(old_path), "snapshot": old["snapshot"],
             "runtime": env, "job_yaml": str(job_path), "preparation_report": str(output / "diagnostics/v3_preparation.json")}
    state_path = submission / "snapshot-switch.json"
    preparation.write_state(state_path, state)
    print(f"[V3 READY] yaml={job_path} output={output} hard={curriculum['hard_groups']} "
          f"anchor={curriculum['matched_anchor_groups']} training_text={text_directory} "
          f"recipe_sha256={env['UI5_TRAIN_RECIPE_SHA256']} PNG_generation=0", flush=True)
    if args.submit:
        mlx = shutil.which(args.mlx_bin)
        if not mlx:
            raise RuntimeError("mlx is missing")
        with reservation.open("x", encoding="utf-8") as handle:
            handle.write(str(state_path) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        preparation.submit_job(mlx, job_path, state_path, state)
    return job_path


def compare():
    env = os.environ
    output = Path(env["OUTPUT_DIR"])
    manifest = preparation.verify_prepared_curriculum(env)
    verify_revision(env["CURRICULUM_DATA_DIR"], expected_recipe_sha=env["UI5_TRAIN_RECIPE_SHA256"],
                    expected_identity=env["UI5_TRAIN_TEXT_IDENTITY"])
    cells = []
    candidates = {"original": env["MODEL_PATH"]}
    if env.get("UI5_V3_DEGRADED_MODEL"):
        candidates["degraded"] = env["UI5_V3_DEGRADED_MODEL"]
    for label, checkpoint in candidates.items():
        subprocess.run([sys.executable, str(ROOT / "scripts/patch_locany_checkpoint.py"),
                        "--checkpoint", checkpoint, "--base-model", env["PROCESSOR_PATH"],
                        "--project-root", str(ROOT), "--force", "--validate-relation-weights"], check=True)
        for mode, policy in (("hybrid", "legacy"), ("hybrid", "boundary_v3"), ("slow", "boundary_v3")):
            destination = output / "diagnostics/decoder_comparison" / f"{label}-{mode}-{policy}"
            command = [sys.executable, str(ROOT / "scripts/run_ui5_curriculum_evaluation.py"),
                       "--evaluation-purpose", "decoder_comparison", "--checkpoint", checkpoint,
                       "--processor-path", env["PROCESSOR_PATH"], "--input-dir", env["UI5_V3_COMPARISON_INPUT"],
                       "--output-dir", str(destination), "--step", "0", "--generation-mode", mode,
                       "--decoder-policy", policy,
                       "--hard-groups-jsonl", str(Path(env["CURRICULUM_DATA_DIR"]) / "hard_groups.jsonl"),
                       "--curriculum-manifest", str(Path(env["CURRICULUM_DATA_DIR"]) / "curriculum_manifest.json"),
                       "--rollout-bundle-root", env["ROLLOUT_BUNDLE_ROOT"], "--frozen-selection", env["FROZEN_SELECTION"],
                       "--expected-hard-groups", str(manifest["hard_groups"]),
                       "--detector-crop-manifest", env["EVAL_DETECTOR_MANIFEST"],
                       "--seed", "42", "--relation-gate-mode", env.get("RELATION_GATE_MODE", "observe"),
                       "--scorer-script", env.get("SCORER_SCRIPT", str(ROOT / "qwen3vl_merge_and_score_fixed_5tasks.py"))]
            for flag, key, default in (("max-new-tokens", "EVAL_MAX_NEW_TOKENS", "4096"),
                                       ("n-future-tokens", "EVAL_N_FUTURE_TOKENS", "6"),
                                       ("temperature", "EVAL_TEMPERATURE", "0.7"),
                                       ("top-p", "EVAL_TOP_P", "0.9"), ("top-k", "EVAL_TOP_K", "0"),
                                       ("repetition-penalty", "EVAL_REPETITION_PENALTY", "1.1"),
                                       ("tile-max-count", "EVAL_TILE_MAX_COUNT", "10"),
                                       ("tile-target-long-side", "EVAL_TILE_TARGET_LONG_SIDE", "1600"),
                                       ("tile-overlap-ratio", "EVAL_TILE_OVERLAP_RATIO", "0.10"),
                                       ("tile-nms-iou", "EVAL_TILE_NMS_IOU", "0.50"),
                                       ("evaluator-iou-threshold", "EVAL_IOU_THRESHOLD", "0.10")):
                command += ["--" + flag, env.get(key, default)]
            print(f"[FORMAL DECODER COMPARISON] model={label} mode={mode} policy={policy} seed=42", flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
            metrics = json.loads((destination / "ui5_metrics.json").read_text())
            audit = audit_evaluation(destination)
            cells.append({"model": label, "mode": mode, "decoder_policy": policy, "metrics": metrics,
                          "raw_audit": audit, "output": str(destination)})
            preparation.write_state(output / "diagnostics/decoder_comparison.json",
                                    {"cells": cells, "complete": False, "selection_used_for_training": False})
    gains = []
    for label in candidates:
        before = next(c for c in cells if c["model"] == label and c["decoder_policy"] == "legacy")
        after = next(c for c in cells if c["model"] == label and c["decoder_policy"] == "boundary_v3" and c["mode"] == "hybrid")
        gains.append({"model": label, "scope": "fixed UI5 subset, same weights and seed",
                      **{key + "_delta": after["metrics"]["overall"][key] - before["metrics"]["overall"][key]
                         for key in ("image_macro_f1", "bbox_macro_f1", "joint_score")},
                      "invalid_delta": sum(r["image_invalid"] for r in after["raw_audit"]["rows"])
                                       - sum(r["image_invalid"] for r in before["raw_audit"]["rows"])})
    preparation.write_state(output / "diagnostics/decoder_comparison.json",
                            {"cells": cells, "complete": True, "decoder_policy_changed": True,
                             "inference_fix_gain": gains, "training_gain_reference": "new full UI5 step-0",
                             "selection_used_for_training": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-submission-dir", type=Path)
    parser.add_argument("--degraded-checkpoint", type=Path)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--mlx-bin", default="mlx")
    parser.add_argument("--run-comparison", action="store_true")
    args = parser.parse_args()
    if args.run_comparison:
        compare()
    elif args.previous_submission_dir is not None:
        prepare(args)
    else:
        parser.error("provide --previous-submission-dir or --run-comparison")


if __name__ == "__main__":
    main()
