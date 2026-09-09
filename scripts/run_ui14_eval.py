#!/usr/bin/env python3
"""Full UI14 evaluation using a shared inference queue and the UI5 scorer."""
from __future__ import annotations
import os
import subprocess
import sys
from uuid import uuid4
from pathlib import Path
from datetime import datetime, timezone
from ui14_common import *


def validate_evaluation_manifest(path):
    rows = read_json(path)["tasks"]
    from eaglevl.ui_task_registry import validate_registry
    validate_registry(rows, 14)
    for spec in rows:
        if spec["split"] != "test": raise ValueError("Evaluation must use test only")
        if spec["skip_figma"] != (spec["task_id"] < 5): raise ValueError("UI5/UI9 scoring policy drift")
        records = list(read_jsonl(spec["test"]))
        if len(records) != spec["expected_records"] or not records: raise ValueError("Test image count changed")
        if spec["view_policy"] == "crops":
            from ui5_eval_detector_cache import validate_eval_detector_cache
            kwargs = {}
            if spec["task_id"] >= 5:
                paths = paths_for(Path(path).parent, spec["task_key"], "test")
                kwargs["expected_task_files"] = {spec["task_key"]: Path(spec.get("detector_input", paths["detector_input"]))}
                count = len({file_digest(r["source_image"]) for r in read_jsonl(paths["normalized"])})
            else:
                count = 1555
                kwargs["input_dir"] = Path(spec["test"]).parent
            validate_eval_detector_cache(Path(spec["cache"]), scan_name=spec["scan_name"],
                expected_unique_images=count, required_cache_scope="full_test",
                require_strict_nonoverlap=True, require_raw_detector_edge_alignment=True,
                require_detector_unique_containment=True, **kwargs)
    return rows


def score_ui9(spec, prediction_dir, destination):
    # Reuse the exact Hungarian IoU and illegal-output behavior; do not pass new
    # source keys through the legacy class-id or filename mapping.
    from qwen3vl_merge_and_score_fixed_5tasks import evaluate_merged_file, build_metrics_summary
    key = spec["task_key"]
    gates = [read_json(p) for p in (prediction_dir / key / "gate").glob("*.json")]
    by_image = {str(Path(r["image_path"]).resolve()): r for r in gates}
    merged, positives, negatives, scores = [], 0, 0, {True: [], False: []}
    for row in read_jsonl(spec["test"]):
        positive = bool(row["boxes_px"])
        positives += positive
        negatives += not positive
        gate = by_image.get(str(Path(row["source_image"]).resolve()))
        if gate is None: raise RuntimeError(f"Incomplete image prediction: {key}/{row['source_image_id']}")
        valid = gate["prediction_status"] in ("ok", "defect")
        pred = {"bbox": gate["final_boxes_pixel_xyxy"], "type": key} if valid else None
        merged.append({"image_id": row["source_image_id"], "task_key": key,
                       "objects": {"bbox": row["boxes_px"], "type": key}, "pred_ans": pred})
        probability = gate.get("p_defect")
        if isinstance(probability, (int, float)): scores[positive].append(probability)
    merged_path = destination / f"{key}.merged.jsonl"
    write_jsonl(merged_path, merged)
    result = build_metrics_summary(evaluate_merged_file(str(merged_path), key, 0.1, include_figma=True))
    result.update(positive_count=positives, negative_count=negatives, source_dataset=spec["source_dataset"],
                  source_version=spec["source_version"], task_id=spec["task_id"], view_policy=spec["view_policy"])
    if "negative_quota_policy" in spec:
        result.update(negative_quota_policy=spec["negative_quota_policy"], evaluation_balance="observed_counts")
    gate_summary = {"samples": len(merged), "positive_count": positives, "negative_count": negatives,
                    "p_defect_pos": sum(scores[True])/len(scores[True]) if scores[True] else None,
                    "p_defect_neg": sum(scores[False])/len(scores[False]) if scores[False] else None,
                    "parse_error": result["invalid_pred"], "gate_filtered": 0,
                    "raw_predicted_positive": result["image"]["tp"] + result["image"]["fp"]}
    for name in ("coarse_recall_03", "coarse_recall_05", "selected_slot_iou", "oracle_8slot_iou",
                 "route_top1_match_accuracy", "predicted_center_diversity", "attention_diversity"):
        values = [r[name] for r in gates if isinstance(r.get(name), (int, float))]
        if values: gate_summary[name] = sum(values)/len(values)
    return result, gate_summary


def evaluation_identity(manifest, checkpoint):
    from collect_ui5_metrics import ui_model_signature
    from eaglevl.ui_answer_grammar import decode_contract
    return {"manifest_digest": file_digest(manifest), "eval_set_id": read_json(manifest).get("eval_set_id"),
            "decoder_contract": decode_contract(os.environ.get("UI_EVAL_ANSWER_GRAMMAR", "legacy")),
            "model_signature": ui_model_signature(Path(checkpoint)),
            "git_commit": os.environ.get("GIT_COMMIT", ""), "config_hash": os.environ.get("UI5_CONFIG_HASH", "")}


def is_complete(output_dir, step, manifest, checkpoint):
    from eaglevl.train.ui5_excel_logger import UI5ExcelLogger
    path = Path(output_dir) / "evaluation" / f"ui14-step-{step}.json"
    if not path.is_file(): return False
    row = read_json(path)
    keys = [t.task_key for t in UI_TASKS]
    return (row.get("status") == "success" and set(row.get("tasks", {})) == set(keys)
            and row.get("identity") == evaluation_identity(manifest, checkpoint)
            and UI5ExcelLogger(Path(output_dir) / "diagnostics" / "ui5_training_evaluation.xlsx", keys).has_eval_step(step))


def run(args):
    # The composite dataset already has CPU image/hash evidence. Periodic
    # evaluation must not reread every source image merely to count cache IDs.
    from ui14_verification import current_checks, verification_session
    root = Path(os.environ["UI_EVAL_MANIFEST"]).parent
    if (root/"negative_extension_manifest.json").is_file() and current_checks() is None:
        with verification_session(root):
            return _run(args)
    return _run(args)


def _run(args):
    from patch_locany_checkpoint import patch_checkpoint
    from run_ui5_eval import build_score_command, run_checked
    from collect_ui5_metrics import (load_history, write_history, build_best_checkpoints_document,
                                     collect_gate_metrics, ui_model_signature, print_metric_summary)
    from eaglevl.train.ui5_excel_logger import UI5ExcelLogger, build_eval_rows
    manifest = Path(os.environ["UI_EVAL_MANIFEST"])
    output = Path(args.output_dir)
    workbook = UI5ExcelLogger(output / "diagnostics" / "ui5_training_evaluation.xlsx", [t.task_key for t in UI_TASKS])
    workbook.initialize()
    print(f"[UI14 eval] step={args.step} full test: 14 tasks | Excel: {workbook.path}", flush=True)
    specs = validate_evaluation_manifest(manifest)
    checkpoint = Path(args.checkpoint)
    from eaglevl.ui_task_registry import validate_registry
    checkpoint_config = read_json(checkpoint / "config.json")
    validate_registry(checkpoint_config.get("ui_task_registry"), checkpoint_config.get("ui_num_tasks"))
    if checkpoint_config.get("ui_num_tasks") != 14: raise ValueError("UI14 evaluation requires a 14-task checkpoint")
    for saved, current in zip(checkpoint_config["ui_task_registry"], specs):
        fields = ("task_id", "task_key", "source_dataset", "source_version", "train", "test", "view_policy",
                  "repair_run_id", "normalization_id", "train_source_sha256", "test_source_sha256", "bbox_config")
        if getattr(args, "external_eval_set", False):
            if not args.skip_patch: raise ValueError("External evaluation must not patch the read-only checkpoint")
            fields = ("task_id", "task_key", "relation_family", "view_policy")
        for field in fields:
            if saved.get(field) != current.get(field): raise ValueError(f"Checkpoint/evaluation registry drift: {current['task_key']}.{field}")
    if not args.skip_patch:
        patch_checkpoint(base_model=Path(args.base_model), checkpoint=checkpoint,
                         project_root=Path(args.project_root), force=True, validate_relation_weights=True)
    identity = evaluation_identity(manifest, checkpoint)
    history_file = output / "evaluation/evaluation_history.json"
    for previous in load_history(history_file):
        if previous.get("eval_set_id") != identity.get("eval_set_id"):
            raise ValueError("Different eval_set_id must use a separate output directory")
    if is_complete(output, args.step, manifest, checkpoint):
        print(f"[UI14 eval] reused complete step={args.step}: 14 tasks and Excel rows verified", flush=True)
        return 0
    history_dir = output / "evaluation"
    prediction = output / f"inference-checkpoint-{args.step}-ui14"
    # The external UI5 scorer creates its run directory with exist_ok=False.
    # Keep every scoring attempt separate, including retries after partial
    # scoring/Excel writes. Prediction paths remain stable for image reuse.
    score_run_name = f"ui14-step-{args.step}-{uuid4().hex}"
    destination = history_dir / "raw" / score_run_name
    state_path = history_dir / f"ui14-step-{args.step}.json"
    started = datetime.now(timezone.utc).isoformat()
    workers_per_gpu = getattr(args, "eval_inference_workers_per_gpu", 2)
    exclusive_gpu_tasks = ["synth_loneword"]
    state = {"status": "running", "sft_step": args.step, "init_checkpoint": str(args.base_model),
             "init_cpt_step": 9000, "identity": identity, "tasks": {}, "started": started,
             "eval_gpu_devices": args.eval_gpu_devices, "inference_workers_per_gpu": workers_per_gpu,
             "exclusive_gpu_tasks": exclusive_gpu_tasks, "evaluation_run_dir": str(destination)}
    repair_metadata = {k: read_json(manifest).get(k) for k in ("repair_run_id", "normalization_id")}
    state.update(repair_metadata)
    write_json(state_path, state)
    command = [sys.executable, str(PROJECT_ROOT / "scripts" / "run_ui5_parallel_inference.py"),
        "--checkpoint", str(checkpoint), "--processor-path", str(checkpoint),
        "--input-dir", str(manifest.parent), "--output-dir", str(prediction),
        "--gpu-devices", args.eval_gpu_devices, "--attn-implementation", args.attn_implementation,
        "--workers-per-gpu", str(workers_per_gpu),
        "--exclusive-gpu-tasks", *exclusive_gpu_tasks,
        "--inference-script", str(PROJECT_ROOT / "scripts" / "inference_ui_defect_locany.py"),
        "--eval-manifest", str(manifest), "--inference-crop-mode", "full_image",
        "--relation-gate-mode", "observe", "--enable-pbd", "--save-raw-answer",
        "--tile-nms-iou", str(args.tile_nms_iou),
        "--runtime-profile", str(history_dir / "task_runtime_profile.json")]
    try:
        run_checked(command, cwd=Path(args.project_root), stage="ui14_parallel_inference")
        # UI5 goes through its unmodified scorer entrypoint and frozen test files.
        args.input_dir = Path(specs[0]["test"]).parent
        print(f"[UI14 eval] scoring attempt: {destination} | existing reports preserved; "
              f"prediction cache: {prediction}", flush=True)
        score_command = build_score_command(args, prediction_dir=prediction,
            raw_evaluation_root=destination.parent, run_name=score_run_name)
        run_checked(score_command, cwd=Path(args.project_root), stage="ui5_score")
        candidates = list(destination.rglob("metrics_summary.json"))
        # The legacy all-task report's filename is fixed by that scorer.
        candidates += list(destination.glob("*metrics*.json"))
        metrics = next((read_json(p) for p in dict.fromkeys(candidates)
                        if set(read_json(p).get("tasks", {})) == {t.task_key for t in UI5_TASKS}), None)
        if metrics is None:
            from collect_ui5_metrics import parse_markdown_report
            metrics = parse_markdown_report(destination / "all_tasks_evaluation.txt")
        gate_metrics = collect_gate_metrics(prediction, args.input_dir, Path(args.scorer_root))
        for spec in specs[5:]:
            result, gate = score_ui9(spec, prediction, destination)
            metrics["tasks"][spec["task_key"]] = result
            gate_metrics[spec["task_key"]] = gate
        gate_metrics.update(collect_gate_metrics(prediction, None,
            task_files={s["task_key"]: Path(s["test"]) for s in specs[5:]}))
        for spec in specs:
            metrics["tasks"][spec["task_key"]].update(source_dataset=spec["source_dataset"],
                source_version=spec["source_version"], view_policy=spec["view_policy"], task_id=spec["task_id"])
        for spec in specs[:5]:
            item=metrics["tasks"][spec["task_key"]]; counts=item["image"]
            if all(counts.get(k) is not None for k in ("tp","fp","fn","tn")):
                item.update(positive_count=counts["tp"]+counts["fn"],negative_count=counts["tn"]+counts["fp"])
        if identity.get("eval_set_id"):
            subset={"eval_set_id":identity["eval_set_id"],"subset":"parent_positive_images", "tasks":{}}
            for spec in specs[7:]:
                if spec.get("parent_positive_test"):
                    subset_path=destination/"parent_positive_inputs"/f"{spec['task_key']}.jsonl"
                    write_jsonl(subset_path,[r for r in read_jsonl(spec["parent_positive_test"]) if r["boxes_px"]])
                    value,_=score_ui9({**spec,"test":str(subset_path)},prediction,destination/"parent_positive_subset")
                    subset["tasks"][spec["task_key"]]=value
            write_json(destination/"parent_positive_metrics.json",subset)
        if set(metrics["tasks"]) != {t.task_key for t in UI_TASKS}: raise RuntimeError("Incomplete UI14 metrics")
        write_json(destination / "ui14_metrics.json", metrics)
        write_json(prediction / "_gate_metrics.json", gate_metrics)
        rows = [r for r in load_history(history_dir / "evaluation_history.json") if int(r.get("step", -1)) != args.step]
        row = {"step": args.step, "sft_step": args.step, "checkpoint": str(checkpoint), **repair_metadata,
               "eval_set_id":identity.get("eval_set_id"),"data_digest":identity["manifest_digest"],
               "evaluation_status": "success", "evaluation_run_dir": str(destination),
               "relation_gate_mode": "observe", "evaluation_split": "test",
               "cache_scope": "full_test", "git_commit": identity["git_commit"], "config_hash": identity["config_hash"],
               "ui_model_signature": ui_model_signature(checkpoint), "init_checkpoint": str(args.base_model), "init_cpt_step": 9000,
               "tasks": metrics["tasks"], "image_macro_f1": metrics["macro"]["image"]["f1"],
               "bbox_macro_f1": metrics["macro"]["bbox"]["f1"], "prediction_dir": str(prediction),
               "evaluation_start_time": started, "evaluation_end_time": datetime.now(timezone.utc).isoformat()}
        rows.append(row)
        best, selections = build_best_checkpoints_document(rows)
        row.update(selections[args.step])
        metadata = {**identity, **row, "run_name": os.environ.get("RUN_NAME"), "tc_msed_stage": "m32"}
        excel_rows = build_eval_rows(step=args.step, checkpoint=str(checkpoint), metrics=metrics,
            gate_metrics=gate_metrics, metadata=metadata,
            audit_context={"evaluation_split": "test", "cache_scope": "full_test", "eval_inference_crop_mode": "task_registry",
                           "recipe_digest": file_digest(args.recipe_path), "cache_digest": file_digest(manifest),
                           "crop_train_mode": "crop_only", "ui_sampling_mode": "task_source_balanced_rotating", "scan_name": SCAN_NAME})
        workbook.append_eval(args.step, excel_rows)
        for step, selection in selections.items():
            workbook.update_checkpoint_status(step, **{k: selection[k] for k in ("is_best_image", "is_best_bbox", "is_4000_milestone", "checkpoint_kept")})
        write_history(history_dir, rows)
        write_json(history_dir / "best_checkpoints.json", best)
        state.update(status="success", tasks=metrics["tasks"], finished=row["evaluation_end_time"])
        write_json(state_path, state)
        print_metric_summary(step=args.step, metrics=metrics, retention=selections[args.step],
                             diagnostics_xlsx=workbook.path)
        print(f"[UI14 eval] complete step={args.step}: 14/14 tasks, {len(excel_rows)} Excel rows; "
              "UI9 task metrics and ui9_macro/micro are saved in Excel and evaluation outputs", flush=True)
        return 0
    except Exception as exc:
        state.update(status="failed", error=str(exc))
        write_json(state_path, state)
        raise


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    raise SystemExit(0 if is_complete(args.output_dir, args.step, args.manifest, args.checkpoint) else 1)
