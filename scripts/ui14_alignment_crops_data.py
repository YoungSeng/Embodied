"""An alignment crop experiment over the frozen available neg11 data."""
from collections import Counter
from pathlib import Path
from ui14_common import read_json, read_jsonl, write_json, file_digest, paths_for, SCAN_NAME
from eaglevl.ui_task_registry import task_from_spec


def prepare_crop_registry(root, evaluation):
    root = Path(root)
    registry = read_json(root / "task_registry.json")
    recipe = read_json(root / "training_recipe.json")
    spec = next(s for s in evaluation["tasks"] if s["task_key"] == "ui_alignment")
    spec.update(view_policy="crops", train=str(paths_for(root, "ui_alignment", "train")["derived"]),
                cache=str(paths_for(root, "ui_alignment", "test")["cache"]), scan_name=SCAN_NAME,
                detector_input=str(paths_for(root, "ui_alignment", "test")["detector_input"]))
    registry["tasks"][5].update(view_policy="crops", train=spec["train"], test=spec["test"])
    recipe["ui_alignment"].update(annotation=spec["train"], view_policy="crops", data_augment=False)
    if registry["tasks"][4]["view_policy"] != "full_image": raise ValueError("content_missing must stay full_image")
    write_json(root / "task_registry.json", registry)
    write_json(root / "training_recipe.json", recipe)
    return {"task_key": "ui_alignment", "task_id": 5, "view_policy": "crops",
            "content_missing_view_policy": "full_image", "status": "cache_pending",
            "gt_used_for_geometry": False, "coordinate_projection_repeated": False}


def finalize_crops(root, output):
    from ui14_alignment_crops_common import PROFILE, protect_context
    from ui14_repair import validate_normalization
    from ui14_crop_materialization import load_completed
    from prepare_ui14_sft import validate_task_cache
    from ui14_verification import current_checks
    from ui14_profile import validate_prepared_profile
    root = Path(root)
    protect_context(root, output)
    report = read_json(root / "cpu_check_report.json")
    if report.get("profile") != PROFILE: raise ValueError("Expected this experiment's prepared crop data")
    if report.get("ready"):
        validate_prepared_profile({"UI14_DATA_ROOT": str(root), "OUTPUT_DIR": str(output), "INIT_CHECKPOINT": report["init_checkpoint"]})
        print("[alignment crops] reused complete recipe, test and cache bindings", flush=True)
        return report
    snapshot = validate_normalization(root)
    parent = Path(report["frozen_parent"]["parent"])
    if file_digest(parent / "cpu_check_report.json") != report["frozen_parent"]["source_report_sha256"]:
        raise ValueError("Frozen parent report changed")
    registry = read_json(root / "task_registry.json")
    task = task_from_spec(registry["tasks"][5])
    if task.view_policy != "crops" or registry["tasks"][4]["view_policy"] != "full_image":
        raise ValueError("Expected ui_alignment=crops and content_missing=full_image")
    result = {"task_key": task.task_key, "task_id": 5, "view_policy": "crops",
              "content_missing_view_policy": "full_image", "gt_used_for_geometry": False,
              "coordinate_projection_repeated": False, "splits": {}}
    artifacts = dict(report["artifact_digests"])
    for split in ("train", "test"):
        paths = paths_for(root, task.task_key, split)
        sources = list(read_jsonl(paths["normalized"]))
        rows = load_completed(root, task, split, sources)
        count = sum(1 for _ in read_jsonl(paths["cache"] / "manifest/unique_images.jsonl"))
        validate_task_cache(root, task, split, count)
        if any(r["view_policy"] != "crops" or r["_ui5_record_kind"] != "crop" or r["split"] != split for r in rows):
            raise ValueError("Alignment crop labels have the wrong input policy/split")
        coverage = read_json(paths["derived"].with_suffix(".coverage.json"))
        result["splits"][split] = {"original_records": len(sources), "source_images": len({r["source_image_id"] for r in sources}),
            "crop_records": len(rows), "positive_crops": sum(bool(r["boxes_px"]) for r in rows),
            "negative_crops": sum(not r["boxes_px"] for r in rows), "gt_count": sum(r["gt_count"] for r in coverage),
            "fully_contained_gt": sum(r["fully_contained_gt"] for r in coverage),
            "fragmented_gt": sum(len(r["uncontained_gt"]) for r in coverage)}
        files = [paths["derived"], paths["derived"].with_suffix(".coverage.json")]
        files += [paths["cache"] / n for n in ("ui14_crop_complete.json", "ui14_label_cache_ready.json",
            "crop_index/images.jsonl", "task_input_manifest.json", "manifest/ui14_prepare_ready.json",
            "manifest/unique_images.jsonl", SCAN_NAME + "/detector_scan_crops.jsonl", SCAN_NAME + "/eval_detector_cache_ready.json")]
        artifacts.update({str(p.relative_to(root)): file_digest(p) for p in files})
        if split == "train": train = rows
    recipe = read_json(root / "training_recipe.json")
    recipe[task.task_key]["length"] = len(train)
    write_json(root / "training_recipe.json", recipe)
    from eaglevl.train.ui_defect_data import (negative_kind, recipe_sampling_ratio, build_task_source_balanced_rotating_plan,
                                             materialize_task_source_balanced_rotating_indices)
    ratio = recipe_sampling_ratio(recipe[task.task_key])
    plan = build_task_source_balanced_rotating_plan(train, negative_to_positive_ratio=ratio)
    draws = materialize_task_source_balanced_rotating_indices(plan, seed=42)
    positive = sum(bool(train[i]["boxes_px"]) for i in draws)
    sampling = read_json(root / "sampling_stats.json")
    sampling[task.task_key].update(train_records=len(train), derived_positive=sum(bool(r["boxes_px"]) for r in train),
        derived_negative=sum(not r["boxes_px"] for r in train), epoch_draws=len(draws), sampled_positive=positive,
        sampled_negative=len(draws)-positive, negative_to_positive_ratio=ratio,
        sampled_negative_kinds=dict(Counter(negative_kind(train[i]) for i in draws)),
        actual_sampled_negative_to_positive_ratio=(len(draws)-positive)/positive if positive else None,
        both_labels_available=bool(any(r["boxes_px"] for r in train) and any(not r["boxes_px"] for r in train)))
    write_json(root / "sampling_stats.json", sampling)
    result["sampling"] = sampling[task.task_key]
    from ui14_alignment_data import training_answer_contract
    label_contract = training_answer_contract(recipe)
    write_json(root / "alignment_answer_contract.json", label_contract)
    from run_ui14_eval import validate_evaluation_manifest
    evaluation = read_json(root / "evaluation_manifest.json")
    parent_eval = read_json(parent / "evaluation_manifest.json")
    if evaluation["eval_set_id"] != parent_eval["eval_set_id"]: raise ValueError("Frozen test identity changed")
    if any(s["test"] != p["test"] for s, p in zip(evaluation["tasks"], parent_eval["tasks"])):
        raise ValueError("This experiment must keep every original-image test list")
    validate_evaluation_manifest(root / "evaluation_manifest.json")
    checks = current_checks()
    if checks is None: raise ValueError("Finalize requires a CPU verification session")
    checks.export_images(root / "verification/image_evidence.jsonl")
    result["status"] = "complete"
    write_json(root / "alignment_crop_check.json", result)
    from ui14_checks import render_formal_yaml
    render_formal_yaml(root, profile=PROFILE)
    artifacts.update({name: file_digest(root / name) for name in ("training_recipe.json", "sampling_stats.json", "alignment_answer_contract.json",
        "alignment_crop_check.json", "verification/image_evidence.jsonl", "formal_job.yaml", "formal_runtime.json")})
    report.update(artifact_digests=artifacts, ready=True, stage="complete", alignment_crop_check=result, training_answer_contract=label_contract,
                  normalization_id=snapshot["normalization_id"])
    write_json(root / "cpu_check_report.json", report)
    try:
        validate_prepared_profile({"UI14_DATA_ROOT": str(root), "OUTPUT_DIR": str(output), "INIT_CHECKPOINT": report["init_checkpoint"]})
    except BaseException:
        report["ready"] = False
        write_json(root / "cpu_check_report.json", report)
        raise
    return report
