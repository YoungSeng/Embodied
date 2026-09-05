#!/usr/bin/env python3
"""CPU-only UI9 normalization, UI14 recipe assembly and production readiness checks."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from pathlib import Path

from ui14_common import *
from ui14_annotations import main_image, source_boxes, training_record, crop_boxes, answer
from eaglevl.train.ui_defect_data import (identify_ui_defect_task, is_positive_ui_defect,
    build_task_source_balanced_rotating_plan, materialize_task_source_balanced_rotating_indices)
from locany_ui5_common import TASK_JSONL


def source_manifest_tasks(manifest):
    value = manifest.get("tasks", manifest.get("datasets", manifest.get("sources", manifest)))
    if isinstance(value, list):
        value = {r.get("task_key", r.get("key", r.get("name"))): r for r in value}
    if not isinstance(value, dict):
        raise ValueError("manifest.json must contain tasks/datasets/sources keyed by task_key")
    missing = {t.task_key for t in UI9_TASKS} - value.keys()
    if missing:
        raise ValueError(f"UI9 manifest is missing tasks: {sorted(missing)}")
    return value


def normalize(args):
    source_root, root = Path(args.ui9_data_root).resolve(), Path(args.output_dir).resolve()
    manifest_path = source_root / "manifest.json"
    manifest = read_json(manifest_path)
    sources = source_manifest_tasks(manifest)
    rows_by_task, stats, registry = {}, {}, []
    identity_cache = {}
    for task in UI_TASKS:
        spec = task.to_dict()
        if task.task_id < 5:
            spec.update(train=str(args.ui5_recipe), test=str(Path(args.ui5_test_dir) / TASK_JSONL[task.task_key]))
            spec["source_version"] = "crop_audit_v4_gt_repair"
        else:
            source = sources[task.task_key]
            if not isinstance(source, dict):
                raise ValueError(f"Bad manifest entry for {task.task_key}")
            spec["source_dataset"] = source.get("source_dataset", task.source_dataset)
            spec["source_version"] = str(source.get("source_version", source.get("version", task.source_dataset if task.task_id >= 7 else manifest.get("version", "v1"))))
            for split in ("train", "test"):
                paths = paths_for(root, task.task_key, split)
                src = source_root / task.task_key / f"{split}.jsonl"
                records, seen_records = [], set()
                for index, raw in enumerate(read_jsonl(src)):
                    # The existing exports own the split. Do not infer it from image names.
                    if raw.get("split", split) != split:
                        raise ValueError(f"Source split mismatch: {src}:{index + 1}")
                    image = main_image(raw, src.parent, task.task_id >= 7)
                    if str(image) not in identity_cache:
                        identity_cache[str(image)] = image_identity(image)
                    content_id, width, height = identity_cache[str(image)]
                    annotation_field = "Objects" if task.task_id >= 7 else "objects"
                    if annotation_field not in raw:
                        raise ValueError(f"Missing {annotation_field}, cannot infer negative label: {src}:{index+1}")
                    boxes = source_boxes(raw, width, height, synthetic=task.task_id >= 7)
                    record_id = str(raw.get("source_record_id", raw.get("id", raw.get("ID", index))))
                    if record_id in seen_records:
                        raise ValueError(f"Duplicate source record ID: {task.task_key}/{split}/{record_id}")
                    seen_records.add(record_id)
                    row = {"source_dataset": spec["source_dataset"], "source_version": spec["source_version"],
                           "source_record_id": record_id, "source_image_id": content_id,
                           "source_image": str(image), "split": split, "task_key": task.task_key,
                           "task_id": task.task_id, "crop_id": "full", "width": width, "height": height,
                           "boxes_px": boxes, "image": str(image), "view_policy": task.view_policy,
                           "source_jsonl": str(src), "source_line": index + 1,
                           "source_metadata": raw}
                    records.append(row)
                if not records:
                    raise ValueError(f"Empty exported split: {src}")
                expected = source.get(f"{split}_count", source.get(f"{split}_records"))
                if isinstance(expected, (int, float)) and int(expected) != len(records):
                    raise ValueError(f"Export count disagrees with manifest: {task.task_key}/{split}: {len(records)} != {expected}")
                write_jsonl(paths["normalized"], records)
                # Detector entrypoint receives no annotations, source messages, or reference images.
                inputs = [{"image": r["source_image"]} for r in records]
                write_jsonl(paths["detector_input"], inputs)
                write_json(paths["detector_inputs"], {task.task_key: str(paths["detector_input"])})
                rows_by_task[task.task_key, split] = records
                spec[split] = str(paths["normalized"])
                stats[f"{task.task_key}/{split}"] = {
                    "records": len(records), "unique_images": len({r["source_image_id"] for r in records}),
                    "positive_count": sum(bool(r["boxes_px"]) for r in records),
                    "negative_count": sum(not r["boxes_px"] for r in records),
                    "input_sha256": file_digest(src), "manifest_entry": source}
        registry.append(spec)
    write_json(root / "task_registry.json", {"schema_version": 1, "tasks": registry,
               "input_manifest": str(manifest_path), "input_manifest_sha256": file_digest(manifest_path)})
    write_json(root / "normalization_stats.json", {"tasks": stats, "originals_modified": False,
               "random_split_used": False, "reference_negatives_added": 0})
    duplicates = image_overlaps([r for values in rows_by_task.values() for r in values])
    write_json(root / "ui9_image_overlap.json", duplicates)
    print(f"Normalized 9 source tasks under {root}; train/test duplicate images={len(duplicates['train_test'])}")


def image_overlaps(rows):
    identities = defaultdict(set)
    for row in rows:
        identities[row["source_image_id"]].add((row["task_key"], row["split"]))
    return {"identity": "sha256(decoded RGB dimensions and pixels)",
            "train_test": [{"source_image_id": key, "uses": sorted(uses)} for key, uses in identities.items()
                           if {split for _, split in uses} == {"train", "test"}],
            "cross_source": [{"source_image_id": key, "uses": sorted(uses)} for key, uses in identities.items()
                             if len({task for task, _ in uses}) > 1]}


def validate_task_cache(root, task, split, expected):
    from ui5_eval_detector_cache import validate_eval_detector_cache
    paths = paths_for(root, task.task_key, split)
    return validate_eval_detector_cache(paths["cache"], scan_name=SCAN_NAME,
        expected_unique_images=expected, required_cache_scope="full_test" if split == "test" else "full_train",
        expected_task_files={task.task_key: paths["detector_input"]},
        require_strict_nonoverlap=True, require_raw_detector_edge_alignment=True,
        require_detector_unique_containment=True)


def crop_annotations(root, task, split, records):
    from PIL import Image, ImageOps
    paths = paths_for(root, task.task_key, split)
    validate_task_cache(root, task, split, len({file_digest(r["source_image"]) for r in records}))
    plans = list(read_jsonl(paths["cache"] / SCAN_NAME / "detector_scan_crops.jsonl"))
    index = {str(Path(p).resolve()): row for row in plans for p in row["image_paths"]}
    derived, coverage = [], []
    for row in records:
        plan = index[row["source_image"]]
        if (plan["width"], plan["height"]) != (row["width"], row["height"]):
            raise ValueError("Crop cache screenshot dimensions changed")
        covered = set()
        with Image.open(row["source_image"]) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
            for i, tile in enumerate(plan["tiles"]):
                crop = tile
                crop_id = f"scan-{i:03d}"
                target = paths["crop_images"] / f"{row['source_image_id']}-{crop_id}.png"
                target.parent.mkdir(parents=True, exist_ok=True)
                expected_crop = image.crop(crop)
                if target.exists():
                    with Image.open(target) as saved:
                        if saved.size != expected_crop.size or saved.convert("RGB").tobytes() != expected_crop.tobytes():
                            raise ValueError(f"Stale training crop content: {target}; use a new data root or remove this derived crop")
                else:
                    expected_crop.save(target)
                boxes, complete = crop_boxes(row["boxes_px"], crop)
                covered.update(complete)
                record = training_record(row, task, target, boxes, crop[2]-crop[0], crop[3]-crop[1], crop_id)
                record.update(crop_box=crop, crop_width=crop[2]-crop[0], crop_height=crop[3]-crop[1],
                              detector_plan_gt_used=False, crop_boxes_px=boxes, crop_image_sha256=file_digest(target))
                derived.append(record)
        coverage.append({"source_record_id": row["source_record_id"], "gt_count": len(row["boxes_px"]),
                         "fully_contained_gt": len(covered), "uncontained_gt": sorted(set(range(len(row["boxes_px"]))) - covered)})
    write_json(paths["derived"].with_suffix(".coverage.json"), coverage)
    return derived


def legacy_train(args):
    recipe_path = Path(args.ui5_recipe).resolve(strict=True)
    recipe = read_json(recipe_path)
    audit_dir = recipe_path.parent.parent
    from run_ui5_crop_audit import validate_training_ready_marker
    from validate_ui5_crop_training_ready import validate_expected_train_mode
    audit = validate_training_ready_marker(audit_dir, recipe_path=recipe_path)
    validate_expected_train_mode(audit, "crop_only")
    task_samples = list(read_jsonl(audit_dir.parent / "manifest" / "task_samples.jsonl"))
    source_by_sample = {r["sample_id"]: r["canonical_path"] for r in task_samples}
    source_by_image = {r["image_id"]: r["canonical_path"] for r in task_samples}
    groups = defaultdict(list)
    identities = {}
    for entry in recipe.values():
        exclusions = set()
        if entry.get("excluded_samples"):
            exclusion_path = Path(entry["excluded_samples"])
            if not exclusion_path.is_absolute(): exclusion_path = recipe_path.parent / exclusion_path
            exclusions = {(str(r["sample_id"]), str(r["task"])) for r in read_jsonl(exclusion_path)}
        annotations = entry["annotation"]
        if isinstance(annotations, str): annotations = [annotations]
        for value in annotations:
            path = Path(value)
            if not path.is_absolute(): path = recipe_path.parent / path
            for record in read_jsonl(path):
                if (str(record.get("_ui5_sample_id", "")), str(record.get("_ui5_task", ""))) in exclusions:
                    if record.get("_ui5_crop_source") == "manual_gt_repair":
                        raise ValueError("Audited exclusion would remove a manual repair")
                    continue
                task_info = identify_ui_defect_task(record)
                if task_info is None or task_info[1] >= 5: raise ValueError("Legacy recipe has an unknown task")
                task = UI_TASKS[task_info[1]]
                row = dict(record)
                image = row["image"]
                if isinstance(image, list): image = image[0]
                base = Path(entry.get("root", recipe_path.parent))
                if not base.is_absolute(): base = recipe_path.parent / base
                row["image"] = str(Path(image) if Path(image).is_absolute() else base / image)
                source = (row.get("_ui5_source_image") or row.get("source_image")
                          or source_by_sample.get(row.get("_ui5_sample_id"))
                          or source_by_image.get(row.get("_ui5_image_id")))
                if not source: raise ValueError("Audited UI5 record lacks source image path")
                source = str(Path(source) if Path(source).is_absolute() else base / source)
                if source not in identities: identities[source] = image_identity(source)[0]
                row.update(task_id=task.task_id, task_key=task.task_key, split="train",
                           source_dataset="ui5", source_version="crop_audit_v4_gt_repair",
                           source_image=source, source_image_id=identities[source],
                           source_record_id=str(row.get("_ui5_sample_id", row.get("id", ""))),
                           crop_id=str(row.get("crop_id", row.get("_ui5_crop_id", Path(row["image"]).stem))),
                           view_policy=task.view_policy)
                groups[task.task_key].append(row)
    if set(groups) != {t.task_key for t in UI5_TASKS}: raise ValueError("Legacy recipe does not contain all UI5 tasks")
    return groups


def finalize(args):
    root = Path(args.output_dir).resolve()
    registry_document = read_json(root / "task_registry.json")
    registry = load_registry(root / "task_registry.json")
    legacy = legacy_train(args)
    recipe, evaluation, stats, all_sources = {}, [], {}, []
    for task in UI_TASKS:
        spec = registry[task.task_id]
        if task.task_id < 5:
            train = legacy[task.task_key]
            paths = paths_for(root, task.task_key, "train")
            write_jsonl(paths["derived"], train)
            test_path = Path(args.ui5_test_dir) / TASK_JSONL[task.task_key]
            # Preserve the test file and scorer's no_figma policy exactly.
            for record in read_jsonl(test_path):
                from qwen3vl_merge_and_score_fixed_5tasks import extract_image_path, is_figma_sample
                if is_figma_sample(record): continue
                image = extract_image_path(record)
                if not image: raise ValueError(f"Missing UI5 test screenshot in {test_path}")
                image = Path(image)
                if not image.is_absolute(): image = test_path.parent / image
                all_sources.append({"source_image_id": image_identity(image)[0], "task_key": task.task_key, "split": "test"})
            cache = str(args.ui5_cache) if task.view_policy == "crops" else None
        else:
            for split in ("train", "test"):
                paths = paths_for(root, task.task_key, split)
                normalized = list(read_jsonl(paths["normalized"]))
                all_sources.extend(normalized)
                if task.view_policy == "crops":
                    records = crop_annotations(root, task, split, normalized)
                else:
                    records = [training_record(r, task, r["source_image"], r["boxes_px"], r["width"], r["height"]) for r in normalized]
                write_jsonl(paths["derived"], records)
                if split == "train": train = records
            original_test = list(read_jsonl(paths_for(root, task.task_key, "test")["normalized"]))
            by_image = {}
            for record in original_test:
                key = record["source_image_id"]
                if key not in by_image:
                    by_image[key] = {k: v for k, v in record.items() if k != "source_metadata"}
                    by_image[key]["source_record_ids"] = []
                    by_image[key]["boxes_px"] = []
                by_image[key]["source_record_ids"].append(record["source_record_id"])
                for box in record["boxes_px"]:
                    if box not in by_image[key]["boxes_px"]: by_image[key]["boxes_px"].append(box)
            test_path = root / "evaluation_inputs" / f"{task.task_key}.jsonl"
            write_jsonl(test_path, by_image.values())
            cache = str(paths_for(root, task.task_key, "test")["cache"]) if task.view_policy == "crops" else None
        all_sources.extend(train)
        paths = paths_for(root, task.task_key, "train")
        spec["train"], spec["test"] = str(paths["derived"]), str(test_path)
        recipe[task.task_key] = {"root": "/", "annotation": str(paths["derived"]), "data_augment": False,
            "repeat_time": 1, "length": len(train), "ui5_crop_recipe": True, "view_policy": task.view_policy,
            "task_key": task.task_key, "task_id": task.task_id, "sampling_weight": 1.0,
            "ui_sampling_mode": "task_source_balanced_rotating"}
        plan = build_task_source_balanced_rotating_plan(train)
        draws = materialize_task_source_balanced_rotating_indices(plan, seed=42)
        stats[task.task_key] = {"train_records": len(train), "source_images": len({r["source_image_id"] for r in train}),
            "positive_records": sum(is_positive_ui_defect(r) for r in train), "negative_records": sum(not is_positive_ui_defect(r) for r in train),
            "sampling_probability": 1/14, "epoch_draws": len(draws),
            "sampled_positive": sum(is_positive_ui_defect(train[i]) for i in draws),
            "sampled_negative": sum(not is_positive_ui_defect(train[i]) for i in draws),
            "manual_repair_count": sum(r.get("_ui5_crop_source") == "manual_gt_repair" for r in train)}
        evaluation.append({**spec, "test": str(test_path), "split": "test", "cache": cache,
                           "skip_figma": task.task_id < 5, "scan_name": SCAN_NAME,
                           "expected_records": sum(1 for _ in read_jsonl(test_path))})
    write_json(root / "task_registry.json", {**registry_document, "tasks": registry})
    write_json(root / "training_recipe.json", recipe)
    write_json(root / "evaluation_manifest.json", {"schema_version": 1, "tasks": evaluation})
    write_json(root / "sampling_stats.json", stats)
    write_json(root / "image_overlap.json", image_overlaps(all_sources))
    check(args)


def check(args):
    root = Path(args.output_dir).resolve()
    registry = load_registry(root / "task_registry.json")
    recipe = read_json(root / "training_recipe.json")
    evaluation = read_json(root / "evaluation_manifest.json")["tasks"]
    errors, results, coverage_results, external_digests = [], {}, {}, {}
    bound_artifacts = {"task_registry.json", "training_recipe.json", "evaluation_manifest.json", "sampling_stats.json", "image_overlap.json", "normalization_stats.json"}
    identity_cache = {}
    expected = {t.task_key for t in UI_TASKS}
    if set(recipe) != expected or {t["task_key"] for t in evaluation} != expected or len(evaluation) != 14:
        errors.append("Recipe/evaluation task sets are not exactly 14")
    for spec in registry:
        task = get_task(spec["task_id"])
        try:
            train = list(read_jsonl(recipe[task.task_key]["annotation"]))
            if not train: raise ValueError("Empty training stream")
            for row in train:
                if row["split"] != "train" or identify_ui_defect_task(row)[1] != task.task_id: raise ValueError("Recipe route/split mismatch")
                if not Path(row["image"]).is_file(): raise FileNotFoundError(row["image"])
            if recipe[task.task_key]["sampling_weight"] != 1.0: raise ValueError("Formal task sampling weights must all be one")
            bound_artifacts.add(str(Path(spec["train"]).relative_to(root)))
            if task.task_id >= 5:
                bound_artifacts.add(str(Path(spec["test"]).relative_to(root)))
                for split in ("train", "test"):
                    p = paths_for(root, task.task_key, split)
                    records = list(read_jsonl(p["normalized"]))
                    for name in ("normalized", "derived", "detector_input", "detector_inputs"):
                        bound_artifacts.add(str(p[name].relative_to(root)))
                    source_path = Path(records[0]["source_jsonl"])
                    external_digests[str(source_path)] = file_digest(source_path)
                    original_stat = read_json(root / "normalization_stats.json")["tasks"][f"{task.task_key}/{split}"]
                    if external_digests[str(source_path)] != original_stat["input_sha256"]:
                        raise ValueError("Original export changed after normalization")
                    for row in records:
                        if row["source_image"] not in identity_cache:
                            identity_cache[row["source_image"]] = image_identity(row["source_image"])
                        current = identity_cache[row["source_image"]]
                        if current != (row["source_image_id"], row["width"], row["height"]):
                            raise ValueError("Normalized screenshot content changed")
                    by_id = {r["source_record_id"]: r for r in records}
                    for derived in read_jsonl(p["derived"]):
                        source = by_id[derived["source_record_id"]]
                        if derived["split"] != split or derived["source_image_id"] != source["source_image_id"] or identify_ui_defect_task(derived)[1] != task.task_id:
                            raise ValueError("Derived annotation source/split/task drift")
                        boxes, w, h = source["boxes_px"], source["width"], source["height"]
                        if task.view_policy == "crops":
                            boxes, _ = crop_boxes(boxes, derived["crop_box"])
                            w, h = derived["crop_width"], derived["crop_height"]
                            if file_digest(derived["image"]) != derived["crop_image_sha256"]:
                                raise ValueError("Cached crop image changed")
                        expected_answer = answer(boxes, w, h, task.prompt_label)
                        if derived["conversations"] != [{"from": "human", "value": "<image>\n" + task.prompt}, {"from": "gpt", "value": expected_answer}]:
                            raise ValueError("Prompt or crop-coordinate annotation drift")
                    if task.view_policy == "crops":
                        validate_task_cache(root, task, split, len({file_digest(r["source_image"]) for r in records}))
                        coverage = read_json(p["derived"].with_suffix(".coverage.json"))
                        coverage_results[f"{task.task_key}/{split}"] = {
                            "gt_count": sum(r["gt_count"] for r in coverage),
                            "fully_contained_gt": sum(r["fully_contained_gt"] for r in coverage),
                            "fragmented_gt": sum(len(r["uncontained_gt"]) for r in coverage),
                            "plan_uses_gt": False, "coverage_policy": "report; never alter input-only plans using GT"}
                        marker = p["cache"] / SCAN_NAME / "eval_detector_cache_ready.json"
                        bound_artifacts.add(str(marker.relative_to(root)))
            results[task.task_key] = "pass"
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            errors.append(f"{task.task_key}: {exc}")
            results[task.task_key] = "fail"
    try:
        from ui5_eval_detector_cache import validate_eval_detector_cache
        validate_eval_detector_cache(Path(args.ui5_cache), scan_name=SCAN_NAME, expected_unique_images=1555,
            input_dir=Path(args.ui5_test_dir), required_cache_scope="full_test", require_strict_nonoverlap=True,
            require_raw_detector_edge_alignment=True, require_detector_unique_containment=True)
        from run_ui14_eval import validate_evaluation_manifest
        validate_evaluation_manifest(root / "evaluation_manifest.json")
        from ui14_checks import validate_initial_checkpoint, render_formal_yaml
        checkpoint_report = validate_initial_checkpoint(Path(args.init_checkpoint))
        yaml_path, runtime_path = render_formal_yaml(root)
        bound_artifacts.update((yaml_path.name, runtime_path.name))
        for path in (Path(args.ui5_recipe), Path(args.ui5_cache) / SCAN_NAME / "eval_detector_cache_ready.json", Path(args.init_checkpoint) / "config.json"):
            external_digests[str(path)] = file_digest(path)
        registry_doc = read_json(root / "task_registry.json")
        external_digests[registry_doc["input_manifest"]] = file_digest(registry_doc["input_manifest"])
        if external_digests[registry_doc["input_manifest"]] != registry_doc["input_manifest_sha256"]:
            raise ValueError("Source manifest changed after normalization")
    except (OSError, ValueError, RuntimeError, KeyError) as exc: errors.append(str(exc))
    overlaps = read_json(root / "image_overlap.json")
    # Frozen exports are never silently resplit or filtered. Surface every original-image overlap.
    report = {"cpu_only": True, "gpu_loaded": False, "tasks": results, "errors": errors,
              "train_test_duplicate_images": len(overlaps["train_test"]), "overlap_report": str(root / "image_overlap.json"),
              "overlap_policy": "report frozen source splits; no repartition or filtering",
              "init_checkpoint": str(args.init_checkpoint), "init_cpt_step": 9000,
              "initialization": locals().get("checkpoint_report"), "sft_start_step": 0,
              "crop_coverage": coverage_results,
              "yaml": str(root / "formal_job.yaml"), "formal_runtime": str(root / "formal_runtime.json"),
              "artifact_digests": {name: file_digest(root / name) for name in sorted(bound_artifacts)},
              "external_digests": external_digests,
              "registry_count": len(registry), "evaluation_count": len(evaluation), "ready": not errors}
    write_json(root / "cpu_check_report.json", report)
    if errors: raise RuntimeError("UI14 CPU check failed: " + "; ".join(errors))
    print(json.dumps(report, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("normalize", "finalize", "check"), required=True)
    parser.add_argument("--ui9-data-root", default=UI9_DATA_ROOT)
    parser.add_argument("--output-dir", default=DATA_ROOT)
    parser.add_argument("--ui5-recipe", default=UI5_RECIPE)
    parser.add_argument("--ui5-test-dir", default=WORKSPACE + "/data")
    parser.add_argument("--ui5-cache", default=WORKSPACE + "/code/Eagle_LocateUI5_v4/Embodied-ui5-det-crop/work_dirs/ui5_eval_detector_cache_horizontal_v5")
    parser.add_argument("--init-checkpoint", default=INIT_CHECKPOINT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    {"normalize": normalize, "finalize": finalize, "check": check}[args.stage](args)
