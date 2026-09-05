"""CPU repair provenance, parser impact audit, and derivative bindings."""
from __future__ import annotations
import ast
from collections import Counter, defaultdict
import math
import re
from ui14_common import *
from ui9_source_parser import is_box


def source_manifest_tasks(manifest):
    value = manifest.get("tasks", manifest.get("datasets", manifest.get("sources", manifest)))
    if isinstance(value, list):
        value = {r.get("task_key", r.get("key", r.get("name"))): r for r in value}
    if not isinstance(value, dict):
        raise ValueError("manifest.json must contain tasks/datasets/sources keyed by task_key")
    if {t.task_key for t in UI9_TASKS} - value.keys():
        raise ValueError("UI9 manifest does not contain all nine registered sources")
    return value


def parser_provenance():
    path = PROJECT_ROOT / "scripts/ui9_source_parser.py"
    provenance = read_json(path.with_name("ui9_parser_provenance.json"))
    nodes = {n.name: n for n in ast.parse(path.read_text(encoding="utf-8")).body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    for name, expected in provenance["symbols"].items():
        actual = hashlib.sha256(ast.dump(nodes[name], include_attributes=False).encode()).hexdigest()
        if actual != expected:
            raise ValueError(f"Extracted preparation parser changed without reconciliation: {name}")
    import ui9_source_parser
    if any(getattr(ui9_source_parser, name) != value for name, value in provenance["constants"].items()):
        raise ValueError("Extracted preparation path constants changed")
    return {**provenance, "module_sha256": file_digest(path),
            "consumer_sha256": file_digest(PROJECT_ROOT / "scripts/ui14_annotations.py"),
            "legacy_consumer_commit": "5d1e07bd4c1eb347b094d2463570ec9249cccf56",
            "legacy_consumer_sha256": file_digest(PROJECT_ROOT / "scripts/ui14_annotations_legacy_v1.py")}


def assert_repair_idle(source_root):
    if (Path(source_root) / ".work/repair_pending.json").exists():
        raise ValueError("UI9 repair publication is pending; the current split is not a complete repair batch")


def capture_repair_snapshot(source_root):
    source_root = Path(source_root).resolve(strict=True)
    assert_repair_idle(source_root)
    manifest = read_json(source_root / "manifest.json")
    repair = read_json(source_root / "repair_summary.json")
    sources = source_manifest_tasks(manifest)
    run_id = repair.get("run_id")
    history = manifest.get("repair_history", [])
    if not run_id or not history or history[-1].get("run_id") != run_id:
        raise ValueError("manifest repair_history and repair_summary run_id disagree")
    if repair.get("check_status") != "PASS" or manifest.get("preparation_errors"):
        raise ValueError("Latest repaired UI9 batch has not passed preparation checks")
    if manifest.get("split_unit") != "global_page_for_synthetic" or not manifest.get("require_page_disjoint"):
        raise ValueError("Manifest does not describe the completed joint synthetic page split")
    repair_tasks = {r["dataset"]: r for r in repair["datasets"]}
    if set(repair_tasks) != {t.task_key for t in UI9_TASKS}:
        raise ValueError("Repair summary must contain exactly nine source tasks")
    files = {str(source_root / name): file_digest(source_root / name)
             for name in ("manifest.json", "repair_summary.json")}
    for task in UI9_TASKS:
        source = sources[task.task_key]
        kind = "synthetic" if task.task_id >= 7 else "annotated"
        if source.get("kind") != kind:
            raise ValueError(f"Manifest kind disagrees with task: {task.task_key}")
        bbox = source.get("bbox", {})
        if kind == "synthetic" and (source.get("split_origin") != "repaired_global_page"
                                   or bbox.get("mode") != "logical-width" or bbox.get("width") != 375):
            raise ValueError(f"Synthetic repair split/375 coordinate contract changed: {task.task_key}")
        for split in ("train", "test"):
            # Never glob recursively: backups, staged files and quarantine are not inputs.
            path = source_root / task.task_key / f"{split}.jsonl"
            files[str(path)] = file_digest(path)
            if source.get(f"{split}_records") != repair_tasks[task.task_key].get(f"after_{split}"):
                raise ValueError(f"Repair and manifest counts disagree: {task.task_key}/{split}")
    remaining = sum(r["after_train"] + r["after_test"] for r in repair_tasks.values())
    if remaining + repair.get("quarantined_records", 0) != repair.get("input_records"):
        raise ValueError("Repair input/quarantine/remaining record totals disagree")
    snapshot = {"schema_version": 2, "source_root": str(source_root), "repair_run_id": run_id,
                "source_files": files, "parser": parser_provenance(), "repair_summary": repair,
                "manifest_split_unit": manifest["split_unit"], "sources": sources}
    return {**snapshot, "normalization_id": digest(snapshot)}


def validate_normalization(root):
    root = Path(root)
    snapshot = read_json(root / "source_snapshot.json")
    assert_repair_idle(snapshot["source_root"])
    if digest({k: v for k, v in snapshot.items() if k != "normalization_id"}) != snapshot["normalization_id"]:
        raise ValueError("Source snapshot identity changed")
    for name, expected in snapshot["source_files"].items():
        if file_digest(name) != expected:
            raise ValueError(f"Repaired source changed after normalization: {name}")
    if snapshot["parser"] != parser_provenance():
        raise ValueError("Parser code changed after normalization; regenerate this repair's artifacts")
    stats = read_json(root / "normalization_stats.json")
    if not stats.get("complete") or stats.get("normalization_id") != snapshot["normalization_id"]:
        raise ValueError("Normalization is incomplete or belongs to another repair batch")
    expected_paths = {str(paths_for(root, t.task_key, s)[k].relative_to(root))
                      for t in UI9_TASKS for s in ("train", "test")
                      for k in ("normalized", "detector_input", "detector_inputs")}
    if set(stats.get("artifact_digests", {})) != expected_paths:
        raise ValueError("Normalization did not bind all 18 source streams and detector inputs")
    for name, expected in stats["artifact_digests"].items():
        if file_digest(root / name) != expected:
            raise ValueError(f"Normalized artifact changed: {name}")
    assert_repair_idle(snapshot["source_root"])
    return snapshot


def cache_label_binding(root, task, split):
    root = Path(root)
    paths = paths_for(root, task.task_key, split)
    snapshot = read_json(root / "source_snapshot.json")
    files = [paths["normalized"], paths["detector_input"], paths["detector_inputs"],
             paths["derived"], paths["derived"].with_suffix(".coverage.json"),
             paths["cache"] / SCAN_NAME / "detector_scan_crops.jsonl",
             paths["cache"] / SCAN_NAME / "eval_detector_cache_ready.json"]
    return {"normalization_id": snapshot["normalization_id"], "repair_run_id": snapshot["repair_run_id"],
            "task_key": task.task_key, "task_id": task.task_id, "split": split,
            "view_policy": task.view_policy, "gt_used_for_plan": False,
            "artifact_digests": {str(p.relative_to(root)): file_digest(p) for p in files}}


def page_statistics(rows):
    pages, image_paths = defaultdict(lambda: defaultdict(Counter)), defaultdict(set)
    per_task = {t.task_key: {s: {"records": 0, "with_page": 0, "pages": set()} for s in ("train", "test")}
                for t in UI9_TASKS}
    for row in rows:
        task, split, page = row["task_key"], row["split"], row.get("source_page_id")
        count = per_task[task][split]
        count["records"] += 1
        if page:
            pages[page][split][task] += 1
            count["with_page"] += 1
            count["pages"].add(page)
        if row["task_id"] >= 7:
            image_paths[row["source_image"]].add(split)
    details = [{"page_id": p, "train": dict(s.get("train", {})), "test": dict(s.get("test", {}))}
               for p, s in sorted(pages.items())]
    synth = {t.task_key for t in UI9_TASKS if t.task_id >= 7}
    overlap = [p for p in details if p["train"] and p["test"]]
    synth_overlap = [p for p in details if synth.intersection(p["train"]) and synth.intersection(p["test"])]
    cross = [p for p in details if len(p["train"].keys() | p["test"].keys()) > 1]
    for splits in per_task.values():
        for count in splits.values():
            count["unique_pages"] = len(count.pop("pages"))
            count["without_page"] = count["records"] - count["with_page"]
    return {"page_identity": "prepare_ui9_datasets.page_key: FigmaKey:FigmaNodeID",
            "tasks": per_task, "unique_pages": len(pages), "cross_source_pages": len(cross),
            "train_test_page_count": len(overlap), "train_test_pages": overlap,
            "synthetic_train_test_page_count": len(synth_overlap), "synthetic_train_test_pages": synth_overlap,
            "synthetic_train_test_image_paths": sorted(p for p, s in image_paths.items() if len(s) > 1),
            "pages": details, "split_policy": "file membership; seven synthetic sources jointly grouped; two annotated splits retained"}


def format_counts(record, synthetic):
    counts = Counter({name + "_records": 0 for name in (
        "numbered_rect_err", "numbered_rect_err_with_rectN", "two_point_box",
        "objects_single_object", "location_direct_coordinates", "location_bbox_wrapper", "bbox_wrapper")})
    counts[f"objects_storage:{type(record.get('Objects' if synthetic else 'objects')).__name__}"] += 1
    flags = set()

    def walk(value, location=False):
        if location:
            counts[f"location_storage:{type(value).__name__}"] += 1
        if isinstance(value, str):
            try: value = json.loads(value)
            except (ValueError, TypeError): return
        if is_box(value):
            if len(value) == 2:
                flags.add("two_point_box")
                counts["two_point_box_occurrences"] += 1
            if location: flags.add("location_direct_coordinates")
            return
        if isinstance(value, dict):
            keys = [k for k in value if re.fullmatch(r"rect_err_?\d+", k, re.I)]
            if keys:
                flags.add("numbered_rect_err")
                counts["numbered_rect_err_fields"] += len(keys)
                if any(re.fullmatch(r"rect\d+", k, re.I) for k in value):
                    flags.add("numbered_rect_err_with_rectN")
            if "bbox" in value:
                flags.add("bbox_wrapper")
                if location: flags.add("location_bbox_wrapper")
            if location and any(k in value for k in ("x1", "xmin", "left", "x")):
                flags.add("location_direct_coordinates")
            for child in value.values(): walk(child)
        elif isinstance(value, list):
            for child in value: walk(child, location=location)

    if synthetic:
        objects = record.get("Objects")
        if isinstance(objects, dict):
            flags.add("objects_single_object")
            objects = [objects]
        for obj in objects if isinstance(objects, list) else []:
            if isinstance(obj, dict): walk(obj.get("Location"), location=True)
    else:
        obj = record.get("objects", {})
        if isinstance(obj, dict):
            counts[f"bbox_storage:{type(obj.get('bbox')).__name__}"] += 1
            counts[f"bbox_type:{obj.get('bbox_type', 'real')}"] += 1
            walk(obj.get("bbox"))
    counts.update({flag + "_records": 1 for flag in flags})
    return counts


def legacy_comparison(raw, task_root, split, synthetic, current, identities, *, record_index=None, seen_record_ids=None):
    from ui14_annotations_legacy_v1 import main_image, source_boxes
    counts, detail = Counter(), {}
    if raw.get("split", split) != split:
        counts["legacy_split_rejection_records"] += 1
        detail["source_split"] = raw.get("split")
    if ("Objects" if synthetic else "objects") not in raw:
        counts["legacy_annotation_field_failure_records"] += 1
        detail["legacy_annotation_error"] = "Missing explicit annotation field"
    if seen_record_ids is not None:
        record_id = str(raw.get("source_record_id", raw.get("id", raw.get("ID", record_index))))
        if record_id in seen_record_ids:
            counts["legacy_duplicate_record_id_records"] += 1
            detail["legacy_record_id_error"] = record_id
        seen_record_ids.add(record_id)
    old_image, old_boxes = None, None
    try:
        old_image = str(main_image(raw, task_root, synthetic))
        if old_image not in identities: identities[old_image] = image_identity(old_image)
        _, w, h = identities[old_image]
    except (ValueError, OSError, TypeError, KeyError, IndexError, AttributeError) as exc:
        counts["legacy_primary_failure_records"] += 1
        detail["legacy_primary_error"] = str(exc)
    if old_image and old_image in identities:
        try:
            old_boxes = source_boxes(raw, w, h, synthetic=synthetic)
        except (ValueError, TypeError, KeyError, IndexError, AttributeError, OverflowError) as exc:
            counts["legacy_bbox_failure_records"] += 1
            detail["legacy_bbox_error"] = str(exc)
    failed = bool(counts["legacy_primary_failure_records"] or counts["legacy_bbox_failure_records"])
    counts["legacy_parse_failure_records"] += int(failed)
    counts["legacy_consumer_failure_records"] += int(failed or any(counts[k] for k in (
        "legacy_split_rejection_records", "legacy_annotation_field_failure_records", "legacy_duplicate_record_id_records")))
    if current and not failed:
        counts["both_parsers_succeeded_records"] += 1
        image_diff = old_image != current["source_image"]
        boxes = current["boxes_px"]
        box_diff = len(old_boxes) != len(boxes) or any(
            not math.isclose(x, y, abs_tol=1e-7, rel_tol=0) for a, b in zip(old_boxes, boxes) for x, y in zip(a, b))
        counts["primary_result_difference_records"] += int(image_diff)
        counts["bbox_result_difference_records"] += int(box_diff)
        counts["parse_result_difference_records"] += int(image_diff or box_diff)
        if image_diff or box_diff:
            detail.update(legacy_image=old_image, current_image=current["source_image"],
                          legacy_boxes_px=old_boxes, current_boxes_px=boxes)
    else:
        counts["not_comparable_records"] += 1
    return counts, detail
