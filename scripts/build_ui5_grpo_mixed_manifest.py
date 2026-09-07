#!/usr/bin/env python3
"""Crop-only four-route selection at the exact bound hour021 publication.

The frozen snapshot's complete AND incomplete streams determine the raw read
boundary. Later raw rows cannot enter the pool. No m31 outcome is consulted.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from eaglevl.train.ui5_grpo_core import GRPOConfig, MixedSampler, TASKS, digest, file_sha, stable_seed
from eaglevl.train.ui5_supervision import corrected_record, validate_answer
from scripts import build_ui5_curriculum_recipe as recipe
from scripts.check_ui5_train_eval_content_overlap import _content_ids
from scripts.analyze_ui5_source_overlap import content_fingerprint
from scripts.merge_ui5_rollout_selections import _validate_snapshot
from scripts.ui5_frozen_selection import resolve_frozen_selection
from scripts.run_ui5_train_rollout_worker import score_prediction, load_module
from scripts.snapshot_ui5_train_rollouts import rollout_technical_issues, snapshot_rollout_payload, visible_jsonl_rows


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_rows(path):
    return recipe._read_jsonl(Path(path))


def frozen_crop_routes(snapshot, rollout_root):
    """Require exact raw-payload equality to the immutable snapshot, not mtime.

    Snapshot reads are sequential; wall time alone is not its exact cutoff.
    Included record/route identities also handle routes that failed on m31.
    """
    expected = {}
    for filename in ("complete8.jsonl", "incomplete_or_technical_error.jsonl"):
        for row in read_rows(snapshot / filename):
            record_id = str(row["record_id"])
            if record_id in expected:
                raise ValueError(f"duplicate snapshot record: {record_id}")
            expected[record_id] = {int(r["rollout_id"]): r for r in row["rollouts"]["crop"]
                                   if r["status"] != "missing"}
    raw = defaultdict(dict)
    evidence, scanned = [], Counter()
    for route in range(4):
        for part in sorted((rollout_root / "raw/crop" / f"rollout_{route}").glob("part-*.jsonl")):
            for line_number, row in enumerate(visible_jsonl_rows(part), 1):
                scanned[route] += 1
                record_id = str(row["record_id"])
                frozen = expected.get(record_id, {}).get(route)
                if frozen is None:
                    continue
                if row["model_id"] != "crop" or int(row["rollout_id"]) != route:
                    raise ValueError("crop raw route ownership mismatch")
                if route in raw[record_id]:
                    raise ValueError(f"duplicate crop raw route {record_id}/{route}")
                if snapshot_rollout_payload("crop", route, row) != frozen:
                    raise ValueError(f"raw differs from frozen route {record_id}/{route}: {part}")
                raw[record_id][route] = row
                evidence.append(dict(record_id=record_id, rollout_id=route, path=str(part),
                                     nonempty_line=line_number, raw_sha256=digest(row)))
    for record_id, routes in expected.items():
        if set(raw[record_id]) != set(routes):
            raise ValueError(f"raw missing a route already present at freeze: {record_id}")
    return raw, evidence, dict(scanned)


def raw_crop_geometry_issues(group, row, crops):
    """Check complete crop coverage/transforms and independently re-merge raw boxes."""
    from scripts.ui5_lossless_tiling import merge_tile_predictions
    outputs = row.get("crop_outputs", [])
    expected = {c["crop_id"]: c for c in crops}
    if len(outputs) != len(expected) or {c.get("crop_id") for c in outputs} != set(expected):
        return ["raw_crop_coverage"]
    pending = []
    try:
        for output in sorted(outputs, key=lambda c: c["crop_index"]):
            crop = expected[output["crop_id"]]
            if any(output.get(k) != crop.get(k) for k in
                   ("crop_index", "crop_xyxy", "gt_local", "gt_global", "coordinate_transforms")):
                return ["raw_crop_geometry_or_annotation"]
            if output.get("parse_status") == "parse_error":
                return ["raw_crop_parse_error"]
            xyxy = crop["crop_xyxy"]
            boxes = recipe._strict_local_boxes(output["pred_local"], label="raw crop prediction",
                        width=xyxy[2] - xyxy[0], height=xyxy[3] - xyxy[1])
            pending.extend(dict(bbox=box, tile_bbox=xyxy, label=group["task"], class_id=0,
                                score=1.0, source_tile_index=crop["crop_index"]) for box in boxes)
        merged = merge_tile_predictions(pending, image_size=(group["width"], group["height"]), iou_threshold=0.5)
        observed = sorted([int(round(x)) for x in item["bbox"]] for item in merged)
        if observed != sorted(row["pred_global"]):
            return ["raw_original_merge_mismatch"]
    except (KeyError, TypeError, ValueError):
        return ["raw_crop_coordinate_anomaly"]
    return []


def select_groups(catalog, routes, scorer, crops_by_sample=None):
    selected, excluded, changed = [], Counter(), []
    for sid, group in sorted(catalog.items()):
        rows = routes.get(str(group["record_id"]), {})
        if set(rows) != {0, 1, 2, 3}:
            excluded["crop_missing_route"] += 1
            continue
        if not group.get("grpo_eligible") or any(group.get(key) for key in (
                "pipeline_coverage_failure", "annotation_anomaly", "coordinate_transform_anomaly")):
            excluded["source_or_coverage_anomaly"] += 1
            continue
        issues = [issue for row in rows.values() for issue in rollout_technical_issues(row)]
        if issues:
            excluded["crop_technical_or_old_parse_invalid"] += 1
            continue
        scores = []
        for route, row in sorted(rows.items()):
            if any(row.get(key) != group[key] for key in ("source_image_id", "task", "gt_global", "prompt")):
                issues.append("raw_source_or_annotation_mismatch")
                continue
            if row.get("grpo_eligible") is False or any(row.get(key) for key in (
                    "pipeline_coverage_failure", "annotation_anomaly", "coordinate_transform_anomaly")):
                issues.append("raw_coordinate_or_coverage_anomaly")
            try:
                predictions = recipe._strict_bbox_list(row["pred_global"], label=f"{sid}/{route}.pred_global")
            except (ValueError, TypeError):
                issues.append("raw_prediction_coordinate_anomaly")
                continue
            if crops_by_sample is not None:
                issues.extend(raw_crop_geometry_issues(group, row, crops_by_sample[sid]))
            for box in predictions:
                if not (0 <= box[0] < box[2] <= group["width"] and 0 <= box[1] < box[3] <= group["height"]):
                    issues.append("prediction_coordinate_anomaly")
            score = score_prediction(scorer, group["gt_global"], predictions,
                                     row["parse_status"], 0.1, (group["width"], group["height"]))
            scores.append(score)
            if score["exact_correct"] != row["exact_correct"]:
                changed.append(dict(sample_id=sid, rollout_id=route,
                                    old=row["exact_correct"], recomputed=score["exact_correct"]))
        if issues:
            excluded["raw_coordinate_or_coverage_anomaly"] += 1
            continue
        count = sum(score["exact_correct"] for score in scores)
        if count not in (1, 2, 3):
            excluded[f"crop_correct_{count}_of_4"] += 1
            continue
        selected.append(dict(group, crop_correct_count=count,
                             crop_exact_correct=[s["exact_correct"] for s in scores],
                             polarity="positive" if group["gt_global"] else "negative"))
    return selected, dict(excluded), changed


def publish_assets(assets, old_directory, destination):
    """Reuse verified PNG paths. Generate only missing PNGs required by mixed."""
    from PIL import Image
    publication = read_json(old_directory / "curriculum_manifest.json")
    asset_root = Path(publication.get("outputs", {}).get("crop_asset_root", old_directory))
    old = {row["crop_id"]: row for row in publication["crop_assets"]}
    result = {}
    for asset in assets:
        cid = asset["crop_id"]
        previous = old.get(cid)
        path = asset_root / previous["relative_path"] if previous else None
        reused = path is not None and path.is_file()
        if previous and (previous["source_image_sha256"] != asset["source_image_sha256"]
                         or previous["crop_xyxy"] != asset["crop_xyxy"]):
            raise ValueError(f"crop cache geometry/content mismatch: {cid}")
        if reused:
            if path.stat().st_size != previous["bytes"] or file_sha(path) != previous["sha256"]:
                raise ValueError(f"corrupt existing crop PNG: {path}")
        else:
            path = destination / "mixed_png" / (digest(cid)[:32] + ".png")
            path.parent.mkdir(parents=True, exist_ok=True)
            if file_sha(asset["source_image"]) != asset["source_image_sha256"]:
                raise ValueError(f"source image changed: {cid}")
            with Image.open(asset["source_image"]) as image:
                image.load()
                if list(image.size) != asset["source_size"]:
                    raise ValueError(f"source image size changed: {cid}")
                temporary = path.with_suffix(".tmp")
                tile = image.crop(asset["crop_xyxy"])
                tile.save(temporary, format="PNG")
                tile.close()
                os.replace(temporary, path)
        with Image.open(path) as image:
            if list(image.size) != asset["crop_size"]:
                raise ValueError(f"crop PNG dimensions changed: {cid}")
        result[cid] = dict(path=str(path.resolve()), sha256=file_sha(path), reused=reused,
                           crop_id=cid, crop_xyxy=asset["crop_xyxy"])
    return result


def verify_manifest(directory):
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    identity = manifest.pop("identity")
    if identity != digest(manifest):
        raise ValueError("mixed manifest identity mismatch")
    for name, expected in manifest["files"].items():
        if file_sha(directory / name) != expected:
            raise ValueError(f"mixed artifact changed: {name}")
    if read_json(directory / "_SUCCESS.json")["identity"] != identity:
        raise ValueError("mixed publication incomplete")
    return dict(manifest, identity=identity)


def build(*, snapshot, frozen_selection, rollout_root, bundle, curriculum_dir, eval_input, output, config):
    output.mkdir(parents=True, exist_ok=True)
    bound = dict(snapshot=str(snapshot), frozen_selection=str(frozen_selection),
                 rollout_root=str(rollout_root), bundle=str(bundle), curriculum_dir=str(curriculum_dir),
                 eval_input=str(eval_input), config=config.to_dict())
    if (output / "_SUCCESS.json").exists():
        manifest = verify_manifest(output)
        if manifest["bound_inputs"] != bound:
            raise ValueError("resume mixed pool has different bound inputs")
        return manifest
    print("[MIXED] verifying hour021 snapshot, frozen selection and complete train bundle", flush=True)
    frozen = resolve_frozen_selection(frozen_selection)
    snapshot_metadata = _validate_snapshot(snapshot)
    sources = read_json(frozen_selection / "manifest.json")["sources"]
    if not any(Path(row["path"]).resolve() == snapshot.resolve()
               and row["manifest_sha256"] == snapshot_metadata["manifest_sha256"] for row in sources):
        raise ValueError("bound snapshot is not the source of the actual frozen selection")
    if read_json(snapshot / "summary.json")["scheduled_hour"] != 21:
        raise ValueError("formal mixed selection must use the bound hour021 snapshot")
    bundle_state, images = recipe._verify_rollout_bundle(bundle)
    crops, _, _ = recipe._bundle_crop_geometry(bundle, images)
    records = recipe._bundle_records(bundle)
    recipe._verify_bundle_record_images(records, images)
    by_sample = defaultdict(list)
    for row in records:
        by_sample[recipe._sample_id(row)].append(row)
    catalog, eligible = recipe._bundle_group_catalog(bundle, crops, recipe._record_group_truth(records))
    scorer = load_module(ROOT / "qwen3vl_merge_and_score_fixed_5tasks.py", "grpo_manifest_scorer")
    routes, evidence, scanned = frozen_crop_routes(snapshot, rollout_root)
    selected, excluded, changed = select_groups(catalog, routes, scorer, crops)
    print(f"[MIXED] crop-only candidates={len(selected)} exclusions={excluded}", flush=True)
    content = {str(path): content_fingerprint(path) for path in images}
    from scripts.run_ui5_curriculum_evaluation import TASK_GT_FILE
    test_ids, test_records = _content_ids([eval_input / name for name in TASK_GT_FILE.values()])
    overlap = set(content.values()) & test_ids
    if overlap:
        raise ValueError(f"train/test content overlap: {len(overlap)}")
    groups, assets, dedup = [], [], {}
    for group in selected:
        sid = group["sample_id"]
        image_path = str((bundle / group["image_relpath"]).resolve())
        key = (content[image_path], group["task"])
        if key in dedup:
            if dedup[key]["gt_global"] != group["gt_global"]:
                raise ValueError("same content/task has conflicting GT")
            excluded["duplicate_content_task"] = excluded.get("duplicate_content_task", 0) + 1
            continue
        dedup[key] = group
        canonical = recipe._canonical_selected_supervision(sid, by_sample[sid], group)
        if group["task"] == "content_missing":
            views = [dict(crop_id="full_image", crop_index=0,
                          crop_xyxy=[0, 0, group["width"], group["height"]], image=image_path,
                          record=recipe._global_view_supervision(canonical, retention=True))]
        else:
            crop_records, group_assets = recipe._selected_crop_supervision(
                sid, canonical, group, crops, asset_namespace="mixed_png")
            assets.extend(group_assets)
            views = [dict(crop_id=r["_ui5_crop_id"], crop_index=r["_ui5_crop_index"],
                          crop_xyxy=r["_ui5_crop_bbox"], record=r) for r in crop_records]
        for view in views:
            validate_answer(view["record"])
        groups.append(dict(group_id=digest([group["source_image_id"], group["task"]]),
                           content_id=key[0], source_image_id=group["source_image_id"],
                           sample_id=sid, task=group["task"], image=image_path,
                           prompt=group["prompt"], gt_global=group["gt_global"],
                           width=group["width"], height=group["height"], views=views,
                           polarity=group["polarity"], crop_correct_count=group["crop_correct_count"],
                           crop_exact_correct=group["crop_exact_correct"]))
    pngs = publish_assets(assets, curriculum_dir, output)
    image_hashes = {str(path): sha for path, sha in images.items()}
    image_hashes.update({value["path"]: value["sha256"] for value in pngs.values()})
    for group in groups:
        for view in group["views"]:
            if view["crop_id"] != "full_image":
                view["image"] = pngs[view["crop_id"]]["path"]
                view["record"]["image"] = view["image"]
            view["image_sha256"] = image_hashes[view["image"]]
    # The three old pools partition the entire eligible source corpus. Merge
    # all three for replay; never use only old global_replay or only mixed IDs.
    replay, replay_seen = [], set()
    for pool in ("hard", "matched_anchor", "global_replay"):
        for row in read_rows(curriculum_dir / f"{pool}.jsonl"):
            row, _ = corrected_record(row)
            validate_answer(row)
            key = (recipe._sample_id(row), row.get("_ui5_crop_id", "full_image"))
            if key in replay_seen:
                raise ValueError("duplicate replay view across original curriculum pools")
            replay_seen.add(key)
            if not Path(row["image"]).is_file():
                cached = pngs.get(row.get("_ui5_crop_id"))
                if cached is None:
                    raise FileNotFoundError(f"replay PNG missing outside mixed pool: {row['image']}")
                row["image"] = cached["path"]
            if row["image"] not in image_hashes:
                image_hashes[row["image"]] = file_sha(row["image"])
            row["_ui5_grpo_image_sha256"] = image_hashes[row["image"]]
            replay.append(row)
    if {sid for sid, _ in replay_seen} != set(eligible):
        raise ValueError("replay must cover the full eligible source corpus, including all old pools")
    sampler = MixedSampler(groups, config.seed)
    diagnostic = []
    for task in TASKS:
        candidates = [g for g in groups if g["task"] == task]
        diagnostic.extend(sorted(candidates, key=lambda g: stable_seed(config.seed, "diagnostic", g["group_id"]))
                          [:config.diagnostic_per_task])
    for name, rows in (("groups.jsonl", groups), ("replay.jsonl", replay),
                       ("raw_evidence.jsonl", evidence), ("train_diagnostic.jsonl", diagnostic)):
        recipe._atomic_jsonl(output / name, rows)
    manifest = dict(schema_version=1, bound_inputs=bound, baseline_sha="b29590f88c4b4a102c742f8410e1c6751b0859d2",
                    scope="UI5 train only; m31 outcomes/completeness are not selection conditions",
                    frozen=frozen, snapshot=snapshot_metadata, bundle=bundle_state,
                    groups=len(groups), replay_records=len(replay), replay_source_groups=len(eligible),
                    excluded=excluded, changed_crop_exact_scores=changed, scanned_raw_routes=scanned,
                    actual_candidate_strata=Counter(f"{g['task']}/{g['polarity']}/{g['crop_correct_count']}" for g in groups),
                    sampling=sampler.report(config.total_steps * config.world_size * config.groups_per_rank),
                    train_test_overlap=0, test_records_checked=test_records,
                    assets=list(pngs.values()), reused_pngs=sum(p["reused"] for p in pngs.values()),
                    generated_pngs=sum(not p["reused"] for p in pngs.values()),
                    diagnostic_scope="fixed mixed TRAIN subset, AR four-trajectory correctness; not formal UI5",
                    files={name: file_sha(output / name) for name in
                           ("groups.jsonl", "replay.jsonl", "raw_evidence.jsonl", "train_diagnostic.jsonl")})
    manifest["identity"] = digest(manifest)
    write_json(output / "manifest.json", manifest)
    write_json(output / "_SUCCESS.json", dict(identity=manifest["identity"]))
    return verify_manifest(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot", "frozen-selection", "rollout-root", "bundle", "curriculum-dir", "eval-input", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    manifest = build(**vars(args), config=GRPOConfig().validate())
    print(json.dumps({k: manifest[k] for k in ("groups", "replay_records", "reused_pngs", "generated_pngs", "identity")}, indent=2))


if __name__ == "__main__":
    main()
