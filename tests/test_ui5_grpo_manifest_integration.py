"""Real fixture bundle -> old crops -> immutable freeze -> crop-only mixed pool.

All records/PNGs here are test fixtures; no internal dataset counts are assumed.
"""
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

from PIL import Image

from eaglevl.train.ui5_grpo_core import GRPOConfig, file_sha
from eaglevl.train.ui5_supervision import validate_answer
from scripts import build_ui5_grpo_mixed_manifest as mixed
from scripts import merge_ui5_rollout_selections as merger
from scripts import snapshot_ui5_train_rollouts as snapshot
from scripts.run_ui5_train_rollout_worker import FORMAL_SEEDS, score_prediction, load_module
from tests import test_ui5_curriculum_recipe as fixtures

recipe, write_jsonl = fixtures.curriculum_recipe, fixtures.write_jsonl


def test_real_manifest_build_reuses_pngs_and_keeps_m31_incomplete_candidate(tmp_path):
    fixture = fixtures.CurriculumRecipeTests(methodName="runTest")
    bundle, old_difficulty = fixture._fixture(tmp_path)
    # Distinct original contents for the fixture's distinct source IDs.
    images = mixed.read_rows(bundle / "manifest/unique_images.jsonl")
    for index, row in enumerate(images):
        image = bundle / row["image_relpath"]
        Image.new("RGB", (100, 100), (index, 30, 50)).save(image)
        row["sha256"] = file_sha(image)
    write_jsonl(bundle / "manifest/unique_images.jsonl", images)
    tasks = mixed.read_rows(bundle / "manifest/task_samples.jsonl")
    for row in tasks:
        row.update(width=100, height=100)
    write_jsonl(bundle / "manifest/task_samples.jsonl", tasks)
    fixture._refresh_bundle_file_inventory(bundle)
    bundle_manifest = mixed.read_json(bundle / "bundle_manifest.json")
    bundle_manifest["rollout_samples"] = len(tasks)
    mixed.write_json(bundle / "bundle_manifest.json", bundle_manifest)
    old = tmp_path / "old_curriculum"
    old_manifest = recipe.build(fixture._args(bundle, old_difficulty, old))
    crop_rows = defaultdict(list)
    for row in mixed.read_rows(bundle / "manifest/crop_samples.jsonl"):
        crop_rows[row["sample_id"]].append(row)
    counts = {row["sample_id"]: row["crop_correct_count"] for row in mixed.read_rows(old_difficulty)}
    scorer = load_module(Path(__file__).resolve().parents[1] / "qwen3vl_merge_and_score_fixed_5tasks.py", "integration_scorer")
    rollout = tmp_path / "rollout"
    for model in ("crop", "m31"):
        for route in range(4):
            raw = []
            for task in tasks:
                sid = task["sample_id"]
                if model == "m31" and sid == "replay-occ-pos":
                    continue  # this mixed group must survive m31's missing routes
                correct = route < counts[sid]
                predictions = deepcopy(task["gt_global"]) if correct else ([] if task["gt_global"] else [[10, 10, 20, 20]])
                status = "defect" if predictions else "ok"
                row = dict(task, model_id=model, rollout_id=route, seed=FORMAL_SEEDS[route],
                           inference_success=True, runtime_error=None, parse_status=status,
                           pred_global=predictions, raw_output="fixture answer", crop_outputs=[])
                if model == "crop":
                    for crop in crop_rows[sid]:
                        xyxy = crop["crop_xyxy"]
                        local = [[b[0] - xyxy[0], b[1] - xyxy[1], b[2] - xyxy[0], b[3] - xyxy[1]]
                                 for b in predictions if xyxy[0] <= b[0] < b[2] <= xyxy[2]
                                 and xyxy[1] <= b[1] < b[3] <= xyxy[3]]
                        output = {key: deepcopy(crop[key]) for key in
                                  ("crop_id", "crop_index", "crop_xyxy", "gt_local", "gt_global", "coordinate_transforms")}
                        output.update(pred_local=local, parse_status="defect" if local else "ok", raw_output="fixture")
                        row["crop_outputs"].append(output)
                    row["raw_output"] = row["crop_outputs"]
                row.update(score_prediction(scorer, task["gt_global"], predictions, status, .1, (100, 100)))
                raw.append(row)
            write_jsonl(rollout / "raw" / model / f"rollout_{route}/part-00000.jsonl", raw)
    records, _ = snapshot.build_difficulty_records(rollout, bundle)
    frozen_source = tmp_path / "hour021"
    complete = [snapshot.classification_projection(row) for row in records if row["cross_model_complete8"]]
    incomplete = [snapshot.classification_projection(row) for row in records if not row["cross_model_complete8"]]
    write_jsonl(frozen_source / "complete8.jsonl", complete)
    write_jsonl(frozen_source / "sample_difficulty.jsonl", [merger._difficulty_projection(row) for row in complete])
    write_jsonl(frozen_source / "incomplete_or_technical_error.jsonl", incomplete)
    summary = dict(snapshot_kind="hourly", scheduled_hour=21, created_at="fixture", previous_snapshot=None)
    mixed.write_json(frozen_source / "summary.json", summary)
    mixed.write_json(frozen_source / "manifest.json", snapshot.build_snapshot_manifest(frozen_source, summary))
    (frozen_source / "_SUCCESS").write_text("fixture complete")
    frozen = tmp_path / "actual_frozen_selection"
    merger.freeze([frozen_source], frozen)
    # Only one PNG needed by mixed is absent; do not regenerate anything else.
    missing = next(asset for asset in old_manifest["crop_assets"] if asset["sample_id"] == "replay-occ-pos")
    (old / missing["relative_path"]).unlink()
    test = tmp_path / "test"
    test.mkdir()
    image = test / "heldout.png"
    Image.new("RGB", (100, 100), (255, 255, 255)).save(image)
    from scripts.run_ui5_curriculum_evaluation import TASK_GT_FILE
    for name in TASK_GT_FILE.values():
        write_jsonl(test / name, [dict(image=str(image))])
    config = GRPOConfig()
    args = dict(snapshot=frozen_source, frozen_selection=frozen, rollout_root=rollout, bundle=bundle,
                curriculum_dir=old, eval_input=test, output=tmp_path / "mixed", config=config)
    result = mixed.build(**args)
    groups = mixed.read_rows(args["output"] / "groups.jsonl")
    assert {group["sample_id"] for group in groups} == {sid for sid, count in counts.items() if count == 2}
    assert "replay-occ-pos" in {group["sample_id"] for group in groups}
    assert result["generated_pngs"] == 1
    assert result["replay_source_groups"] == len(tasks)
    assert result["train_test_overlap"] == 0
    replay = mixed.read_rows(args["output"] / "replay.jsonl")
    assert len(replay) == sum(len(mixed.read_rows(old / (pool + ".jsonl"))) for pool in recipe.POOLS)
    for group in groups:
        assert len(group["views"]) == (1 if group["task"] == "content_missing" else 2)
        for view in group["views"]:
            validate_answer(view["record"])
            assert file_sha(view["image"]) == view["image_sha256"]
    for row in replay:
        validate_answer(row)
        assert file_sha(row["image"]) == row["_ui5_grpo_image_sha256"]
    assert mixed.build(**args)["identity"] == result["identity"]
