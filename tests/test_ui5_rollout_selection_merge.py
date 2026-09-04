from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


merge = load_script("merge_ui5_rollout_selections", "merge_ui5_rollout_selections.py")
curriculum = load_script("build_ui5_curriculum_recipe_for_merge", "build_ui5_curriculum_recipe.py")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def complete_row(sample_id: str, *, task: str = "occlusion", correct: int = 0) -> dict:
    gt = [[1, 2, 10, 20]]
    route_correct = [index < correct for index in range(8)]
    rollouts = {}
    for model_index, model in enumerate(merge.MODELS):
        rollouts[model] = [
            {
                "model_id": model,
                "rollout_id": rollout_id,
                "seed": merge.FORMAL_SEEDS[rollout_id],
                "status": "completed",
                "reward": route_correct[model_index * 4 + rollout_id],
                "exact_correct": route_correct[model_index * 4 + rollout_id],
                "parse_status": "ok",
                "contains_crop_parse_error": False,
                "runtime_error": None,
                "oom_final_failure": False,
                "image_confusion": (
                    "TP" if route_correct[model_index * 4 + rollout_id] else "FN"
                ),
                "gt_global": gt,
                "pred_global": (
                    gt if route_correct[model_index * 4 + rollout_id] else []
                ),
            }
            for rollout_id in merge.ROLLOUT_IDS
        ]
    m31 = sum(route_correct[:4])
    crop = sum(route_correct[4:])
    return {
        "record_id": f"record-{sample_id}",
        "sample_id": sample_id,
        "source_image_id": f"image-{sample_id}",
        "task": task,
        "image_relpath": f"images/{sample_id}.png",
        "prompt": "find the defect",
        "gt_global": gt,
        "source_records": [],
        "original_training_record": {},
        "m31_correct_count": m31,
        "crop_correct_count": crop,
        "total_correct_count": correct,
        "success_rate": correct / 8.0,
        "difficulty": "easy" if correct == 8 else "hard" if correct == 0 else "medium",
        "grpo_ready_m31": False,
        "grpo_ready_crop": False,
        "grpo_source_eligible": True,
        "pipeline_coverage_failure": False,
        "annotation_anomaly": False,
        "coordinate_transform_anomaly": False,
        "grpo_parse_clean_m31": True,
        "grpo_parse_clean_crop": True,
        "m31_complete4": True,
        "crop_complete4": True,
        "m31_completed_rollout_count": 4,
        "crop_completed_rollout_count": 4,
        "completed_rollout_count": 8,
        "runtime_error_count": 0,
        "runtime_errors": [],
        "parse_error_count": 0,
        "cross_model_complete8": True,
        "technical_error_free": True,
        "technical_issues": {"m31": {}, "crop": {}},
        "exclusion_reason": None,
        "rollouts": rollouts,
    }


def selection(path: Path, rows: list[dict]) -> Path:
    write_jsonl(path / "complete8.jsonl", rows)
    write_jsonl(
        path / "sample_difficulty.jsonl",
        [merge._difficulty_projection(row) for row in rows],
    )
    return path


def snapshot(path: Path, rows: list[dict], *, hour: int = 3) -> Path:
    selection(path, rows)
    files = []
    for file in sorted(path.iterdir()):
        files.append(
            {
                "path": file.name,
                "bytes": file.stat().st_size,
                "sha256": merge._sha256(file),
                "jsonl_records": merge._row_count(file),
            }
        )
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 4,
                "snapshot_kind": "hourly",
                "scheduled_hour": hour,
                "append_only": True,
                "atomic_publish": True,
                "success_marker": "_SUCCESS",
                "files": files,
            }
        ),
        encoding="utf-8",
    )
    (path / "_SUCCESS").write_text("2026-09-05T00:00:00+00:00\n", encoding="utf-8")
    return path


class RolloutSelectionMergeTests(unittest.TestCase):
    def test_merges_snapshots_and_selection_with_identical_deduplication(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = complete_row("one", correct=0)
            second = complete_row("two", task="cropping", correct=3)
            source_a = snapshot(root / "hour_003", [first])
            source_b = selection(root / "selection", [first, second])
            output = root / "frozen"

            summary = merge.freeze([source_a, source_b], output)

            self.assertEqual(summary["input_rows"], 3)
            self.assertEqual(summary["unique_complete8_samples"], 2)
            self.assertEqual(summary["deduplicated_rows"], 1)
            self.assertTrue((output / "_SUCCESS").is_file())
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["immutable"])
            self.assertEqual(manifest["training_input_policy"], "resolve_once_at_run_start_no_hot_reload")
            self.assertEqual(len(manifest["sources"]), 2)
            rows, state = curriculum._load_difficulty_rows(output / "complete8.jsonl")
            self.assertEqual(len(rows), 2)
            self.assertEqual(state["authoritative_path"], str((output / "complete8.jsonl").resolve()))

    def test_conflicting_sample_is_rejected_without_publishing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = complete_row("same", correct=0)
            changed = complete_row("same", correct=1)
            output = root / "frozen"
            with self.assertRaisesRegex(ValueError, "sample_id conflict"):
                merge.freeze(
                    [selection(root / "a", [first]), selection(root / "b", [changed])],
                    output,
                )
            self.assertFalse(output.exists())

    def test_conflicting_record_id_across_different_samples_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = complete_row("one")
            second = complete_row("two")
            second["record_id"] = first["record_id"]
            with self.assertRaisesRegex(ValueError, "record_id conflict"):
                merge.freeze(
                    [selection(root / "a", [first]), selection(root / "b", [second])],
                    root / "frozen",
                )

    def test_incomplete_or_technically_dirty_row_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = complete_row("dirty")
            row["parse_error_count"] = 1
            source = selection(root / "selection", [row])
            with self.assertRaisesRegex(ValueError, "parse_error_count"):
                merge.freeze([source], root / "frozen")

    def test_route_identity_and_rewards_are_recomputed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = complete_row("bad-route", correct=2)
            row["rollouts"]["crop"][2]["seed"] = 17
            source = selection(root / "selection", [row])
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                merge.freeze([source], root / "frozen")

    def test_freeze_rescores_stale_route_rewards_before_classification(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = complete_row("stale", correct=0)
            # Simulate an hour_006 artifact scored by the old matcher: the
            # serialized prediction is exact, but all cached rewards/counts
            # still say 0/8 hard.  Freezing must derive labels from boxes.
            for model in merge.MODELS:
                for route in row["rollouts"][model]:
                    route["pred_global"] = row["gt_global"]
            row["grpo_m31_group"] = {"rewards_exact": [False] * 4}
            row["grpo_crop_group"] = {"rewards_exact": [False] * 4}
            row["visualization_rollouts"] = {
                "crop": [{"exact_correct": False}]
            }
            source = selection(root / "selection", [row])
            output = root / "frozen"

            summary = merge.freeze([source], output)

            frozen = json.loads(
                (output / "complete8.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(frozen["m31_correct_count"], 4)
            self.assertEqual(frozen["crop_correct_count"], 4)
            self.assertEqual(frozen["difficulty"], "easy")
            self.assertEqual(summary["changed_route_scores"], 8)
            self.assertEqual(summary["changed_sample_classifications"], 1)
            self.assertEqual(summary["formal_crop_hard_groups"], 0)
            self.assertEqual(
                summary["crop_correct_count_distribution"],
                {"0": 0, "1": 0, "2": 0, "3": 0, "4": 1},
            )
            self.assertEqual(
                frozen["scoring_policy"]["matcher"],
                "max_qualified_cardinality_then_iou",
            )
            for derived_key in (
                "grpo_m31_group",
                "grpo_crop_group",
                "visualization_rollouts",
            ):
                self.assertNotIn(derived_key, frozen)

    def test_projected_index_must_exactly_match_authoritative_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = complete_row("projection")
            source = selection(root / "selection", [row])
            projected = merge._difficulty_projection(row)
            projected["crop_correct_count"] = 4
            write_jsonl(source / "sample_difficulty.jsonl", [projected])
            with self.assertRaisesRegex(ValueError, "exact projection"):
                merge.freeze([source], root / "frozen")

    def test_snapshot_manifest_hash_and_three_hour_boundary_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = complete_row("manifest")
            damaged = snapshot(root / "damaged", [row])
            with (damaged / "complete8.jsonl").open("a", encoding="utf-8") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "inventory mismatch"):
                merge.freeze([damaged], root / "out-a")

            wrong_hour = snapshot(root / "wrong-hour", [row], hour=4)
            with self.assertRaisesRegex(ValueError, "three-hour boundary"):
                merge.freeze([wrong_hour], root / "out-b")

    def test_existing_output_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = selection(root / "selection", [complete_row("one")])
            output = root / "frozen"
            output.mkdir()
            sentinel = output / "keep.txt"
            sentinel.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "immutable output"):
                merge.freeze([source], output)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_prior_frozen_artifact_can_feed_the_next_frozen_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = selection(root / "selection", [complete_row("one")])
            first = root / "frozen-one"
            second = root / "frozen-two"
            merge.freeze([source], first)
            summary = merge.freeze([first], second)
            self.assertEqual(summary["unique_complete8_samples"], 1)
            manifest = json.loads((second / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["sources"][0]["source_kind"], "frozen_selection")

    def test_script_has_no_gpu_framework_dependency(self) -> None:
        source = (PROJECT_ROOT / "scripts" / "merge_ui5_rollout_selections.py").read_text(
            encoding="utf-8"
        ).lower()
        self.assertNotIn("import torch", source)
        self.assertNotIn("cuda", source)


if __name__ == "__main__":
    unittest.main()
