from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from PIL import Image

from eaglevl.train.ui5_curriculum import (
    UI5CurriculumSchedule,
    curriculum_artifact_identity,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "build_ui5_curriculum_recipe.py"
SPEC = importlib.util.spec_from_file_location("build_ui5_curriculum_recipe", SCRIPT)
assert SPEC and SPEC.loader
curriculum_recipe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(curriculum_recipe)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


class CurriculumRecipeTests(unittest.TestCase):
    @staticmethod
    def _schedule() -> UI5CurriculumSchedule:
        return UI5CurriculumSchedule(
            total_steps=1200,
            hard_ratios=(0.60, 0.45, 0.30),
            matched_anchor_ratios=(0.25, 0.35, 0.30),
            global_replay_ratios=(0.15, 0.20, 0.40),
            llm_lrs=(1e-6, 7e-7, 5e-7),
            expected_hard_groups=3,
        )

    def _fixture(self, root: Path) -> tuple[Path, Path]:
        bundle = root / "bundle"
        (bundle / "images").mkdir(parents=True)
        groups = [
            ("hard-pos", "occlusion", True, 0),
            ("hard-neg", "text_overflow", False, 0),
            ("hard-content", "content_missing", True, 0),
            ("anchor-pos", "occlusion", True, 4),
            ("anchor-neg", "text_overflow", False, 4),
            ("anchor-content", "content_missing", True, 4),
            ("replay", "cropping", True, 2),
            ("replay-two", "content_missing", False, 2),
        ]
        source_rows = []
        difficulty_rows = []
        unique_rows = []
        crop_rows = []
        base_plans = {}
        for sample_id, task, positive, correct in groups:
            image = bundle / "images" / f"{sample_id}.png"
            Image.new("RGB", (100, 100), color=(12, 34, 56)).save(image)
            unique_rows.append(
                {
                    "image_id": f"image-{sample_id}",
                    "image_relpath": f"images/{sample_id}.png",
                    "sha256": curriculum_recipe._sha256_file(image),
                    "width": 100,
                    "height": 100,
                }
            )
            answer = "<box><1><1><2><2></box>" if positive else "<box>none</box>"
            source_rows.append(
                {
                    "source_record_id": f"source-{sample_id}",
                    "sample_id": sample_id,
                    "image_id": f"image-{sample_id}",
                    "task": task,
                    "gt_boxes_1000": [[1, 1, 2, 2]] if positive else [],
                    "gt_boxes_global_xyxy": [[1, 1, 2, 2]] if positive else [],
                    "portable_training_record": {
                        "image": f"images/{sample_id}.png",
                        "conversations": [
                            {"from": "human", "value": "locate"},
                            {"from": "gpt", "value": answer},
                        ],
                    },
                }
            )
            difficulty_rows.append(
                {
                    "record_id": sample_id,
                    "sample_id": sample_id,
                    "source_image_id": f"image-{sample_id}",
                    "task": task,
                    "image_relpath": f"images/{sample_id}.png",
                    "prompt": "locate",
                    "gt_global": [[1, 1, 2, 2]] if positive else [],
                    "crop_correct_count": correct,
                    "m31_complete4": True,
                    "crop_complete4": True,
                    "cross_model_complete8": True,
                    "technical_error_free": True,
                    "runtime_error_count": 0,
                    "parse_error_count": 0,
                    "grpo_source_eligible": True,
                    "pipeline_coverage_failure": False,
                    "annotation_anomaly": False,
                    "coordinate_transform_anomaly": False,
                }
            )
            tiles = (
                [[0, 0, 100, 100]]
                if task == "content_missing"
                else [[0, 0, 100, 50], [0, 50, 100, 100]]
            )
            if task != "content_missing":
                base_plans[f"image-{sample_id}"] = {
                    "image_id": f"image-{sample_id}",
                    "width": 100,
                    "height": 100,
                    "base_tiles": tiles,
                    "geometry_digest": f"geometry-{sample_id}",
                    "gt_used": False,
                    "source": "crop_root/base_scan_plans.json",
                }
            sample_gt = [[1, 1, 2, 2]] if positive else []
            for crop_index, crop in enumerate(tiles):
                contained = [
                    box
                    for box in sample_gt
                    if crop[0] <= box[0] < box[2] <= crop[2]
                    and crop[1] <= box[1] < box[3] <= crop[3]
                ]
                crop_width, crop_height = crop[2] - crop[0], crop[3] - crop[1]
                local = [
                    [
                        box[0] - crop[0],
                        box[1] - crop[1],
                        box[2] - crop[0],
                        box[3] - crop[1],
                    ]
                    for box in contained
                ]
                local_1000 = [
                    [
                        round(box[0] / crop_width * 1000),
                        round(box[1] / crop_height * 1000),
                        round(box[2] / crop_width * 1000),
                        round(box[3] / crop_height * 1000),
                    ]
                    for box in local
                ]
                crop_rows.append(
                    {
                        "record_id": sample_id,
                        "sample_id": sample_id,
                        "source_image_id": f"image-{sample_id}",
                        "task": task,
                        "prompt": "locate",
                        "image_relpath": f"images/{sample_id}.png",
                        "crop_id": f"crop-{sample_id}-{crop_index}",
                        "crop_index": crop_index,
                        "crop_xyxy": crop,
                        "crop_size": [crop_width, crop_height],
                        "gt_local": local,
                        "gt_local_1000": local_1000,
                        "gt_global": contained,
                        "sample_gt_global": sample_gt,
                        "coordinate_transforms": [
                            {
                                "global_bbox_xyxy": global_box,
                                "local_bbox_xyxy": local_box,
                                "local_bbox_1000": norm_box,
                            }
                            for global_box, local_box, norm_box in zip(
                                contained, local, local_1000
                            )
                        ],
                        "partial_gt_indices": [],
                        "pipeline_coverage_failure": False,
                        "coordinate_transform_anomaly": False,
                        "geometry_source": "base_scan_plans.base_tiles",
                        "gt_used_for_geometry": False,
                    }
                )
        source_records = bundle / "manifest" / "source_records.jsonl"
        unique_images = bundle / "manifest" / "unique_images.jsonl"
        crop_samples = bundle / "manifest" / "crop_samples.jsonl"
        base_scan_plans = bundle / "base_scan_plans.json"
        write_jsonl(source_records, source_rows)
        write_jsonl(unique_images, unique_rows)
        write_jsonl(crop_samples, crop_rows)
        base_scan_plans.write_text(json.dumps(base_plans), encoding="utf-8")
        files = {}
        for path in (source_records, unique_images, crop_samples, base_scan_plans):
            relative = path.relative_to(bundle).as_posix()
            files[relative] = {
                "bytes": path.stat().st_size,
                "sha256": curriculum_recipe._sha256_file(path),
            }
        (bundle / "bundle_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "complete": True,
                    "unique_images": len(unique_rows),
                    "files": files,
                }
            ),
            encoding="utf-8",
        )
        difficulty = root / "sample_difficulty.jsonl"
        write_jsonl(difficulty, difficulty_rows)
        return bundle, difficulty

    @staticmethod
    def _args(bundle: Path, difficulty: Path, output: Path) -> Namespace:
        return Namespace(
            base_recipe=None,
            rollout_difficulty=difficulty,
            rollout_bundle_root=bundle,
            output_dir=output,
            expected_hard_groups=3,
            seed=42,
        )

    @staticmethod
    def _refresh_bundle_file_inventory(bundle: Path) -> None:
        manifest_path = bundle / "bundle_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for relative, metadata in manifest["files"].items():
            path = bundle / relative
            metadata["bytes"] = path.stat().st_size
            metadata["sha256"] = curriculum_recipe._sha256_file(path)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    @staticmethod
    def _append_crop_gt(
        bundle: Path, sample_id: str, global_box: list[int]
    ) -> None:
        path = bundle / "manifest" / "crop_samples.jsonl"
        rows = curriculum_recipe._read_jsonl(path)
        matched = False
        for row in rows:
            if row["sample_id"] != sample_id:
                continue
            matched = True
            row["sample_gt_global"].append(global_box)
            crop = row["crop_xyxy"]
            if not (
                crop[0] <= global_box[0] < global_box[2] <= crop[2]
                and crop[1] <= global_box[1] < global_box[3] <= crop[3]
            ):
                continue
            local = [
                global_box[0] - crop[0],
                global_box[1] - crop[1],
                global_box[2] - crop[0],
                global_box[3] - crop[1],
            ]
            width, height = row["crop_size"]
            norm = [
                round(local[0] / width * 1000),
                round(local[1] / height * 1000),
                round(local[2] / width * 1000),
                round(local[3] / height * 1000),
            ]
            row["gt_global"].append(global_box)
            row["gt_local"].append(local)
            row["gt_local_1000"].append(norm)
            row["coordinate_transforms"].append(
                {
                    "global_bbox_xyxy": global_box,
                    "local_bbox_xyxy": local,
                    "local_bbox_1000": norm,
                }
            )
        if not matched:
            raise AssertionError(f"fixture sample not found: {sample_id}")
        write_jsonl(path, rows)

    @staticmethod
    def _project_difficulty(difficulty: Path) -> Path:
        authoritative_rows = curriculum_recipe._read_jsonl(difficulty)
        authoritative = difficulty.parent / "complete8.jsonl"
        write_jsonl(authoritative, authoritative_rows)
        projection_keys = (
            "record_id",
            "sample_id",
            "source_image_id",
            "task",
            "image_relpath",
            "crop_correct_count",
            "crop_complete4",
        )
        write_jsonl(
            difficulty,
            [
                {key: row.get(key) for key in projection_keys}
                for row in authoritative_rows
            ],
        )
        return authoritative

    def test_builds_disjoint_hard_anchor_and_replay_pools(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, difficulty = self._fixture(root)
            output = root / "curriculum"
            summary = curriculum_recipe.build(self._args(bundle, difficulty, output))

            self.assertEqual(summary["hard_groups"], 3)
            self.assertEqual(summary["matched_anchor_groups"], 3)
            self.assertEqual(summary["pools"]["global_replay"]["sample_groups"], 2)
            self.assertEqual(summary["pools"]["hard"]["training_records"], 5)
            self.assertEqual(summary["pools"]["hard"]["crop_training_records"], 4)
            self.assertEqual(
                summary["pools"]["hard"]["content_missing_global_records"], 1
            )
            self.assertEqual(
                summary["pools"]["global_replay"]["retention_full_image_records"], 2
            )
            recipe = json.loads(
                (output / "ui5_crop_rollout4_curriculum.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                {
                    entry["curriculum_pool"] for entry in recipe.values()
                },
                {"hard", "matched_anchor", "global_replay"},
            )
            self.assertTrue(recipe["ui5_curriculum_hard"]["ui5_crop_recipe"])
            self.assertTrue(
                recipe["ui5_curriculum_matched_anchor"]["ui5_crop_recipe"]
            )
            self.assertFalse(
                recipe["ui5_curriculum_global_replay"]["ui5_crop_recipe"]
            )
            self.assertTrue(
                recipe["ui5_curriculum_global_replay"]["ui5_retention_recipe"]
            )
            hard_records = curriculum_recipe._read_jsonl(output / "hard.jsonl")
            self.assertEqual(
                {row["_ui5_sample_id"] for row in hard_records},
                {"hard-pos", "hard-neg", "hard-content"},
            )
            self.assertTrue(
                all(Path(row["image"]).is_absolute() for row in hard_records)
            )
            hard_region = [
                row
                for row in hard_records
                if row["_ui5_task"] != "content_missing"
            ]
            self.assertEqual(len(hard_region), 4)
            self.assertTrue(
                all(row["_ui5_record_kind"] == "crop" for row in hard_region)
            )
            self.assertTrue(
                all(row["_ui5_gt_used_for_geometry"] is False for row in hard_region)
            )
            self.assertTrue(
                all(row["_ui5_partial_gt_indices"] == [] for row in hard_region)
            )
            self.assertTrue(all(Path(row["image"]).is_file() for row in hard_region))
            hard_content = [
                row
                for row in hard_records
                if row["_ui5_task"] == "content_missing"
            ]
            self.assertEqual(len(hard_content), 1)
            self.assertEqual(hard_content[0]["_ui5_record_kind"], "global_view")
            self.assertEqual(
                hard_content[0]["_ui5_crop_source"], "content_missing_global"
            )
            replay_records = curriculum_recipe._read_jsonl(
                output / "global_replay.jsonl"
            )
            self.assertTrue(
                all(row["_ui5_record_kind"] == "full_image" for row in replay_records)
            )
            self.assertTrue(
                all(row["_ui5_retention_view"] is True for row in replay_records)
            )
            assets = curriculum_recipe._read_jsonl(output / "crop_assets.jsonl")
            self.assertEqual(len(assets), 8)
            self.assertEqual(
                {Path(row["image"]).resolve() for row in hard_region},
                {
                    (output / row["relative_path"]).resolve()
                    for row in assets
                    if row["sample_id"].startswith("hard-")
                    and row["sample_id"] != "hard-content"
                },
            )
            success = json.loads((output / "_SUCCESS.json").read_text())
            self.assertTrue(
                all(
                    isinstance(metadata, dict)
                    and set(metadata) == {"bytes", "sha256"}
                    for metadata in success["files"].values()
                )
            )
            identity = curriculum_artifact_identity(
                output / "ui5_crop_rollout4_curriculum.json", self._schedule()
            )
            self.assertEqual(set(identity["verified_files"]), set(success["files"]))
            anchors = curriculum_recipe._read_jsonl(
                output / "matched_anchor_groups.jsonl"
            )
            self.assertEqual(
                {row["matched_hard_sample_id"] for row in anchors},
                {"hard-pos", "hard-neg", "hard-content"},
            )
            self.assertTrue((output / "_SUCCESS.json").is_file())

    def test_selected_group_uses_one_snapshot_verified_union_gt_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, difficulty = self._fixture(root)
            source_records = bundle / "manifest" / "source_records.jsonl"
            source_rows = curriculum_recipe._read_jsonl(source_records)
            second = json.loads(
                json.dumps(
                    next(row for row in source_rows if row["sample_id"] == "hard-pos")
                )
            )
            second["source_record_id"] = "source-hard-pos-second"
            second["gt_boxes_1000"] = [[3, 3, 4, 4]]
            second["gt_boxes_global_xyxy"] = [[3, 3, 4, 4]]
            second["portable_training_record"]["conversations"][-1]["value"] = (
                "<box><3><3><4><4></box>"
            )
            source_rows.append(second)
            write_jsonl(source_records, source_rows)
            self._append_crop_gt(bundle, "hard-pos", [3, 3, 4, 4])
            self._refresh_bundle_file_inventory(bundle)

            difficulty_rows = curriculum_recipe._read_jsonl(difficulty)
            hard = next(
                row for row in difficulty_rows if row["sample_id"] == "hard-pos"
            )
            hard["gt_global"] = [[1, 1, 2, 2], [3, 3, 4, 4]]
            write_jsonl(difficulty, difficulty_rows)

            output = root / "curriculum"
            summary = curriculum_recipe.build(self._args(bundle, difficulty, output))
            hard_records = curriculum_recipe._read_jsonl(output / "hard.jsonl")
            selected = next(
                row for row in hard_records if row["_ui5_sample_id"] == "hard-pos"
            )

            self.assertEqual(summary["schema_version"], 3)
            self.assertEqual(len(hard_records), 5)
            self.assertNotIn("_ui5_source_gt_global", selected)
            self.assertNotIn("_ui5_source_gt_1000", selected)
            self.assertEqual(selected["_ui5_union_source_record_count"], 2)
            self.assertEqual(
                selected["_ui5_union_gt_global"],
                [[1, 1, 2, 2], [3, 3, 4, 4]],
            )
            self.assertEqual(
                selected["conversations"][-1]["value"],
                "<box><10><20><20><40></box><box><30><60><40><80></box>",
            )
            selected_rows = [
                row for row in hard_records if row["_ui5_sample_id"] == "hard-pos"
            ]
            self.assertEqual(len(selected_rows), 2)
            self.assertEqual(
                selected_rows[1]["conversations"][-1]["value"],
                "<box>none</box>",
            )

    def test_global_replay_also_uses_one_verified_union_gt_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, difficulty = self._fixture(root)
            source_records = bundle / "manifest" / "source_records.jsonl"
            source_rows = curriculum_recipe._read_jsonl(source_records)
            second = json.loads(
                json.dumps(
                    next(row for row in source_rows if row["sample_id"] == "replay")
                )
            )
            second["source_record_id"] = "source-replay-second"
            second["gt_boxes_1000"] = [[3, 3, 4, 4]]
            second["gt_boxes_global_xyxy"] = [[3, 3, 4, 4]]
            second["portable_training_record"]["conversations"][-1]["value"] = (
                "<box><3><3><4><4></box>"
            )
            source_rows.append(second)
            write_jsonl(source_records, source_rows)
            self._refresh_bundle_file_inventory(bundle)

            difficulty_rows = curriculum_recipe._read_jsonl(difficulty)
            replay = next(
                row for row in difficulty_rows if row["sample_id"] == "replay"
            )
            replay["gt_global"] = [[1, 1, 2, 2], [3, 3, 4, 4]]
            write_jsonl(difficulty, difficulty_rows)

            output = root / "curriculum"
            curriculum_recipe.build(self._args(bundle, difficulty, output))
            replay_records = curriculum_recipe._read_jsonl(
                output / "global_replay.jsonl"
            )
            selected = next(
                row for row in replay_records if row["_ui5_sample_id"] == "replay"
            )

            self.assertEqual(len(replay_records), 2)
            self.assertEqual(selected["_ui5_union_source_record_count"], 2)
            self.assertEqual(
                selected["conversations"][-1]["value"],
                "<box><1><1><2><2></box><box><3><3><4><4></box>",
            )

    def test_crop_record_canonicalizes_equivalent_image_alias(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, difficulty = self._fixture(root)
            source_path = bundle / "manifest" / "source_records.jsonl"
            rows = curriculum_recipe._read_jsonl(source_path)
            target = next(row for row in rows if row["sample_id"] == "hard-pos")
            target["portable_training_record"]["images"] = [
                target["portable_training_record"]["image"]
            ]
            write_jsonl(source_path, rows)
            self._refresh_bundle_file_inventory(bundle)

            output = root / "curriculum"
            curriculum_recipe.build(self._args(bundle, difficulty, output))
            selected = [
                row
                for row in curriculum_recipe._read_jsonl(output / "hard.jsonl")
                if row["_ui5_sample_id"] == "hard-pos"
            ]
            self.assertEqual(len(selected), 2)
            self.assertTrue(all("images" not in row for row in selected))
            self.assertTrue(all(Path(row["image"]).is_file() for row in selected))

    def test_ineligible_anomalous_replay_is_excluded_from_every_pool(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, difficulty = self._fixture(root)
            rows = curriculum_recipe._read_jsonl(difficulty)
            replay = next(row for row in rows if row["sample_id"] == "replay")
            replay["grpo_source_eligible"] = False
            replay["annotation_anomaly"] = True
            replay["gt_global"] = []
            write_jsonl(difficulty, rows)

            output = root / "curriculum"
            summary = curriculum_recipe.build(self._args(bundle, difficulty, output))
            trained_ids = {
                row["_ui5_sample_id"]
                for filename in ("hard.jsonl", "matched_anchor.jsonl", "global_replay.jsonl")
                for row in curriculum_recipe._read_jsonl(output / filename)
            }

            self.assertNotIn("replay", trained_ids)
            self.assertEqual(
                {
                    row["_ui5_sample_id"]
                    for row in curriculum_recipe._read_jsonl(
                        output / "global_replay.jsonl"
                    )
                },
                {"replay-two"},
            )
            self.assertEqual(summary["formal_eligibility"]["authoritative_groups"], 8)
            self.assertEqual(summary["formal_eligibility"]["source_eligible_groups"], 7)
            self.assertEqual(summary["formal_eligibility"]["structural_anomaly_groups"], 1)
            self.assertEqual(
                summary["formal_eligibility"]["fully_eligible_rollout_groups"], 7
            )

    def test_incomplete8_replay_is_excluded_from_every_pool(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, difficulty = self._fixture(root)
            rows = curriculum_recipe._read_jsonl(difficulty)
            replay = next(row for row in rows if row["sample_id"] == "replay")
            replay["m31_complete4"] = False
            replay["cross_model_complete8"] = False
            replay["technical_error_free"] = False
            write_jsonl(difficulty, rows)

            output = root / "curriculum"
            curriculum_recipe.build(self._args(bundle, difficulty, output))
            trained_ids = {
                row["_ui5_sample_id"]
                for filename in ("hard.jsonl", "matched_anchor.jsonl", "global_replay.jsonl")
                for row in curriculum_recipe._read_jsonl(output / filename)
            }

            self.assertNotIn("replay", trained_ids)
            self.assertIn("replay-two", trained_ids)

    def test_selected_group_rejects_source_union_that_differs_from_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, difficulty = self._fixture(root)
            rows = curriculum_recipe._read_jsonl(difficulty)
            hard = next(row for row in rows if row["sample_id"] == "hard-pos")
            hard["gt_global"] = [[1, 1, 2, 2], [3, 3, 4, 4]]
            write_jsonl(difficulty, rows)

            with self.assertRaisesRegex(
                ValueError, "source GT union does not match snapshot gt_global"
            ):
                curriculum_recipe.build(
                    self._args(bundle, difficulty, root / "curriculum")
                )

    def test_selected_group_without_gt_provenance_rejects_multiple_records(self) -> None:
        record = {
            "_ui5_sample_id": "sample",
            "_ui5_task": "occlusion",
            "_ui5_positive": True,
            "image": "/image.png",
            "conversations": [
                {"from": "human", "value": "locate"},
                {"from": "gpt", "value": "<box><1><1><2><2></box>"},
            ],
        }
        with self.assertRaisesRegex(
            ValueError, "multiple training records without source GT provenance"
        ):
            curriculum_recipe._canonical_selected_supervision(
                "sample",
                [record, dict(record)],
                {"gt_global": [[1, 1, 2, 2]]},
            )

    def test_grpo_source_ineligible_group_cannot_be_hard_or_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, difficulty = self._fixture(root)
            rows = curriculum_recipe._read_jsonl(difficulty)
            hard = next(row for row in rows if row["sample_id"] == "hard-pos")
            hard["grpo_source_eligible"] = False
            write_jsonl(difficulty, rows)

            with self.assertRaisesRegex(
                ValueError, "hard group count mismatch: expected=3, observed=2"
            ):
                curriculum_recipe.build(
                    self._args(bundle, difficulty, root / "curriculum")
                )

    def test_anomaly_is_filtered_before_untrusted_polarity_is_consulted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, difficulty = self._fixture(root)
            rows = curriculum_recipe._read_jsonl(difficulty)
            hard = next(row for row in rows if row["sample_id"] == "hard-pos")
            hard["grpo_source_eligible"] = False
            hard["annotation_anomaly"] = True
            hard["gt_global"] = []
            write_jsonl(difficulty, rows)

            with self.assertRaisesRegex(
                ValueError, "hard group count mismatch: expected=3, observed=2"
            ):
                curriculum_recipe.build(
                    self._args(bundle, difficulty, root / "curriculum")
                )

    def test_rejects_snapshot_that_marks_anomaly_source_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, difficulty = self._fixture(root)
            rows = curriculum_recipe._read_jsonl(difficulty)
            rows[0]["annotation_anomaly"] = True
            write_jsonl(difficulty, rows)

            with self.assertRaisesRegex(
                ValueError, "anomalous sample grpo_source_eligible=true"
            ):
                curriculum_recipe.build(
                    self._args(bundle, difficulty, root / "curriculum")
                )

    def test_selected_region_group_must_include_every_base_tile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, difficulty = self._fixture(root)
            crop_path = bundle / "manifest" / "crop_samples.jsonl"
            rows = curriculum_recipe._read_jsonl(crop_path)
            rows = [
                row
                for row in rows
                if not (
                    row["sample_id"] == "hard-pos" and row["crop_index"] == 1
                )
            ]
            write_jsonl(crop_path, rows)
            self._refresh_bundle_file_inventory(bundle)

            with self.assertRaisesRegex(ValueError, "every base tile exactly once"):
                curriculum_recipe.build(
                    self._args(bundle, difficulty, root / "curriculum")
                )

    def test_selected_region_group_rejects_partial_gt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, difficulty = self._fixture(root)
            crop_path = bundle / "manifest" / "crop_samples.jsonl"
            rows = curriculum_recipe._read_jsonl(crop_path)
            target = next(
                row
                for row in rows
                if row["sample_id"] == "hard-pos" and row["crop_index"] == 0
            )
            target["partial_gt_indices"] = [0]
            write_jsonl(crop_path, rows)
            self._refresh_bundle_file_inventory(bundle)

            with self.assertRaisesRegex(ValueError, "partial_gt_indices"):
                curriculum_recipe.build(
                    self._args(bundle, difficulty, root / "curriculum")
                )

    def test_selected_region_group_rejects_crop_union_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, difficulty = self._fixture(root)
            crop_path = bundle / "manifest" / "crop_samples.jsonl"
            rows = curriculum_recipe._read_jsonl(crop_path)
            for row in rows:
                if row["sample_id"] == "hard-pos":
                    row["sample_gt_global"] = []
            write_jsonl(crop_path, rows)
            self._refresh_bundle_file_inventory(bundle)

            with self.assertRaisesRegex(
                ValueError, "sample_gt_global differs from authoritative union"
            ):
                curriculum_recipe.build(
                    self._args(bundle, difficulty, root / "curriculum")
                )

    def test_crop_geometry_must_be_explicitly_gt_free(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, difficulty = self._fixture(root)
            crop_path = bundle / "manifest" / "crop_samples.jsonl"
            rows = curriculum_recipe._read_jsonl(crop_path)
            rows[0]["gt_used_for_geometry"] = True
            write_jsonl(crop_path, rows)
            self._refresh_bundle_file_inventory(bundle)

            with self.assertRaisesRegex(ValueError, "not explicitly GT-free"):
                curriculum_recipe.build(
                    self._args(bundle, difficulty, root / "curriculum")
                )

    def test_existing_curriculum_rejects_tampered_crop_asset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, difficulty = self._fixture(root)
            output = root / "curriculum"
            args = self._args(bundle, difficulty, output)
            summary = curriculum_recipe.build(args)
            asset = output / summary["crop_assets"][0]["relative_path"]
            asset.write_bytes(b"tampered")

            with self.assertRaisesRegex(RuntimeError, "artifact changed"):
                curriculum_recipe.build(args)

    def test_fails_closed_when_bundle_manifest_is_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, difficulty = self._fixture(root)
            manifest_path = bundle / "bundle_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["complete"] = False
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "manifest is not complete"):
                curriculum_recipe.build(
                    self._args(bundle, difficulty, root / "curriculum")
                )
            self.assertFalse((root / "curriculum").exists())

    def test_fails_closed_when_declared_bundle_file_hash_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, difficulty = self._fixture(root)
            source_records = bundle / "manifest" / "source_records.jsonl"
            payload = bytearray(source_records.read_bytes())
            payload[-1] = ord(" ")
            source_records.write_bytes(payload)

            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                curriculum_recipe.build(
                    self._args(bundle, difficulty, root / "curriculum")
                )

    def test_fails_closed_when_unique_image_hash_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, difficulty = self._fixture(root)
            (bundle / "images" / "hard-pos.png").write_bytes(b"evil-image")

            with self.assertRaisesRegex(RuntimeError, "image SHA-256 mismatch"):
                curriculum_recipe.build(
                    self._args(bundle, difficulty, root / "curriculum")
                )

    def test_fails_closed_when_source_record_uses_unverified_image(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, difficulty = self._fixture(root)
            (bundle / "images" / "unverified.png").write_bytes(b"not-in-unique-images")
            source_records = bundle / "manifest" / "source_records.jsonl"
            rows = curriculum_recipe._read_jsonl(source_records)
            rows[-1]["portable_training_record"]["image"] = "images/unverified.png"
            write_jsonl(source_records, rows)
            self._refresh_bundle_file_inventory(bundle)

            with self.assertRaisesRegex(
                ValueError, "training record images are not verified"
            ):
                curriculum_recipe.build(
                    self._args(bundle, difficulty, root / "curriculum")
                )

    def test_existing_curriculum_rejects_changed_bundle_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, difficulty = self._fixture(root)
            output = root / "curriculum"
            args = self._args(bundle, difficulty, output)
            summary = curriculum_recipe.build(args)
            self.assertEqual(curriculum_recipe.build(args), summary)
            manifest_path = bundle / "bundle_manifest.json"
            manifest_path.write_text(
                manifest_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(RuntimeError, r"changed \(rollout_bundle\)"):
                curriculum_recipe.build(args)

    def test_existing_v1_curriculum_is_not_reused_after_supervision_fix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, difficulty = self._fixture(root)
            output = root / "curriculum"
            args = self._args(bundle, difficulty, output)
            curriculum_recipe.build(args)
            for filename in ("curriculum_manifest.json", "_SUCCESS.json"):
                path = output / filename
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["schema_version"] = 1
                path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "builder schema differs"):
                curriculum_recipe.build(args)

    def test_rejects_duplicate_difficulty_id_even_if_first_row_is_ineligible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, difficulty = self._fixture(root)
            rows = curriculum_recipe._read_jsonl(difficulty)
            duplicate = dict(rows[0])
            duplicate["crop_complete4"] = False
            write_jsonl(difficulty, [duplicate, *rows])

            with self.assertRaisesRegex(
                ValueError, "duplicate .*rollout difficulty.*hard-pos"
            ):
                curriculum_recipe.build(
                    self._args(bundle, difficulty, root / "curriculum")
                )

    def test_projected_difficulty_uses_verified_complete8_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, difficulty = self._fixture(root)
            authoritative = self._project_difficulty(difficulty)

            summary = curriculum_recipe.build(
                self._args(bundle, difficulty, root / "curriculum")
            )

            state = summary["inputs"]["rollout_difficulty_authoritative"]
            self.assertTrue(state["projection_verified"])
            self.assertEqual(state["authoritative_path"], str(authoritative.resolve()))
            self.assertEqual(state["rows"], 8)

    def test_projected_difficulty_requires_complete8_sibling(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, difficulty = self._fixture(root)
            rows = curriculum_recipe._read_jsonl(difficulty)
            write_jsonl(
                difficulty,
                [
                    {
                        "sample_id": row["sample_id"],
                        "task": row["task"],
                        "crop_correct_count": row["crop_correct_count"],
                        "crop_complete4": row["crop_complete4"],
                    }
                    for row in rows
                ],
            )

            with self.assertRaisesRegex(FileNotFoundError, "complete8.jsonl"):
                curriculum_recipe.build(
                    self._args(bundle, difficulty, root / "curriculum")
                )

    def test_projected_difficulty_must_match_complete8(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, difficulty = self._fixture(root)
            self._project_difficulty(difficulty)
            projected = curriculum_recipe._read_jsonl(difficulty)
            projected[0]["crop_correct_count"] = 1
            write_jsonl(difficulty, projected)

            with self.assertRaisesRegex(
                ValueError, "projected/authoritative.*crop_correct_count"
            ):
                curriculum_recipe.build(
                    self._args(bundle, difficulty, root / "curriculum")
                )

    def test_authoritative_parse_error_is_not_treated_as_clean(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, difficulty = self._fixture(root)
            authoritative = self._project_difficulty(difficulty)
            rows = curriculum_recipe._read_jsonl(authoritative)
            rows[0]["parse_error_count"] = 1
            write_jsonl(authoritative, rows)

            with self.assertRaisesRegex(ValueError, "hard group count mismatch"):
                curriculum_recipe.build(
                    self._args(bundle, difficulty, root / "curriculum")
                )

    def test_authoritative_polarity_conflict_with_training_record_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, difficulty = self._fixture(root)
            rows = curriculum_recipe._read_jsonl(difficulty)
            rows[0]["gt_global"] = []
            write_jsonl(difficulty, rows)

            with self.assertRaisesRegex(ValueError, "polarity conflict.*hard-pos"):
                curriculum_recipe.build(
                    self._args(bundle, difficulty, root / "curriculum")
                )

    def test_authoritative_task_conflict_with_training_record_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, difficulty = self._fixture(root)
            rows = curriculum_recipe._read_jsonl(difficulty)
            rows[0]["task"] = "cropping"
            write_jsonl(difficulty, rows)

            with self.assertRaisesRegex(ValueError, "task conflict.*hard-pos"):
                curriculum_recipe.build(
                    self._args(bundle, difficulty, root / "curriculum")
                )

    def test_fails_closed_when_expected_hard_count_differs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, difficulty = self._fixture(root)
            with self.assertRaisesRegex(ValueError, "hard group count mismatch"):
                curriculum_recipe.build(
                    Namespace(
                        base_recipe=None,
                        rollout_difficulty=difficulty,
                        rollout_bundle_root=bundle,
                        output_dir=root / "curriculum",
                        expected_hard_groups=72,
                        seed=42,
                    )
                )


if __name__ == "__main__":
    unittest.main()
