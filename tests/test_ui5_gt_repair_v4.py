from __future__ import annotations

import json
import random
import shutil
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_ui5_crop_training_recipe as recipe_builder  # noqa: E402
import locany_ui5_common  # noqa: E402
import run_locany_ui5_local_debug  # noqa: E402
import run_ui5_parallel_inference  # noqa: E402
import run_ui5_gt_repair as gt_repair  # noqa: E402
import submit_locany_ui5  # noqa: E402
from analyze_ui5_source_overlap import content_fingerprint  # noqa: E402
from run_ui5_crop_audit import (  # noqa: E402
    V4_FINAL_TRAINING_GATE_CONDITIONS,
    atomic_write_json,
    atomic_write_jsonl,
    validate_recipe_repair_mapping_summary,
)
from ui5_lossless_tiling import (  # noqa: E402
    assert_lossless_coverage,
    generate_lossless_tiles,
    global_bbox_to_tile,
    merge_tile_predictions,
    tile_bbox_to_global,
)


def repair_sample(
    sample_id: str, gt: list[int], *, task: str = "ui_text_overflow", split: str = "train"
) -> dict:
    return {
        "sample_id": sample_id,
        "image_id": "image_shared",
        "task": task,
        "split": split,
        "width": 1200,
        "height": 1200,
        "gt_boxes": [gt],
        "gt_boxes_1000": [gt],
    }


def partial_source(sample_id: str, gt: list[int], crop: list[int]) -> dict:
    return {
        "crop_boxes": [crop],
        "failures": [
            {
                "sample_id": sample_id,
                "task": "ui_text_overflow",
                "gt_index": 0,
                "gt_bbox": gt,
                "failure_type": "partial_intersection",
            }
        ],
    }


class GTRepairGeometryTests(unittest.TestCase):
    def test_106_valid_failures_have_one_task_scoped_action_each(self) -> None:
        detections = []
        actions = []
        sequence = 0
        for task, count in gt_repair.EXPECTED_VALID_REPAIRS_BY_TASK.items():
            for _ in range(count):
                row = {
                    "task": task,
                    "sample_id": f"sample_{sequence}",
                    "gt_index": 0,
                    "source": "manual_gt_repair",
                    "split": "train",
                }
                detections.append(row)
                actions.append({key: row[key] for key in ("task", "sample_id", "gt_index")})
                sequence += 1
        gt_repair.validate_repair_inventory(
            detections,
            actions,
            expected_count=106,
            expected_by_task=gt_repair.EXPECTED_VALID_REPAIRS_BY_TASK,
        )

    def test_text_overflow_one_pixel_bottom_extension_keeps_gt(self) -> None:
        gt = [364, 284, 432, 355]
        sample = repair_sample("sample_5ac3ab2fad4252026615", gt)
        boxes, detections, actions, provenance = gt_repair.repair_sample_geometry(
            None,
            sample=sample,
            source_result=partial_source(sample["sample_id"], gt, [0, 0, 1178, 354]),
            detection={},
            task_rule={"context_ratio": 0.2},
            source_audit_name="crop_audit_v3",
            max_crops=10,
            boundary_margin_ratio=0.01,
        )
        self.assertEqual(boxes, [[0, 0, 1178, 355]])
        self.assertEqual(detections[0]["bbox"], gt)
        self.assertTrue(actions[0]["gt_unchanged"])
        self.assertEqual(actions[0]["before_crop_bbox"][3], 354)
        self.assertEqual(actions[0]["after_crop_bbox"][3], 355)
        self.assertEqual(provenance, ["manual_gt_repair"])

    def test_text_overflow_two_pixel_bottom_extension_keeps_gt(self) -> None:
        gt = [289, 188, 427, 240]
        sample = repair_sample("sample_6f82ad6a361c4afe81f1", gt)
        boxes, _, actions, _ = gt_repair.repair_sample_geometry(
            None,
            sample=sample,
            source_result=partial_source(sample["sample_id"], gt, [0, 0, 644, 238]),
            detection={},
            task_rule={"context_ratio": 0.2},
            source_audit_name="crop_audit_v3",
            max_crops=10,
            boundary_margin_ratio=0.01,
        )
        self.assertEqual(boxes, [[0, 0, 644, 240]])
        self.assertEqual(actions[0]["bbox"], gt)

    def test_repair_is_refused_outside_training_split(self) -> None:
        gt = [10, 10, 20, 20]
        sample = repair_sample("sample_val", gt, split="val")
        with self.assertRaisesRegex(RuntimeError, "training-only"):
            gt_repair.repair_sample_geometry(
                None,
                sample=sample,
                source_result=partial_source(sample["sample_id"], gt, [0, 0, 15, 15]),
                detection={},
                task_rule={"context_ratio": 0.2},
                source_audit_name="crop_audit_v3",
                max_crops=10,
                boundary_margin_ratio=0.01,
            )

    def test_uncovered_repair_is_task_scoped_and_does_not_mutate_raw_detection(self) -> None:
        gt = [800, 700, 850, 760]
        sample = repair_sample("sample_occ", gt, task="ui_occlusion")
        source = {
            "crop_boxes": [[0, 0, 300, 300]],
            "failures": [
                {
                    "sample_id": sample["sample_id"],
                    "task": sample["task"],
                    "gt_index": 0,
                    "gt_bbox": gt,
                    "failure_type": "uncovered",
                }
            ],
        }
        raw = {
            "text_detections": [{"bbox": [1, 1, 20, 20], "score": 0.5}],
            "icon_detections": [{"bbox": [30, 30, 50, 50], "score": 0.5}],
        }
        raw_before = json.loads(json.dumps(raw))

        def fake_proposal(_cropper, augmented, _rule, **_kwargs):
            self.assertEqual(len(augmented["text_detections"]), 1)
            self.assertEqual(len(augmented["icon_detections"]), 2)
            repair = augmented["icon_detections"][-1]
            self.assertEqual(repair["task"], "ui_occlusion")
            self.assertEqual(repair["sample_id"], "sample_occ")
            return {"crop_boxes": [[790, 690, 870, 780]]}

        with mock.patch.object(gt_repair, "proposal_crops", side_effect=fake_proposal):
            boxes, detections, actions, _ = gt_repair.repair_sample_geometry(
                object(),
                sample=sample,
                source_result=source,
                detection=raw,
                task_rule={"context_ratio": 0.2, "min_context_image_ratio": 0.01},
                source_audit_name="crop_audit_v3",
                max_crops=10,
                boundary_margin_ratio=0.01,
            )
        self.assertEqual(raw, raw_before)
        self.assertEqual(detections[0]["source"], "manual_gt_repair")
        self.assertEqual(detections[0]["split"], "train")
        self.assertTrue(any(gt_repair.rect_contains(box, gt) for box in boxes))
        self.assertEqual(actions[0]["action"], "add_task_scoped_regenerated_crop")

    def test_repair_four_panel_is_written_from_saved_detection_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.png"
            Image.new("RGB", (120, 80), "white").save(source)
            actions = [
                {
                    "sample_id": "sample",
                    "task": "ui_occlusion",
                    "failure_type": "partial_intersection",
                    "gt_index": 0,
                    "bbox": [40, 20, 80, 60],
                    "action": "expand_max_intersection_crop",
                }
            ]
            gt_repair.render_repair_visualizations(
                target_audit=root / "audit",
                actions=actions,
                unique_by_id={"image": {"image_path": str(source)}},
                detections={
                    "image": {
                        "text_detections": [{"bbox": [5, 5, 30, 20]}],
                        "icon_detections": [{"bbox": [10, 30, 35, 60]}],
                    }
                },
                source_results_by_sample={"sample": {"crop_boxes": [[0, 0, 60, 80]]}},
                repaired_results_by_sample={
                    "sample": {
                        "image_id": "image",
                        "crop_boxes": [[0, 0, 80, 80]],
                    }
                },
            )
            panel = Path(actions[0]["visualization_4panel"])
            self.assertTrue(panel.is_file())
            with Image.open(panel) as rendered:
                self.assertEqual(rendered.format, "PNG")
                self.assertGreaterEqual(rendered.width, 4 * 120)
            self.assertTrue(
                (root / "audit" / "gt_repair_visualizations" / "gallery" / "index.html").is_file()
            )

    def test_excluded_annotation_evidence_is_copied_without_deleting_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_audit = root / "crop_audit_v3"
            target_audit = root / "crop_audit_v4_gt_repair"
            source_image = root / "source.png"
            old_panel = root / "old_panel.png"
            Image.new("RGB", (100, 80), "white").save(source_image)
            Image.new("RGB", (400, 80), "gray").save(old_panel)
            atomic_write_jsonl(
                source_audit / "gt_failures_visualized.jsonl",
                [
                    {
                        "sample_id": gt_repair.EXCLUDED_SAMPLE_ID,
                        "gt_index": 0,
                        "gt_bbox": [816, 789, 847, 1039],
                        "visualization_4panel": str(old_panel),
                    }
                ],
            )
            exclusions = gt_repair.build_exclusion_evidence(
                target_audit=target_audit,
                source_audit=source_audit,
                samples_by_id={
                    gt_repair.EXCLUDED_SAMPLE_ID: {
                        "sample_id": gt_repair.EXCLUDED_SAMPLE_ID,
                        "image_id": "image",
                        "task": gt_repair.EXCLUDED_TASK,
                        "canonical_path": str(source_image),
                        "source_records": [],
                    }
                },
            )
            evidence = Path(exclusions[0]["evidence_dir"])
            self.assertTrue(source_image.is_file())
            for name in (
                "gt_record.json", "task_sample_record.json", "visualization_4panel.png",
                "source_image.png", "README.md",
            ):
                self.assertTrue((evidence / name).is_file(), name)
            self.assertFalse(
                json.loads((evidence / "gt_record.json").read_text(encoding="utf-8"))[
                    "source_data_deleted"
                ]
            )


class LosslessTilingTests(unittest.TestCase):
    def test_horizontal_vertical_long_small_and_random_images_have_full_coverage(self) -> None:
        shapes = [(5000, 600), (600, 5000), (8000, 1400), (500, 400)]
        rng = random.Random(20260826)
        shapes.extend((rng.randint(200, 7000), rng.randint(200, 7000)) for _ in range(50))
        for width, height in shapes:
            with self.subTest(width=width, height=height):
                tiles = generate_lossless_tiles(width, height)
                self.assertGreaterEqual(len(tiles), 1)
                self.assertLessEqual(len(tiles), 10)
                assert_lossless_coverage(width, height, tiles)

    def test_content_missing_always_uses_the_full_image(self) -> None:
        self.assertEqual(
            generate_lossless_tiles(8000, 3000, task="ui_content_missing"),
            [[0, 0, 8000, 3000]],
        )

    def test_forward_inverse_bbox_transform_is_within_one_pixel(self) -> None:
        tile = [401, 203, 1601, 1403]
        global_box = [515, 315, 1001, 1200]
        local = global_bbox_to_tile(global_box, tile)
        restored = tile_bbox_to_global(local, tile, image_size=(2200, 1800))
        self.assertLessEqual(max(abs(a - b) for a, b in zip(global_box, restored)), 1)

    def test_cross_tile_dedup_happens_after_global_mapping(self) -> None:
        merged = merge_tile_predictions(
            [
                {"bbox": [90, 20, 140, 70], "tile_bbox": [0, 0, 200, 100], "label": "x", "score": 0.9},
                {"bbox": [0, 20, 50, 70], "tile_bbox": [90, 0, 290, 100], "label": "x", "score": 0.8},
            ],
            image_size=(300, 100),
            iou_threshold=0.5,
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["bbox"], [90.0, 20.0, 140.0, 70.0])


class RecipeBuilderTests(unittest.TestCase):
    def _make_fixture(self, root: Path, *, include_excel: bool = True) -> Namespace:
        output = root / "output"
        audit = output / "crop_audit_v4_gt_repair"
        recipes = audit / "training_recipes"
        audit.mkdir(parents=True)
        image = root / "source.png"
        crop_raw = root / "crop_raw.png"
        crop_repair = root / "crop_repair.png"
        for path, color in ((image, "white"), (crop_raw, "blue"), (crop_repair, "red")):
            Image.new("RGB", (100, 80), color).save(path)

        annotation = root / "ui_text_overflow_train.jsonl"
        valid_record = {
            "image": str(image),
            "conversations": [
                {"from": "human", "value": "locate"},
                {"from": "gpt", "value": "<box><100><100><300><300></box>"},
            ],
        }
        excluded_record = {
            "image": str(image),
            "conversations": [
                {"from": "human", "value": "locate"},
                {"from": "gpt", "value": "<box><816><789><847><1039></box>"},
            ],
        }
        atomic_write_jsonl(annotation, [valid_record, excluded_record])
        base_meta = root / "base_meta.json"
        atomic_write_json(
            base_meta,
            {"ui_text_overflow": {"annotation": [str(annotation)], "root": ""}},
        )
        parent_manifest = output / "manifest" / "task_samples.jsonl"
        atomic_write_jsonl(
            parent_manifest,
            [
                {
                    "sample_id": "sample_valid",
                    "image_id": "image",
                    "task": "ui_text_overflow",
                    "split": "train",
                    "source_records": [{"source_file": str(annotation), "line_no": 1}],
                },
                {
                    "sample_id": gt_repair.EXCLUDED_SAMPLE_ID,
                    "image_id": "image",
                    "task": gt_repair.EXCLUDED_TASK,
                    "split": "train",
                    "source_records": [{"source_file": str(annotation), "line_no": 2}],
                },
            ],
        )
        conversations = [
            {"from": "human", "value": "locate"},
            {"from": "gpt", "value": "<box><0><0><1000><1000></box>"},
        ]
        task_manifest = audit / "task_aware_manifest.jsonl"
        atomic_write_jsonl(
            task_manifest,
            [
                {
                    "sample_id": "sample_valid",
                    "image_id": "image",
                    "task": "ui_text_overflow",
                    "training_records": [
                        {
                            "image": str(crop_raw),
                            "conversations": conversations,
                            "_ui5_sample_id": "sample_valid",
                            "_ui5_image_id": "image",
                            "_ui5_task": "ui_text_overflow",
                            "_ui5_split": "train",
                            "_ui5_record_kind": "crop",
                            "_ui5_crop_source": "raw_detector",
                            "_ui5_contained_gt_indices": [0],
                            "_ui5_manual_repair_gt_indices": [],
                            "_ui5_training_eligible": True,
                        },
                        {
                            "image": str(crop_repair),
                            "conversations": conversations,
                            "_ui5_sample_id": "sample_valid",
                            "_ui5_image_id": "image",
                            "_ui5_task": "ui_text_overflow",
                            "_ui5_split": "train",
                            "_ui5_record_kind": "crop",
                            "_ui5_crop_source": "manual_gt_repair",
                            "_ui5_contained_gt_indices": [0],
                            "_ui5_manual_repair_gt_indices": [0],
                            "_ui5_training_eligible": True,
                        },
                    ],
                }
            ],
        )
        excluded = audit / "excluded_training_samples.jsonl"
        atomic_write_jsonl(
            excluded,
            [
                {
                    "sample_id": gt_repair.EXCLUDED_SAMPLE_ID,
                    "task": gt_repair.EXCLUDED_TASK,
                    "reason": "annotation_error",
                }
            ],
        )
        conditions = {name: True for name in V4_FINAL_TRAINING_GATE_CONDITIONS}
        conditions.update(
            {
                "excluded_sample_absent_from_text_overflow_recipe": False,
                "crop_training_recipe_written_successfully": False,
                "crop_training_recipe_contains_crop_records": False,
            }
        )
        state = {"schema_version": 4, "test": True}
        atomic_write_json(audit / "audit_state.json", state)
        atomic_write_json(
            audit / "summary.json",
            {
                "recommended_config": gt_repair.REPAIR_CONFIG,
                "audit_state_digest": gt_repair.audit_state_digest(state),
                "input_snapshot_digest": "snapshot",
                "repair_metrics": {
                    "training_materialization_gt_recall_after_repair": 1.0,
                    "repaired_valid_failure_count": 1,
                },
                "next_stage_gate": {"conditions": conditions},
            },
        )
        for path in (
            audit / "materialization_summary.json",
            audit / "statistics.csv",
            audit / "gt_repair_detections.jsonl",
            audit / "gt_repair_visualizations" / "gallery" / "index.html",
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("test\n", encoding="utf-8")
        atomic_write_jsonl(
            audit / "gt_repair_actions.jsonl",
            [{"sample_id": "sample_valid", "gt_index": 0, "task": "ui_text_overflow"}],
        )
        if include_excel:
            (audit / "ui5_crop_audit.xlsx").write_bytes(b"xlsx")
        return Namespace(
            audit_dir=audit,
            base_meta=base_meta,
            task_aware_manifest=task_manifest,
            excluded_samples=excluded,
            mode="full_plus_crop",
            output_dir=recipes,
            require_valid_gt_recall=1.0,
        )

    def test_recipe_excludes_bad_annotation_and_marker_is_digest_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self._make_fixture(Path(temporary))
            result = recipe_builder.build(args)
            self.assertEqual(result["full_image_records"], 1)
            self.assertEqual(result["crop_records"], 2)
            self.assertEqual(result["gt_repair_crop_records"], 1)
            self.assertEqual(result["gt_repair_action_count"], 1)
            self.assertEqual(result["gt_repair_action_mapped_count"], 1)
            self.assertTrue(result["all_gt_repair_actions_mapped"])
            combined = [
                json.loads(line)
                for line in (args.output_dir / "ui_defect_5class_train_full_plus_crop.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertFalse(
                any(row.get("_ui5_sample_id") == gt_repair.EXCLUDED_SAMPLE_ID for row in combined)
            )
            self.assertTrue(any(row.get("_ui5_crop_source") == "manual_gt_repair" for row in combined))
            marker_path = args.audit_dir / "training_ready.json"
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            summary_path = args.audit_dir / "summary.json"
            self.assertEqual(marker["summary_file_digest"], content_fingerprint(summary_path))
            self.assertEqual(
                marker["excluded_samples_digest"], content_fingerprint(args.excluded_samples)
            )
            self.assertTrue(marker["created_after_all_checks"])
            validated_counts = validate_recipe_repair_mapping_summary(
                json.loads(
                    (args.output_dir / "recipe_summary.json").read_text(encoding="utf-8")
                )
            )
            self.assertEqual(validated_counts, (1, 1))

    def test_crop_only_recipe_has_no_local_full_image_and_keeps_content_global(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self._make_fixture(root)
            content_annotation = root / "ui_content_missing_train.jsonl"
            content_image = root / "content.png"
            Image.new("RGB", (100, 80), "green").save(content_image)
            content_record = {
                "image": str(content_image),
                "conversations": [
                    {"from": "human", "value": "locate content missing"},
                    {"from": "gpt", "value": "<box>none</box>"},
                ],
            }
            atomic_write_jsonl(content_annotation, [content_record])
            base = json.loads(args.base_meta.read_text(encoding="utf-8"))
            base["ui_content_missing"] = {
                "annotation": [str(content_annotation)],
                "root": "",
            }
            atomic_write_json(args.base_meta, base)
            parent = args.audit_dir.parent / "manifest" / "task_samples.jsonl"
            parent_rows = gt_repair.read_jsonl(parent)
            parent_rows.append(
                {
                    "sample_id": "sample_content",
                    "image_id": "content_image",
                    "task": "ui_content_missing",
                    "split": "train",
                    "source_records": [
                        {"source_file": str(content_annotation), "line_no": 1}
                    ],
                }
            )
            atomic_write_jsonl(parent, parent_rows)
            manifest_rows = gt_repair.read_jsonl(args.task_aware_manifest)
            manifest_rows.append(
                {
                    "sample_id": "sample_content",
                    "image_id": "content_image",
                    "task": "ui_content_missing",
                    "training_records": [],
                    "content_missing_global_view": True,
                }
            )
            atomic_write_jsonl(args.task_aware_manifest, manifest_rows)
            atomic_write_json(
                args.task_aware_manifest.parent / "data_split_overlap.json",
                {
                    "train_validation_content_overlap_count": 0,
                    "train_test_content_overlap_count": 0,
                    "validation_test_content_overlap_count": 0,
                },
            )
            args.mode = "crop_only"
            result = recipe_builder.build(args)
            rows = gt_repair.read_jsonl(
                args.output_dir / "ui_defect_5class_train_crop_only.jsonl"
            )
            self.assertEqual(result["crop_only_local_task_full_image_records"], 0)
            self.assertEqual(result["crop_only_content_missing_global_records"], 1)
            self.assertEqual(sum(row["_ui5_record_kind"] == "crop" for row in rows), 2)
            self.assertEqual(
                sum(row["_ui5_record_kind"] == "global_view" for row in rows), 1
            )
            self.assertTrue(
                json.loads((args.audit_dir / "training_ready.json").read_text())["training_ready"]
            )

    def test_missing_report_fails_closed_and_leaves_no_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self._make_fixture(Path(temporary), include_excel=False)
            with self.assertRaisesRegex(RuntimeError, "final training gate failed"):
                recipe_builder.build(args)
            self.assertFalse((args.audit_dir / "training_ready.json").exists())

    def test_unmapped_gt_repair_action_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self._make_fixture(Path(temporary))
            manifest = gt_repair.read_jsonl(args.task_aware_manifest)
            manifest[0]["training_records"][1]["_ui5_manual_repair_gt_indices"] = [1]
            atomic_write_jsonl(args.task_aware_manifest, manifest)
            with self.assertRaisesRegex(RuntimeError, "do not map"):
                recipe_builder.build(args)
            self.assertFalse((args.audit_dir / "training_ready.json").exists())

    def test_byte_identical_source_tree_alias_maps_without_basename_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self._make_fixture(root)
            base_payload = json.loads(args.base_meta.read_text(encoding="utf-8"))
            original = Path(base_payload["ui_text_overflow"]["annotation"][0])
            alias = root / "moved_tree" / "renamed_source.jsonl"
            alias.parent.mkdir(parents=True)
            shutil.copyfile(original, alias)
            base_payload["ui_text_overflow"]["annotation"] = [str(alias)]
            atomic_write_json(args.base_meta, base_payload)

            result = recipe_builder.build(args)
            self.assertEqual(result["source_mapping_exact_records"], 0)
            self.assertEqual(result["source_mapping_content_alias_records"], 2)

    def test_different_source_content_does_not_fall_back_to_line_number(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self._make_fixture(root)
            base_payload = json.loads(args.base_meta.read_text(encoding="utf-8"))
            original = Path(base_payload["ui_text_overflow"]["annotation"][0])
            alias = root / "moved_tree" / "renamed_source.jsonl"
            alias.parent.mkdir(parents=True)
            alias.write_bytes(original.read_bytes() + b"\n")
            base_payload["ui_text_overflow"]["annotation"] = [str(alias)]
            atomic_write_json(args.base_meta, base_payload)

            with self.assertRaisesRegex(ValueError, "byte-identical source-file fingerprint"):
                recipe_builder.build(args)
            self.assertFalse((args.audit_dir / "training_ready.json").exists())

    def test_manual_repair_in_validation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self._make_fixture(Path(temporary))
            rows = [
                json.loads(line)
                for line in args.task_aware_manifest.read_text(encoding="utf-8").splitlines()
            ]
            rows[0]["training_records"][1]["_ui5_split"] = "val"
            atomic_write_jsonl(args.task_aware_manifest, rows)
            with self.assertRaisesRegex(RuntimeError, "validation/test"):
                recipe_builder.build(args)
            self.assertFalse((args.audit_dir / "training_ready.json").exists())


class CropTrainingWiringTests(unittest.TestCase):
    def test_crop_only_defaults_to_all_record_sampling_across_local_and_submit(self) -> None:
        audit = PROJECT_ROOT / "work_dirs" / "test_v5_croponly_audit"
        local_args = run_locany_ui5_local_debug.parse_args(
            [
                "--machine", "a800", "--gpus", "4", "--max-steps", "20",
                "--use-detection-crops", "--crop-audit-dir", str(audit),
                "--crop-train-mode", "crop_only", "--project-root", str(PROJECT_ROOT),
            ]
        )
        local_env = run_locany_ui5_local_debug.build_environment(local_args, base_env={})
        self.assertEqual(local_env["UI5_CROP_TRAIN_MODE"], "crop_only")
        self.assertEqual(
            local_env["UI5_UI_SAMPLING_MODE"], "task_balanced_all_records"
        )

        submit_args = submit_locany_ui5.parse_args(
            [
                "--machine", "a800", "--resource-group", "aiai_locate", "--gpus", "4",
                "--use-detection-crops", "--crop-audit-dir", "/mnt/audit/v5",
                "--crop-train-mode", "crop_only",
                "--eval-input-dir", "/mnt/validation",
                "--eval-data-split", "validation",
                "--require-cache-scope", "validation",
                "--render-only",
            ]
        )
        rendered, runtime = submit_locany_ui5.render_job(submit_args)
        self.assertEqual(runtime["UI5_CROP_TRAIN_MODE"], "crop_only")
        self.assertEqual(
            runtime["UI5_UI_SAMPLING_MODE"], "task_balanced_all_records"
        )
        self.assertEqual(runtime["EVAL_DATA_SPLIT"], "validation")
        self.assertEqual(runtime["EVAL_INPUT_DIR"], "/mnt/validation")
        self.assertIn("UI5_UI_SAMPLING_MODE", rendered)
        self.assertIn("EVAL_DATA_SPLIT", rendered)

    def test_local_debug_resolves_all_four_crop_parameters(self) -> None:
        audit = PROJECT_ROOT / "work_dirs" / "test_v4_audit"
        args = run_locany_ui5_local_debug.parse_args(
            [
                "--machine", "a800", "--gpus", "4", "--max-steps", "20",
                "--use-detection-crops", "--crop-audit-dir", str(audit),
                "--crop-train-mode", "full_plus_crop", "--project-root", str(PROJECT_ROOT),
            ]
        )
        env = run_locany_ui5_local_debug.build_environment(args, base_env={})
        self.assertEqual(env["UI5_USE_DETECTION_CROPS"], "1")
        self.assertEqual(env["UI5_CROP_AUDIT_DIR"], str(audit.resolve()))
        self.assertEqual(env["UI5_CROP_TRAIN_MODE"], "full_plus_crop")
        self.assertEqual(env["GRADIENT_ACCUMULATION_STEPS"], "2")

    def test_submit_merlin_yaml_contains_crop_and_lossless_eval_parameters(self) -> None:
        args = submit_locany_ui5.parse_args(
            [
                "--machine", "a800", "--resource-group", "aiai_locate", "--gpus", "4",
                "--max-num-tokens", "12800", "--use-detection-crops",
                "--crop-audit-dir", "/mnt/audit/crop_audit_v4_gt_repair",
                "--crop-train-mode", "full_plus_crop",
                "--eval-inference-crop-mode", "lossless_tiling", "--render-only",
            ]
        )
        rendered, runtime = submit_locany_ui5.render_job(args)
        self.assertEqual(runtime["UI5_USE_DETECTION_CROPS"], 1)
        self.assertTrue(str(runtime["META_PATH"]).endswith("full_plus_crop.json"))
        self.assertEqual(runtime["EVAL_INFERENCE_CROP_MODE"], "lossless_tiling")
        for name in (
            "UI5_USE_DETECTION_CROPS", "UI5_CROP_AUDIT_DIR", "UI5_CROP_TRAIN_MODE",
            "UI5_CROP_META_PATH", "EVAL_INFERENCE_CROP_MODE",
        ):
            self.assertIn(name, rendered)

    def test_parallel_inference_forwards_lossless_tiling_without_gt_repair(self) -> None:
        args = Namespace(
            python="python", inference_script=Path("inference.py"), checkpoint=Path("checkpoint"),
            processor_path=Path("processor"), input_dir=Path("input"), output_dir=Path("output"),
            attn_implementation="sdpa", relation_gate_mode="observe",
            relation_gate_threshold=0.5, overwrite=False, max_images_per_task=1,
            inference_crop_mode="lossless_tiling", tile_max_count=10,
            tile_target_long_side=1600, tile_overlap_ratio=0.1, tile_nms_iou=0.5,
        )
        command = run_ui5_parallel_inference.build_command(
            args, "occlusion", "0", Path("summary.json")
        )
        rendered = " ".join(map(str, command))
        self.assertIn("--inference-crop-mode lossless_tiling", rendered)
        self.assertNotIn("manual_gt_repair", rendered)

    def test_train_shell_selects_crop_meta_before_environment_preflight_and_torchrun(self) -> None:
        script = (PROJECT_ROOT / "shell" / "train_locany_ui_defect.sh").read_text(
            encoding="utf-8"
        )
        marker_check = script.index("validate_ui5_crop_training_ready.py")
        final_meta = script.index('META_PATH="${UI5_CROP_META_PATH}"')
        preflight = script.index("check_locany_environment.py")
        torchrun = script.index("if torchrun")
        meta_argument = script.index('--meta_path "${META_PATH}"')
        self.assertLess(marker_check, final_meta)
        self.assertLess(final_meta, preflight)
        self.assertLess(preflight, torchrun)
        self.assertGreater(meta_argument, torchrun)

    def test_runtime_config_crop_recipe_is_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires UI5_CROP_AUDIT_DIR"):
            locany_ui5_common.resolve_runtime_config(
                {
                    "MACHINE_TYPE": "a800", "GPU_COUNT": "4",
                    "CUDA_DEVICES": "0,1,2,3", "UI5_USE_DETECTION_CROPS": "1",
                }
            )

if __name__ == "__main__":
    unittest.main()
