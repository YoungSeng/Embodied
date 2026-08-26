from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from render_ui5_crop_failures import (  # noqa: E402
    ROOT_CAUSES,
    render_failure_visualizations,
    validate_expected_failures,
)
from run_ui5_crop_audit import (  # noqa: E402
    FINAL_TRAINING_GATE_CONDITIONS,
    atomic_write_jsonl,
    build_final_training_gate,
)


class FailureVisualizationTest(unittest.TestCase):
    def make_audit(self, root: Path) -> tuple[Path, Path]:
        output = root / "output"
        audit = output / "crop_audit_v3"
        source = root / "source.png"
        Image.new("RGB", (120, 80), "white").save(source)
        unique = {
            "image_id": "img",
            "content_id": "content",
            "image_path": str(source),
            "width": 120,
            "height": 80,
        }
        sample = {
            "sample_id": "sample",
            "image_id": "img",
            "task": "ui_occlusion",
            "gt_boxes": [[40, 20, 80, 60], [90, 10, 110, 30]],
        }
        detection = {
            "image_id": "img",
            "width": 120,
            "height": 80,
            "text_detections": [{"bbox": [5, 5, 55, 25], "score": 0.9123}],
            "icon_detections": [{"bbox": [15, 35, 45, 65], "confidence": 0.8234}],
        }
        geometry = {
            "image_id": "img",
            "config": "X",
            "sample_results": [
                {
                    "sample_id": "sample",
                    "task": "ui_occlusion",
                    "crop_boxes": [[0, 0, 60, 80]],
                    "detail": {"detection_density": "sparse"},
                }
            ],
        }
        failures = [
            {
                "config": "X",
                "sample_id": "sample",
                "image_id": "img",
                "task": "ui_occlusion",
                "gt_index": 0,
                "gt_bbox": [40, 20, 80, 60],
                "intersecting_crop_ids": [1],
                "intersecting_crop_bboxes": [[0, 0, 60, 80]],
                "failure_type": "partial_intersection",
                "detection_density": "sparse",
                "required_compensation_px": {
                    "left": 0,
                    "top": 0,
                    "right": 20,
                    "bottom": 0,
                },
                "required_max_single_side_px": 20,
                "required_total_px": 20,
                "compensation_bucket": "medium_17_64px",
                "visualization": "",
            },
            {
                "config": "X",
                "sample_id": "sample",
                "image_id": "img",
                "task": "ui_occlusion",
                "gt_index": 1,
                "gt_bbox": [90, 10, 110, 30],
                "intersecting_crop_ids": [],
                "intersecting_crop_bboxes": [],
                "failure_type": "uncovered",
                "detection_density": "sparse",
                "required_compensation_px": None,
                "required_max_single_side_px": None,
                "required_total_px": None,
                "compensation_bucket": "not_applicable",
                "visualization": "",
            },
        ]
        atomic_write_jsonl(output / "manifest" / "unique_images.jsonl", [unique])
        atomic_write_jsonl(output / "manifest" / "task_samples.jsonl", [sample])
        atomic_write_jsonl(
            output / "detections" / "merged" / "detections.jsonl", [detection]
        )
        atomic_write_jsonl(
            audit / "candidate_X" / "geometry" / "shard_00000.jsonl", [geometry]
        )
        atomic_write_jsonl(audit / "gt_failures.jsonl", failures)
        return output, audit

    def test_renders_all_failures_and_preserves_original_jsonl(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, audit = self.make_audit(Path(temporary))
            original = (audit / "gt_failures.jsonl").read_bytes()
            result = render_failure_visualizations(
                output_dir=output,
                crop_audit_name="crop_audit_v3",
                config_name="X",
                expected_failures=2,
                expected_partial=1,
                expected_uncovered=1,
                resume=True,
            )
            self.assertEqual(result["rendered"], 2)
            self.assertEqual(result["uncovered"], 1)
            visualized = [
                json.loads(line)
                for line in (audit / "gt_failures_visualized.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(visualized), 2)
            self.assertTrue(all(row["text_detection_count"] == 1 for row in visualized))
            self.assertTrue(all(row["icon_detection_count"] == 1 for row in visualized))
            self.assertTrue(all(row["crop_count_for_task"] == 1 for row in visualized))
            self.assertTrue(all(row["manual_root_cause"] == "" for row in visualized))
            for row in visualized:
                path = Path(row["visualization_4panel"])
                self.assertTrue(path.is_file())
                with Image.open(path) as image:
                    self.assertEqual(image.format, "PNG")
                    self.assertGreater(image.width, 4 * 120)
                    self.assertGreater(image.height, 80)
            for filename in (
                "index.html",
                "uncovered_all.html",
                "representative_partial.html",
                "diagnosis_summary.html",
            ):
                self.assertTrue(
                    (audit / "failure_visualizations" / "gallery" / filename).is_file()
                )
            self.assertEqual((audit / "gt_failures.jsonl").read_bytes(), original)

    def test_resume_reuses_panels_and_preserves_manual_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, audit = self.make_audit(Path(temporary))
            first = render_failure_visualizations(
                output_dir=output,
                crop_audit_name="crop_audit_v3",
                config_name="X",
                expected_failures=2,
                expected_partial=1,
                expected_uncovered=1,
                resume=True,
            )
            rows = [
                json.loads(line)
                for line in Path(first["visualized_jsonl"])
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            rows[0]["manual_root_cause"] = ROOT_CAUSES[0]
            rows[0]["manual_note"] = "reviewed"
            atomic_write_jsonl(Path(first["visualized_jsonl"]), rows)
            second = render_failure_visualizations(
                output_dir=output,
                crop_audit_name="crop_audit_v3",
                config_name="X",
                expected_failures=2,
                expected_partial=1,
                expected_uncovered=1,
                resume=True,
            )
            self.assertEqual(second["rendered"], 0)
            self.assertEqual(second["reused"], 2)
            updated = [
                json.loads(line)
                for line in Path(second["visualized_jsonl"])
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(updated[0]["manual_root_cause"], ROOT_CAUSES[0])
            self.assertEqual(updated[0]["manual_note"], "reviewed")

    def test_distribution_validation_fails_closed(self):
        rows = [
            {
                "config": "X",
                "task": "ui_occlusion",
                "failure_type": "uncovered",
                "detection_density": "sparse",
            }
        ]
        with self.assertRaisesRegex(ValueError, "distribution by task differs"):
            validate_expected_failures(
                rows,
                config_name="X",
                expected_failures=1,
                expected_partial=0,
                expected_uncovered=1,
                expected_by_task={
                    "ui_occlusion": {
                        "partial_intersection": 1,
                        "uncovered": 0,
                    }
                },
            )

    def test_other_manual_cause_requires_note(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, _ = self.make_audit(Path(temporary))
            result = render_failure_visualizations(
                output_dir=output,
                crop_audit_name="crop_audit_v3",
                config_name="X",
                expected_failures=2,
                expected_partial=1,
                expected_uncovered=1,
            )
            visualized = Path(result["visualized_jsonl"])
            rows = [json.loads(line) for line in visualized.read_text().splitlines()]
            for row in rows:
                row["manual_root_cause"] = "context_too_small"
            rows[0]["manual_root_cause"] = "other"
            rows[0]["manual_note"] = ""
            atomic_write_jsonl(visualized, rows)
            with self.assertRaisesRegex(ValueError, "other.*manual_note"):
                render_failure_visualizations(
                    output_dir=output,
                    crop_audit_name="crop_audit_v3",
                    config_name="X",
                    expected_failures=2,
                    expected_partial=1,
                    expected_uncovered=1,
                    summary_only=True,
                    require_manual_review=True,
                )


class FinalGateTest(unittest.TestCase):
    def candidate_gate(self) -> dict:
        return {
            "config": "X",
            "conditions": {
                "region_overall_recall_at_least_0_99": True,
                "each_region_task_recall_at_least_0_98": True,
                "detector_boundary_cut_count_zero": True,
                "region_roundtrip_error_over_1_count_zero": True,
                "partial_crop_training_eligible_count_zero": True,
                "hard_negative_max_one_per_image_task": True,
            },
            "passes": True,
        }

    def build(self, **overrides) -> dict:
        values = {
            "same_content_cross_train_val_count": 0,
            "content_missing_recall": 1.0,
            "content_missing_normalized_gt_mismatch_count": 0,
            "input_snapshot_unchanged": True,
            "all_reports_written_successfully": True,
        }
        values.update(overrides)
        return build_final_training_gate(self.candidate_gate(), **values)

    def test_final_gate_contains_all_eleven_conditions(self):
        gate = self.build()
        self.assertTrue(gate["passes"])
        self.assertEqual(len(gate["conditions"]), 11)
        self.assertEqual(set(gate["conditions"]), FINAL_TRAINING_GATE_CONDITIONS)
        self.assertEqual(gate["failed_conditions"], [])

    def test_leakage_content_and_snapshot_failures_are_explicit(self):
        cases = (
            (
                {"same_content_cross_train_val_count": 1},
                "same_content_cross_train_val_count_zero",
            ),
            ({"content_missing_recall": 0.99}, "content_missing_recall_equals_1"),
            (
                {"content_missing_normalized_gt_mismatch_count": 1},
                "content_missing_normalized_gt_mismatch_count_zero",
            ),
            ({"input_snapshot_unchanged": False}, "input_snapshot_unchanged"),
            (
                {"all_reports_written_successfully": False},
                "all_reports_written_successfully",
            ),
        )
        for override, failed_name in cases:
            with self.subTest(failed_name=failed_name):
                gate = self.build(**override)
                self.assertFalse(gate["passes"])
                self.assertFalse(gate["training_ready"])
                self.assertIn(failed_name, gate["failed_conditions"])


if __name__ == "__main__":
    unittest.main()
