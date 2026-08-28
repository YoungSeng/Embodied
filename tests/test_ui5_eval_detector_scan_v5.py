from __future__ import annotations

import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from run_ui5_crop_audit import (  # noqa: E402
    detection_worker_command,
    preflight_icon_runtime,
    preflight_text_runtime,
)
from ui5_eval_detector_cache import validate_eval_detector_cache  # noqa: E402
from ui5_lossless_tiling import generate_detector_scan_plan  # noqa: E402


class HorizontalSeamV7RegressionTest(unittest.TestCase):
    def assert_hard_gates(self, plan: dict) -> None:
        self.assertEqual(plan["lossless_pixel_coverage_ratio"], 1.0)
        self.assertEqual(plan["detector_bbox_containment_rate"], 1.0)
        self.assertEqual(plan["uncontained_detector_bbox_count"], 0)
        self.assertEqual(plan["full_tile_in_multi_plan_count"], 0)
        self.assertEqual(plan["duplicate_tile_count"], 0)
        self.assertEqual(plan["nested_tile_count"], 0)
        self.assertTrue(plan["strict_vertical_partition"])
        self.assertEqual(plan["adjacent_overlap_pixels_total"], 0)
        self.assertEqual(plan["adjacent_gap_pixels_total"], 0)
        self.assertEqual(plan["duplicate_pixel_area"], 0)
        self.assertEqual(plan["processed_pixel_ratio"], 1.0)
        self.assertEqual(plan["seam_crossed_detector_bbox_count"], 0)
        self.assertEqual(plan["detector_boundary_cut_count"], 0)
        self.assertEqual(plan["balanced_fallback_seam_count"], 0)
        self.assertEqual(plan["non_edge_seam_count"], 0)
        self.assertEqual(plan["gap_interior_seam_count"], 0)
        self.assertEqual(plan["guarded_bbox_crossed_by_seam_count"], 0)
        self.assertEqual(plan["guarded_bbox_unique_containment_rate"], 1.0)
        self.assertTrue(plan["every_seam_is_guarded_detector_edge"])
        self.assertTrue(
            set(plan["horizontal_seams"]).issubset(plan["detector_edge_candidates"])
        )
        self.assertEqual(
            plan["detector_bbox_unique_containment_count"],
            plan["detector_box_count"],
        )
        self.assertTrue(
            all(left[3] == right[1] for left, right in zip(plan["tiles"], plan["tiles"][1:]))
        )

    def test_dense_y_chain_reduces_to_full_image_instead_of_cutting(self) -> None:
        boxes = [[20, 0, 160, 3000]]
        plan = generate_detector_scan_plan(1000, 3000, boxes)
        self.assert_hard_gates(plan)
        self.assertEqual(plan["tile_count"], 1)
        self.assertEqual(plan["tiles"], [[0, 0, 1000, 3000]])

    def test_wide_gap_is_preferred_and_no_gap_reduces_count(self) -> None:
        gap_plan = generate_detector_scan_plan(
            800, 2000, [[0, 100, 200, 800], [0, 1200, 200, 1900]], target_tile_height=1000
        )
        self.assertEqual(gap_plan["seam_source"], ["detector_edge"])
        self.assertIn(gap_plan["horizontal_seams"][0], gap_plan["detector_edge_candidates"])
        dense_plan = generate_detector_scan_plan(
            800,
            3000,
            [[20, 0, 300, 3000]],
        )
        self.assert_hard_gates(dense_plan)
        self.assertEqual(dense_plan["tile_count"], 1)
        self.assertEqual(dense_plan["seam_source"], [])

    def test_old_full_plus_lower_half_shapes_cannot_recur(self) -> None:
        for boxes in (
            [[10, 700, 100, 900], [900, 720, 1000, 880]],
            [[10, 770, 100, 850], [10, 1500, 100, 1650]],
            [[10, 10, 100, 500], [10, 800, 100, 1000], [10, 1700, 100, 1900]],
        ):
            plan = generate_detector_scan_plan(1125, 2436, boxes)
            self.assert_hard_gates(plan)
            self.assertNotIn([0, 0, 1125, 2436], plan["tiles"])
            self.assertEqual(plan["near_full_tile_count"], 0)

    def test_153784_shape_reduces_from_three_parts_to_two_safe_parts(self) -> None:
        plan = generate_detector_scan_plan(
            1000, 2160, [[0, 100, 100, 1000], [0, 1160, 100, 2060]]
        )
        self.assert_hard_gates(plan)
        self.assertEqual(plan["desired_tile_count"], 3)
        self.assertEqual(plan["actual_tile_count"], 2)
        self.assertIn(plan["horizontal_seams"][0], plan["detector_edge_candidates"])

    def test_153790_shape_uses_far_safe_second_seam(self) -> None:
        plan = generate_detector_scan_plan(
            1000,
            2880,
            [[0, 100, 100, 850], [0, 1050, 100, 2200], [0, 2400, 100, 2800]],
        )
        self.assert_hard_gates(plan)
        self.assertEqual(plan["actual_tile_count"], 3)
        self.assertIn(plan["horizontal_seams"][1], plan["detector_edge_candidates"])

    def test_153781_shape_uses_safe_seam_instead_of_balanced_fallback(self) -> None:
        plan = generate_detector_scan_plan(
            1000,
            2160,
            [[0, 100, 100, 620], [0, 820, 100, 1120], [0, 1300, 100, 2050]],
        )
        self.assert_hard_gates(plan)
        self.assertEqual(plan["actual_tile_count"], 3)
        self.assertIn(plan["horizontal_seams"][1], plan["detector_edge_candidates"])

    def test_named_v7_gap_midpoint_regressions_never_recur(self) -> None:
        cases = {
            "153793": (2160, 960),
            "153781": (2160, 720),
            "153787_first": (2436, 812),
            "153787_second": (2436, 1624),
            "153789": (2160, 960),
            "153796": (2460, 820),
            "153797": (2460, 820),
        }
        boxes = [
            {"bbox": [0, 100, 100, 600], "source": "text"},
            {"bbox": [0, 1200, 100, 1500], "source": "icon"},
            {"bbox": [0, 1900, 100, 2050], "source": "text"},
        ]
        for name, (height, forbidden) in cases.items():
            plan = generate_detector_scan_plan(1000, height, boxes)
            self.assert_hard_gates(plan)
            if forbidden not in plan["detector_edge_candidates"]:
                self.assertNotIn(forbidden, plan["horizontal_seams"], name)

    def test_random_realistic_boxes_remain_lossless_and_contained(self) -> None:
        rng = random.Random(20260828)
        for _ in range(100):
            width, height = rng.randint(300, 2200), rng.randint(1000, 8000)
            boxes = []
            for _ in range(rng.randint(0, 100)):
                x1, y1 = rng.randrange(width - 1), rng.randrange(height - 1)
                boxes.append(
                    [x1, y1, min(width, x1 + rng.randint(1, 300)), min(height, y1 + rng.randint(1, 240))]
                )
            self.assert_hard_gates(generate_detector_scan_plan(width, height, boxes))


class DualEnvironmentAndReadonlyCacheTest(unittest.TestCase):
    def _worker_args(self) -> SimpleNamespace:
        return SimpleNamespace(
            text_python="/env/UI5PaddleOCR/bin/python",
            icon_python="/env/LocateAnything/bin/python",
            detector_worker_script=Path("worker.py"), parser_root=Path("parser"),
            output_dir=Path("output"), gpus="0,1", workers_per_gpu=1,
            image_loader_threads=4, shard_size=750, max_unique_images=0,
            progress_interval_seconds=10, progress_every_images=25,
            text_long_side=1920, text_box_threshold=0.3, icon_long_side=1920,
            icon_confidence=0.05, source_dir=None, locany_data_dir=None,
            text_model_dir=None, icon_model=Path("model.pt"), resume=True,
            enable_mkldnn=False,
        )

    def test_worker_commands_use_their_explicit_python(self) -> None:
        args = self._worker_args()
        self.assertEqual(detection_worker_command(args, "text", 0, 4)[0], args.text_python)
        self.assertEqual(detection_worker_command(args, "icon", 0, 4)[0], args.icon_python)

    def test_text_and_icon_preflight_fail_before_worker_launch(self) -> None:
        text_failure = subprocess.CompletedProcess([], 1, stdout="", stderr="No module named paddle")
        with mock.patch("run_ui5_crop_audit.subprocess.run", return_value=text_failure):
            with self.assertRaisesRegex(RuntimeError, "尚未启动 GPU worker"):
                preflight_text_runtime("text-python", gpu="0", parser_root=Path("parser"), model_dir=None)
        icon_failure = subprocess.CompletedProcess([], 1, stdout="", stderr="No module named torch")
        with mock.patch("run_ui5_crop_audit.subprocess.run", return_value=icon_failure):
            with self.assertRaisesRegex(RuntimeError, "尚未启动 GPU worker"):
                preflight_icon_runtime("icon-python", gpu="0", model_path=Path("model.pt"))

    def test_readonly_cache_missing_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError):
                validate_eval_detector_cache(
                    Path(temporary), scan_name="horizontal_scan_v4_detector_edge_aligned"
                )

    def test_readonly_eval_code_guards_detector_build(self) -> None:
        source = (SCRIPTS / "run_ui5_eval.py").read_text(encoding="utf-8")
        guard = source.index('if args.eval_detector_cache_mode == "build"')
        stage_all = source.index('"--stage", "all"', guard)
        readonly_log = source.index("detector cache: readonly validated", stage_all)
        self.assertLess(guard, stage_all)
        self.assertLess(stage_all, readonly_log)


if __name__ == "__main__":
    unittest.main()
