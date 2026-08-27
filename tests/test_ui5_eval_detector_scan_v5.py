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


class HorizontalSeamV5RegressionTest(unittest.TestCase):
    def assert_hard_gates(self, plan: dict) -> None:
        self.assertEqual(plan["lossless_pixel_coverage_ratio"], 1.0)
        self.assertEqual(plan["detector_bbox_containment_rate"], 1.0)
        self.assertEqual(plan["uncontained_detector_bbox_count"], 0)
        self.assertEqual(plan["full_tile_in_multi_plan_count"], 0)
        self.assertEqual(plan["duplicate_tile_count"], 0)
        self.assertEqual(plan["nested_tile_count"], 0)

    def test_dense_y_chain_does_not_trigger_full_image_fallback(self) -> None:
        boxes = [[20 + index % 3 * 250, index * 75, 160 + index % 3 * 250, index * 75 + 115] for index in range(39)]
        plan = generate_detector_scan_plan(1000, 3000, boxes)
        self.assert_hard_gates(plan)
        self.assertGreater(plan["tile_count"], 1)
        self.assertNotIn([0, 0, 1000, 3000], plan["tiles"])
        self.assertIn("balanced_fallback", plan["seam_source"])

    def test_wide_gap_is_preferred_and_no_gap_uses_balanced_overlap(self) -> None:
        gap_plan = generate_detector_scan_plan(
            800, 2000, [[0, 100, 200, 800], [0, 1200, 200, 1900]], target_tile_height=1000
        )
        self.assertEqual(gap_plan["seam_source"], ["detector_gap"])
        self.assertTrue(800 <= gap_plan["horizontal_seams"][0] <= 1200)
        dense_plan = generate_detector_scan_plan(
            800,
            3000,
            [
                [20 + index % 2 * 350, index * 100, 300 + index % 2 * 350, index * 100 + 180]
                for index in range(29)
            ],
        )
        self.assert_hard_gates(dense_plan)
        self.assertIn("balanced_fallback", dense_plan["seam_source"])
        self.assertGreater(dense_plan["seam_crossed_detector_bbox_count"], 0)

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
                validate_eval_detector_cache(Path(temporary), scan_name="horizontal_scan_v2")

    def test_readonly_eval_code_guards_detector_build(self) -> None:
        source = (SCRIPTS / "run_ui5_eval.py").read_text(encoding="utf-8")
        guard = source.index('if args.eval_detector_cache_mode == "build"')
        stage_all = source.index('"--stage", "all"', guard)
        readonly_log = source.index("detector cache: readonly validated", stage_all)
        self.assertLess(guard, stage_all)
        self.assertLess(stage_all, readonly_log)


if __name__ == "__main__":
    unittest.main()
