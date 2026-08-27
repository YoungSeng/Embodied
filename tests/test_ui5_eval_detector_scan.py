from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from ui5_lossless_tiling import (  # noqa: E402
    assert_lossless_coverage,
    detector_boundary_cut_count,
    generate_detector_scan_plan,
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


parallel = load_module(
    "ui5_parallel_detector_scan_test", SCRIPTS / "run_ui5_parallel_inference.py"
)
common = load_module("ui5_common_detector_scan_test", SCRIPTS / "locany_ui5_common.py")
preparer = load_module(
    "ui5_eval_detector_preparer_test",
    SCRIPTS / "prepare_ui5_eval_detector_crops.py",
)


class DetectorScanGeometryTest(unittest.TestCase):
    def test_left_and_right_detections_protect_complete_horizontal_strip(self) -> None:
        boxes = [[20, 820, 180, 1060], [1010, 830, 1180, 1040]]
        plan = generate_detector_scan_plan(
            1200,
            3000,
            boxes,
            target_tile_height=1000,
            overlap_ratio=0.10,
        )
        self.assertEqual(plan["lossless_pixel_coverage_ratio"], 1.0)
        # A seam may cross a detector on a dense row; containment, not the
        # historical boundary-cut diagnostic, is the correctness gate.
        self.assertEqual(plan["detector_bbox_containment_rate"], 1.0)
        self.assertEqual(plan["uncontained_detector_bbox_count"], 0)
        self.assertTrue(all(tile[0] == 0 and tile[2] == 1200 for tile in plan["tiles"]))
        # Any pixel between left/right neighbours is included because every
        # scan spans the complete source width.
        self.assertTrue(
            any(tile[1] <= 900 and tile[3] >= 1000 for tile in plan["tiles"])
        )
        self.assertEqual(plan["connected_band_count"], 1)

    def test_random_page_sizes_are_lossless_and_never_cut_detector_boxes(self) -> None:
        randomizer = random.Random(20260827)
        for _ in range(100):
            width = randomizer.randint(240, 2400)
            height = randomizer.randint(240, 8000)
            boxes = []
            for _box in range(randomizer.randint(0, 80)):
                x1 = randomizer.randrange(0, width - 1)
                y1 = randomizer.randrange(0, height - 1)
                x2 = randomizer.randrange(x1 + 1, width + 1)
                y2 = randomizer.randrange(y1 + 1, height + 1)
                boxes.append([x1, y1, x2, y2])
            plan = generate_detector_scan_plan(width, height, boxes)
            self.assertLessEqual(len(plan["tiles"]), 10)
            assert_lossless_coverage(width, height, plan["tiles"])
            self.assertEqual(plan["detector_bbox_containment_rate"], 1.0)
            self.assertEqual(plan["uncontained_detector_bbox_count"], 0)

    def test_dense_connected_page_is_balanced_instead_of_full_view(self) -> None:
        boxes = [[10, 50, 100, 950], [200, 900, 300, 1950], [400, 1900, 500, 2950]]
        plan = generate_detector_scan_plan(1000, 3000, boxes)
        self.assertGreater(len(plan["tiles"]), 1)
        self.assertNotIn([0, 0, 1000, 3000], plan["tiles"])
        self.assertEqual(plan["detector_bbox_containment_rate"], 1.0)
        self.assertEqual(plan["nested_tile_count"], 0)

    def test_content_missing_keeps_global_view_without_gt(self) -> None:
        plan = generate_detector_scan_plan(
            1000, 5000, [[10, 10, 20, 20]], task="ui_content_missing"
        )
        self.assertEqual(plan["tiles"], [[0, 0, 1000, 5000]])
        self.assertFalse(plan["gt_used"])

    def test_detector_empty_long_page_still_uses_balanced_scans(self) -> None:
        plan = generate_detector_scan_plan(1000, 5000, [])
        self.assertGreater(len(plan["tiles"]), 1)
        self.assertEqual(plan["lossless_pixel_coverage_ratio"], 1.0)


class DetectorScanPipelineTest(unittest.TestCase):
    def test_eval_pipeline_defaults_to_readonly_cache_before_locany_workers(self) -> None:
        source = (SCRIPTS / "run_ui5_eval.py").read_text(encoding="utf-8")
        prepare_position = source.index('if args.eval_detector_cache_mode == "build"')
        inference_position = source.index("run_ui5_parallel_inference.py")
        self.assertLess(prepare_position, inference_position)
        self.assertIn("--detector-crop-manifest", source)
        self.assertIn('"--stage", "all"', source)
        self.assertIn("detector cache: readonly validated", source)
        self.assertNotIn("manual_gt_repair", source)

    def test_crop_stage_writes_visualization_gallery_and_statistics_without_gt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "page.png"
            Image.new("RGB", (600, 2400), "white").save(image_path)
            manifest = root / "manifest"
            merged = root / "detections" / "merged"
            manifest.mkdir(parents=True)
            merged.mkdir(parents=True)
            unique_row = {
                "image_id": "eval_1",
                "content_id": "content_1",
                "image_path": str(image_path),
                "image_paths": [str(image_path)],
                "width": 600,
                "height": 2400,
                "tasks": ["occlusion"],
            }
            (manifest / "unique_images.jsonl").write_text(
                json.dumps(unique_row) + "\n", encoding="utf-8"
            )
            (manifest / "task_samples.jsonl").write_text(
                json.dumps(
                    {
                        "task": "occlusion",
                        "task_index": 0,
                        "image_id": "eval_1",
                        "content_id": "content_1",
                        "image_path": str(image_path),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            detected = {
                "image_id": "eval_1",
                "content_id": "content_1",
                "image": str(image_path),
                "width": 600,
                "height": 2400,
                "text_detections": [{"bbox": [20, 900, 100, 1020], "score": 0.9}],
                "icon_detections": [{"bbox": [500, 910, 580, 1030], "score": 0.8}],
            }
            (merged / "detections.jsonl").write_text(
                json.dumps(detected) + "\n", encoding="utf-8"
            )
            task_files = {}
            for task_name in common.TASKS:
                task_file = root / common.TASK_JSONL[task_name]
                task_file.write_text(json.dumps({"images": [str(image_path)]}) + "\n", encoding="utf-8")
                task_files[task_name] = str(task_file)
            (manifest / "selection_config.json").write_text(
                json.dumps({"task_files": task_files}), encoding="utf-8"
            )
            (root / "detections" / "detector_config.json").write_text(
                json.dumps(
                    {
                        "parser_commit": "06eaebf8eb4ea01e61b690f2ff972bf614915918",
                        "text": {"model_dir": None},
                        "icon": {"model": "model.pt"},
                    }
                ),
                encoding="utf-8",
            )
            for stage_name in ("text", "icon"):
                stage_dir = root / "detections" / stage_name
                stage_dir.mkdir(parents=True)
                (stage_dir / "stage_summary.json").write_text(
                    json.dumps({"images": 1, "workers": 1, "runtime": {"python": stage_name}}),
                    encoding="utf-8",
                )
                (stage_dir / "shard_00000.jsonl").write_text(
                    json.dumps({"image_id": "eval_1"}) + "\n", encoding="utf-8"
                )
                (stage_dir / "shard_00000.done.json").write_text(
                    json.dumps({"stage": stage_name, "count": 1}), encoding="utf-8"
                )
            args = argparse.Namespace(
                output_dir=root,
                scan_name="horizontal_scan_v2",
                scan_max_crops=10,
                scan_target_height=960,
                scan_overlap_ratio=0.12,
                scan_vertical_link_ratio=0.015,
                scan_context_ratio=0.10,
                scan_min_context_image_ratio=0.01,
                scan_dense_band_ratio=0.80,
                scan_detector_margin_ratio=0.003,
                scan_seam_search_ratio=0.25,
                scan_context_pixels=48,
                scan_minimum_core_height_ratio=0.35,
                visualization_samples=1,
                save_preview_crops=True,
                resume=True,
                progress_interval_seconds=10.0,
            )
            rows = preparer.build_scan_crops(args)
            self.assertEqual(len(rows), 1)
            self.assertFalse(rows[0]["gt_used"])
            self.assertEqual(rows[0]["detector_boundary_cut_count"], 0)
            self.assertTrue((root / "horizontal_scan_v2" / "summary.json").is_file())
            self.assertTrue((root / "horizontal_scan_v2" / "statistics.csv").is_file())
            self.assertTrue((root / "horizontal_scan_v2" / "gallery" / "index.html").is_file())
            self.assertTrue((root / "horizontal_scan_v2" / "eval_detector_cache_ready.json").is_file())
            self.assertTrue(list((root / "horizontal_scan_v2" / "preview_crops").glob("*.png")))
            original_draw = preparer._draw_preview
            try:
                preparer._draw_preview = lambda *unused, **unused_kwargs: self.fail(
                    "valid --resume must not re-render previews"
                )
                cached = preparer.build_scan_crops(args)
            finally:
                preparer._draw_preview = original_draw
            self.assertEqual(cached, rows)

    def test_parallel_worker_receives_detector_manifest(self) -> None:
        args = argparse.Namespace(
            python="python",
            inference_script=Path("inference.py"),
            checkpoint=Path("checkpoint"),
            processor_path=Path("processor"),
            input_dir=Path("input"),
            output_dir=Path("output"),
            attn_implementation="sdpa",
            relation_gate_mode="observe",
            inference_crop_mode="detector_scan",
            tile_max_count=10,
            tile_target_long_side=1600,
            tile_overlap_ratio=0.12,
            tile_nms_iou=0.5,
            detector_crop_manifest=Path("detector_scan_crops.jsonl"),
            relation_gate_threshold=None,
            overwrite=False,
            max_images_per_task=0,
        )
        command = parallel.build_command(
            args, "occlusion", "0", Path("summary.json")
        )
        self.assertIn("detector_scan", command)
        self.assertIn("--detector-crop-manifest", command)
        self.assertIn("detector_scan_crops.jsonl", command)

    def test_runtime_defaults_to_detector_scan_and_exposes_detector_settings(self) -> None:
        config = common.resolve_runtime_config(
            {"MACHINE_TYPE": "a800", "GPU_COUNT": "4"}
        )
        self.assertEqual(config["EVAL_INFERENCE_CROP_MODE"], "detector_scan")
        self.assertTrue(config["PROJECT_ROOT"].endswith("Embodied-ui5-det-crop"))
        self.assertTrue(config["EVAL_PARSER_ROOT"].endswith("ui-region-parser"))
        self.assertTrue(config["EVAL_ICON_MODEL"].endswith("icon_detect_v3/model.pt"))
        self.assertEqual(config["EVAL_DETECTOR_WORKERS_PER_GPU"], 1)
        self.assertEqual(config["EVAL_DETECTOR_CACHE_MODE"], "readonly")
        self.assertEqual(config["EVAL_SCAN_NAME"], "horizontal_scan_v2")


if __name__ == "__main__":
    unittest.main()
