from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import analyze_ui5_tiled_evaluation as analyzer  # noqa: E402
from locany_ui5_common import TASK_JSONL, TASKS  # noqa: E402


class TiledEvaluationDiagnosticsTest(unittest.TestCase):
    def _fixture(self, root: Path, *, with_raw: bool) -> Namespace:
        predictions = root / "predictions"
        gt = root / "gt"
        scorer = root / "scorer"
        output = root / "output"
        predictions.mkdir()
        gt.mkdir()
        scorer.mkdir()
        image = root / "page.png"
        image.write_bytes(b"fixture")
        (scorer / "qwen3vl_merge_and_score_fixed_5tasks.py").write_text(
            "def get_gt_payload(sample):\n"
            "    return sample\n\n"
            "def extract_bboxes_for_issue(payload, issue):\n"
            "    return payload.get('gt_boxes', [])\n",
            encoding="utf-8",
        )
        for task in TASKS:
            (gt / TASK_JSONL[task]).write_text(
                json.dumps({"images": str(image), "gt_boxes": []}) + "\n",
                encoding="utf-8",
            )
            if not with_raw:
                continue
            raw_dir = predictions / task / "raw"
            raw_dir.mkdir(parents=True)
            raw = {
                "image_path": str(image),
                "image_size": {"width": 100, "height": 200},
                "parse": {"pixel_boxes_xyxy": [[10, 10, 20, 20]]},
                "inference_crop": {
                    "mode": "detector_scan",
                    "tiles": [
                        {
                            "tile_index": 0,
                            "tile_bbox": [0, 0, 100, 100],
                            "local_pixel_boxes": [[10, 10, 20, 20]],
                        },
                        {
                            "tile_index": 1,
                            "tile_bbox": [0, 100, 100, 200],
                            "local_pixel_boxes": [],
                        },
                    ],
                },
            }
            (raw_dir / "page.json").write_text(
                json.dumps(raw) + "\n", encoding="utf-8"
            )
        return Namespace(
            prediction_dir=predictions,
            gt_dir=gt,
            scorer_root=scorer,
            output_dir=output,
            iou_threshold=0.1,
        )

    def test_missing_raw_sidecars_is_explicit_failure_not_zero_amplification(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self._fixture(Path(temporary), with_raw=False)
            summary = analyzer.build(args)
            self.assertEqual(summary["status"], "missing_raw_sidecars")
            self.assertEqual(summary["raw_sidecar_record_count"], 0)
            for task in TASKS:
                task_summary = summary["tasks"][task]
                self.assertEqual(task_summary["status"], "missing_raw_sidecars")
                self.assertIsNone(task_summary["source_image_fp_amplification"])
                self.assertEqual(
                    task_summary["source_image_fp_amplification_status"],
                    "missing_raw_sidecars",
                )

    def test_raw_sidecars_report_tile_nms_and_fp_amplification(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self._fixture(Path(temporary), with_raw=True)
            summary = analyzer.build(args)
            self.assertEqual(summary["status"], "ok")
            self.assertEqual(summary["raw_sidecar_record_count"], 5)
            for task in TASKS:
                task_summary = summary["tasks"][task]
                self.assertEqual(task_summary["tile_count_distribution"], {"2": 1})
                self.assertEqual(task_summary["boxes_before_global_nms"], 1)
                self.assertEqual(task_summary["boxes_after_global_nms"], 1)
                self.assertEqual(
                    task_summary["false_positive_tiles_on_negative_images"], 1
                )
                self.assertEqual(task_summary["source_image_false_positive_count"], 1)
                self.assertEqual(task_summary["source_image_fp_amplification"], 1.0)
            self.assertTrue(
                (args.output_dir / "text_ellipsis_fp_gallery.html").is_file()
            )


if __name__ == "__main__":
    unittest.main()
