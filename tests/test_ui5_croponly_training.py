import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from PIL import Image

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_ui5_croponly_training_manifest as croponly  # noqa: E402
from run_ui5_crop_audit import read_jsonl  # noqa: E402


TASKS = (
    "ui_occlusion",
    "ui_cropping",
    "ui_text_overflow",
    "ui_text_ellipsis",
    "ui_content_missing",
)


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class CropOnlyTrainingManifestTests(unittest.TestCase):
    def _fixture(self, root: Path) -> Namespace:
        output = root / "ui5_crop_audit"
        audit = output / "crop_audit_v4_gt_repair"
        audit.mkdir(parents=True)
        image_path = root / "source.png"
        Image.new("RGB", (100, 1200), color="white").save(image_path)
        samples = []
        for task in TASKS:
            gt_boxes = [[5, 390, 25, 410]] if task == "ui_occlusion" else []
            samples.append(
                {
                    "sample_id": f"sample_{task}",
                    "image_id": "image_0001",
                    "task": task,
                    "split": "train",
                    "canonical_path": str(image_path.resolve()),
                    "width": 100,
                    "height": 1200,
                    "gt_boxes": gt_boxes,
                    "gt_boxes_1000": [[50, 325, 250, 342]] if gt_boxes else [],
                }
            )
        write_jsonl(output / "manifest" / "task_samples.jsonl", samples)
        write_jsonl(
            output / "detections" / "merged" / "detections.jsonl",
            [
                {
                    "image_id": "image_0001",
                    "width": 100,
                    "height": 1200,
                    "text_detections": [
                        {"bbox": [5, 300, 80, 400], "score": 0.9}
                    ],
                    "icon_detections": [],
                }
            ],
        )
        write_jsonl(
            audit / "gt_repair_actions.jsonl",
            [
                {
                    "sample_id": "sample_ui_occlusion",
                    "task": "ui_occlusion",
                    "gt_index": 0,
                }
            ],
        )
        return Namespace(
            audit_dir=audit,
            output_name="crop_only_horizontal_v5_train_repair",
            detections=None,
            max_crops=10,
            target_height=960,
            progress_interval_seconds=999.0,
            resume=False,
        )

    def test_all_negative_strips_and_train_only_seam_repair_are_retained(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self._fixture(Path(temporary))
            summary = croponly.build(args)
            manifest = read_jsonl(
                args.audit_dir / args.output_name / "task_aware_manifest.jsonl"
            )
            cropping = next(row for row in manifest if row["task"] == "ui_cropping")
            self.assertGreaterEqual(len(cropping["training_records"]), 2)
            self.assertTrue(
                all(not record["_ui5_positive"] for record in cropping["training_records"])
            )
            occlusion = next(row for row in manifest if row["task"] == "ui_occlusion")
            self.assertEqual(occlusion["removed_gt_crossing_seams"], [400])
            self.assertEqual(occlusion["final_tiles"], [[0, 0, 100, 1200]])
            self.assertEqual(
                occlusion["training_records"][0]["_ui5_crop_source"],
                "manual_gt_repair",
            )
            self.assertEqual(summary["partial_negative_count"], 0)
            self.assertTrue(summary["all_legal_strips_retained"])
            self.assertTrue(summary["all_repair_gt_mapped"])

    def test_resume_reuses_matching_manifest_without_detector_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self._fixture(Path(temporary))
            first = croponly.build(args)
            args.resume = True
            second = croponly.build(args)
            self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
