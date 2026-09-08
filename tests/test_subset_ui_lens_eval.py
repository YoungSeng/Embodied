from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from locany_ui5_common import TASK_JSONL
from subset_ui_lens_eval import subset


class SubsetUILensTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.output = self.root / "douyin_subset"
        self.rows = []
        for index, (app, boxes) in enumerate([
            ("DouYin", [[1.5, 2, 10, 20]]), ([" 抖音 "], []),
            ("douyin_lite", []), ("wechat", [[3, 4, 5, 6]]),
        ]):
            # Parent directory deliberately contains douyin: filenames must not
            # participate in app selection. Extraction does not decode images.
            image = self.root / "douyin_parent" / f"{index}.png"
            image.parent.mkdir(exist_ok=True)
            image.write_bytes(b"not decoded by the subset filter")
            self.rows.append({
                "id": index, "images": [str(image)],
                "answer": {"bbox": boxes, "types": ["issue"] * len(boxes)},
                "extra_info": {"original_infos": {"app_name": app, "box_list": boxes},
                               "bbox_conversion": {"boundary_policy": "clip", "clipped_boxes": []}},
            })
        for task in TASK_JSONL:
            self.write(task, self.rows)

    def write(self, task, rows):
        (self.data / TASK_JSONL[task]).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    def run_subset(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return subset(self.data, self.output, **kwargs)

    def test_preserves_positive_negative_and_original_records_with_unequal_tasks(self):
        self.write("content_missing", self.rows[1:])
        report = self.run_subset()
        self.assertEqual(report["unique_images_by_path"], 2)
        self.assertEqual(report["image_task_records"], 9)
        for task, filename in TASK_JSONL.items():
            expected = self.rows[1:2] if task == "content_missing" else self.rows[:2]
            saved = (self.output / filename).read_text(encoding="utf-8")
            self.assertEqual(saved, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in expected))
            self.assertEqual(report["tasks"][task]["positive"], 0 if task == "content_missing" else 1)
            self.assertEqual(report["tasks"][task]["negative"], 1)
        self.assertTrue((self.output / "subset_summary.json").is_file())
        self.assertEqual(len((self.output / "selection_manifest.jsonl").read_text().splitlines()), 9)

    def test_app_inventory_and_explicit_alias(self):
        with contextlib.redirect_stdout(io.StringIO()):
            report = subset(self.data, list_apps=True)
        self.assertEqual(report["available_app_names"], {"douyin": 5, "douyin_lite": 5, "wechat": 5, "抖音": 5})
        self.assertFalse(self.output.exists())
        report = self.run_subset(apps=("douyin_lite",))
        self.assertEqual(report["image_task_records"], 5)
        self.assertTrue(all(s["positive"] == 0 for s in report["tasks"].values()))

    def test_missing_app_metadata_fails_before_writing(self):
        del self.rows[2]["extra_info"]["original_infos"]["app_name"]
        self.write("occlusion", self.rows)
        with self.assertRaisesRegex(ValueError, "lack valid app metadata"):
            self.run_subset()
        self.assertFalse(self.output.exists())

    def test_no_matches_fails_before_writing(self):
        with self.assertRaisesRegex(ValueError, "No matches"):
            self.run_subset(apps=("unknown_app",))
        self.assertFalse(self.output.exists())

    def test_empty_task_stays_empty_and_is_reported(self):
        self.write("content_missing", self.rows[2:])
        report = self.run_subset()
        self.assertEqual(report["empty_tasks"], ["content_missing"])
        self.assertEqual((self.output / TASK_JSONL["content_missing"]).read_text(), "")

    def test_rejects_relative_paths_and_missing_selected_images(self):
        self.rows[0]["images"] = ["relative.png"]
        self.write("occlusion", self.rows)
        with self.assertRaisesRegex(ValueError, "must be absolute"):
            self.run_subset()
        self.rows[0]["images"] = [str(self.root / "missing.png")]
        self.write("occlusion", self.rows)
        with self.assertRaisesRegex(ValueError, "selected image is missing"):
            self.run_subset()
        self.assertFalse(self.output.exists())

    def test_never_overwrites_an_existing_directory(self):
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_text("existing result")
        with self.assertRaises(FileExistsError):
            self.run_subset()
        self.assertEqual(marker.read_text(), "existing result")

    def test_real_conversion_subset_and_gt_inspection_round_trip(self):
        from PIL import Image
        from prepare_ui_lens_eval import TASKS, prepare
        from inspect_ui_lens_eval import read_converted

        raw = self.root / "raw"
        labels = raw / "label_for_single_UIs_cn"
        labels.mkdir(parents=True)
        (raw / "single_UIs_cn").mkdir()
        for index in range(3):
            Image.new("RGB", (100, 200)).save(raw / "single_UIs_cn" / f"{index}.png")
        for source, (task, _) in TASKS.items():
            rows = []
            for index, name in enumerate(("douyin", "抖音", "wechat")):
                boxes = [[90, 20, 30, 40]] if index == 0 else []
                rows.append({"id": index, "infos": {
                    "app_name": name, "image_path": f"single_UIs/{index}.png",
                    "image_size": [[100, 200]], "target_problem": task,
                    "box_list": boxes, "label_list": ["element"] * len(boxes),
                }})
            (labels / f"{source}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        converted = self.root / "converted"
        prepare(raw, converted, "clip")
        with contextlib.redirect_stdout(io.StringIO()):
            report = subset(converted, self.output)
        inspected = read_converted(self.output)
        for task, records in inspected.items():
            self.assertEqual(len(records), 2)
            self.assertTrue(all(record["conversion_ok"] for record in records))
            self.assertEqual(report["tasks"][task]["clipped_boxes"], 1)


if __name__ == "__main__":
    unittest.main()
