import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import inspect_ui_lens_eval as inspector


class InspectUILensTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "converted"
        self.data.mkdir()
        for label in ("positive", "negative"):
            Image.new("RGB", (100, 200), "white").save(self.root / f"{label}.png")
        for task, issue in inspector.TASKS.values():
            rows = []
            for label, boxes, raw_boxes in (("positive", [[10, 20, 40, 60]], [[10, 20, 30, 40]]), ("negative", [], [])):
                rows.append({"id": label, "images": [str(self.root / f"{label}.png")],
                             "answer": {"bbox": boxes, "types": [issue] * len(boxes)},
                             "extra_info": {"original_infos": {"box_list": raw_boxes, "image_size": [[100, 200]]}}})
            self.write_rows(task, rows)

    def write_rows(self, task, rows):
        (self.data / f"test_ui_{task}_wcnt_no_figma.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    def inspect(self, *args, **kwargs):
        self.stdout = io.StringIO()
        with contextlib.redirect_stdout(self.stdout):
            return inspector.inspect(self.data, *args, **kwargs)

    def test_full_statistics_are_independent_of_preview_sample_count(self):
        output = self.root / "inspection"
        summary = self.inspect(output, samples_per_task=1)
        self.assertEqual(summary["unique_image_paths"], 2)
        self.assertEqual(summary["total_image_task_records"]["samples"], 10)
        self.assertEqual(summary["total_image_task_records"]["positive"], 5)
        self.assertEqual(summary["total_image_task_records"]["negative"], 5)
        self.assertEqual(summary["total_image_task_records"]["boxes"], 5)
        self.assertEqual(summary["total_image_task_records"]["conversion_mismatches"], 0)
        self.assertEqual(summary["visualization"]["rendered"], 5)
        self.assertIn("TOTAL(image-task)", self.stdout.getvalue())
        self.assertTrue((output / "dataset_stats.csv").is_file())
        saved = json.loads((output / "dataset_stats.json").read_text(encoding="utf-8"))
        self.assertEqual(saved, summary)
        gallery = (output / "index.html").read_text(encoding="utf-8")
        for sample in json.loads((output / "samples.json").read_text(encoding="utf-8")):
            self.assertTrue((output / sample["preview"]).is_file())
            self.assertIn(sample["preview"], gallery)

    def test_render_uses_saved_xyxy_coordinates_and_leaves_original_intact(self):
        output = self.root / "inspection"
        self.inspect(output, samples_per_task=0)
        with Image.open(output / "occlusion" / "000001_positive.png") as pair:
            self.assertEqual(pair.getpixel((10, 20 + 64)), (255, 255, 255))
            self.assertEqual(pair.getpixel((280 + 16 + 40, 40 + 64)), (0, 184, 107))
            self.assertEqual(pair.getpixel((280 + 16 + 50, 40 + 64)), (255, 255, 255))
        with Image.open(output / "occlusion" / "000002_negative.png") as pair:
            self.assertEqual(pair.getpixel((280 + 16 + 40, 40 + 64)), (255, 255, 255))

    def test_mismatch_is_counted_and_prioritized_for_preview(self):
        path = self.data / "test_ui_occlusion_wcnt_no_figma.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows[1]["extra_info"]["original_infos"]["box_list"] = [[10, 20, 30, 40]]
        self.write_rows("occlusion", rows)
        output = self.root / "inspection"
        summary = self.inspect(output, samples_per_task=1)
        self.assertEqual(summary["tasks"]["occlusion"]["conversion_mismatches"], 1)
        selected = [s for s in json.loads((output / "samples.json").read_text(encoding="utf-8")) if s["task"] == "occlusion"]
        self.assertEqual(selected[0]["line"], 2)
        self.assertFalse(selected[0]["conversion_ok"])

    def test_sampling_is_balanced_reproducible_and_handles_single_polarity(self):
        records = [{"line": i, "positive": i < 20, "conversion_ok": True} for i in range(30)]
        selected = inspector.select_samples(records, 10, 42)
        self.assertEqual(sum(r["positive"] for r in selected), 5)
        self.assertEqual(selected, inspector.select_samples(records, 10, 42))
        self.assertEqual(len(inspector.select_samples(records[:20], 10, 42)), 10)
        self.assertEqual(inspector.select_samples(records, 0, 42), records)

    def test_stats_only_does_not_write_files(self):
        before = sorted(self.root.rglob("*"))
        summary = self.inspect()
        self.assertEqual(summary["total_image_task_records"]["samples"], 10)
        self.assertEqual(before, sorted(self.root.rglob("*")))

    def add_clipped_sample(self):
        path = self.data / "test_ui_occlusion_wcnt_no_figma.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows[0]["answer"]["bbox"] = [[90, 20, 100, 60]]
        rows[0]["extra_info"]["original_infos"]["box_list"] = [[90, 20, 30, 40]]
        rows[0]["extra_info"]["bbox_conversion"] = {
            "boundary_policy": "clip", "clipped_boxes": [{
                "box_index": 0, "original_xywh": [90, 20, 30, 40],
                "original_xyxy": [90, 20, 120, 60], "clipped_xyxy": [90, 20, 100, 60],
                "max_clip_pixels": 20, "removed_area_ratio": 1 - 10 / 30,
            }],
        }
        self.write_rows("occlusion", rows)
        return rows

    def test_declared_clipping_is_audited_counted_and_drawn_at_visible_edge(self):
        self.add_clipped_sample()
        output = self.root / "inspection"
        stats = self.inspect(output, samples_per_task=1)
        task = stats["tasks"]["occlusion"]
        self.assertEqual((task["conversion_mismatches"], task["clipped_images"], task["clipped_boxes"]), (0, 1, 1))
        self.assertEqual((task["positive"], task["negative"]), (1, 1))
        self.assertEqual(task["max_clip_pixels"], 20)
        self.assertIn("clip_box", self.stdout.getvalue())
        with Image.open(output / "occlusion" / "000001_positive.png") as pair:
            self.assertEqual(pair.getpixel((280 + 16 + 99, 40 + 64)), (224, 128, 22))
        gallery = (output / "index.html").read_text(encoding="utf-8")
        self.assertIn('data-clipped="yes"', gallery)
        self.assertIn("裁剪边缘最多 20 px", gallery)

    def test_silent_clipping_and_incorrect_audit_are_mismatches(self):
        rows = self.add_clipped_sample()
        rows[0]["extra_info"]["bbox_conversion"]["clipped_boxes"] = []
        self.write_rows("occlusion", rows)
        self.assertEqual(self.inspect()["tasks"]["occlusion"]["conversion_mismatches"], 1)
        del rows[0]["extra_info"]["bbox_conversion"]
        self.write_rows("occlusion", rows)
        self.assertEqual(self.inspect()["tasks"]["occlusion"]["conversion_mismatches"], 1)

    def test_clipped_samples_are_prioritized_after_mismatches(self):
        records = [{"line": i, "positive": True, "conversion_ok": True, "clipped_boxes": []} for i in range(30)]
        records[20]["clipped_boxes"] = [{"box_index": 0}]
        records[25]["conversion_ok"] = False
        self.assertEqual([r["line"] for r in inspector.select_samples(records, 2, 42)], [20, 25])


if __name__ == "__main__":
    unittest.main()
