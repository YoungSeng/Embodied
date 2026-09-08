import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

SPEC = importlib.util.spec_from_file_location(
    "prepare_ui_lens_eval", Path(__file__).resolve().parents[1] / "scripts" / "prepare_ui_lens_eval.py"
)
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


class PrepareUILensTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "label_for_single_UIs_cn").mkdir()
        (self.root / "single_UIs_cn").mkdir()
        for name in ("positive", "negative"):
            Image.new("RGB", (100, 200)).save(self.root / "single_UIs_cn" / f"{name}.png")
        for source, (task, _) in adapter.TASKS.items():
            rows = []
            for name, boxes in (("positive", [[10, 20, 30, 40]]), ("negative", [])):
                rows.append({"id": name, "infos": {
                    "image_path": f"single_UIs/{name}.png", "image_size": [100, 200],
                    "target_problem": source.replace("_", " ").title(),
                    "label_list": ["element"] * len(boxes), "box_list": boxes,
                }})
            self.write_rows(source, rows)

    def write_rows(self, source, rows):
        (self.root / "label_for_single_UIs_cn" / f"{source}.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    def modify_first(self, key, value):
        path = self.root / "label_for_single_UIs_cn" / "container_overlap.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["infos"][key] = value
        self.write_rows("container_overlap", rows)

    def test_coordinates_and_negative_preservation(self):
        output = self.root / "converted"
        report = adapter.prepare(self.root, output)
        self.assertEqual(report["unique_images"], 2)
        for task in report["tasks"]:
            self.assertEqual(report["tasks"][task]["positive"], 1)
            self.assertEqual(report["tasks"][task]["negative"], 1)
            rows = [json.loads(line) for line in (output / f"test_ui_{task}_wcnt_no_figma.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["answer"]["bbox"], [[10, 20, 40, 60]])
            self.assertEqual(rows[1]["answer"], {"bbox": [], "types": []})
            self.assertTrue(Path(rows[0]["images"][0]).is_file())

    def test_rejects_bad_coordinates_before_writing(self):
        self.modify_first("box_list", [[90, 20, 30, 40]])
        output = self.root / "converted"
        with self.assertRaisesRegex(ValueError, "no automatic clipping"):
            adapter.prepare(self.root, output)
        self.assertFalse(output.exists())

    def test_reported_nine_pixel_overflow(self):
        raw = [[2778, 4140, 849, 348]]
        self.assertEqual(adapter.convert_boxes(raw, 3618, 7866, "clip"), [[2778, 4140, 3618, 4488]])
        self.assertEqual(raw, [[2778, 4140, 849, 348]])
        with self.assertRaisesRegex(ValueError, "no automatic clipping"):
            adapter.convert_boxes(raw, 3618, 7866)

    def test_clipping_preserves_labels_and_records_each_correction(self):
        self.modify_first("box_list", [[90, 20, 30, 40]])
        output = self.root / "converted"
        report = adapter.prepare(self.root, output, bbox_boundary_policy="clip")
        self.assertEqual(report["bbox_boundary_policy"], "clip")
        self.assertEqual(report["clipped_image_task_records"], 1)
        self.assertEqual(report["clipped_boxes"], 1)
        stats = report["tasks"]["occlusion"]
        self.assertEqual((stats["positive"], stats["negative"], stats["boxes"]), (1, 1, 1))
        self.assertEqual((stats["clipped_images"], stats["clipped_boxes"], stats["max_clip_pixels"]), (1, 1, 20))
        self.assertAlmostEqual(stats["max_removed_area_ratio"], 2 / 3)
        rows = [json.loads(line) for line in (output / "test_ui_occlusion_wcnt_no_figma.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(rows[0]["answer"]["bbox"], [[90, 20, 100, 60]])
        self.assertEqual(rows[0]["extra_info"]["original_infos"]["box_list"], [[90, 20, 30, 40]])
        self.assertEqual(rows[1]["answer"], {"bbox": [], "types": []})
        corrections = rows[0]["extra_info"]["bbox_conversion"]["clipped_boxes"]
        self.assertEqual(corrections[0]["original_xyxy"], [90, 20, 120, 60])
        self.assertEqual(corrections[0]["clipped_xyxy"], [90, 20, 100, 60])
        audit = [json.loads(line) for line in (output / "bbox_clipping_audit.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(audit), 1)
        self.assertEqual(audit[0]["source_line"], 1)
        self.assertEqual(audit[0]["clipped_boxes"], corrections)

    def test_clipping_all_edges_preserves_fractional_coordinates(self):
        self.assertEqual(adapter.convert_boxes([[-2.5, -3, 105, 205]], 100, 200, "clip"), [[0, 0, 100, 200]])
        self.assertEqual(adapter.convert_boxes([[99.5, 190.25, 1, 12]], 100, 200, "clip"), [[99.5, 190.25, 100, 200]])

    def test_clipping_does_not_drop_invalid_or_fully_external_boxes(self):
        for raw in ([[100, 20, 3, 4]], [[-5, 20, 5, 4]], [[5, 210, 3, 4]], [[5, 20, 0, 4]], [[5, 20, -3, 4]]):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                adapter.convert_boxes(raw, 100, 200, "clip")
        self.modify_first("box_list", [[100, 20, 3, 4]])
        output = self.root / "converted"
        with self.assertRaisesRegex(ValueError, "refusing to drop"):
            adapter.prepare(self.root, output, bbox_boundary_policy="clip")
        self.assertFalse(output.exists())

    def test_missing_annotations_are_not_negatives(self):
        self.modify_first("box_list", None)
        with self.assertRaisesRegex(ValueError, "explicit"):
            adapter.prepare(self.root)

    def test_checks_image_dimensions_and_task(self):
        self.modify_first("image_size", [200, 100])
        with self.assertRaisesRegex(ValueError, "actual size"):
            adapter.prepare(self.root)
        self.modify_first("image_size", [100, 200])
        self.modify_first("target_problem", "unknown category")
        with self.assertRaisesRegex(ValueError, "Unrecognized target_problem"):
            adapter.prepare(self.root)

    def test_singleton_wrapped_image_size(self):
        self.modify_first("image_size", [[100, 200]])
        output = self.root / "converted"
        report = adapter.prepare(self.root, output)
        self.assertEqual(report["tasks"]["occlusion"]["rows"], 2)
        row = json.loads((output / "test_ui_occlusion_wcnt_no_figma.jsonl").read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(row["extra_info"]["original_infos"]["image_size"], [[100, 200]])
        self.assertEqual(row["answer"]["bbox"], [[10, 20, 40, 60]])

    def test_wrapped_image_size_still_checks_actual_dimensions(self):
        self.modify_first("image_size", [[200, 100]])
        with self.assertRaisesRegex(ValueError, "actual size"):
            adapter.prepare(self.root)
        for invalid in ([[100, 200], [100, 200]], [True, 200], None, [[100]], [[[100, 200]]]):
            with self.subTest(size=invalid), self.assertRaises(ValueError):
                adapter.normalize_image_size(invalid)

    def test_rejects_duplicate_images_within_task(self):
        path = self.root / "label_for_single_UIs_cn" / "container_overlap.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.write_rows("container_overlap", rows + [rows[0]])
        with self.assertRaisesRegex(ValueError, "Repeated image"):
            adapter.prepare(self.root)

    def test_missing_images_fail_instead_of_silently_skipping(self):
        self.modify_first("image_path", "single_UIs/absent.png")
        with self.assertRaisesRegex(ValueError, "0 matches"):
            adapter.prepare(self.root)


if __name__ == "__main__":
    unittest.main()
