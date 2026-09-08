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
