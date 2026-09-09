import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image, ImageDraw
from pypdf import PdfReader
from pypdf.generic import ContentStream

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from export_ui_lens_bbox_pdf import export, pdf_rect
from locany_ui5_common import TASK_JSONL


def fixture(root):
    rows = {key: [] for key in TASK_JSONL}
    for name, task, box in [("788.png", "occlusion", [40, 100, 240, 250]),
                            ("819.png", "cropping", [400, 850, 640, 1136])]:
        path = root / name
        with Image.new("RGB", (640, 1136), "white") as image:
            draw = ImageDraw.Draw(image)
            draw.rectangle(box, fill="#b0c4de")
            draw.text((30, 30), "SYNTHETIC COORDINATE CHECK", fill="black")
            image.save(path)
        rows[task].append({"images": [str(path)], "answer": {"bbox": [box]}})
    rows["content_missing"].append({"images": [str(root / "788.png")], "answer": {"bbox": []}})
    for key, filename in TASK_JSONL.items():
        (root / filename).write_text("".join(json.dumps(row) + "\n" for row in rows[key]), encoding="utf-8")


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "output"
        fixture(self.root)

    def run_export(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()) as log:
            result = export(self.root, self.output, **kwargs)
        return result, log.getvalue()

    def test_pdf_preserves_pixels_and_draws_vector_boxes(self):
        result, log = self.run_export()
        self.assertEqual([r["task"] for r in result], ["occlusion", "cropping"])
        self.assertIn("positive, boxes=1", log)
        self.assertIn("negative, boxes=0", log)
        saved = json.loads((self.output / "export_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(saved[0]["bbox_xyxy"], [[40, 100, 240, 250]])
        for sample in result:
            reader = PdfReader(sample["pdf"])
            self.assertEqual(len(reader.pages), 1)
            page = reader.pages[0]
            self.assertEqual([float(page.mediabox.width), float(page.mediabox.height)], [320, 568])
            images = [obj.get_object() for obj in page["/Resources"]["/XObject"].values()]
            self.assertEqual(len(images), 1)
            self.assertEqual((images[0]["/Width"], images[0]["/Height"]), (640, 1136))
            self.assertNotIn("DCTDecode", str(images[0]["/Filter"]))
            operations = ContentStream(page.get_contents(), reader).operations
            rectangles = [[float(v) for v in args] for args, op in operations if op == b"re"]
            expected = pdf_rect(sample["bbox_xyxy"][0], 640, 1136, .5, 1.5)
            self.assertEqual(rectangles, [list(expected)])
            with Image.open(sample["preview"]) as preview:
                self.assertEqual(preview.size, (640, 1136))

    def test_flip_and_boundary_stroke(self):
        self.assertEqual(pdf_rect([40, 100, 240, 250], 640, 1136, .5, 1.5), (20, 443, 100, 75))
        self.assertEqual(pdf_rect([0, 0, 640, 1136], 640, 1136, .5, 1.5), (.75, .75, 318.5, 566.5))

    def test_ambiguous_tasks_fail_before_writing(self):
        path = self.root / TASK_JSONL["text_overflow"]
        path.write_text(json.dumps({"images": [str(self.root / "788.png")], "answer": {"bbox": [[1, 2, 3, 4]]}}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "multiple positive"):
            self.run_export()
        self.assertFalse(self.output.exists())
        result, _ = self.run_export(image_names=["788.png"], task="occlusion")
        self.assertEqual(result[0]["task"], "occlusion")

    def test_invalid_second_image_fails_before_writing(self):
        (self.root / "819.png").unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_export()
        self.assertFalse(self.output.exists())

    def test_invalid_box_fails_before_writing(self):
        path = self.root / TASK_JSONL["cropping"]
        row = json.loads(path.read_text(encoding="utf-8"))
        row["answer"]["bbox"][0][2] = 641
        path.write_text(json.dumps(row), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Out-of-bounds"):
            self.run_export()
        self.assertFalse(self.output.exists())

    def test_existing_pdf_is_not_overwritten(self):
        self.output.mkdir()
        existing = self.output / "788_bbox.pdf"
        existing.write_bytes(b"existing")
        with self.assertRaises(FileExistsError):
            self.run_export()
        self.assertEqual(existing.read_bytes(), b"existing")


if __name__ == "__main__":
    unittest.main()
