from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from PIL import Image, ImageDraw, PngImagePlugin

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_ui_lens_duplicates import audit, fingerprint
from locany_ui5_common import TASK_JSONL


class DuplicateAuditTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images = self.root / "images"
        self.images.mkdir()
        self.output = self.root / "report"

    def page(self):
        page = Image.new("RGB", (200, 400), "#f0f2f6")
        draw = ImageDraw.Draw(page)
        draw.rectangle((0, 0, 200, 45), fill="#152755")
        draw.rectangle((15, 70, 185, 180), fill="#c44131")
        for y in (210, 250, 290, 330):
            draw.rectangle((20, y, 130, y + 10), fill="#333333")
        return page

    def run_audit(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return audit(self.images, self.output, interval=60, **kwargs)

    def test_file_vs_pixel_duplicates_and_readonly_source(self):
        page = self.page()
        a, b, c = [self.images / f"{name}.png" for name in "abc"]
        page.save(a)
        b.write_bytes(a.read_bytes())
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("note", "different PNG metadata")
        page.save(c, pnginfo=metadata)
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.images.iterdir()}
        result = self.run_audit()
        self.assertEqual(result["identical_file_extra_copies"], 1)
        self.assertEqual(result["identical_pixel_extra_copies"], 2)
        self.assertEqual(result["exact_pixel_pairs"], 3)
        self.assertEqual(result["near_candidate_pairs_excluding_exact"], 0)
        self.assertEqual(result["pairs_compared"], 3)
        self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.images.iterdir()})
        self.assertTrue((self.output / "index.html").is_file())
        self.assertTrue(list((self.output / "assets").glob("*.jpg")))

    def test_minor_variant_and_resolution_variant_are_candidates(self):
        page = self.page()
        page.save(self.images / "original.png")
        page.resize((400, 800)).save(self.images / "resized.png")
        draw = ImageDraw.Draw(page)
        draw.rectangle((170, 20, 175, 25), fill="white")
        page.save(self.images / "variant.png")
        result = self.run_audit()
        self.assertEqual(result["near_candidate_pairs_excluding_exact"], 3)
        self.assertEqual(result["identical_pixel_groups"], 0)
        self.assertEqual(result["images_in_exact_or_near_groups"], 3)
        self.assertEqual(len(list((self.output / "assets").glob("diff-*.png"))), 3)
        curve = list(result["near_candidate_counts_by_threshold_excluding_exact"].values())
        self.assertEqual(curve, sorted(curve))

    def test_alpha_differences_are_not_exact_pixel_matches(self):
        Image.new("RGBA", (40, 80), (255, 0, 0, 0)).save(self.images / "transparent.png")
        Image.new("RGBA", (40, 80), (255, 0, 0, 255)).save(self.images / "opaque.png")
        a, _ = fingerprint(self.images / "transparent.png")
        b, _ = fingerprint(self.images / "opaque.png")
        self.assertNotEqual(a["pixel_sha256"], b["pixel_sha256"])

    def test_exif_orientation_is_normalized_before_pixel_comparison(self):
        original = self.images / "original.png"
        rotated = self.images / "rotated.png"
        page = self.page()
        page.save(original)
        exif = Image.Exif()
        exif[274] = 6
        page.transpose(Image.Transpose.ROTATE_90).save(rotated, exif=exif)
        a, _ = fingerprint(original)
        b, _ = fingerprint(rotated)
        self.assertNotEqual(a["file_sha256"], b["file_sha256"])
        self.assertEqual(a["pixel_sha256"], b["pixel_sha256"])
        self.assertEqual(a["phash"], b["phash"])

    def test_connected_groups_are_not_reported_as_all_pair_matches(self):
        for index, color in enumerate(("red", "green", "blue")):
            Image.new("RGB", (40, 80), color).save(self.images / f"{index}.png")
        def controlled_hash(path):
            record, (_, dh, small) = fingerprint(path)
            ph = (0b0000, 0b0011, 0b1111)[int(path.stem)]
            record["phash"] = f"{ph:016x}"
            return record, (ph, dh, small)
        with mock.patch("audit_ui_lens_duplicates.fingerprint", side_effect=controlled_hash):
            result = self.run_audit(threshold=2)
        self.assertEqual(result["near_candidate_pairs_excluding_exact"], 2)
        self.assertEqual(result["connected_groups"], 1)
        self.assertEqual(result["largest_connected_group"], 3)

    def test_aspect_filter_and_csv_caps_do_not_change_full_counts(self):
        for index in range(3):
            Image.new("RGB", (40, 80), "white").save(self.images / f"{index}.png")
        Image.new("RGB", (80, 40), "white").save(self.images / "wide.png")
        result = self.run_audit(max_pairs=1, max_preview_pairs=0, max_preview_groups=0)
        self.assertEqual(result["exact_pixel_pairs"], 3)
        self.assertEqual(result["aspect_rejected_nonexact_pairs"], 3)
        self.assertEqual(result["csv_pairs_written"], 1)
        self.assertEqual(result["csv_pairs_omitted"], 2)
        with (self.output / "pairs.csv").open(encoding="utf-8-sig", newline="") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 1)
        groups = json.loads((self.output / "near_groups.json").read_text())
        self.assertEqual(len(groups["groups"][0]), 3)

    def test_label_disagreement_is_reported_without_filtering(self):
        page = self.page()
        a, b = self.images / "a.png", self.images / "b.png"
        page.save(a)
        page.save(b)
        data = self.root / "eval"
        data.mkdir()
        for filename in TASK_JSONL.values():
            rows = [{"images": [str(p)], "answer": {"bbox": boxes},
                     "extra_info": {"original_infos": {"app_name": ["抖音"]}}}
                    for p, boxes in ((a, []), (b, [[1, 2, 3, 4]]))]
            (data / filename).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        result = self.run_audit(eval_dir=data)
        self.assertEqual(result["candidate_pairs_with_task_label_disagreement"], 1)
        self.assertEqual(result["images_by_app"], {"抖音": 2})
        self.assertEqual(result["exact_pixel_pairs"], 1)

    def test_unreadable_file_is_reported_and_valid_images_still_finish(self):
        self.page().save(self.images / "good.png")
        (self.images / "bad.png").write_bytes(b"invalid png")
        result = self.run_audit()
        self.assertEqual(result["status"], "complete_with_errors")
        self.assertEqual(result["decoded_images"], 1)
        self.assertEqual(result["unreadable_images"], 1)
        self.assertEqual(len(json.loads((self.output / "errors.json").read_text())), 1)

    def test_existing_output_and_invalid_settings_are_rejected(self):
        self.page().save(self.images / "one.png")
        with self.assertRaises(ValueError):
            self.run_audit(threshold=65)
        self.assertFalse(self.output.exists())
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            self.run_audit()


if __name__ == "__main__":
    unittest.main()
