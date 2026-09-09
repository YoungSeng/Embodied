from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_ui_lens_duplicates import fingerprint
from audit_ui_lens_task_quality import analyze, bbox_valid, classify_task, quality_metrics
from locany_ui5_common import TASK_JSONL


def row(path, boxes=None, clipped=False):
    boxes = boxes or []
    return {"path": path, "bbox": boxes, "positive": bool(boxes), "valid_bbox": True, "clipped_gt": clipped}


def features(pixel_hash, phash="0"):
    return {"pixel_sha256": pixel_hash, "file_sha256": pixel_hash, "phash": phash, "width": 300, "height": 600}


class TaskQualityTest(unittest.TestCase):
    def test_within_task_duplicates_and_cross_task_sharing_are_separate(self):
        f = {"a": features("same"), "b": features("same"), "c": features("same")}
        q = {p: {"flags": []} for p in f}
        cross = {"same": {"occlusion", "cropping"}}
        stats, _, _, _ = classify_task("occlusion", [row("a", [[1, 2, 3, 4]]), row("b")], f, q, cross, 8, .03)
        self.assertEqual(stats["pixel_extra_copies"], 1)
        self.assertEqual(stats["pixel_duplicate_images"], 2)
        self.assertEqual(stats["exact_polarity_conflict_groups"], 1)
        self.assertEqual(stats["exact_bbox_disagreement_groups"], 1)
        self.assertEqual(stats["cross_task_shared_content_images"], 2)
        other, _, _, _ = classify_task("cropping", [row("c")], f, q, cross, 8, .03)
        self.assertEqual(other["pixel_extra_copies"], 0)
        self.assertEqual(other["cross_task_shared_content_images"], 1)
        self.assertEqual(other["exact_polarity_conflict_groups"], 0)

    def test_other_task_cannot_bridge_two_dissimilar_images(self):
        f = {"a": features("a", "0"), "b": features("b", "3"), "c": features("c", "f")}
        q = {p: {"flags": []} for p in f}
        stats, groups, _, _ = classify_task("occlusion", [row("a"), row("c")], f, q, {}, 2, .03)
        self.assertEqual(stats["near_pairs_excluding_exact"], 0)
        self.assertEqual(groups["near_components_excluding_exact_edges"], [])
        stats, groups, _, _ = classify_task("occlusion", [row("a"), row("b"), row("c")], f, q, {}, 2, .03, max_pairs=0)
        self.assertEqual(stats["near_pairs_excluding_exact"], 2)
        self.assertEqual(stats["near_csv_pairs_omitted"], 2)
        self.assertEqual(len(groups["near_components_excluding_exact_edges"][0]), 3)

    def test_bbox_order_does_not_create_conflict_and_clipping_is_separate(self):
        f = {"a": features("same"), "b": features("same")}
        q = {p: {"flags": []} for p in f}
        boxes = [[1, 2, 3, 4], [4, 5, 6, 7]]
        stats, _, _, _ = classify_task("occlusion", [row("a", boxes, True), row("b", boxes[::-1])], f, q, {}, 8, .03)
        self.assertEqual(stats["exact_bbox_disagreement_groups"], 0)
        self.assertEqual(stats["clipped_gt_rows"], 1)
        self.assertEqual(stats["quality_candidate_images"], 0)

    def test_quality_flags_use_union_and_keep_positive_negative_counts(self):
        rows = [row("a", [[1, 2, 3, 4]]), row("b")]
        q = {"a": {"flags": ["low_resolution", "near_blank"]}, "b": {"flags": ["low_detail"]}}
        stats, _, _, _ = classify_task("content_missing", rows, {}, q, {}, 8, .03)
        self.assertEqual(stats["quality_candidate_images"], 2)
        self.assertEqual(stats["quality_candidate_positive"], 1)
        self.assertEqual(stats["quality_candidate_negative"], 1)
        self.assertEqual(stats["duplicate_unassessed_rows"], 2)

    def test_quality_metrics_flag_blank_and_detect_loss_of_detail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.new("RGB", (64, 128), "white").save(root / "blank.png")
            metrics = quality_metrics(root / "blank.png")
            self.assertEqual(metrics["flags"], ["low_resolution", "near_blank"])
            page = Image.new("RGB", (300, 600), "white")
            draw = ImageDraw.Draw(page)
            for y in range(10, 590, 20):
                draw.rectangle((20, y, 280, y+5), fill="black")
            page.save(root / "sharp.png")
            page.filter(ImageFilter.GaussianBlur(4)).save(root / "blurred.png")
            sharp = quality_metrics(root / "sharp.png")
            blurred = quality_metrics(root / "blurred.png")
            self.assertGreater(sharp["laplacian_variance"], blurred["laplacian_variance"])
            midpoint = (sharp["laplacian_variance"] + blurred["laplacian_variance"]) / 2
            self.assertNotIn("low_detail", quality_metrics(root / "sharp.png", low_detail_variance=midpoint)["flags"])
            self.assertIn("low_detail", quality_metrics(root / "blurred.png", low_detail_variance=midpoint)["flags"])

    def fixture(self, root):
        image_dir, data, audit_dir = root / "images", root / "eval", root / "audit"
        for p in (image_dir, data, audit_dir):
            p.mkdir()
        page = Image.new("RGB", (1000, 2000), "white")
        ImageDraw.Draw(page).rectangle((20, 50, 900, 500), fill="navy")
        images = [image_dir / f"{i}.png" for i in range(2)]
        hashes = []
        for i, path in enumerate(images):
            page.save(path)
            fp, _ = fingerprint(path)
            hashes.append({"id": i, "path": str(path), **fp})
        (audit_dir / "images.jsonl").write_text("".join(json.dumps(r)+"\n" for r in hashes), encoding="utf-8")
        (audit_dir / "summary.json").write_text(json.dumps({"phash_threshold": 8, "aspect_tolerance": .03}), encoding="utf-8")
        for task, filename in TASK_JSONL.items():
            selected = images if task == "occlusion" else images[:1]
            # Very narrow border box tests thumbnail drawing after downscaling.
            records = [{"images": [str(path)], "answer": {"bbox": [[999.9, 1999, 1000, 2000]] if i == 0 else []}}
                       for i, path in enumerate(selected)]
            (data / filename).write_text("".join(json.dumps(r)+"\n" for r in records), encoding="utf-8")
        return audit_dir, data, images

    def test_end_to_end_report_and_small_border_gt_preview(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit_dir, data, _ = self.fixture(root)
            output = root / "report"
            with contextlib.redirect_stdout(io.StringIO()):
                result = analyze(audit_dir, data, output, interval=60)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["image_task_records"], 6)
            self.assertEqual(result["tasks"]["occlusion"]["pixel_extra_copies"], 1)
            self.assertEqual(result["tasks"]["cropping"]["pixel_extra_copies"], 0)
            self.assertEqual(result["tasks"]["occlusion"]["exact_polarity_conflict_groups"], 1)
            for filename in ("index.html", "report.md", "task_summary.csv", "image_quality.jsonl", "task_samples.jsonl", "groups.json"):
                self.assertTrue((output / filename).is_file())
            self.assertTrue(list((output / "assets").glob("*.jpg")))

    def test_stale_or_missing_images_are_not_classified_using_old_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit_dir, data, images = self.fixture(root)
            Image.new("RGB", (1000, 2000), "red").save(images[0])
            images[1].unlink()
            with contextlib.redirect_stdout(io.StringIO()):
                result = analyze(audit_dir, data, root / "report", interval=60)
            self.assertEqual(result["duplicate_unassessed_unique_paths"], 2)
            self.assertEqual(result["tasks"]["occlusion"]["pixel_extra_copies"], 0)
            self.assertEqual(result["tasks"]["occlusion"]["unreadable"], 1)
            self.assertEqual({row["reason"] for row in result["cache_issues"]},
                             {"unreadable", "image_changed_since_previous_audit"})

    def test_empty_task_and_bbox_validation(self):
        stats, _, _, _ = classify_task("occlusion", [], {}, {}, {}, 8, .03)
        self.assertEqual(stats["rows"], 0)
        self.assertEqual(stats["pixel_extra_copy_rate"], 0)
        self.assertTrue(bbox_valid([[99.9, 20, 100, 200]], (100, 200)))
        for box in ([0, 0, 101, 10], [5, 5, 5, 6], [0, 0, float("nan"), 8]):
            self.assertFalse(bbox_valid([box], (100, 200)))


if __name__ == "__main__":
    unittest.main()
