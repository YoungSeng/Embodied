"""EXIF=0 and old failed-global-report recovery; all fixtures and checks are CPU-only."""
import contextlib
import io
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from tests.test_ui14_pipeline import source_fixture
from ui14_common import UI_TASKS, paths_for, image_identity, file_digest, read_json, read_jsonl, write_json, write_jsonl
from ui14_repair import validate_normalization
from ui14_normalize_resume import split_state_paths, SplitResume
import prepare_ui14_sft as prepare
from PIL import Image


class NormalizeResumeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source, self.data = self.root / "input", self.root / "data"
        source_fixture(self.source)
        # Two alignment rows per split: one EXIF=0 and one ordinary success.
        # Legacy outputs will have partial successes, exactly like the reported run.
        for split in ("train", "test"):
            folder = self.source / "ui_alignment"
            path = folder / f"{split}.jsonl"
            row = next(read_jsonl(path))
            image = Path(row["images"][0])
            with Image.open(image) as opened:
                pixels = opened.convert("RGB")
            pixels.putpixel((1, 2), (255, 0, 0))
            exif = Image.Exif(); exif[274] = 0
            pixels.save(image, exif=exif)
            normal = folder / "sample_imgs" / f"ordinary-{split}.png"
            pixels.save(normal)
            write_jsonl(path, [row, {**row, "id": "ordinary", "images": [str(normal)]}])
        manifest = read_json(self.source / "manifest.json")
        for source in manifest["datasets"]:
            if source["key"] == "ui_alignment": source.update(train_records=2, test_records=2)
        write_json(self.source / "manifest.json", manifest)
        repair = read_json(self.source / "repair_summary.json")
        repair["input_records"] = 20
        for item in repair["datasets"]:
            if item["dataset"] == "ui_alignment":
                item.update(before_train=2, before_test=2, after_train=2, after_test=2)
        write_json(self.source / "repair_summary.json", repair)
        # A reused split must retain a real old/new-parser disagreement detail.
        path = self.source / "synth_radius/train.jsonl"
        raw = next(read_jsonl(path))
        raw["Objects"][0]["Location"] = {"rect_err_1": [15, 45, 90, 75], "rect1": [10, 40, 100, 80]}
        write_jsonl(path, [raw])
        self.args = SimpleNamespace(ui9_data_root=self.source, output_dir=self.data,
                                   ui5_recipe="audited.json", ui5_test_dir=self.root / "old_test")

    def normalize(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            prepare.normalize(self.args)
        return output.getvalue(), read_json(self.data / "cpu_check_report.json")

    def legacy_failure(self):
        actual_main = prepare.main_image

        def reject_zero(*args):
            image = actual_main(*args)
            with Image.open(image) as opened:
                if opened.getexif().get(274, 1) == 0:
                    raise ValueError("Screenshot has EXIF orientation; preparation's raw canvas and detector canvas disagree")
            return image

        with mock.patch.object(prepare, "main_image", side_effect=reject_zero), self.assertRaisesRegex(RuntimeError, "intake failed"):
            self.normalize()
        # Reproduce the on-disk legacy schema: only the failed global report,
        # shared audit details, snapshot and existing normalized/detector outputs.
        legacy_markers = (self.data / "normalization_splits").resolve()
        self.assertEqual(legacy_markers.parent, self.data.resolve())
        shutil.rmtree(legacy_markers)
        report = read_json(self.data / "cpu_check_report.json")
        report.pop("normalization_resume")
        write_json(self.data / "cpu_check_report.json", report)
        stats = read_json(self.data / "normalization_stats.json")
        self.assertFalse(stats["complete"])
        self.assertEqual(sum(s["failed_records"] for s in stats["tasks"].values()), 2)
        self.assertEqual(stats["tasks"]["ui_alignment/train"]["normalized_records"], 1)
        return stats

    @contextlib.contextmanager
    def no_images(self):
        with mock.patch.object(Image, "open", side_effect=AssertionError("reuse opened an image")), \
             mock.patch.object(prepare, "inspect_image", side_effect=AssertionError("reuse inspected an image")), \
             mock.patch.object(prepare, "image_identity", side_effect=AssertionError("reuse hashed pixels")), \
             mock.patch.object(prepare, "iter_records", side_effect=AssertionError("reuse reparsed source records")):
            yield

    def test_orientation_zero_preserves_pixels_bbox_and_original_file(self):
        path = self.source / "ui_alignment/sample_imgs/figma-train.png"
        before, identity = file_digest(path), image_identity(path)
        with Image.open(path) as image:
            self.assertEqual(image.getexif()[274], 0)
            pixels, size = image.convert("RGB").tobytes(), image.size
        _, report = self.normalize()
        row = next(read_jsonl(paths_for(self.data, "ui_alignment", "train")["normalized"]))
        self.assertEqual((row["source_image_id"], row["width"], row["height"]), identity)
        self.assertEqual(row["boxes_px"], [[20, 80, 200, 160]])
        self.assertEqual(file_digest(path), before)
        with Image.open(path) as image:
            self.assertEqual((image.size, image.convert("RGB").tobytes()), (size, pixels))
        self.assertTrue(report["normalization_complete"])
        self.assertFalse(report["ready"])
        self.assertEqual(report["normalization_resume"]["normalized_records"], 20)

    def test_old_failed_global_reuses_16_splits_and_only_rebuilds_alignment(self):
        old = self.legacy_failure()
        old_issues = [r for r in read_jsonl(self.data / "parser_compatibility_issues.jsonl") if r["task_key"] != "ui_alignment"]
        original_digests = {k: v for k, v in old["artifact_digests"].items() if "ui_alignment" not in k}
        original_open = Image.open
        opened = []

        def alignment_only(path, *args, **kwargs):
            self.assertIn("ui_alignment", Path(path).parts, "reused split read original pixels")
            opened.append(str(path))
            return original_open(path, *args, **kwargs)

        with mock.patch.object(Image, "open", side_effect=alignment_only), \
             mock.patch.object(prepare, "rebuild_normalized_split", wraps=prepare.rebuild_normalized_split) as rebuild:
            log, report = self.normalize()
        self.assertEqual([(c.args[2].task_key, c.args[3]) for c in rebuild.call_args_list],
                         [("ui_alignment", "train"), ("ui_alignment", "test")])
        self.assertTrue(opened)
        self.assertEqual(report["normalization_id"], old["normalization_id"])
        result = report["normalization_resume"]
        self.assertEqual((result["reused_splits"], result["rebuilt_splits"], result["reused_records"], result["rebuilt_records"]), (16, 2, 16, 4))
        self.assertEqual((result["normalized_records"], result["failed_records"]), (20, 0))
        self.assertIn("[normalize reused] synth_radius/train", log)
        self.assertIn("[normalize rebuilt] ui_alignment/train", log)
        self.assertTrue(read_json(self.data / "normalization_stats.json")["complete"])
        self.assertTrue(report["normalization_complete"])
        self.assertFalse(report["ready"])
        self.assertEqual(original_digests, {k: file_digest(self.data / k) for k in original_digests})
        self.assertEqual(old_issues, [r for r in read_jsonl(self.data / "parser_compatibility_issues.jsonl") if r["task_key"] != "ui_alignment"])
        validate_normalization(self.data)
        with self.no_images():
            _, repeated = self.normalize()
        self.assertEqual((repeated["normalization_resume"]["reused_splits"], repeated["normalization_resume"]["rebuilt_splits"]), (18, 0))

    def test_interrupt_after_alignment_commits_still_migrates_other_16_legacy_splits(self):
        old = self.legacy_failure()
        actual_record = SplitResume.record

        def stop_after_alignment(instance, task, split, *args):
            actual_record(instance, task, split, *args)
            if (task.task_key, split) == ("ui_alignment", "test"):
                raise KeyboardInterrupt("fixture interruption after split commit")

        with mock.patch.object(SplitResume, "record", stop_after_alignment), self.assertRaises(KeyboardInterrupt):
            self.normalize()
        self.assertFalse(read_json(self.data / "normalization_stats.json")["complete"])
        self.assertEqual(read_json(self.data / "normalization_stats.json")["artifact_digests"], old["artifact_digests"])
        with self.no_images():
            _, report = self.normalize()
        self.assertEqual(report["normalization_resume"]["reused_splits"], 18)
        self.assertTrue(report["normalization_complete"])

    def test_interrupted_partial_split_is_rebuilt_but_previous_commit_survives(self):
        original = prepare.save_split_state

        def interrupt_before_marker(root, task, split, *args):
            if (task.task_key, split) == ("ui_alignment", "test"):
                raise KeyboardInterrupt("fixture interruption after labels before split commit")
            return original(root, task, split, *args)

        with mock.patch.object(prepare, "save_split_state", side_effect=interrupt_before_marker), self.assertRaises(KeyboardInterrupt):
            self.normalize()
        self.assertTrue(read_json(split_state_paths(self.data, "ui_alignment", "train")[0])["complete"])
        self.assertFalse(read_json(split_state_paths(self.data, "ui_alignment", "test")[0])["complete"])
        with mock.patch.object(prepare, "rebuild_normalized_split", wraps=prepare.rebuild_normalized_split) as rebuild:
            _, report = self.normalize()
        self.assertNotIn(("ui_alignment", "train"), [(c.args[2].task_key, c.args[3]) for c in rebuild.call_args_list])
        self.assertEqual((report["normalization_resume"]["reused_splits"], report["normalization_resume"]["rebuilt_splits"]), (1, 17))

    def test_output_digest_mismatch_rebuilds_only_that_split(self):
        self.normalize()
        path = paths_for(self.data, "synth_radius", "test")["detector_input"]
        path.write_text("{}\n", encoding="utf-8")
        with mock.patch.object(prepare, "rebuild_normalized_split", wraps=prepare.rebuild_normalized_split) as rebuild:
            _, report = self.normalize()
        self.assertEqual([(c.args[2].task_key, c.args[3]) for c in rebuild.call_args_list], [("synth_radius", "test")])
        self.assertEqual(report["normalization_resume"]["reused_splits"], 17)
        validate_normalization(self.data)

    def test_changed_source_snapshot_never_reuses_stale_normalization_ids(self):
        self.normalize()
        previous = read_json(self.data / "source_snapshot.json")["normalization_id"]
        path = self.source / "synth_radius/test.jsonl"
        raw = next(read_jsonl(path)); raw["source_revision"] = "changed source"
        write_jsonl(path, [raw])
        _, report = self.normalize()
        self.assertNotEqual(report["normalization_id"], previous)
        self.assertEqual(report["normalization_resume"]["reused_splits"], 0)
        validate_normalization(self.data)

    def test_global_page_failure_is_recomputed_even_when_all_splits_are_reused(self):
        path = self.source / "synth_radius/test.jsonl"
        raw = next(read_jsonl(path)); raw["FigmaNodeID"] = "train"
        write_jsonl(path, [raw])
        with self.assertRaisesRegex(RuntimeError, "intake failed"):
            self.normalize()
        with self.no_images(), self.assertRaisesRegex(RuntimeError, "intake failed"):
            self.normalize()
        report = read_json(self.data / "cpu_check_report.json")
        self.assertEqual(report["normalization_resume"]["reused_splits"], 18)
        self.assertGreater(report["ui9_page_split"]["synthetic_train_test_page_count"], 0)
        self.assertFalse(report["normalization_complete"])

    def test_orientation_2_to_8_keep_existing_rejection_policy(self):
        path = self.source / "ui_alignment/sample_imgs/figma-train.png"
        with Image.open(path) as image:
            pixels = image.convert("RGB")
        for orientation in range(2, 9):
            with self.subTest(orientation=orientation):
                exif = Image.Exif(); exif[274] = orientation
                pixels.save(path, exif=exif)
                with self.assertRaisesRegex(RuntimeError, "intake failed"):
                    self.normalize()
                report = read_json(self.data / "cpu_check_report.json")
                self.assertEqual(report["tasks"]["ui_alignment/train"]["failed_records"], 1)
                self.assertTrue(any("EXIF orientation" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
