"""Repaired UI9 CPU contracts; all image/data fixtures are synthetic, no GPU."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from tests.test_ui14_pipeline import source_fixture, detector_fixture
from ui14_common import UI_TASKS, UI9_TASKS, SCAN_NAME, paths_for, read_json, read_jsonl, write_json, write_jsonl, file_digest
from ui14_annotations import source_boxes, source_box_details, main_image, crop_boxes, norm1000
from ui9_source_parser import gt_boxes, location_boxes, primary_image
from ui14_repair import (parser_provenance, format_counts, legacy_comparison,
                        capture_repair_snapshot, validate_normalization, cache_label_binding)
import prepare_ui14_sft as prepare
from PIL import Image


class UI9ParserCompatibilityTests(unittest.TestCase):
    def test_extracted_cpu_parser_matches_preparation_ast_provenance(self):
        info = parser_provenance()
        self.assertEqual(info["source_version"], "2.1")
        self.assertEqual(info["source_sha256"], "72a43e551c4388fdeaef985a9206105da400e4351bc817ef4d7cdedbce3c084e")
        self.assertEqual(len(info["symbols"]), 14)

    def test_numbered_priority_case_insensitivity_and_multiple_boxes(self):
        err, other = [10, 20, 30, 40], [40, 50, 60, 70]
        self.assertEqual(location_boxes({"rect_err_1": err, "rect1": other}), [(err, "Location.rect_err_1")])
        self.assertEqual([b for b, _ in location_boxes({"RECT_ERR1": err, "rect_err_2": other, "rect_mbr": err})], [err, other])
        for higher, lower in (("rect_mbr_2", "rect1_shift"), ("rect1_shift", "rect_combine_2"), ("rect_combine_2", "rect1")):
            with self.subTest(higher=higher):
                self.assertEqual([b for b, _ in location_boxes({higher: err, lower: other})], [err])
        self.assertEqual([b for b, _ in location_boxes({"rect_err_1": [], "rect1": other})], [other])

    def test_direct_two_point_single_object_and_wrapped_coordinates(self):
        xyxy = [10, 20, 30, 40]
        for location in ([[10, 20], [30, 40]], xyxy, {"bbox": [[10, 20], [30, 40]]},
                         {"xmin": 10, "ymin": 20, "xmax": 30, "ymax": 40},
                         {"x": 10, "y": 20, "width": 20, "height": 20},
                         json.dumps({"rect_err_1": [[10, 20], [30, 40]], "rect1": [50, 60, 70, 80]})):
            for objects in ({"Location": location}, [{"Location": location}]):
                with self.subTest(location=location, objects_type=type(objects).__name__):
                    raw = {"Objects": objects, "BBoxCanvasWidth": 375}
                    self.assertEqual(source_boxes(raw, 750, 1600, synthetic=True), [[20, 40, 60, 80]])
                    self.assertEqual([b for b, _ in gt_boxes(raw, "synthetic")], [xyxy])
        self.assertEqual(source_boxes({"objects": {"bbox": [[10, 20], [30, 40]], "bbox_type": "real"}},
                                      750, 1600, synthetic=False), [xyxy])

    def test_declared_annotated_scales_follow_preparation_without_heuristics(self):
        cases = [("real", [10, 20, 30, 40], {}, [[10, 20, 30, 40]]),
                 ("norm1", [.1, .2, .3, .4], {}, [[75, 320, 225, 640]]),
                 ("norm1000", [100, 200, 300, 400], {}, [[75, 320, 225, 640]]),
                 ("real", [10, 20, 30, 40], {"bbox": {"mode": "canvas", "width": 375, "height": 800}}, [[20, 40, 60, 80]]),
                 ("real", [10, 20, 30, 40], {"bbox": {"mode": "scale", "sx": 3, "sy": 4}}, [[30, 80, 90, 160]])]
        for mode, box, config, expected in cases:
            with self.subTest(mode=mode, config=config):
                self.assertEqual(source_boxes({"objects": {"bbox": box, "bbox_type": mode}}, 750, 1600,
                                              synthetic=False, task_config=config), expected)
        raw = {"RawImgURL": "reference", "objects": {"bbox": [10, 20, 30, 40], "bbox_type": "real"}}
        self.assertEqual(source_boxes(raw, 750, 1600, synthetic=False,
                         task_config={"bbox": {"mode": "raw-export", "export_scale": 2}},
                         images={"reference": {"ok": True, "width": 375}}), [[40, 80, 120, 160]])
        with self.assertRaisesRegex(ValueError, "坐标尺度"):
            source_boxes({"objects": {"bbox": [1, 2, 3, 4], "bbox_type": "normalized"}}, 750, 1600, synthetic=False)

    def test_repaired_logical_coordinates_are_projected_once_then_crop_offset(self):
        raw = {"BBoxCanvasWidth": 375, "Objects": {"Location": {"rect_err_1": [[183, 449], [269, 479]]}}}
        details = source_box_details(raw, 941, 2048, synthetic=True)
        self.assertEqual(details["scale_xy"], [941/375, 941/375])
        local, contained = crop_boxes(details["boxes_px"], [0, 1000, 941, 1500])
        self.assertEqual(contained, [0])
        self.assertEqual(norm1000(details["boxes_px"][0], 941, 2048), [488, 550, 717, 587])
        self.assertAlmostEqual(local[0][1] + 1000, 449*941/375)
        with self.assertRaisesRegex(ValueError, "outside screenshot"):
            source_boxes({"Objects": [{"Location": [1, 2, 400, 500]}]}, 750, 1600, synthetic=True)
        with self.assertRaisesRegex(ValueError, "375"):
            source_boxes({**raw, "BBoxCanvasWidth": 750}, 941, 2048, synthetic=True)

    def test_main_image_roles_nested_storage_and_explicit_negative(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("sample.png", "normal.png", "raw.png"):
                Image.new("RGB", (375, 800)).save(root / name)
            record = {"ScreenShotURL": {"url": str(root / "sample.png")}, "LocalImgURL": str(root / "normal.png"),
                      "RawImgURL": str(root / "raw.png"), "Objects": []}
            self.assertEqual(main_image(record, root, True), root / "sample.png")
            del record["ScreenShotURL"]
            self.assertEqual(main_image(record, root, True), root / "normal.png")
            record["Objects"] = [{"Location": [1, 2, 3, 4]}]
            with self.assertRaises(ValueError): main_image(record, root, True)
            annotated = {"messages": [{"content": [{"image_url": {"url": str(root / "sample.png")}}]}]}
            self.assertEqual(main_image(annotated, root, False), root / "sample.png")
            with self.assertRaises(ValueError):
                primary_image({"images": [str(root / "sample.png"), str(root / "normal.png")]}, "annotated")
        for raw in ({}, {"Objects": [{}]}, {"Objects": None}, {"Objects": "[]"}):
            with self.assertRaises(ValueError): gt_boxes(raw, "synthetic")

    def test_audit_distinguishes_formats_failures_and_successful_parse_differences(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); image = root / "page.png"; Image.new("RGB", (750, 1600)).save(image)
            raw = {"ScreenShotURL": str(image), "BBoxCanvasWidth": 375, "split": "test",
                   "Objects": [{"Location": {"rect_err_1": [10, 20, 30, 40], "rect1": [40, 50, 60, 70]}}]}
            current = {"source_image": str(image), "boxes_px": [[20, 40, 60, 80]]}
            counts, detail = legacy_comparison(raw, root, "train", True, current, {})
            self.assertEqual(counts["legacy_parse_failure_records"], 0)
            self.assertEqual(counts["legacy_split_rejection_records"], 1)
            self.assertEqual(counts["legacy_consumer_failure_records"], 1)
            self.assertEqual(counts["parse_result_difference_records"], 1)
            self.assertEqual(detail["legacy_boxes_px"], [[80, 100, 120, 140]])
            self.assertEqual(format_counts(raw, True)["numbered_rect_err_with_rectN_records"], 1)
            for objects in ([{"Location": {"rect1": [[10, 20], [30, 40]]}}],
                            {"Location": [10, 20, 30, 40]}, [{"Location": {"bbox": [10, 20, 30, 40]}}]):
                raw["Objects"] = objects
                counts, _ = legacy_comparison(raw, root, "train", True, current, {})
                self.assertEqual(counts["legacy_parse_failure_records"], 1)
                self.assertEqual(counts["parse_result_difference_records"], 0)


class RepairedIntakeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.data = self.root / "data"
        source_fixture(self.source)
        self.args = SimpleNamespace(ui9_data_root=self.source, output_dir=self.data,
                                   ui5_recipe="old_recipe.json", ui5_test_dir=self.root / "old_test")

    def normalize(self):
        with contextlib.redirect_stdout(io.StringIO()): prepare.normalize(self.args)

    def test_repair_counts_split_metadata_page_groups_and_quarantine_exclusion(self):
        write_jsonl(self.source / "repair_backups" / "fake" / "quarantine.jsonl", [{"INVALID": "never read"}])
        self.normalize()
        report = read_json(self.data / "cpu_check_report.json")
        self.assertTrue(report["normalization_complete"])
        self.assertFalse(report["ready"])  # GPU cache and final CPU checks still required
        self.assertEqual(len(report["source_files"]), 20)
        self.assertEqual(sum(s["records"] for s in report["tasks"].values()), 18)
        self.assertEqual(report["parser_comparison"]["legacy_split_rejection_records"], 14)
        self.assertEqual(report["parser_comparison"]["legacy_parse_failure_records"], 0)
        self.assertEqual(report["ui9_page_split"]["synthetic_train_test_page_count"], 0)
        self.assertEqual(report["ui9_page_split"]["cross_source_pages"], 2)
        self.assertEqual(report["ui9_train_test_duplicate_images"], 1)
        row = next(read_jsonl(paths_for(self.data, "synth_cropping", "train")["normalized"]))
        self.assertEqual((row["split"], row["source_split"], row["source_metadata"]["split"]), ("train", "test", "test"))
        self.assertEqual(row["repair_run_id"], "fixture-repair-v2.1")
        validate_normalization(self.data)

    def test_reject_partial_publication_mismatched_batch_and_counts(self):
        manifest_path = self.source / "manifest.json"
        original = read_json(manifest_path)
        write_json(self.source / ".work/repair_pending.json", {"run_id": "in progress"})
        with self.assertRaisesRegex(ValueError, "pending"): capture_repair_snapshot(self.source)
        (self.source / ".work/repair_pending.json").unlink()
        write_json(manifest_path, {**original, "repair_history": [{"run_id": "different"}]})
        with self.assertRaisesRegex(ValueError, "run_id"): self.normalize()
        write_json(manifest_path, original)
        test = self.source / "synth_radius/test.jsonl"
        write_jsonl(test, list(read_jsonl(test)) * 2)
        with self.assertRaisesRegex(RuntimeError, "intake failed"): self.normalize()
        self.assertFalse(read_json(self.data / "cpu_check_report.json")["normalization_complete"])

    def test_reject_cross_source_synthetic_page_leak_without_resplitting(self):
        path = self.source / "synth_radius/test.jsonl"
        record = next(read_jsonl(path)); record["FigmaNodeID"] = "train"
        write_jsonl(path, [record]); before = file_digest(path)
        with self.assertRaisesRegex(RuntimeError, "intake failed"): self.normalize()
        self.assertEqual(file_digest(path), before)
        report = read_json(self.data / "cpu_check_report.json")
        self.assertEqual(report["ui9_page_split"]["synthetic_train_test_page_count"], 1)

    def test_gt_repair_relabels_unchanged_detector_plan_and_replaces_completion(self):
        self.normalize()
        task = UI_TASKS[7]; p = paths_for(self.data, task.task_key, "train")
        with contextlib.redirect_stdout(io.StringIO()):
            detector_fixture(self.data, task, "train")
            before = prepare.crop_annotations(self.data, task, "train", list(read_jsonl(p["normalized"])))
        plan = p["cache"] / SCAN_NAME / "detector_scan_crops.jsonl"
        plan_digest = file_digest(plan)
        old_binding = read_json(p["cache"] / "ui14_label_cache_ready.json")
        path = self.source / task.task_key / "train.jsonl"
        raw = next(read_jsonl(path)); raw["Objects"][0]["Location"] = {"rect_err_1": [[15, 45], [90, 75]]}
        write_jsonl(path, [raw])
        with self.assertRaisesRegex(ValueError, "source changed"): validate_normalization(self.data)
        for name in ("manifest.json", "repair_summary.json"):
            value = read_json(self.source / name)
            if name == "manifest.json": value["repair_history"].append({"run_id": "fixture-repair-next"})
            else: value["run_id"] = "fixture-repair-next"
            write_json(self.source / name, value)
        self.normalize()
        self.assertNotEqual(old_binding, cache_label_binding(self.data, task, "train"))
        after = prepare.crop_annotations(self.data, task, "train", list(read_jsonl(p["normalized"])))
        self.assertEqual(file_digest(plan), plan_digest)
        self.assertNotEqual(before[0]["conversations"], after[0]["conversations"])
        self.assertEqual(after[0]["repair_run_id"], "fixture-repair-next")
        self.assertEqual(read_json(p["cache"] / "ui14_label_cache_ready.json"), cache_label_binding(self.data, task, "train"))
        # A split's image/path selection still cannot reuse an old index.
        write_jsonl(p["detector_input"], [{"image": "changed-page.png"}])
        with self.assertRaises(RuntimeError): prepare.validate_task_cache(self.data, task, "train", 1)

    def test_normalized_annotations_cannot_be_changed_after_snapshot(self):
        self.normalize()
        path = paths_for(self.data, "ui_alignment", "train")["normalized"]
        value = next(read_jsonl(path)); value["boxes_px"] = []
        write_jsonl(path, [value])
        with self.assertRaisesRegex(ValueError, "Normalized artifact changed"): validate_normalization(self.data)

    def test_sft_resume_cannot_cross_repair_batches(self):
        from ui14_profile import validate_run_data_binding
        runtime = {"OUTPUT_DIR": str(self.root / "run"), "UI14_DATA_ROOT": str(self.data)}
        snapshot = {"normalization_id": "first", "repair_run_id": "run1"}
        validate_run_data_binding(runtime, snapshot, create=True)
        validate_run_data_binding(runtime, snapshot)
        with self.assertRaisesRegex(RuntimeError, "another repair batch"):
            validate_run_data_binding(runtime, {"normalization_id": "second", "repair_run_id": "run2"})
        runtime["OUTPUT_DIR"] = str(self.root / "unbound")
        (Path(runtime["OUTPUT_DIR"]) / "checkpoint-1000").mkdir(parents=True)
        with self.assertRaisesRegex(RuntimeError, "no repair data binding"):
            validate_run_data_binding(runtime, snapshot, create=True)

    def test_finalization_failure_keeps_real_intake_statistics_in_cpu_report(self):
        self.normalize()
        self.args.stage = "finalize"
        with self.assertRaises(OSError): prepare.run_stage(self.args)  # audited UI5 recipe unavailable in fixture
        report = read_json(self.data / "cpu_check_report.json")
        self.assertFalse(report["ready"])
        self.assertEqual(report["repair_run_id"], "fixture-repair-v2.1")
        self.assertEqual(len(report["source_files"]), 20)
        self.assertEqual(sum(s["records"] for s in report["tasks"].values()), 18)
        self.assertTrue(report["errors"])


if __name__ == "__main__": unittest.main()
