from __future__ import annotations

import json
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_ui_lens_app_generalization import (
    load_training_apps,
    path_mentions_app,
    prediction_audit,
    main,
    resumable_audit,
    ui_lens_rows,
    write_partitions,
)
from locany_ui5_common import TASK_ISSUE_NAMES, TASK_JSONL


class AppGeneralizationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "eval"
        self.data.mkdir()
        self.checkpoint = self.root / "checkpoint-9000"
        self.checkpoint.mkdir()
        self.image_dir = self.root / "images"
        self.image_dir.mkdir()
        self.rows = []
        for index, app in enumerate(("DouYin", "WeChat"), 1):
            image = self.image_dir / f"{index}.png"
            image.write_bytes(b"x")
            self.rows.append({"id": index, "images": [str(image)],
                              "answer": {"bbox": [[1, 2, 3, 4]] if index == 1 else [], "types": []},
                              "extra_info": {"original_infos": {"app_name": app}}})
        for filename in TASK_JSONL.values():
            Path(self.data / filename).write_text(
                "".join(json.dumps(row) + "\n" for row in self.rows), encoding="utf-8")

    def test_training_apps_use_metadata_then_exact_path_token(self):
        training = self.root / "train.jsonl"
        training.write_text(
            json.dumps({"infos": {"app_name": "DouYin"}, "images": ["/x/one.png"]}) + "\n" +
            json.dumps({"image": "/datasets/wechat/two.png"}) + "\n", encoding="utf-8")
        apps, evidence = load_training_apps([], None, [training], ["douyin", "wechat"])
        self.assertEqual(apps, {"douyin", "wechat"})
        self.assertEqual(evidence["metadata"], {"douyin": 1})
        self.assertEqual(evidence["path_token"], {"wechat": 1})
        self.assertIsNone(path_mentions_app("/x/douyin-wechat/a.png", ["douyin", "wechat"]))

    def test_partition_is_disjoint_and_preserves_records(self):
        records, inventory = ui_lens_rows(self.data)
        self.assertEqual(inventory, {"douyin": 5, "wechat": 5})
        output = self.root / "out"
        stats = write_partitions(records, output, {"douyin"})
        self.assertEqual(stats["train_source_apps"]["image_task_records"], 5)
        self.assertEqual(stats["other_apps"]["image_task_records"], 5)
        for task, filename in TASK_JSONL.items():
            seen = [json.loads(x) for x in (output / "subsets/train_source_apps" / filename).read_text().splitlines()]
            unseen = [json.loads(x) for x in (output / "subsets/other_apps" / filename).read_text().splitlines()]
            self.assertEqual([x["id"] for x in seen], [1])
            self.assertEqual([x["id"] for x in unseen], [2])
            self.assertNotIn("_normalized_app_name", seen[0])

    def test_prediction_audit_checks_every_task_and_manifest_identity(self):
        records, _ = ui_lens_rows(self.data)
        pred = self.root / "predictions"
        pred.mkdir()
        manifest = {"checkpoint": str(self.checkpoint), "inference_crop": {"mode": "detector_scan"}}
        (pred / "_run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        for task in TASK_JSONL:
            task_dir = pred / task
            task_dir.mkdir()
            (task_dir / "1.json").write_text("[]", encoding="utf-8")
            (task_dir / "2.json").write_text("[]", encoding="utf-8")
        audit = prediction_audit(pred, records, self.checkpoint, "detector_scan")
        self.assertTrue(audit["usable"])
        (pred / "cropping/2.json").unlink()
        audit = prediction_audit(pred, records, self.checkpoint, "detector_scan")
        self.assertFalse(audit["usable"])
        self.assertEqual(audit["tasks"]["cropping"]["missing"], 1)

    def test_parse_error_is_not_reused_as_formal_result(self):
        records, _ = ui_lens_rows(self.data)
        pred = self.root / "predictions"
        pred.mkdir()
        (pred / "_run_manifest.json").write_text(json.dumps({
            "checkpoint": str(self.checkpoint), "inference_crop": {"mode": "detector_scan"}}), encoding="utf-8")
        for task in TASK_JSONL:
            task_dir = pred / task; task_dir.mkdir()
            (task_dir / "1.json").write_text("[]", encoding="utf-8")
            (task_dir / "2.json").write_text("[]", encoding="utf-8")
        (pred / "occlusion/1.json").unlink()
        (pred / "occlusion/1_parse_error.json").write_text("[]", encoding="utf-8")
        audit = prediction_audit(pred, records, self.checkpoint, "detector_scan")
        self.assertFalse(audit["usable"])
        self.assertEqual(audit["tasks"]["occlusion"]["invalid"], 1)
        self.assertFalse(resumable_audit(audit))

    def test_end_to_end_reuses_predictions_and_writes_both_and_per_app_metrics(self):
        for task, filename in TASK_JSONL.items():
            rows = []
            for index, app in enumerate(("DouYin", "WeChat"), 1):
                boxes = [[1, 2, 3, 4]] if index == 1 else []
                rows.append({"id": index, "images": [str(self.image_dir / f"{index}.png")],
                             "answer": {"bbox": boxes, "types": [TASK_ISSUE_NAMES[task]] * len(boxes)},
                             "extra_info": {"original_infos": {"app_name": app}}})
            (self.data / filename).write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        pred = self.root / "predictions"; pred.mkdir()
        (pred / "_run_manifest.json").write_text(json.dumps({
            "checkpoint": str(self.checkpoint), "inference_crop": {"mode": "detector_scan"}}), encoding="utf-8")
        for task in TASK_JSONL:
            task_dir = pred / task; task_dir.mkdir()
            (task_dir / "1.json").write_text(json.dumps([
                {"bbox_2d": [1, 2, 3, 4], "label": TASK_ISSUE_NAMES[task]}]), encoding="utf-8")
            (task_dir / "2.json").write_text("[]", encoding="utf-8")
        output = self.root / "result"
        project = Path(__file__).resolve().parents[1]
        fake_metrics = {"macro": {"image": {"f1": 1.0}, "bbox": {"f1": 1.0}}, "tasks": {}}
        with contextlib.redirect_stdout(io.StringIO()), mock.patch(
            "evaluate_ui_lens_app_generalization.score_subset", return_value=fake_metrics
        ) as score:
            code = main(["--project-root", str(project), "--checkpoint", str(self.checkpoint),
                         "--eval-dir", str(self.data), "--prediction-dir", str(pred),
                         "--output-dir", str(output), "--train-app-name", "douyin",
                         "--no-run-if-missing"])
        self.assertEqual(code, 0)
        summary = json.loads((output / "app_generalization_summary.json").read_text(encoding="utf-8"))
        self.assertTrue(summary["prediction_reused"])
        self.assertEqual(summary["training_source_apps"], ["douyin"])
        self.assertEqual(set(summary["per_app"]), {"douyin", "wechat"})
        self.assertEqual(summary["metrics"]["train_source_apps"]["macro"]["image"]["f1"], 1.0)
        self.assertEqual(score.call_count, 4)
        self.assertTrue((output / "app_generalization_report.md").is_file())


if __name__ == "__main__":
    unittest.main()
