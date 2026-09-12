from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np


def load_scorer():
    scipy = types.ModuleType("scipy")
    optimize = types.ModuleType("scipy.optimize")
    optimize.linear_sum_assignment = lambda matrix: (
        np.arange(min(matrix.shape), dtype=int), np.arange(min(matrix.shape), dtype=int)
    )
    scipy.optimize = optimize
    path = Path(__file__).resolve().parents[1] / "qwen3vl_merge_and_score_fixed_5tasks.py"
    spec = importlib.util.spec_from_file_location("qwen_scorer_parse_error_test", path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"scipy": scipy, "scipy.optimize": optimize}):
        spec.loader.exec_module(module)
    return module


class ParseErrorScoringTest(unittest.TestCase):
    def test_locateanything_parse_error_is_invalid_not_empty_prediction(self):
        scorer = load_scorer()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gt = root / "gt.jsonl"
            pred = root / "pred"; pred.mkdir()
            merged = root / "merged.jsonl"
            gt.write_text(json.dumps({
                "images": [str(root / "1.png")],
                "answer": {"bbox": [[1, 2, 3, 4]], "types": ["元素重叠"]},
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            (pred / "1_parse_error.json").write_text("[]\n", encoding="utf-8")
            merge_stats = scorer.merge_gt_and_yolo_dir_preds(
                str(gt), str(pred), str(merged), "xyxy", "元素重叠"
            )
            row = json.loads(merged.read_text(encoding="utf-8"))
            self.assertIsNone(row["pred_ans"])
            self.assertEqual(row["pred_parse_info"]["parse_status"], "model_output_parse_error")
            self.assertEqual(merge_stats["parse_errors"], 1)
            metrics = scorer.evaluate_merged_file(str(merged), "元素重叠", 0.1, False)
            self.assertEqual(metrics["invalid_pred"], 1)
            self.assertEqual(metrics["img_fn"], 1)
            self.assertEqual(metrics["fn"], 1)

    def test_all_task_json_keeps_invalid_count(self):
        scorer = load_scorer()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "all_tasks_evaluation.txt"
            task = {
                "bbox": {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                         "tp": 0, "fp": 0, "fn": 1, "count_accuracy": 0.0},
                "image": {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                          "tp": 0, "fp": 0, "fn": 1, "tn": 0, "accuracy": 0.0},
                "total_samples": 1,
                "invalid_pred": 1,
            }
            scorer.write_all_tasks_summary({"occlusion": task}, str(output))
            saved = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(saved["tasks"]["occlusion"]["invalid_pred"], 1)
            self.assertEqual(saved["tasks"]["occlusion"]["total_samples"], 1)


if __name__ == "__main__":
    unittest.main()
