import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from eaglevl.train.cpt_excel import build_cpt_workbook


class CPTExcelTest(unittest.TestCase):
    def test_exactly_three_sheets_and_missing_values_remain_blank(self):
        from openpyxl import load_workbook

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "split_summary.json").write_text(
                json.dumps(
                    {
                        "group_intersection": 0,
                        "tasks": {
                            "vqa": {
                                "train_rows": 98,
                                "val_rows": 2,
                                "val_fast_rows": 2,
                                "train_groups": 90,
                                "val_groups": 2,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            (root / "cpt_eval_metrics.jsonl").write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        {
                            "checkpoint": "checkpoint-100",
                            "step": 100,
                            "split": "heldout",
                            "task": "vqa",
                            "primary_metric": 0.9,
                            "eval_token_ce": 1.2,
                        },
                        {
                            "checkpoint": "checkpoint-200",
                            "step": 200,
                            "split": "heldout",
                            "task": "vqa",
                            "primary_metric": 0.8,
                            "eval_token_ce": 1.0,
                        },
                        {
                            "checkpoint": "checkpoint-200",
                            "step": 200,
                            "split": "heldout",
                            "task": "__task_macro__",
                            "primary_metric": 0.8,
                        },
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "metrics.xlsx"
            self.assertTrue(build_cpt_workbook(root, output))
            workbook = load_workbook(output, data_only=False)
            self.assertEqual(workbook.sheetnames, ["Overview", "TrainMetrics", "EvalMetrics"])
            overview = workbook["Overview"]
            self.assertEqual(overview.max_row, 2)
            headers = {cell.value: cell.column for cell in overview[1]}
            self.assertEqual(overview.cell(2, headers["task"]).value, "vqa")
            self.assertEqual(overview.cell(2, headers["primary_metric"]).value, 0.8)
            self.assertEqual(overview.cell(2, headers["best_step"]).value, 100)
            for sheet in workbook.worksheets:
                self.assertEqual(sheet.freeze_panes, "A2")
                self.assertIsNotNone(sheet.auto_filter.ref)

    def test_excel_import_failure_is_only_a_warning(self):
        with tempfile.TemporaryDirectory() as directory:
            original_import = __import__

            def rejecting_import(name, *args, **kwargs):
                if name == "openpyxl" or name.startswith("openpyxl."):
                    raise ImportError("intentional test failure")
                return original_import(name, *args, **kwargs)

            with mock.patch("builtins.__import__", side_effect=rejecting_import):
                with self.assertWarns(UserWarning):
                    self.assertFalse(build_cpt_workbook(directory))


if __name__ == "__main__":
    unittest.main()
