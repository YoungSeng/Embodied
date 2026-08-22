from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from openpyxl import load_workbook

from eaglevl.train.ui5_excel_logger import (
    EXPECTED_SHEETS,
    UI5ExcelLogger,
    build_eval_rows,
)


def training_metrics(step: int) -> dict:
    return {
        "epoch": step / 1000,
        "gpu_num": 4,
        "max_num_tokens": 12800,
        "learning_rate": 2e-5,
        "loss_total": 1.0,
        "tasks": {},
    }


def scorer_metrics(value: float = 0.5) -> dict:
    tasks = {}
    for task in (
        "text_overflow",
        "text_ellipsis",
        "occlusion",
        "cropping",
        "content_missing",
    ):
        tasks[task] = {
            "image": {
                "precision": value,
                "recall": value,
                "f1": value,
                "tp": 2,
                "fp": 1,
                "fn": 1,
                "tn": 10,
                "accuracy": 0.8,
            },
            "bbox": {
                "precision": value,
                "recall": value,
                "f1": value,
                "tp": 2,
                "fp": 1,
                "fn": 1,
                "count_accuracy": 0.7,
            },
        }
    return {
        "tasks": tasks,
        "macro": {
            "image": {"precision": value, "recall": value, "f1": value},
            "bbox": {"precision": value, "recall": value, "f1": value},
        },
    }


class UI5ExcelLoggerTest(unittest.TestCase):
    def test_train_resume_deduplicates_and_keeps_exactly_two_sheets(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "diagnostics" / "ui5_training_evaluation.xlsx"
            logger = UI5ExcelLogger(path)
            self.assertTrue(logger.update_train(100, training_metrics(100)))
            self.assertTrue(logger.update_train(200, training_metrics(200)))
            self.assertFalse(logger.update_train(100, training_metrics(100)))
            # A run ending at 250 writes no partial window; resume adds only 300.
            self.assertTrue(logger.update_train(300, training_metrics(300)))
            workbook = load_workbook(path, read_only=True)
            try:
                self.assertEqual(tuple(workbook.sheetnames), EXPECTED_SHEETS)
                steps = [
                    row[0]
                    for row in workbook["train_100steps"].iter_rows(
                        min_row=2, values_only=True
                    )
                ]
                self.assertEqual(steps, [100, 200, 300])
                self.assertFalse(
                    any(
                        isinstance(cell.value, str) and cell.value.startswith("=")
                        for sheet in workbook.worksheets
                        for row in sheet.iter_rows()
                        for cell in row
                    )
                )
            finally:
                workbook.close()

    def test_eval_has_five_tasks_and_two_macro_rows_per_step(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ui5_training_evaluation.xlsx"
            logger = UI5ExcelLogger(path)
            for step, value in ((0, 0.4), (1000, 0.5)):
                rows = build_eval_rows(
                    step=step,
                    checkpoint=f"checkpoint-{step}",
                    metrics=scorer_metrics(value),
                )
                self.assertTrue(logger.append_eval(step, rows))
            self.assertFalse(
                logger.append_eval(
                    1000,
                    build_eval_rows(
                        step=1000,
                        checkpoint="checkpoint-1000",
                        metrics=scorer_metrics(0.5),
                    ),
                )
            )
            workbook = load_workbook(path, read_only=True)
            try:
                rows = list(
                    workbook["eval_1000steps"].iter_rows(
                        min_row=2, values_only=True
                    )
                )
                self.assertEqual(len(rows), 24)
                self.assertEqual(sum(row[0] == 0 for row in rows), 12)
                self.assertEqual(sum(row[0] == 1000 for row in rows), 12)
                f1_change_index = 18
                self.assertTrue(
                    all(
                        abs(row[f1_change_index] - 0.1) < 1e-9
                        for row in rows
                        if row[0] == 1000
                    )
                )
            finally:
                workbook.close()

    def test_failed_atomic_replace_leaves_original_workbook_readable(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ui5_training_evaluation.xlsx"
            logger = UI5ExcelLogger(path)
            logger.update_train(100, training_metrics(100))
            with mock.patch(
                "eaglevl.train.ui5_excel_logger.os.replace",
                side_effect=OSError("simulated interruption"),
            ):
                with self.assertRaises(OSError):
                    logger.update_train(200, training_metrics(200))
            workbook = load_workbook(path, read_only=True)
            try:
                steps = [
                    row[0]
                    for row in workbook["train_100steps"].iter_rows(
                        min_row=2, values_only=True
                    )
                ]
                self.assertEqual(steps, [100])
            finally:
                workbook.close()


if __name__ == "__main__":
    unittest.main()
