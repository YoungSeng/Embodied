"""CPU-only completion, persistence and identity regressions for interrupted evaluation."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ui14_incremental_eval import run_incremental_inference
from ui14_common import UI_TASKS, write_json
from eaglevl.train.ui5_excel_logger import UI5ExcelLogger, build_eval_rows, build_task_eval_rows
from openpyxl import load_workbook


def metric():
    return {"image": dict(tp=2, fp=1, fn=1, tn=3, precision=2/3, recall=2/3, f1=2/3),
            "bbox": dict(tp=2, fp=1, fn=1, precision=2/3, recall=2/3, f1=2/3),
            "view_policy": "crops"}


def task_rows(task, *, step=1000, data="data-v1"):
    return build_task_eval_rows(task_key=task, step=step, checkpoint="checkpoint-1000",
        metrics={"tasks": {task: metric()}},
        metadata={"eval_set_id": data, "data_digest": data, "model_signature": "model-v1"})


class IncrementalEvalTests(unittest.TestCase):
    def test_old_rendered_environment_cannot_remove_inner_margin_exclusivity(self):
        from locany_ui5_common import ui14_exclusive_gpu_tasks
        self.assertEqual(ui14_exclusive_gpu_tasks("synth_loneword change_line_illegal_v3"),
                         ["synth_loneword", "change_line_illegal_v3", "synth_inner_margin"])

    def test_excel_is_visible_before_other_process_fails_and_late_success_is_saved(self):
        # This real CPU subprocess waits for Excel publication before it emits an OOM event.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            book = UI5ExcelLogger(root / "metrics.xlsx", [t.task_key for t in UI_TASKS])
            book.initialize()
            worker = root / "fake_scheduler.py"
            worker.write_text("""
import argparse,json,os,time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--completion-dir',type=Path)
p.add_argument('--ack',type=Path)
a=p.parse_args()
def event(task,code):
    f=a.completion_dir/(task+'.json')
    tmp=f.with_suffix('.tmp')
    tmp.write_text(json.dumps({'task':task,'return_code':code}))
    os.replace(tmp,f)
event('ui_alignment',0)
deadline=time.monotonic()+20
while not a.ack.exists():
    if time.monotonic()>deadline: raise RuntimeError('Excel was not published while inference was running')
    time.sleep(.02)
event('synth_inner_margin',1)
time.sleep(.3)
event('synth_small_margin',0)
raise SystemExit(1)
""", encoding="utf-8")
            events = []
            def completed(event):
                events.append(event)
                if event["return_code"]: return
                book.append_eval_tasks(1000, task_rows(event["task"]))
                current = load_workbook(book.path)
                try:
                    self.assertEqual(current["eval_1000steps"].max_row, 1 + 2 * sum(e["return_code"] == 0 for e in events))
                finally: current.close()
                self.assertFalse(book.has_eval_step(1000))
                (root / "ack").touch()
            with self.assertRaises(subprocess.CalledProcessError):
                run_incremental_inference([sys.executable, str(worker), "--ack", str(root/"ack")],
                    cwd=root, completion_dir=root/"events", on_complete=completed)
            self.assertEqual({e["task"] for e in events}, {"ui_alignment", "synth_inner_margin", "synth_small_margin"})
            book.mark_eval_failed(1000)
            current = load_workbook(book.path)
            try: self.assertEqual(current["eval_1000steps"].max_row, 5)
            finally: current.close()

    def test_partial_upsert_preserves_train_counts_and_requires_all_36_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            book = UI5ExcelLogger(Path(tmp)/"metrics.xlsx", [t.task_key for t in UI_TASKS])
            book.update_train(100, {"loss_total": .9})
            for task in UI_TASKS:
                for _ in range(2): book.append_eval_tasks(1000, task_rows(task.task_key))
            self.assertFalse(book.has_eval_step(1000))
            current = load_workbook(book.path)
            try:
                self.assertEqual(current["train_100steps"].max_row, 2)
                sheet = current["eval_1000steps"]
                self.assertEqual(sheet.max_row, 29)
                columns = [c.value for c in sheet[1]]
                for row in sheet.iter_rows(min_row=2, values_only=True):
                    value = dict(zip(columns, row))
                    self.assertEqual(value["view_policy"], "crops")
                    self.assertEqual(value["tp"], 2)
                    self.assertEqual(value["tn"], 3 if value["granularity"] == "image" else None)
            finally: current.close()
            with self.assertRaisesRegex(ValueError, "Different evaluation data"):
                book.append_eval_tasks(1000, task_rows("synth_inner_margin", data="other"))
            rows = build_eval_rows(step=1000, checkpoint="checkpoint-1000",
                metrics={"tasks": {t.task_key: metric() for t in UI_TASKS}},
                metadata={"eval_set_id":"data-v1", "data_digest":"data-v1", "model_signature":"model-v1"})
            book.append_eval(1000, rows)
            self.assertTrue(book.has_eval_step(1000))
            before = book.path.read_bytes()
            self.assertFalse(book.append_eval_tasks(1000, task_rows("ui_alignment")))
            self.assertEqual(book.path.read_bytes(), before)

    def test_scheduler_publishes_receipt_before_round_finishes(self):
        import run_ui5_parallel_inference as parallel
        from tests.test_ui14_inference_workers import fixture
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv = fixture(root) + ["--completion-dir", str(root/"events"), "--tasks", "ui_alignment"]
            def child(command, **kwargs):
                write_json(root/"pred/ui_alignment/sample.json", {})
                return subprocess.CompletedProcess(command, 0)
            with mock.patch.object(sys, "argv", argv), mock.patch.object(parallel.subprocess, "run", side_effect=child):
                self.assertEqual(parallel.main(), 0)
            event = json.loads((root/"events/ui_alignment.json").read_text())
            self.assertEqual((event["task"], event["return_code"]), ("ui_alignment", 0))


if __name__ == "__main__":
    unittest.main()
