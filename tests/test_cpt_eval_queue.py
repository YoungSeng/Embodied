from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from eaglevl.train.cpt_eval_queue import (
    claim_next_eval,
    enqueue_pending_eval,
    finish_eval,
    read_eval_queue,
)
from eaglevl.train.cpt_observability import CPT_TASKS
from scripts.run_locany_cpt_eval_queue import validate_eval_summary


class CPTEvalQueueTest(unittest.TestCase):
    def test_queue_deduplicates_claims_and_records_terminal_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            queue = Path(temporary) / "diagnostics/cpt_eval_queue.jsonl"
            row = {"step": 20, "checkpoint": "/run/checkpoint-20"}
            self.assertTrue(enqueue_pending_eval(queue, row))
            self.assertFalse(enqueue_pending_eval(queue, row))
            claimed = claim_next_eval(queue, worker="unit-test")
            self.assertIsNotNone(claimed)
            assert claimed is not None
            self.assertEqual(claimed["status"], "running")
            self.assertEqual(claimed["attempt"], 1)
            self.assertIsNone(claim_next_eval(queue))
            completed = finish_eval(
                queue,
                claimed["queue_id"],
                status="completed",
                details={"summary": "/run/eval/checkpoint-20/summary.json"},
            )
            self.assertEqual(completed["status"], "completed")
            self.assertEqual(len(read_eval_queue(queue)), 1)

    def test_failed_row_requires_explicit_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            queue = Path(temporary) / "queue.jsonl"
            enqueue_pending_eval(queue, {"step": 10, "checkpoint": "/run/checkpoint-10"})
            row = claim_next_eval(queue)
            assert row is not None
            finish_eval(queue, row["queue_id"], status="failed", details={"error": "boom"})
            self.assertIsNone(claim_next_eval(queue))
            retry = claim_next_eval(queue, retry_failed=True)
            self.assertIsNotNone(retry)
            assert retry is not None
            self.assertEqual(retry["attempt"], 2)

    def test_summary_gate_requires_ten_tasks_ce_and_zero_errors(self):
        per_task = {
            task: {
                "eval_main_token_ce": 1.5,
                "inference_error_count": 0,
            }
            for task in CPT_TASKS
        }
        summary = {
            "split": "heldout",
            "teacher_forced": True,
            "task_counts": {task: 10 for task in CPT_TASKS},
            "base": {
                "per_task": per_task,
                "heldout_task_macro_primary": 0.5,
            },
            "checkpoint_metrics": {
                "per_task": per_task,
                "heldout_task_macro_primary": 0.8,
            },
        }
        validate_eval_summary(summary, samples_per_task=10, require_zero_errors=True)
        summary["checkpoint_metrics"]["per_task"][CPT_TASKS[0]] = {
            "eval_main_token_ce": 1.5,
            "inference_error_count": 1,
        }
        with self.assertRaisesRegex(RuntimeError, "inference_error_count=1"):
            validate_eval_summary(summary, samples_per_task=10, require_zero_errors=True)


if __name__ == "__main__":
    unittest.main()
