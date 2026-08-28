from __future__ import annotations

import errno
import json
import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from eaglevl.train import cpt_eval_queue
from eaglevl.train.cpt_eval_queue import (
    claim_next_eval,
    enqueue_pending_eval,
    finish_eval,
    read_eval_queue,
)
from eaglevl.train.cpt_observability import CPT_TASKS
from scripts.run_locany_cpt_eval_queue import validate_eval_summary


class CPTEvalQueueTest(unittest.TestCase):
    def test_flock_unsupported_falls_back_to_atomic_directory_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            queue = Path(temporary) / "diagnostics" / "cpt_eval_queue.jsonl"
            with mock.patch.object(cpt_eval_queue, "_acquire_flock", return_value=None):
                self.assertTrue(
                    enqueue_pending_eval(
                        queue,
                        {"step": 10, "checkpoint": "/run/checkpoint-10"},
                    )
                )
            self.assertEqual(read_eval_queue(queue)[0]["step"], 10)
            self.assertFalse(
                Path(str(queue.with_suffix(queue.suffix + ".lock")) + ".mkdir").exists()
            )

    def test_fsync_unsupported_keeps_atomic_queue_publish(self):
        with tempfile.TemporaryDirectory() as temporary:
            queue = Path(temporary) / "diagnostics" / "cpt_eval_queue.jsonl"
            unsupported = OSError(errno.ENOSYS, "Function not implemented")
            with mock.patch.object(cpt_eval_queue.os, "fsync", side_effect=unsupported):
                self.assertTrue(
                    enqueue_pending_eval(
                        queue,
                        {"step": 10, "checkpoint": "/run/checkpoint-10"},
                    )
                )
            self.assertEqual(read_eval_queue(queue)[0]["step"], 10)

    def test_dead_same_host_directory_lock_is_reclaimed_immediately(self):
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "base.json.lock"
            directory = Path(str(lock_path) + ".mkdir")
            directory.mkdir()
            (directory / "owner.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "token": "dead-owner",
                        "hostname": socket.gethostname(),
                        "pid": 999999999,
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.object(
                cpt_eval_queue, "_acquire_flock", return_value=None
            ), mock.patch.object(cpt_eval_queue, "_pid_is_alive", return_value=False):
                with cpt_eval_queue.exclusive_file_lock(lock_path):
                    owner = json.loads(
                        (directory / "owner.json").read_text(encoding="utf-8")
                    )
                    self.assertEqual(owner["pid"], os.getpid())
                    self.assertNotEqual(owner["token"], "dead-owner")
            self.assertFalse(directory.exists())

    def test_live_same_host_owner_is_not_stolen_even_when_age_threshold_is_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "queue.lock.mkdir"
            directory.mkdir()
            (directory / "owner.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "token": "live-owner",
                        "hostname": socket.gethostname(),
                        "pid": os.getpid(),
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.dict(
                os.environ,
                {
                    "CPT_EVAL_QUEUE_LOCK_TIMEOUT_SECONDS": "0",
                    "CPT_EVAL_QUEUE_LOCK_STALE_SECONDS": "0",
                },
            ):
                with self.assertRaisesRegex(TimeoutError, "live-owner"):
                    with cpt_eval_queue._directory_queue_lock(
                        Path(temporary) / "queue.lock"
                    ):
                        self.fail("live owner lock must not be stolen")

    def test_expired_legacy_empty_lock_can_be_reclaimed(self):
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "base.json.lock"
            directory = Path(str(lock_path) + ".mkdir")
            directory.mkdir()
            with mock.patch.dict(
                os.environ,
                {"CPT_EVAL_QUEUE_LEGACY_LOCK_STALE_SECONDS": "0"},
            ), mock.patch.object(cpt_eval_queue, "_acquire_flock", return_value=None):
                with cpt_eval_queue.exclusive_file_lock(lock_path):
                    self.assertTrue((directory / "owner.json").is_file())
            self.assertFalse(directory.exists())

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
