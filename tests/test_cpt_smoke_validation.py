from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eaglevl.train.cpt_excel import build_cpt_workbook
from scripts.validate_locany_cpt_smoke import CPT_TASKS, validate_run


class CPTSmokeValidationTest(unittest.TestCase):
    def _fixture(self, root: Path, *, with_eval: bool = False) -> Path:
        diagnostics = root / "diagnostics"
        diagnostics.mkdir(parents=True)
        (diagnostics / "cpt_run_config.json").write_text(
            json.dumps(
                {
                    "world_size": 2,
                    "datasets": [{"task": task} for task in sorted(CPT_TASKS)],
                }
            ),
            encoding="utf-8",
        )
        rows = []
        for task in sorted(CPT_TASKS):
            rows.append(
                {
                    "step": 20,
                    "scope": "lifetime_global",
                    "task": task,
                    "attempted_samples": 3,
                    "accepted_samples": 2,
                    "trained_samples": 2,
                    "oversize_skipped_samples": 1,
                    "main_supervised_tokens": 10,
                    "mtp_supervised_tokens": 5,
                    "total_supervised_tokens": 15,
                    "main_loss_tokens": 10,
                    "train_main_token_ce": 2.0,
                    "train_mtp_token_ce": 2.2,
                    "train_total_token_ce": 2.1,
                    "row_coverage": 0.5,
                    "group_coverage": 0.5,
                    "effective_epoch": 0.2,
                    "repeat_factor": 1.0,
                    "packing_efficiency": 0.9,
                    "window_oversize_record_hashes": [1],
                    "global_attempted_samples": 30,
                    "global_trained_samples": 20,
                    "global_total_supervised_tokens": 150,
                }
            )
        (diagnostics / "cpt_train_metrics.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        (diagnostics / "cpt_eval_queue.jsonl").write_text(
            json.dumps({"step": 10, "status": "pending"})
            + "\n"
            + json.dumps({"step": 20, "status": "completed" if with_eval else "pending"})
            + "\n",
            encoding="utf-8",
        )
        if with_eval:
            eval_rows = []
            for task in sorted(CPT_TASKS):
                eval_rows.append(
                    {
                        "step": 20,
                        "split": "heldout",
                        "task": task,
                        "samples_per_task": 10,
                        "eval_token_ce": 1.5,
                        "primary_metric": 0.8,
                        "metrics": {"inference_error_count": 0},
                    }
                )
            eval_rows.append(
                {
                    "step": 20,
                    "split": "heldout",
                    "task": "__task_macro__",
                    "samples_per_task": 10,
                    "eval_token_ce": 1.5,
                    "primary_metric": 0.8,
                    "complete_ten_task_heldout": True,
                }
            )
            (diagnostics / "cpt_eval_metrics.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in eval_rows),
                encoding="utf-8",
            )
        for step in (10, 20):
            checkpoint = root / f"checkpoint-{step}"
            checkpoint.mkdir()
            (checkpoint / "checkpoint_complete.json").write_text(
                json.dumps({"global_step": step}), encoding="utf-8"
            )
            for rank in range(2):
                (checkpoint / f"dataloader_state_rank{rank}.pt").write_bytes(b"state")
        (root / "done.txt").write_text("done", encoding="utf-8")
        return root

    def test_validates_resume_state_and_ten_task_schema(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = validate_run(
                self._fixture(Path(temporary)), require_excel=False
            )
            self.assertEqual(report["final_step"], 20)
            self.assertEqual(set(report["tasks"]), CPT_TASKS)
            self.assertEqual(report["eval_queue_steps"], [10, 20])

    def test_rejects_sample_identity_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._fixture(Path(temporary))
            path = root / "diagnostics" / "cpt_train_metrics.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]["attempted_samples"] = 4
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "sample identity"):
                validate_run(root, require_excel=False)

    def test_require_eval_checks_real_ten_task_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._fixture(Path(temporary), with_eval=True)
            self.assertTrue(build_cpt_workbook(root / "diagnostics"))
            report = validate_run(
                root,
                require_eval=True,
                eval_samples_per_task=10,
            )
            self.assertEqual(report["heldout_eval"]["task_macro_primary"], 0.8)


if __name__ == "__main__":
    unittest.main()
