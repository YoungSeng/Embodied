from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import unittest
from unittest import mock

from scripts import restart_ui5_after_detail_audit as restart
from tests import test_ui5_curriculum_snapshot_prepare as fixtures


class DetailAuditRestartTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.SnapshotPrepareTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.previous = self.fixture.workspace / "gui_logs/ui5_curriculum/previous-submission"
        self.fixture.previous.mkdir(parents=True)
        legacy = self.fixture.make_legacy_caption_failure()
        job_path = self.fixture.execute_retry(legacy)
        self.state_path = job_path.parent / "snapshot-switch.json"
        self.state = json.loads(self.state_path.read_text())
        self.output = Path(self.state["runtime"]["OUTPUT_DIR"])
        (self.output / "logs").mkdir(parents=True)
        self.training = self.output / "train-20260906-120000.log"
        self.training.write_text(
            "[rank0]: RuntimeError: Initial Detail Pyramid scale weights are not thirds: [[0.32,0.33,0.35]]\n"
            "TRAIN_EXIT_CODE: 1\nTRAIN_STATUS: FAILED\n"
        )
        self.pipeline = self.output / "logs/curriculum-20260906T120000Z-123.log"
        self.pipeline.write_text(
            "[LOCANY FATAL] script=/project/shell/run_locany_ui5_crop_rollout4_curriculum_h20x2.sh "
            "line=552 exit_code=1\n"
        )
        (self.output / "checkpoints.json").write_text('{"evaluations":[{"step":0}]}')
        self.fixture.calls.clear()

    def execute(self):
        with mock.patch.object(restart.preparation, "PROJECT_ROOT", self.fixture.project), \
             mock.patch.object(restart.shutil, "which", return_value="/bin/mlx"), \
             mock.patch.object(restart.subprocess, "check_output", return_value="c" * 40), \
             mock.patch.object(restart.subprocess, "run", side_effect=self.fixture.run_command), \
             contextlib.redirect_stdout(io.StringIO()):
            return restart.restart(self.fixture.previous)

    def test_previous_directory_resolves_actual_caption_retry_not_latest_directory(self):
        # An unrelated newer run must not be selected by timestamps or a glob.
        unrelated = self.state_path.parent.parent / "unrelated-newer-run"
        unrelated.mkdir()
        (unrelated / "snapshot-switch.json").write_text("{}")
        self.assertEqual(restart.resolve_submitted_state(self.fixture.previous), self.state_path.resolve())

    def test_restart_submits_once_without_rebuilding_and_preserves_failed_run(self):
        old_state_bytes = self.state_path.read_bytes()
        old_log_bytes = self.training.read_bytes()
        path = self.execute()
        self.assertEqual(len(self.fixture.calls), 1)
        self.assertEqual(self.fixture.calls[0][0], ["/bin/mlx", "job", "submitv2", "--path", str(path)])
        state = json.loads((path.parent / "snapshot-switch.json").read_text())
        for key in ("FROZEN_SELECTION", "CURRICULUM_DATA_DIR", "MODEL_PATH", "PROCESSOR_PATH"):
            self.assertEqual(state["runtime"][key], self.state["runtime"][key])
        self.assertNotEqual(state["runtime"]["OUTPUT_DIR"], str(self.output))
        self.assertEqual(state["runtime"]["CODE_REVISION"], "c" * 40)
        self.assertEqual(state["failure_evidence"]["completed_optimizer_steps"], 0)
        self.assertEqual(self.state_path.read_bytes(), old_state_bytes)
        self.assertEqual(self.training.read_bytes(), old_log_bytes)
        self.fixture.calls.clear()
        with self.assertRaisesRegex(RuntimeError, "already has a restart reservation"):
            self.execute()
        self.assertEqual(self.fixture.calls, [])

    def test_running_or_other_failure_is_not_resubmitted(self):
        for content in (
            "Initial Detail Pyramid scale weights are not thirds: ...\n",
            "CUDA out of memory\nTRAIN_EXIT_CODE: 1\nTRAIN_STATUS: FAILED\n",
            "Initial Detail Pyramid scale weights are not thirds: ...\nTRAIN_EXIT_CODE: 0\nTRAIN_STATUS: SUCCESS\n",
        ):
            self.training.write_text(content)
            with self.subTest(content=content), self.assertRaises(RuntimeError):
                self.execute()
        self.assertEqual(self.fixture.calls, [])

    def test_waits_for_controller_terminal_failure(self):
        self.pipeline.write_text("[TRAIN SEGMENT] starting\n")
        with self.assertRaisesRegex(RuntimeError, "controller has not recorded"):
            self.execute()
        self.assertEqual(self.fixture.calls, [])

    def test_any_optimizer_checkpoint_requires_exact_resume_instead(self):
        (self.output / "resume/latest").mkdir(parents=True)
        with self.assertRaisesRegex(RuntimeError, "checkpoint state exists"):
            self.execute()
        self.assertEqual(self.fixture.calls, [])

    def test_nonzero_evaluations_are_never_discarded(self):
        (self.output / "checkpoints.json").write_text('{"evaluations":[{"step":0},{"step":200}]}')
        with self.assertRaisesRegex(RuntimeError, "nonzero training/evaluation"):
            self.execute()
        self.assertEqual(self.fixture.calls, [])

    def test_pointer_cannot_escape_known_submission_root(self):
        outside = self.fixture.root / "snapshot-switch.json"
        outside.write_text("{}")
        (self.fixture.previous / "snapshot-switch-submit.started").write_text(str(outside))
        with self.assertRaisesRegex(RuntimeError, "known run-log root"):
            self.execute()
        self.assertEqual(self.fixture.calls, [])


if __name__ == "__main__":
    unittest.main()
