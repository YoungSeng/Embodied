from __future__ import annotations

import contextlib
import errno
import io
import json
from pathlib import Path
import unittest
from unittest import mock

from scripts import restart_ui5_after_storage_failure as restart
from tests import test_ui5_detail_audit_restart as fixtures


class StorageRestartTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.DetailAuditRestartTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.state_path = self.fixture.state_path
        self.state = self.fixture.state
        self.output = self.fixture.output
        self.train = self.fixture.training
        self.train.write_text(
            f"OSError: [Errno 122] Disk quota exceeded: '{self.output}/checkpoint-200/continuity_state.json.tmp-4518'\n"
            "TRAIN_EXIT_CODE: 1\nTRAIN_STATUS: FAILED\n"
        )
        self.checkpoint = self.output / "checkpoint-200"
        self.checkpoint.mkdir()
        (self.checkpoint / "model.safetensors").write_bytes(b"incomplete-save-must-not-be-used")
        (self.checkpoint / "trainer_state.json").write_text('{"global_step":200}')
        model = Path(self.state["runtime"]["MODEL_PATH"])
        (model / "config.json").write_text('{}')
        (model / "model.safetensors").write_bytes(b"base-model")
        processor = Path(self.state["runtime"]["PROCESSOR_PATH"])
        (processor / "tokenizer_config.json").write_text('{}')
        (processor / "preprocessor_config.json").write_text('{}')
        self.calls = self.fixture.fixture.calls
        self.calls.clear()

    def execute(self, **options):
        kwargs = dict(restart_from_base=True, confirm_job_stopped=True, confirm_storage_reclaimed=True)
        kwargs.update(options)
        with mock.patch.object(restart.preparation, "PROJECT_ROOT", self.fixture.fixture.project), \
             mock.patch.object(restart.shutil, "which", return_value="/bin/mlx"), \
             mock.patch.object(restart.subprocess, "check_output", return_value="d" * 40), \
             mock.patch.object(restart.subprocess, "run", side_effect=self.fixture.fixture.run_command), \
             contextlib.redirect_stdout(io.StringIO()):
            return restart.restart(self.state_path.parent, **kwargs)

    def test_explicit_restart_reuses_data_and_preserves_all_failed_files(self):
        before = {p.relative_to(self.output): p.read_bytes() for p in self.output.rglob('*') if p.is_file()}
        state_before = self.state_path.read_bytes()
        path = self.execute()
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0][0], ["/bin/mlx", "job", "submitv2", "--path", str(path)])
        state = json.loads((path.parent / "snapshot-switch.json").read_text())
        self.assertEqual(state["status"], "submitted")
        self.assertEqual(state["runtime"]["CODE_REVISION"], "d" * 40)
        self.assertEqual(state["failure_evidence"]["failed_save_step"], 200)
        self.assertEqual(state["failure_evidence"]["restart_global_step"], 0)
        self.assertFalse(state["failure_evidence"]["incomplete_checkpoint_trusted"])
        self.assertIsNone(state["storage_check"]["user_quota_available_bytes"])
        self.assertNotEqual(state["runtime"]["OUTPUT_DIR"], str(self.output))
        for key in ("FROZEN_SELECTION", "CURRICULUM_DATA_DIR", "MODEL_PATH", "PROCESSOR_PATH"):
            self.assertEqual(state["runtime"][key], self.state["runtime"][key])
        for key, value in restart.preparation.FORMAL_ENV.items():
            self.assertEqual(state["runtime"][key], value)
        after = {p.relative_to(self.output): p.read_bytes() for p in self.output.rglob('*') if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(self.state_path.read_bytes(), state_before)

    def test_repeat_submission_is_blocked_without_removing_reservation(self):
        path = self.execute()
        self.calls.clear()
        marker = self.state_path.parent / "storage-restart.started"
        self.assertEqual(Path(marker.read_text().strip()), path.parent / "snapshot-switch.json")
        with self.assertRaisesRegex(RuntimeError, "successor reservation"):
            self.execute()
        self.assertEqual(self.calls, [])

    def test_existing_other_restart_is_not_overridden(self):
        for name in ("detail-audit-restart.started", "caption-retry.started"):
            marker = self.state_path.parent / name
            marker.write_text("another-task")
            with self.assertRaisesRegex(RuntimeError, "successor reservation"):
                self.execute()
            self.assertEqual(marker.read_text(), "another-task")
            marker.unlink()
        self.assertEqual(self.calls, [])

    def test_no_implicit_optimizer_reset_or_job_stopped_assumption(self):
        for key in ("restart_from_base", "confirm_job_stopped", "confirm_storage_reclaimed"):
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, "require --restart-from-base"):
                self.execute(**{key: False})
        self.assertEqual(self.calls, [])

    def test_read_only_check_does_not_reserve_or_submit(self):
        self.execute(check_only=True)
        self.assertFalse((self.state_path.parent / "storage-restart.started").exists())
        self.assertEqual(self.calls, [])

    def test_complete_or_rolling_checkpoint_requires_exact_resume(self):
        marker = self.checkpoint / "checkpoint_complete.json"
        marker.write_text('{"global_step":200}')
        with self.assertRaisesRegex(RuntimeError, "completion marker"):
            self.execute()
        marker.unlink()
        rolling = self.output / "resume/latest"
        rolling.mkdir(parents=True)
        with self.assertRaisesRegex(RuntimeError, "exact resume"):
            self.execute()
        self.assertEqual(self.calls, [])

    def test_later_checkpoint_or_evaluation_prevents_restart(self):
        extra = self.output / "checkpoint-400"
        extra.mkdir()
        with self.assertRaisesRegex(RuntimeError, "unexpected checkpoint"):
            self.execute()
        extra.rmdir()
        (self.output / "checkpoints.json").write_text('{"evaluations":[{"step":0},{"step":200}]}')
        with self.assertRaisesRegex(RuntimeError, "nonzero evaluations"):
            self.execute()
        self.assertEqual(self.calls, [])

    def test_error_must_be_first_save_storage_failure(self):
        for text in ("CUDA out of memory", "Disk quota exceeded: 'evaluation/report.json'",
                     "Disk quota exceeded: 'checkpoint-400/continuity_state.json'",
                     "Disk quota exceeded\ncheckpoint-200/continuity_state.json"):
            self.train.write_text(text + "\nTRAIN_EXIT_CODE: 1\nTRAIN_STATUS: FAILED\n")
            with self.subTest(text=text), self.assertRaisesRegex(RuntimeError, "storage failure saving checkpoint-200"):
                self.execute()
        self.assertEqual(self.calls, [])

    def test_quota_full_log_can_lack_footer_with_explicit_stop_confirmation(self):
        self.train.write_text("OSError: [Errno 122] Disk quota exceeded: 'checkpoint-200/continuity_state.json'\n")
        self.fixture.pipeline.write_text("log truncated by quota\n")
        self.execute(check_only=True)
        self.assertEqual(self.calls, [])

    def test_success_in_latest_log_is_not_restarted(self):
        with self.train.open("a") as handle:
            handle.write("TRAIN_STATUS: SUCCESS\n")
        with self.assertRaisesRegex(RuntimeError, "does not end with failed"):
            self.execute()
        self.assertEqual(self.calls, [])

    def test_deleted_source_model_prevents_submission(self):
        (Path(self.state["runtime"]["MODEL_PATH"]) / "model.safetensors").unlink()
        with self.assertRaisesRegex(RuntimeError, "original crop checkpoint is missing/incomplete"):
            self.execute()
        self.assertEqual(self.calls, [])

    def test_insufficient_space_and_quota_probe_failure_prevent_reservation(self):
        with mock.patch.object(restart.shutil, "disk_usage", return_value=mock.Mock(free=1)):
            with self.assertRaisesRegex(RuntimeError, "filesystem free space"):
                self.execute()
        with mock.patch.object(restart.os, "fsync", side_effect=OSError(errno.EDQUOT, "Disk quota exceeded")):
            with self.assertRaises(OSError):
                self.execute()
        self.assertEqual(self.calls, [])
        self.assertFalse((self.state_path.parent / "storage-restart.started").exists())

    def test_curriculum_identity_mismatch_stops_without_rebuild(self):
        manifest = Path(self.state["runtime"]["CURRICULUM_DATA_DIR"]) / "curriculum_manifest.json"
        data = json.loads(manifest.read_text())
        data["hard_groups"] += 1
        manifest.write_text(json.dumps(data))
        with self.assertRaisesRegex(RuntimeError, "identity/publication/reuse"):
            self.execute()
        self.assertEqual(self.calls, [])

    def test_platform_error_keeps_reservation_and_does_not_claim_submitted(self):
        def reject(command, **kwargs):
            self.calls.append((command, kwargs))
            kwargs["stdout"].write('{"errCode":"QuotaExceeded","errMsg":"platform rejected job"}')
            return mock.Mock(returncode=0)

        with mock.patch.object(self.fixture.fixture, "run_command", side_effect=reject):
            with self.assertRaisesRegex(RuntimeError, "submission_failed"):
                self.execute()
        marker = self.state_path.parent / "storage-restart.started"
        state = json.loads(Path(marker.read_text().strip()).read_text())
        self.assertEqual(state["status"], "submission_failed")
        self.assertEqual(len(self.calls), 1)
        with self.assertRaisesRegex(RuntimeError, "successor reservation"):
            self.execute()
        self.assertEqual(len(self.calls), 1)


if __name__ == "__main__":
    unittest.main()
