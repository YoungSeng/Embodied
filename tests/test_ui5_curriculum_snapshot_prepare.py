from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from scripts import prepare_ui5_curriculum_snapshot as prepare


class SnapshotPrepareTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.workspace = self.root / "workspace"
        self.snapshot = self.workspace / "gui_rollouts/job/snapshots/hour_018_20260905T030758Z"
        self.snapshot.mkdir(parents=True)
        (self.snapshot / "_SUCCESS").write_text("complete")
        (self.snapshot / "manifest.json").write_text("{}")
        self.source = self.workspace / "gui_data/ui5_curriculum/hour009-s42-v1"
        self.source.mkdir(parents=True)
        (self.source / "_SUCCESS.json").write_text('{"complete": true}')
        self.previous = self.workspace / "previous-submission"
        self.previous.mkdir()
        self.project = self.root / "project"
        (self.project / "jobs").mkdir(parents=True)
        self.template = yaml.safe_load((prepare.PROJECT_ROOT / "jobs/ui5_train_rollouts_h20x2_merlin.yaml").read_text())
        (self.project / "jobs/ui5_train_rollouts_h20x2_merlin.yaml").write_text(yaml.safe_dump(self.template))
        (self.workspace / "gui_models/Embodied-ui5-det-crop/locany-ui5-v5-croponly-sourcebalanced-a800x4-20260830/checkpoint-12000").mkdir(parents=True)
        (self.workspace / "hf_home/hub/models--nvidia--LocateAnything-3B/snapshots/c32291ca5e996f5a7a485845b4f57a233936bba0").mkdir(parents=True)
        evaluation = self.workspace / "gui_models/ui5_eval_detector_cache_horizontal_v5/detector_scan_crops.h20.jsonl"
        evaluation.parent.mkdir(parents=True)
        evaluation.write_text("{}")
        self.calls = []

    def args(self):
        return argparse.Namespace(
            snapshot=self.snapshot, reuse_crops_from=self.source,
            previous_submission_dir=self.previous, take_over_builder_pid=None,
            workspace=self.workspace, mlx_bin="mlx",
        )

    def run_command(self, command, **kwargs):
        self.calls.append((command, kwargs))
        if "build_ui5_curriculum_recipe.py" in str(command):
            destination = Path(command[command.index("--output-dir") + 1])
            destination.mkdir(parents=True)
            (destination / "curriculum_manifest.json").write_text(json.dumps({
                "identity_digest": "identity", "hard_groups": 17,
                "crop_assets": [{"crop_id": "a"}, {"crop_id": "b"}],
                "crop_asset_reuse": {"reused_crop_assets": 2, "generated_crop_assets": 0},
            }))
            (destination / "_SUCCESS.json").write_text(json.dumps({
                "complete": True, "identity_digest": "identity",
            }))
        return mock.Mock(returncode=0)

    def execute(self, side_effect=None):
        with mock.patch.object(prepare, "PROJECT_ROOT", self.project), \
             mock.patch.object(prepare.shutil, "which", return_value="/bin/mlx"), \
             mock.patch.object(prepare.subprocess, "check_output", return_value="a" * 40), \
             mock.patch.object(prepare.subprocess, "run", side_effect=side_effect or self.run_command), \
             contextlib.redirect_stdout(io.StringIO()):
            return prepare.prepare(self.args())

    def test_freeze_reuse_then_submit_in_order_with_cpu_only_preparation(self):
        path = self.execute()
        commands = [item[0] for item in self.calls]
        script_names = [Path(command[2]).name for command in commands[:-1]]
        self.assertEqual(script_names, [
            "merge_ui5_rollout_selections.py", "ui5_frozen_selection.py",
            "relocate_ui5_eval_detector_manifest.py", "build_ui5_curriculum_recipe.py",
        ])
        self.assertEqual(commands[-1], ["/bin/mlx", "job", "submitv2", "--path", str(path)])
        for _, kwargs in self.calls[:-1]:
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
            self.assertTrue(kwargs["check"])
        builder = commands[-2]
        self.assertEqual(builder[builder.index("--reuse-crops-from") + 1], str(self.source.resolve()))
        self.assertNotIn("--force", builder)
        job = yaml.safe_load(path.read_text())
        env = job["jobRunParams"]["envsList"]
        self.assertIn("hour_018_20260905T030758Z", env["FROZEN_SELECTION"])
        self.assertNotEqual(env["CURRICULUM_DATA_DIR"], str(self.source))
        self.assertEqual(env["CURRICULUM_REUSE_CROPS_FROM"], str(self.source.resolve()))
        for key, value in prepare.FORMAL_ENV.items():
            self.assertEqual(env[key], value)
        self.assertNotIn("EXPECTED_HARD_GROUPS", env)
        self.assertEqual(job["jobDefVersion"]["resource"], self.template["jobDefVersion"]["resource"])
        self.assertEqual(job["jobDefVersion"]["imageMeta"], self.template["jobDefVersion"]["imageMeta"])
        self.assertIn("run_locany_ui5_crop_rollout4_curriculum_h20x2.sh", job["jobRunParams"]["entrypointFullScript"])
        self.assertNotIn("run_ui5_train_rollouts_h20x2.sh", job["jobRunParams"]["entrypointFullScript"])
        state = json.loads((path.parent / "snapshot-switch.json").read_text())
        self.assertEqual(state["status"], "submitted")
        self.assertTrue((self.previous / "snapshot-switch-submit.started").is_file())

    def test_second_invocation_cannot_submit_again_with_new_run_name(self):
        self.execute()
        self.calls.clear()
        with self.assertRaisesRegex(RuntimeError, "already attempted"):
            self.execute()
        self.assertEqual(self.calls, [])

    def test_builder_failure_prevents_submission(self):
        def fail_builder(command, **kwargs):
            if "build_ui5_curriculum_recipe.py" in str(command):
                raise RuntimeError("bad reused PNG")
            return self.run_command(command, **kwargs)
        with self.assertRaisesRegex(RuntimeError, "bad reused PNG"):
            self.execute(fail_builder)
        self.assertFalse(any(command[0] == "/bin/mlx" for command, _ in self.calls))
        self.assertFalse((self.previous / "snapshot-switch-submit.started").exists())

    def test_any_regenerated_png_blocks_submission(self):
        def generated(command, **kwargs):
            result = self.run_command(command, **kwargs)
            if "build_ui5_curriculum_recipe.py" in str(command):
                path = Path(command[command.index("--output-dir") + 1]) / "curriculum_manifest.json"
                state = json.loads(path.read_text())
                state["crop_asset_reuse"]["generated_crop_assets"] = 1
                path.write_text(json.dumps(state))
            return result
        with self.assertRaisesRegex(RuntimeError, "all crops reused"):
            self.execute(generated)
        self.assertFalse(any(command[0] == "/bin/mlx" for command, _ in self.calls))

    def test_old_submission_marker_or_log_stops_before_prepare(self):
        for name, payload in (("foreground-submit.started", ""),
                              ("prepare-and-submit.log", "[STAGE 3/3] submit")):
            with self.subTest(name=name):
                path = self.previous / name
                path.write_text(payload)
                with self.assertRaisesRegex(RuntimeError, "already attempted"):
                    self.execute()
                path.unlink()
        self.assertEqual(self.calls, [])

    def test_incomplete_source_is_not_rebuilt(self):
        (self.source / "_SUCCESS.json").unlink()
        with self.assertRaisesRegex(RuntimeError, "refusing to regenerate"):
            self.execute()
        self.assertEqual(self.calls, [])

    def test_pid_reuse_does_not_count_as_live_process(self):
        identity = {"pid": 123, "start_ticks": "old", "state": "S"}
        with mock.patch.object(prepare, "read_process", return_value={**identity, "start_ticks": "new"}):
            self.assertFalse(prepare.process_active(identity))
            self.assertFalse(prepare.signal_process(identity, 0))

    def test_retiring_submitter_never_signals_the_builder(self):
        parent = {"pid": 100, "start_ticks": "parent"}
        with mock.patch.object(prepare, "process_active", side_effect=[True, False]), \
             mock.patch.object(prepare, "signal_process") as send, \
             mock.patch.object(prepare.signal, "SIGCONT", 18, create=True), \
             contextlib.redirect_stdout(io.StringIO()):
            prepare.retire_held_submitter(parent)
        self.assertEqual([call.args[0] for call in send.call_args_list], [parent, parent])

    def test_takeover_only_holds_the_verified_parent_and_not_the_crop_builder(self):
        builder = {"pid": 123, "ppid": 100, "start_ticks": "child", "state": "S",
                   "argv": ["python", "-u", "scripts/build_ui5_curriculum_recipe.py",
                            "--output-dir", str(self.source.resolve())]}
        parent = {"pid": 100, "ppid": 99, "start_ticks": "parent", "state": "S",
                  "argv": ["python", "-u", "-"]}
        def read(pid):
            return builder if pid == 123 else parent
        with mock.patch.object(prepare, "require_takeover_platform"), \
             mock.patch.object(prepare, "read_process", side_effect=read), \
             mock.patch.object(prepare, "signal_process", return_value=True) as send, \
             mock.patch.object(prepare.signal, "SIGSTOP", 19, create=True), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(
                prepare.hold_legacy_submitter(123, self.source.resolve(), self.previous),
                (builder, parent),
            )
        send.assert_called_once_with(parent, 19)

    def test_unrelated_builder_pid_cannot_be_signalled(self):
        builder = {"pid": 123, "ppid": 100, "start_ticks": "child", "state": "S",
                   "argv": ["python", "scripts/build_ui5_curriculum_recipe.py",
                            "--output-dir", str(self.root / "another-task")]}
        with mock.patch.object(prepare, "require_takeover_platform"), \
             mock.patch.object(prepare, "read_process", return_value=builder), \
             mock.patch.object(prepare, "signal_process") as send:
            with self.assertRaisesRegex(RuntimeError, "different curriculum"):
                prepare.hold_legacy_submitter(123, self.source.resolve(), self.previous)
        send.assert_not_called()

    def test_wait_does_not_retire_parent_until_builder_finished(self):
        builder = {"pid": 123}
        parent = {"pid": 100}
        with mock.patch.object(prepare, "process_active", side_effect=[True, False]), \
             mock.patch.object(prepare.time, "sleep") as sleep, \
             mock.patch.object(prepare, "retire_held_submitter") as retire, \
             contextlib.redirect_stdout(io.StringIO()):
            prepare.wait_for_source(self.source, builder, parent)
        sleep.assert_called_once_with(10)
        retire.assert_called_once_with(parent)

    def test_template_must_remain_h20x2(self):
        self.template["jobDefVersion"]["resource"]["arnoldConfig"]["roles"][0]["gpu"] = 4
        with self.assertRaisesRegex(RuntimeError, "H20x2"):
            prepare.render_job(self.template, {}, "test")


if __name__ == "__main__":
    unittest.main()
