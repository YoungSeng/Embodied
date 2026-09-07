"""CPU-only resource selection, YAML binding and shell dispatch regressions."""
import contextlib
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import yaml
import locany_ui5_common as common
import submit_locany_ui5 as submit
import ui14_checks as checks
from ui14_common import file_digest, read_json, write_json
from ui14_profile import profile_environment


class SubmissionResourcesTests(unittest.TestCase):
    def args(self, root, resource=None):
        words = ["--profile", "m32-cpt9000-ui14-v1", "--machine", "a800", "--gpus", "4", "--ui14-data-root", str(root)]
        if resource is not None:
            words += ["--resource-group", resource]
        return submit.parse_args(words)

    def test_both_resources_render_and_restore_the_same_formal_training_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            reference = None
            for resource, group in ((None, 2146), ("aiai_locate", 2146), ("yg", 1602), ("default", 1602)):
                rendered, runtime = submit.render_job(self.args(tmp, resource))
                checks.validate_formal_yaml(rendered, runtime)
                parsed = yaml.safe_load(rendered)
                arnold = parsed["jobDefVersion"]["resource"]["arnoldConfig"]
                self.assertEqual(arnold["clusterId"], 24)
                self.assertEqual(arnold["groupIds"], [group])
                self.assertEqual(runtime["RESOURCE_GROUP_ID"], group)
                self.assertEqual(len(arnold["roles"]), 1)
                self.assertEqual(arnold["roles"][0]["gpu"], 4)
                if group == 1602:
                    self.assertNotIn("queueName", arnold["roles"][0])
                    self.assertEqual(runtime["RESOURCE_GROUP"], "default")
                else:
                    self.assertEqual(arnold["roles"][0]["queueName"], "compute-3302-yg-cloudnative-ai-aiai.locate-guarantee")
                settings = {k: v for k, v in runtime.items() if not k.startswith("RESOURCE_")}
                if reference is None:
                    reference = settings
                self.assertEqual(settings, reference)
                # Training startup resolves the selected resource from the rendered env.
                env = {str(k): str(v) for k, v in parsed["jobRunParams"]["envsList"].items()}
                restored = common.resolve_runtime_config(env)
                self.assertEqual(restored["RESOURCE_GROUP"], runtime["RESOURCE_GROUP"])
                self.assertEqual(restored["EVAL_FAIL_POLICY"], "stop")
                self.assertEqual(restored["EVAL_INFERENCE_WORKERS_PER_GPU"], 2)
            self.assertEqual(common.machine_resource_config("a800", resource_group="yg")["group_id"], 1602)
            self.assertEqual(common.resolve_runtime_config({**profile_environment(), "GPU_COUNT": "4"})["RESOURCE_GROUP"], "aiai_locate")
            ui5 = submit.parse_args(["--machine", "a800", "--gpus", "4"])
            self.assertEqual(submit.render_job(ui5)[1]["RESOURCE_GROUP"], "default")

    def test_unknown_resource_and_formal_resource_drift_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "Unknown resource group"):
                submit.render_job(self.args(tmp, "unknown"))
            env = {**profile_environment(), "RESOURCE_GROUP": "unknown"}
            with self.assertRaisesRegex(ValueError, "Unknown resource group"):
                common.resolve_runtime_config(env)
            for resource in ("yg", "aiai_locate"):
                rendered, runtime = submit.render_job(self.args(tmp, resource))
                parsed = yaml.safe_load(rendered)
                parsed["jobDefVersion"]["resource"]["arnoldConfig"]["groupIds"] = [9999]
                with self.assertRaisesRegex(ValueError, "four-card A800"):
                    checks.validate_formal_yaml(yaml.safe_dump(parsed), runtime)
            args = self.args(tmp, "yg")
            args.machine = "h20"
            with self.assertRaisesRegex(ValueError, "a800"):
                submit.render_job(args)

    def test_switch_preserves_finalized_report_and_writes_separate_submission_bindings(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            canonical, canonical_runtime = checks.render_formal_yaml(root)
            report_path = root / "cpu_check_report.json"
            write_json(report_path, {"ready": True, "normalization_id": "fixture-normalized", "repair_run_id": "fixture-repair",
                "artifact_digests": {p.name: file_digest(p) for p in (canonical, canonical_runtime)}})
            protected = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (canonical, canonical_runtime, report_path)}
            for resource in ("yg", "aiai_locate"):
                args = self.args(root, resource)
                output = root / "submissions" / f"formal_{'default' if resource == 'yg' else resource}.yaml"
                def prepared(runtime):
                    self.assertFalse(output.exists())
                    self.assertEqual(runtime["RESOURCE_GROUP"], "default" if resource == "yg" else resource)
                with mock.patch.object(submit, "parse_args", return_value=args), \
                     mock.patch("ui14_profile.validate_prepared_profile", side_effect=prepared) as validate, \
                     mock.patch.object(submit.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as mlx:
                    self.assertEqual(submit.main(), 0)
                validate.assert_called_once()
                mlx.assert_called_once_with(["mlx", "job", "submitv2", "--path", str(output)], check=False)
                binding = read_json(output.with_suffix(".binding.json"))
                self.assertFalse(binding["render_only"])
                self.assertEqual(binding["yaml_sha256"], file_digest(output))
                self.assertEqual(binding["cpu_check_report_sha256"], file_digest(report_path))
                self.assertEqual(binding["normalization_id"], "fixture-normalized")
                self.assertEqual(binding["repair_run_id"], "fixture-repair")
                self.assertEqual(protected, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in protected})
            rendered, runtime = submit.render_job(self.args(root, "yg"))
            with self.assertRaisesRegex(ValueError, "overwrite CPU-checked"):
                checks.write_submission_artifacts(canonical, rendered, runtime, render_only=True)
            self.assertEqual(protected, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in protected})

    def test_render_only_never_submits_and_failed_data_check_prevents_publication(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args = self.args(tmp, "yg")
            with mock.patch.object(submit, "parse_args", return_value=args), \
                 mock.patch("ui14_profile.validate_prepared_profile", side_effect=RuntimeError("fixture not ready")), \
                 mock.patch.object(submit.subprocess, "run") as mlx:
                with self.assertRaisesRegex(RuntimeError, "not ready"):
                    submit.main()
                mlx.assert_not_called()
            self.assertFalse((Path(tmp) / "submissions").exists())
            args.render_only = True
            with mock.patch.object(submit, "parse_args", return_value=args), \
                 mock.patch("ui14_profile.validate_prepared_profile", side_effect=AssertionError("render checked image data")), \
                 mock.patch.object(submit.subprocess, "run") as mlx:
                self.assertEqual(submit.main(), 0)
                mlx.assert_not_called()
            binding = read_json(Path(tmp) / "submissions/formal_default.binding.json")
            self.assertTrue(binding["render_only"])
            self.assertIsNone(binding["normalization_id"])

    def test_shell_optional_resource_and_environment_are_forwarded(self):
        bash = r"C:\Program Files\Git\bin\bash.exe" if os.name == "nt" else shutil.which("bash")
        if not bash or not Path(bash).is_file():
            self.skipTest("Bash unavailable")
        source = (Path(__file__).resolve().parents[1] / "shell/ui14_cpt9000_a800.sh").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "entry.sh"
            path.write_text(source, encoding="utf-8", newline="\n")
            env = {**os.environ, "UI14_PYTHON": "/bin/echo", "UI14_RESOURCE_GROUP": "aiai_locate"}
            for words, expected in (([], "aiai_locate"), (["--resource-group", "yg"], "yg"),
                                    (["--resource-group=default", "--render-only"], "default")):
                result = subprocess.run([bash, "--noprofile", "--norc", path.as_posix(), "submit", *words], env=env,
                                        text=True, capture_output=True, check=True)
                self.assertIn(f"--resource-group {expected} --gpus 4", result.stdout)
                if "--render-only" in words:
                    self.assertIn("--render-only", result.stdout)
            result = subprocess.run([bash, "--noprofile", "--norc", path.as_posix(), "submit"],
                env={**env, "UI14_RESOURCE_GROUP": "yg"}, text=True, capture_output=True, check=True)
            self.assertIn("--resource-group yg", result.stdout)
            result = subprocess.run([bash, "--noprofile", "--norc", path.as_posix(), "submit-status", "--watch", "--pid", "123"],
                env=env, text=True, capture_output=True, check=True)
            self.assertIn("scripts/ui14_submit_status.py --data-root", result.stdout)
            self.assertIn("--watch --pid 123", result.stdout)
            self.assertNotIn("scripts/submit_locany_ui5.py", result.stdout)
            for words in (["--resource-group"], ["--resource-group="], ["--invalid"]):
                result = subprocess.run([bash, "--noprofile", "--norc", path.as_posix(), "submit", *words],
                    env=env, text=True, capture_output=True)
                self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
