"""CPU regressions: visible submit checks, reusable evidence, read-only old-process observation."""
import contextlib
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from PIL import Image
import submit_locany_ui5 as submit
import ui14_progress as progress
import ui14_submit_status as status
import ui14_verification as verification
from ui14_common import read_json, write_json, write_jsonl


class SubmitProgressTests(unittest.TestCase):
    def test_status_visible_before_validation_and_before_mlx(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()) as captured:
            root = Path(tmp)
            args = submit.parse_args(["--profile", "m32-cpt9000-ui14-v1", "--machine", "a800", "--gpus", "4",
                                      "--ui14-data-root", tmp])
            seen = []
            def validate(runtime):
                value = read_json(root / "progress/submit.json")
                self.assertEqual(value["status"], "running")
                self.assertIn("提交前 CPU 检查", value["phases"][-1]["label"])
                self.assertIn("尚未调用 mlx", captured.getvalue())
                self.assertFalse((root / "submissions").exists())
                seen.append("checked")
            def run(command, check):
                self.assertEqual(seen, ["checked"])
                value = read_json(root / "progress/submit.json")
                self.assertIn("mlx 提交请求", value["phases"][-1]["label"])
                self.assertIsNone(value["phases"][-1]["eta_seconds"])
                self.assertTrue(Path(command[-1]).is_file())
                return SimpleNamespace(returncode=0)
            with mock.patch.object(submit, "parse_args", return_value=args), \
                 mock.patch("ui14_profile.validate_prepared_profile", side_effect=validate), \
                 mock.patch.object(submit.subprocess, "run", side_effect=run) as mlx:
                self.assertEqual(submit.main(), 0)
                mlx.assert_called_once()
            self.assertEqual(read_json(root / "progress/submit.json")["status"], "completed")
            self.assertIn("保存提交 YAML", (root / "progress/submit.log").read_text(encoding="utf-8"))

    def test_failed_mlx_is_failed_progress_and_never_automatically_retried(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args = submit.parse_args(["--profile", "m32-cpt9000-ui14-v1", "--machine", "a800", "--gpus", "4",
                                      "--ui14-data-root", tmp])
            with mock.patch.object(submit, "parse_args", return_value=args), \
                 mock.patch("ui14_profile.validate_prepared_profile"), \
                 mock.patch.object(submit.subprocess, "run", return_value=SimpleNamespace(returncode=17)) as mlx:
                with self.assertRaises(SystemExit) as error:
                    submit.main()
                self.assertEqual(error.exception.code, 17)
                mlx.assert_called_once()
            self.assertEqual(read_json(Path(tmp) / "progress/submit.json")["status"], "failed")

    def test_second_coordinator_cannot_overwrite_active_progress_or_submit(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args = submit.parse_args(["--profile", "m32-cpt9000-ui14-v1", "--machine", "a800", "--gpus", "4",
                                      "--ui14-data-root", tmp])
            with verification.preparation_lock(tmp, filename=".ui14-submission.lock"), \
                 mock.patch.object(submit, "parse_args", return_value=args), \
                 mock.patch.object(submit.subprocess, "run") as mlx:
                with self.assertRaisesRegex(RuntimeError, "coordinator is running"):
                    submit.main()
                mlx.assert_not_called()
            self.assertFalse((Path(tmp) / "progress/submit.json").exists())

    def test_overlapping_file_progress_does_not_restore_finished_worker(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), \
             progress.ProgressSession("submit", tmp) as session:
            first, second = progress.file_activity("a", 10), progress.file_activity("b", 20)
            first.__enter__(); second.__enter__()
            first.__exit__(None, None, None)
            self.assertEqual(session.activity.label, "b")
            second.__exit__(None, None, None)
            self.assertIsNone(session.activity)


class EvidenceResumeTests(unittest.TestCase):
    def fixture(self, root, count=4):
        paths = []
        with verification.verification_session(root) as checks:
            for i in range(count):
                path = root / f"image-{i}.png"
                Image.new("RGB", (20, 30), (i, 10, 20)).save(path)
                checks.rgb_identity(path); checks.sha256(path)
                paths.append(path)
            evidence = root / "image_evidence.jsonl"
            checks.export_images(evidence)
        return paths, evidence

    def test_unchanged_images_only_stat_once_per_unique_image_in_parallel(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp); paths, evidence = self.fixture(root)
            real_signature = verification.signature
            barrier = threading.Barrier(2); seen = []; guard = threading.Lock()
            def inspect(path):
                with guard: seen.append(str(path)); call = len(seen)
                if call <= 2: barrier.wait(timeout=5)
                return real_signature(path)
            with verification.verification_session(root) as checks, \
                 mock.patch.dict(os.environ, {"UI14_SUBMIT_CHECK_WORKERS": "2"}), \
                 mock.patch.object(verification, "signature", side_effect=inspect), \
                 mock.patch.object(verification, "stable_sha256", side_effect=AssertionError("image rehashed")), \
                 mock.patch.object(Image, "open", side_effect=AssertionError("image opened")):
                self.assertEqual(checks.validate_images(evidence), {"images": 4, "reused": 4, "refreshed": 0, "workers": 2})
            self.assertCountEqual(seen, list(map(str, paths)))

    def test_interrupted_check_reuses_completed_hashes_and_does_not_trust_changed_pixels(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp); paths, evidence = self.fixture(root, count=3)
            # An attribute-only update requires one new content check, then the
            # journal can supply it even if the enclosing submit never finishes.
            for path in paths:
                old = path.stat(); os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns + 1000000000))
            original_rows = list(json.loads(line) for line in evidence.read_text().splitlines())
            write_jsonl(evidence, [row for row in original_rows if row["key"].endswith(str(paths[0]))])
            with self.assertRaises(KeyboardInterrupt), verification.verification_session(root) as checks:
                result = checks.validate_images(evidence)
                self.assertEqual(result["refreshed"], 1)
                raise KeyboardInterrupt()
            with verification.verification_session(root) as checks, \
                 mock.patch.object(Image, "open", side_effect=AssertionError("completed image decoded again")), \
                 mock.patch.object(verification, "stable_sha256", side_effect=AssertionError("completed image rehashed")):
                self.assertEqual(checks.validate_images(evidence)["reused"], 1)
            # Restore the full immutable evidence; only two pending images need work.
            write_jsonl(evidence, original_rows)
            with verification.verification_session(root) as checks:
                result = checks.validate_images(evidence)
                self.assertEqual((result["reused"], result["refreshed"]), (1, 2))
            with verification.verification_session(root) as checks, \
                 mock.patch.object(Image, "open", side_effect=AssertionError("completed image decoded again")), \
                 mock.patch.object(verification, "stable_sha256", side_effect=AssertionError("completed image rehashed")):
                self.assertEqual(checks.validate_images(evidence)["reused"], 3)
            Image.new("RGB", (20, 30), "red").save(paths[1])
            for _ in range(2):
                with verification.verification_session(root) as checks:
                    with self.assertRaisesRegex(RuntimeError, "Prepared image content changed"):
                        checks.validate_images(evidence)


class OldSubmitObserverTests(unittest.TestCase):
    def make_proc(self, proc, pid, words, start=100):
        folder = proc / str(pid); folder.mkdir()
        (folder / "cmdline").write_bytes(b"\0".join(word.encode() for word in words) + b"\0")
        fields = ["S", *("0" for _ in range(21))]; fields[19] = str(start)
        (folder / "stat").write_text(f"{pid} (python worker) " + " ".join(fields))
        (proc / "uptime").write_text("1000.0 500.0")
        (folder / "fd").mkdir(); (folder / "fdinfo").mkdir()
        (folder / "task" / str(pid)).mkdir(parents=True)
        (folder / "task" / str(pid) / "children").write_text("")
        return folder

    def test_exact_selection_and_pid_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = Path(tmp)
            words = ["python", "scripts/submit_locany_ui5.py", "--profile=m32-cpt9000-ui14-v1", "--ui14-data-root=/data/ui14"]
            folder = self.make_proc(proc, 123, words)
            self.make_proc(proc, 456, [*words[:-1], "--ui14-data-root=/data/another"])
            self.assertEqual(status.find_processes("/data/ui14", proc), [123])
            self.assertEqual(status.snapshot(123, proc / "data", proc)["start_ticks"], 100)
            (folder / "stat").write_text("123 (python) Z")
            self.assertIsNone(status.snapshot(123, proc / "data", proc))

    def test_old_reader_and_mlx_detection_without_reading_target_contents(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = Path(tmp); folder = self.make_proc(proc, 123, ["python", "submit_locany_ui5.py"])
            evidence = proc / "image_evidence.jsonl"; evidence.write_bytes(b" " * 1000)
            (folder / "fd/3").touch(); (folder / "fd/4").touch()
            (folder / "fdinfo/3").write_text(f"pos:\t250\nflags:\t{os.O_RDONLY:o}\n")
            (folder / "fdinfo/4").write_text(f"pos:\t1000\nflags:\t{os.O_APPEND | os.O_WRONLY:o}\n")
            with mock.patch.object(status.os, "readlink", return_value=str(evidence)):
                value = status.snapshot(123, proc / "data", proc)
            self.assertEqual(len(value["readers"]), 1)
            observer = status.Observer()
            self.assertIn("估算中", observer.describe(value, now=0))
            value["readers"][0]["position"] = 500
            text = observer.describe(value, now=10)
            self.assertIn("50.0%", text); self.assertIn("00:00:20", text)
            self.assertIn("非整次提交 ETA", text)
            self.make_proc(proc, 321, ["python", "/bin/mlx", "job", "submitv2", "--path", "formal.yaml"])
            (folder / "task/123/children").write_text("321")
            with mock.patch.object(status.os, "readlink", return_value=str(evidence)):
                value = status.snapshot(123, proc / "data", proc)
            self.assertEqual(value["mlx_pids"], [321])
            self.assertIn("等待集群响应", observer.describe(value))

    def test_new_progress_and_stale_pid_are_distinguished(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.make_proc(root, 123, ["python"])
            payload = {"stage": "submit", "status": "running", "pid": 123, "updated_at": datetime.now(timezone.utc).isoformat(),
                       "phases": [{"label": "stat", "completed": 8, "total": 10, "unit": "images", "eta_seconds": 2}]}
            write_json(root / "progress/submit.json", payload)
            self.assertIn("8/10 images", status.Observer().describe(status.snapshot(123, root, root)))
            payload["updated_at"] = "2000-01-01T00:00:00+00:00"
            write_json(root / "progress/submit.json", payload)
            self.assertNotIn("progress", status.snapshot(123, root, root))

    def test_observer_has_no_mutating_process_or_submission_operations(self):
        import ast
        tree = ast.parse(Path(status.__file__).read_text(encoding="utf-8"))
        forbidden = {"run", "Popen", "kill", "terminate", "write_text", "write_bytes", "unlink", "replace"}
        called = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        self.assertFalse(called & forbidden)


if __name__ == "__main__":
    unittest.main()
