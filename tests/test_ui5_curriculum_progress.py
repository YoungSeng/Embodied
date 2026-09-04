from __future__ import annotations

import ast
import importlib.util
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ui5_curriculum_progress.py"
SPEC = importlib.util.spec_from_file_location("ui5_curriculum_progress", SCRIPT)
assert SPEC and SPEC.loader
progress_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(progress_module)
BuildProgress = progress_module.BuildProgress


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class CurriculumProgressTests(unittest.TestCase):
    def latest(self, root: Path) -> dict:
        return json.loads((root / "progress" / "build_progress.json").read_text(encoding="utf-8"))

    def records(self, root: Path) -> list[dict]:
        return [json.loads(line) for line in (root / "progress" / "build_progress.jsonl").read_text(encoding="utf-8").splitlines()]

    def test_stage_eta_requires_progress_and_is_not_full_build_eta(self) -> None:
        clock = FakeClock()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with BuildProgress(root, stream=io.StringIO(), clock=clock) as progress:
                with progress.stage("materialize", total=100, unit="crops") as stage:
                    self.assertIsNone(self.latest(root)["eta_seconds"])
                    clock.advance(10)
                    stage.update(25, detail="sample-25", bytes_processed=1024)
                    record = self.latest(root)
                    self.assertEqual(record["percent"], 25)
                    self.assertEqual(record["speed_per_second"], 2.5)
                    self.assertEqual(record["eta_seconds"], 30)
                    self.assertEqual(record["eta_scope"], "current_stage")
                    self.assertEqual(record["bytes_processed"], 1024)
                    clock.advance(5)
                    stage.update(100, force=True)
                    self.assertEqual(self.latest(root)["eta_seconds"], 0)
                with progress.stage("different_cost", total=50) as stage:
                    self.assertIsNone(self.latest(root)["eta_seconds"])
                    self.assertEqual(self.latest(root)["build_elapsed_seconds"], 15)
                    clock.advance(10)
                    stage.update(10)
                    self.assertEqual(self.latest(root)["eta_seconds"], 40)

    def test_zero_and_unknown_totals_do_not_divide_by_zero_or_invent_eta(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with BuildProgress(root, stream=io.StringIO(), clock=FakeClock()) as progress:
                with progress.stage("empty", total=0):
                    record = self.latest(root)
                    self.assertEqual(record["eta_seconds"], 0)
                    self.assertEqual(record["percent"], 100)
                    self.assertIsNone(record["speed_per_second"])
                with progress.stage("count_unknown") as stage:
                    stage.update(3, force=True)
                    record = self.latest(root)
                    self.assertIsNone(record["eta_seconds"])
                    self.assertIsNone(record["percent"])
                    self.assertIsNone(record["total"])

    def test_throttle_force_and_immediate_stage_boundaries(self) -> None:
        clock = FakeClock()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with BuildProgress(root, stream=io.StringIO(), clock=clock) as progress:
                with progress.stage("source_images", total=10) as stage:
                    self.assertEqual(len(self.records(root)), 2)
                    stage.update(1)
                    stage.set_detail("loading slow.png")
                    self.assertEqual(len(self.records(root)), 2)
                    clock.advance(10)
                    stage.update(2)
                    self.assertEqual(self.latest(root)["detail"], "loading slow.png")
                    self.assertEqual(len(self.records(root)), 3)
                    stage.update(3, force=True)
                    self.assertEqual(len(self.records(root)), 4)
                self.assertEqual(self.latest(root)["event"], "stage_completed")
            self.assertEqual(self.latest(root)["event"], "build_completed")

    def test_background_heartbeat_reports_blocked_item_without_fake_progress(self) -> None:
        heartbeat_seen = threading.Event()

        class CapturingStream(io.StringIO):
            def write(self, value: str) -> int:
                result = super().write(value)
                if "event=heartbeat" in value:
                    heartbeat_seen.set()
                return result

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stream = CapturingStream()
            with BuildProgress(root, interval_seconds=0.02, stream=stream) as progress:
                with progress.stage("read_images", total=2) as stage:
                    stage.set_detail("waiting on slow-image.png")
                    # No update is possible during this simulated blocking read.
                    self.assertTrue(heartbeat_seen.wait(2), "background heartbeat did not run")
                    with progress._lock:
                        record = self.latest(root)
                    self.assertEqual(record["event"], "heartbeat")
                    self.assertEqual(record["completed"], 0)
                    self.assertEqual(record["detail"], "waiting on slow-image.png")
                    self.assertGreater(record["last_progress_age_seconds"], 0)
                    self.assertIsNone(record["eta_seconds"])
            self.assertFalse(progress._thread.is_alive())

    def test_failed_build_retains_original_exception_and_no_completion(self) -> None:
        failure = RuntimeError("corrupt source image")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(RuntimeError) as caught:
                with BuildProgress(root, stream=io.StringIO()) as progress:
                    with progress.stage("images", total=2) as stage:
                        stage.update(1, force=True)
                        raise failure
            self.assertIs(caught.exception, failure)
            record = self.latest(root)
            self.assertEqual(record["event"], "build_failed")
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["stage_status"], "failed")
            self.assertIn("corrupt source image", record["error"])
            self.assertEqual(record["completed"], 1)
            self.assertNotIn("build_completed", [row["event"] for row in self.records(root)])

    def test_logging_io_error_does_not_mask_build_exception(self) -> None:
        failure = ValueError("original error")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "progress").write_text("not a directory", encoding="utf-8")
            stream = io.StringIO()
            with self.assertRaises(ValueError) as caught:
                with BuildProgress(root, stream=stream):
                    raise failure
            self.assertIs(caught.exception, failure)
            self.assertIn("cannot save progress", stream.getvalue())
            self.assertIn("build_failed", stream.getvalue())

    def test_restart_keeps_history_with_distinct_run_ids_and_atomic_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real_replace = progress_module.os.replace
            published: list[dict] = []

            def check_replace(source: Path, destination: Path) -> None:
                self.assertEqual(Path(source).parent, Path(destination).parent)
                published.append(json.loads(Path(source).read_text(encoding="utf-8")))
                real_replace(source, destination)

            with patch.object(progress_module.os, "replace", side_effect=check_replace):
                for _ in range(2):
                    with BuildProgress(root, stream=io.StringIO()) as progress:
                        with progress.stage("validate", total=1) as stage:
                            stage.update(1, force=True)
            records = self.records(root)
            self.assertEqual(len({row["run_id"] for row in records}), 2)
            self.assertEqual(len(published), len(records))
            self.assertEqual(self.latest(root), records[-1])
            self.assertEqual(list((root / "progress").glob("*.tmp")), [])

    def test_nested_stages_restore_parent_for_heartbeat(self) -> None:
        clock = FakeClock()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with BuildProgress(root, stream=io.StringIO(), clock=clock) as progress:
                with progress.stage("parent", total=2) as outer:
                    outer.set_detail("parent operation")
                    with progress.stage("child", total=1) as inner:
                        inner.update(1, force=True)
                    clock.advance(10)
                    outer.update(1)
                    self.assertEqual(self.latest(root)["stage"], "parent")
                    self.assertEqual(self.latest(root)["completed"], 1)
                    self.assertEqual(self.latest(root)["detail"], "parent operation")

    def test_progress_module_only_imports_standard_library(self) -> None:
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module.split(".")[0])
        self.assertTrue(imported <= {
            "__future__", "json", "math", "os", "sys", "threading", "time",
            "uuid", "datetime", "pathlib", "typing",
        }, imported)


if __name__ == "__main__":
    unittest.main()
