"""CPU progress contracts: real ETA, idle heartbeat, unchanged artifacts and failure propagation."""
import contextlib
from datetime import datetime, timezone
import hashlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ui14_progress as progress
from ui14_common import file_digest, read_json, read_jsonl, write_json, write_jsonl


class ProgressTests(unittest.TestCase):
    def test_eta_uses_completed_work_and_does_not_invent_a_total(self):
        with mock.patch.object(progress.time, "monotonic", return_value=100):
            counter = progress.Counter("images", 100, "images")
        self.assertIsNone(counter.snapshot(105)["eta_seconds"])
        counter.advance(20)
        state = counter.snapshot(110)
        self.assertEqual(state["eta_seconds"], 40)
        self.assertEqual(state["percent"], 20)
        counter.advance(80)
        self.assertEqual(counter.snapshot(120)["eta_seconds"], 0)
        counter.status = "failed"
        self.assertIsNone(counter.snapshot(120)["eta_seconds"])
        self.assertIsNone(progress.Counter("unknown").snapshot(200)["eta_seconds"])
        mixed = progress.Counter("unequal jobs", 14, estimate=False)
        mixed.advance(7)
        self.assertEqual(mixed.snapshot(mixed.started + 10)["percent"], 50)
        self.assertIsNone(mixed.snapshot(mixed.started + 10)["eta_seconds"])

    def test_idle_heartbeat_updates_without_advancing_or_completing_work(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            session = progress.ProgressSession("normalize", tmp, interval=.01)
            heartbeat_seen = threading.Event()
            original = session.emit

            def emit():
                original()
                if threading.current_thread() is session.thread and session.frames:
                    heartbeat_seen.set()

            with mock.patch.object(session, "emit", side_effect=emit), session:
                with progress.phase("blocked image read", 100, "images"):
                    self.assertTrue(heartbeat_seen.wait(2), "no heartbeat while foreground made no progress")
                    state = read_json(Path(tmp) / "progress/normalize.json")
                    self.assertEqual(state["status"], "running")
                    self.assertEqual(state["phases"][-1]["completed"], 0)
                    self.assertIsNone(state["phases"][-1]["eta_seconds"])
            self.assertFalse(session.thread.is_alive())
            self.assertEqual(read_json(Path(tmp) / "progress/normalize.json")["status"], "completed")

    def test_partial_failure_and_interrupt_are_not_reported_as_success(self):
        for exception, status in ((RuntimeError("fixture failure"), "failed"), (KeyboardInterrupt(), "interrupted")):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(type(exception)), progress.ProgressSession("finalize", tmp):
                    with progress.phase("check", 10, "images") as counter:
                        counter.advance(3)
                        raise exception
                state = read_json(Path(tmp) / "progress/finalize.json")
                self.assertEqual(state["status"], status)
                self.assertEqual(counter.completed, 3)
                self.assertEqual(counter.status, status)
                self.assertIsNone(progress._current)

    def test_tracking_is_lazy_and_counts_after_consumer_finishes(self):
        consumed = []

        def stream():
            for value in range(3):
                consumed.append(value)
                yield value

        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            with progress.ProgressSession("normalize", tmp) as session:
                iterator = progress.track(stream(), "stream", total=3)
                self.assertEqual(consumed, [])
                for index, value in enumerate(iterator):
                    self.assertEqual(value, index)
                    self.assertEqual(session.frames[-1].completed, index)
                self.assertEqual(consumed, [0, 1, 2])
                self.assertEqual(session.frames, [])
            with contextlib.redirect_stdout(io.StringIO()) as captured:
                self.assertEqual(list(progress.track(range(3), "quiet without session")), [0, 1, 2])
            self.assertEqual(captured.getvalue(), "")

    def test_io_contents_and_hashes_do_not_depend_on_progress(self):
        rows = [{"image": "示例.png", "bbox": [1, 2, 3, 4]}, {"value": "x" * 20000}]
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            write_jsonl(root / "plain.jsonl", rows)
            expected = (root / "plain.jsonl").read_bytes()
            with progress.ProgressSession("normalize", root):
                write_jsonl(root / "tracked.jsonl", iter(rows))
                self.assertEqual(list(read_jsonl(root / "tracked.jsonl")), rows)
                self.assertEqual(file_digest(root / "tracked.jsonl"), hashlib.sha256(expected).hexdigest())
                from ui5_eval_detector_cache import sha256_file
                self.assertEqual(sha256_file(root / "tracked.jsonl"), hashlib.sha256(expected).hexdigest())
            self.assertEqual((root / "tracked.jsonl").read_bytes(), expected)

    def test_detector_status_ignores_previous_run_then_mirrors_worker_eta(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            path = root / "run_status.json"
            status = dict(stage="text", status="completed", completed=100, total=100, unit="images",
                          percent=1, elapsed_seconds=10, rate_per_second=10, eta_seconds=0,
                          updated_at="2020-01-01T00:00:00+00:00")
            write_json(path, status)
            with mock.patch.object(progress.time, "time", return_value=2000000000), \
                 progress.ProgressSession("cache", root) as session, progress.detector_status(path):
                session.emit()
                self.assertIsNone(read_json(root / "progress/cache.json")["detector"])
                status.update(status="running", completed=20, percent=.2, eta_seconds=40,
                              updated_at=datetime.fromtimestamp(2000000001, timezone.utc).isoformat())
                write_json(path, status)
                session.emit()
                self.assertEqual(read_json(root / "progress/cache.json")["detector"]["eta_seconds"], 40)
                self.assertIn("detector text: running | 20/100 images", (root / "progress/cache.log").read_text(encoding="utf-8"))
                status["rate_per_second"] = None
                write_json(path, status)
                session.emit()
                self.assertIsNone(read_json(root / "progress/cache.json")["detector"])

    def test_log_write_error_cannot_mask_preparation_exception(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "real error"), progress.ProgressSession("cache", tmp):
                with mock.patch.object(progress.os, "replace", side_effect=OSError("log disk failure")):
                    progress._current.emit()
                    raise RuntimeError("real error")

    def test_invalid_interval_fails_before_creating_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            for interval in (0, -1, float("nan"), float("inf")):
                with self.assertRaises(ValueError):
                    progress.ProgressSession("cache", tmp, interval)
            self.assertFalse((Path(tmp) / "progress").exists())


class PreparationProgressTests(unittest.TestCase):
    def test_normalization_and_crop_artifacts_are_identical_with_progress(self):
        from tests.test_ui14_pipeline import source_fixture, detector_fixture
        from ui14_common import paths_for, UI_TASKS
        from ui14_repair import validate_normalization
        import prepare_ui14_sft as prepare
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp); source_fixture(root / "input")
            args = SimpleNamespace(ui9_data_root=root / "input", output_dir=root / "data",
                                   ui5_recipe="audited.json", ui5_test_dir=root / "old_test")
            prepare.normalize(args)
            # Per-invocation reuse/rebuild counters belong to the CPU report;
            # normalized data, split commits and every bound artifact remain identical.
            digests = {p: file_digest(p) for p in args.output_dir.rglob("*.json*") if p.name != "cpu_check_report.json"}
            with progress.ProgressSession("normalize", args.output_dir, .01):
                prepare.normalize(args)
            self.assertEqual(digests, {p: file_digest(p) for p in digests})
            validate_normalization(args.output_dir)
            task, split = UI_TASKS[7], "test"
            detector_fixture(args.output_dir, task, split)
            paths = paths_for(args.output_dir, task.task_key, split)
            records = list(read_jsonl(paths["normalized"]))
            prepare.crop_annotations(args.output_dir, task, split, records)
            expected = (paths["derived"].read_bytes(), (paths["cache"] / "ui14_label_cache_ready.json").read_bytes())
            with progress.ProgressSession("cache", args.output_dir, .01):
                prepare.crop_annotations(args.output_dir, task, split, records)
            self.assertEqual(expected, (paths["derived"].read_bytes(), (paths["cache"] / "ui14_label_cache_ready.json").read_bytes()))

    def test_cache_dispatch_keeps_all_14_splits_and_propagates_gpu_failure(self):
        from tests.test_ui14_pipeline import source_fixture
        from ui14_common import UI9_TASKS, SCAN_NAME
        import prepare_ui14_sft as prepare
        import prepare_ui14_detector_crops as cache
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp); source_fixture(root / "input")
            prepare.normalize(SimpleNamespace(ui9_data_root=root / "input", output_dir=root / "data",
                                             ui5_recipe="audited.json", ui5_test_dir=root / "old_test"))
            write_json(root / "old_cache/detections/detector_config.json", {
                "text": {"long_side": 960, "box_threshold": .3}, "icon": {"long_side": 640, "confidence": .1}})
            args = SimpleNamespace(data_root=root / "data", ui5_cache=root / "old_cache", parser_root=str(root),
                text_python=sys.executable, icon_python=sys.executable, gpus="0,1,2,3",
                text_model_dir=None, icon_model=None, progress_interval_seconds=5)
            args.stage = "detect"
            with mock.patch.object(cache.subprocess, "run") as worker, \
                 mock.patch.object(prepare, "crop_annotations") as labels, \
                 mock.patch.object(cache, "validate_prepared", return_value=1):
                with progress.ProgressSession("cache", args.data_root):
                    cache.run(args)
                labels.assert_not_called()
                self.assertEqual(worker.call_count, 28)
                self.assertEqual([call.args[0][call.args[0].index("--stage") + 1] for call in worker.call_args_list],
                                 ["text"] * 14 + ["icon"] * 14)
                for call in worker.call_args_list:
                    command = call.args[0]
                    self.assertEqual(command[1], "-u")
                    for flag, value in (("--progress-interval-seconds", "5"), ("--gpus", "0,1,2,3"), ("--scan-name", SCAN_NAME)):
                        self.assertEqual(command[command.index(flag) + 1], value)
                    self.assertTrue(call.kwargs["check"])
                    self.assertIn("--resume", command)
            with mock.patch.object(cache.subprocess, "run", side_effect=subprocess.CalledProcessError(3, "detector")) as worker, \
                 mock.patch.object(prepare, "crop_annotations") as labels, \
                 mock.patch.object(cache, "validate_prepared", return_value=1):
                with self.assertRaises(subprocess.CalledProcessError), progress.ProgressSession("cache", args.data_root):
                    cache.run(args)
                self.assertEqual(worker.call_count, 1)
                labels.assert_not_called()
                self.assertEqual(read_json(args.data_root / "progress/cache.json")["status"], "failed")


if __name__ == "__main__":
    unittest.main()
