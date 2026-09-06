"""CPU-only tests of GPU worker scheduling; detectors and GPU probes are mocked."""
import contextlib
import io
import itertools
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from PIL import Image
import prepare_ui5_eval_detector_crops as entry
import run_ui5_crop_audit as audit


class DetectorWorkersTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        redirect = contextlib.redirect_stdout(io.StringIO())
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)

    def args(self, count=4):
        return entry.parse_args([
            "--output-dir", str(self.root / "cache"), "--parser-root", str(self.root),
            "--workers-per-gpu", str(count), "--allow-multiple-processes-per-gpu",
            "--text-python", sys.executable, "--icon-python", sys.executable,
            "--resume", "--image-loader-threads", "2",
        ])

    def fixture(self, args, stage, image_count=27, completed_ids=(0, 1, 2)):
        paths = audit.AuditPaths(args.output_dir)
        unique = []
        for index in range(image_count):
            image_path = self.root / f"image{index}.png"
            Image.new("RGB", (12, 24), (index, 0, 0)).save(image_path)
            row = {"image_id": str(index), "image_path": str(image_path), "width": 12, "height": 24}
            unique.append(row)
            shard = paths.shards / f"shard_{index:05d}.jsonl"
            audit.atomic_write_jsonl(shard, [row])
            if index in completed_ids:
                audit.atomic_write_jsonl(paths.stage_dir(stage) / shard.name, [
                    {**row, f"{stage}_detections": [], "inference_ms": 10}])
                audit.atomic_write_json(paths.stage_dir(stage) / f"{shard.stem}.done.json", {
                    "stage": stage, "count": 1, "image_id_digest": audit.digest_ids([str(index)])})
        audit.atomic_write_jsonl(paths.unique_images, unique)
        audit.ensure_detector_config(paths.detector_config, audit.detector_config(args))
        protected = [paths.unique_images, paths.detector_config, *paths.shards.glob("*.jsonl"),
                     *paths.stage_dir(stage).glob("shard_*")]
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in protected}
        # An interrupted shard with no completion marker must be recomputed.
        if image_count > 3 and 3 not in completed_ids:
            audit.atomic_write_jsonl(paths.stage_dir(stage) / "shard_00003.jsonl", [])
        return paths, before

    def fake_module(self, seen):
        class Detector:
            def __init__(self, *args, **kwargs):
                pass

            def predict(self, image, **kwargs):
                seen.append(image.getpixel((0, 0))[0])
                return [{"bbox": [1, 2, 4, 8], "score": .8}]
        return SimpleNamespace(PaddleTextDetector=Detector, OmniParserYOLOv9Detector=Detector)

    def test_flags_forward_to_workers_and_leave_detector_identity_unchanged(self):
        baseline = audit.detector_config(self.args(1))
        for count in (1, 2, 4, 5):
            args = self.args(count)
            audit.validate_detector_worker_count(args)
            self.assertEqual(audit.detector_config(args), baseline)
            for stage in ("text", "icon"):
                command = audit.detection_worker_command(args, stage, 3, count * 4)
                child = entry.parse_args(command[2:])
                audit.validate_detector_worker_count(child)
                self.assertEqual(child.workers_per_gpu, count)
                self.assertEqual(child.worker_count, count * 4)
        legacy = self.args(2)
        legacy.allow_multiple_processes_per_gpu = False
        legacy.allow_two_processes_per_gpu = True
        audit.validate_detector_worker_count(legacy)
        self.assertIn("--allow-two-processes-per-gpu", audit.detection_worker_command(legacy, "text", 0, 8))
        legacy.workers_per_gpu = 4
        with self.assertRaises(ValueError):
            audit.validate_detector_worker_count(legacy)
        legacy.workers_per_gpu = 6
        with self.assertRaises(ValueError):
            audit.validate_detector_worker_count(legacy)
        legacy_cli = audit.parse_args(["--source-dir", ".", "--locany-data-dir", ".",
            "--parser-root", ".", "--output-dir", ".", "--workers-per-gpu", "5",
            "--allow-multiple-processes-per-gpu"])
        audit.validate_detector_worker_count(legacy_cli)

    def test_four_and_five_workers_resume_without_duplicate_or_missing_images(self):
        for count, stage in itertools.product((4, 5), ("text", "icon")):
            with self.subTest(count=count, stage=stage):
                args = self.args(count)
                args.output_dir = self.root / f"cache-{count}-{stage}"
                paths, before = self.fixture(args, stage)
                args.detector_stage = stage
                args.worker_count = count * 4
                seen = []
                with mock.patch.object(audit, "load_parser_module", return_value=self.fake_module(seen)):
                    for index in range(args.worker_count):
                        args.worker_index = index
                        audit.run_detector_worker(args)
                self.assertEqual(sorted(seen), list(range(3, 27)))
                self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before})
                for shard in paths.shards.glob("*.jsonl"):
                    self.assertTrue(audit.completed_shard_valid(shard, paths.stage_dir(stage) / shard.name,
                        paths.stage_dir(stage) / f"{shard.stem}.done.json", stage))
                # A different worker count also resumes without loading any model or image.
                args.worker_count = 20 if count == 4 else 16
                with mock.patch.object(audit, "load_parser_module", side_effect=AssertionError("loaded model")), \
                     mock.patch.object(audit, "open_raw_image", side_effect=AssertionError("opened image")):
                    for index in range(args.worker_count):
                        args.worker_index = index
                        audit.run_detector_worker(args)

    def test_coordinator_launches_per_gpu_slots_and_counts_only_new_throughput(self):
        for count in (4, 5):
            with self.subTest(count=count):
                args = self.args(count)
                args.output_dir = self.root / f"parent-{count}"
                paths, before = self.fixture(args, "text")
                stale = paths.stage_dir("text") / "progress/worker_99.json"
                audit.atomic_write_json(stale, {"current_shard": "shard_00026.jsonl", "current_shard_completed": 999})
                seen, launched = [], []

                def launch(command, env):
                    self.assertFalse(stale.exists())
                    child = entry.parse_args(command[2:])
                    audit.validate_detector_worker_count(child)
                    launched.append((child.worker_index, child.worker_count, env["CUDA_VISIBLE_DEVICES"]))
                    audit.run_detector_worker(child)
                    return SimpleNamespace(poll=lambda: 0, returncode=0)

                with mock.patch.object(audit, "print_stage_preflight"), \
                     mock.patch.object(audit.shutil, "which", return_value="nvidia-smi"), \
                     mock.patch.object(audit.subprocess, "run", return_value=SimpleNamespace(stdout="0\n1\n2\n3\n")), \
                     mock.patch.object(audit, "preflight_text_runtime", return_value={}), \
                     mock.patch.object(audit, "load_parser_module", return_value=self.fake_module(seen)), \
                     mock.patch.object(audit.time, "perf_counter", side_effect=itertools.count()), \
                     mock.patch.object(audit.subprocess, "Popen", side_effect=launch):
                    audit.run_detection_stage(args, "text")
                self.assertEqual(launched, [(i, 4 * count, str(i % 4)) for i in range(4 * count)])
                self.assertEqual(sorted(seen), list(range(3, 27)))
                self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before})
                summary_path = paths.stage_dir("text") / "stage_summary.json"
                summary_bytes = summary_path.read_bytes()
                summary = json.loads(summary_bytes)
                self.assertEqual((summary["reused_images"], summary["new_images"]), (3, 24))
                self.assertAlmostEqual(summary["throughput_images_per_second"] * summary["wall_seconds"], 24, places=3)
                with mock.patch.object(audit, "print_stage_preflight"), \
                     mock.patch.object(audit.subprocess, "Popen", side_effect=AssertionError("GPU started")), \
                     mock.patch.object(audit.subprocess, "run", side_effect=AssertionError("GPU probed")):
                    args.workers_per_gpu = 5 if count == 4 else 4
                    audit.run_detection_stage(args, "text")
                self.assertEqual(summary_path.read_bytes(), summary_bytes)

    def test_failed_worker_or_interrupted_launch_stops_siblings(self):
        for interrupt in (False, True):
            with self.subTest(interrupt=interrupt):
                args = self.args()
                args.output_dir = self.root / f"failure-{interrupt}"
                paths, before = self.fixture(args, "text")
                processes = []

                def launch(command, env):
                    if interrupt and processes:
                        raise KeyboardInterrupt()
                    process = mock.Mock()
                    process.poll.return_value = 1 if not interrupt and not processes else None
                    processes.append(process)
                    return process

                with mock.patch.object(audit, "print_stage_preflight"), \
                     mock.patch.object(audit.shutil, "which", return_value="nvidia-smi"), \
                     mock.patch.object(audit.subprocess, "run", return_value=SimpleNamespace(stdout="0\n1\n2\n3\n")), \
                     mock.patch.object(audit, "preflight_text_runtime", return_value={}), \
                     mock.patch.object(audit.subprocess, "Popen", side_effect=launch):
                    with self.assertRaises(KeyboardInterrupt if interrupt else RuntimeError):
                        audit.run_detection_stage(args, "text")
                for process in (processes if interrupt else processes[1:]):
                    process.terminate.assert_called_once()
                    process.wait.assert_called_once()
                self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before})
                self.assertFalse((paths.stage_dir("text") / "stage_summary.json").exists())

    def test_small_split_and_sparse_resume_use_all_four_gpus(self):
        for count, stage, sparse in itertools.product((4, 5), ("text", "icon"), (False, True)):
            with self.subTest(count=count, stage=stage, sparse=sparse):
                args = self.args(count)
                args.output_dir = self.root / f"balanced-{count}-{stage}-{sparse}"
                # 5907 images / 750 => 8 shards. Tiny CPU fixtures retain that
                # topology; sparse resume also defeats merely reordering GPUs
                # while keeping the old index % worker_count assignment.
                image_count = 27 if sparse else 8
                pending_ids = [0, 4, 16, 20] if sparse else list(range(8))
                completed = set(range(image_count)) - set(pending_ids)
                paths, before = self.fixture(args, stage, image_count, completed)
                seen, launched, dispatched = [], [], []

                def launch(command, env):
                    child = entry.parse_args(command[2:])
                    self.assertTrue(child.assigned_shards)
                    launched.append(env["CUDA_VISIBLE_DEVICES"])
                    dispatched.extend(child.assigned_shards)
                    audit.run_detector_worker(child)
                    return SimpleNamespace(poll=lambda: 0, returncode=0)

                with mock.patch.object(audit, "print_stage_preflight"), \
                     mock.patch.object(audit.shutil, "which", return_value="nvidia-smi"), \
                     mock.patch.object(audit.subprocess, "run", return_value=SimpleNamespace(stdout="0\n1\n2\n3\n")), \
                     mock.patch.object(audit, f"preflight_{stage}_runtime", return_value={}), \
                     mock.patch.object(audit, "load_parser_module", return_value=self.fake_module(seen)), \
                     mock.patch.object(audit.subprocess, "Popen", side_effect=launch):
                    audit.run_detection_stage(args, stage)
                self.assertEqual(launched, ["0", "1", "2", "3"] * (1 if sparse else 2))
                self.assertEqual(sorted(dispatched), [f"shard_{i:05d}.jsonl" for i in pending_ids])
                self.assertEqual(sorted(seen), pending_ids)
                self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before})
                summary = json.loads((paths.stage_dir(stage) / "stage_summary.json").read_text())
                self.assertEqual(summary["active_workers_per_gpu"], {str(i): 1 if sparse else 2 for i in range(4)})
                self.assertEqual(summary["new_images"], len(pending_ids))
                self.assertEqual(summary["reused_images"], len(completed))
                self.assertEqual(len(summary["worker_assignments"]), len(pending_ids))

    def test_dispatch_balances_slots_and_keeps_every_pending_shard_exactly_once(self):
        gpus = ["2", "5", "7", "9"]
        for count, size in itertools.product((1, 4, 5), (0, 1, 2, 3, 4, 8, 16, 19, 23, 71)):
            pending = [Path(f"shard_{i * 16:05d}.jsonl") for i in range(size)]
            plan = audit.detector_worker_assignments(gpus, count, pending)
            self.assertEqual(len(plan), min(size, count * 4))
            self.assertEqual([gpu for gpu, _, _ in plan], [gpus[i % 4] for i in range(len(plan))])
            self.assertTrue(all(slot < count and shards for _, slot, shards in plan))
            self.assertEqual(sorted(p for _, _, shards in plan for p in shards), pending)
        args = self.args()
        command = audit.detection_worker_command(args, "text", 0, 4, [Path("shard_00016.jsonl")])
        # Both the legacy audit and detector-only entrypoints accept explicit dispatch.
        legacy = audit.parse_args(command[2:] + ["--source-dir", ".", "--locany-data-dir", "."])
        self.assertEqual(legacy.assigned_shards, ["shard_00016.jsonl"])

    def test_worker_rejects_duplicate_or_unknown_assigned_shards(self):
        args = self.args()
        self.fixture(args, "text")
        args.detector_stage, args.worker_index, args.worker_count = "text", 0, 4
        for names in (["shard_00003.jsonl"] * 2, ["../shard_00003.jsonl"], ["shard_99999.jsonl"]):
            args.assigned_shards = names
            with mock.patch.object(audit, "load_parser_module", side_effect=AssertionError("model loaded")):
                with self.assertRaisesRegex(ValueError, "assigned shards"):
                    audit.run_detector_worker(args)


class ImagePrefetchTests(unittest.TestCase):
    def test_prefetch_is_bounded_ordered_and_releases_pending_images_on_close(self):
        images = []
        lock = threading.Lock()
        filled = threading.Event()

        def load(path):
            image = Image.new("RGB", (10, 20))
            with lock:
                images.append(image)
                if len(images) == 4:
                    filled.set()
            return image

        rows = [{"image_id": str(i), "image_path": str(i), "width": 10, "height": 20} for i in range(100)]
        with mock.patch.object(audit, "open_raw_image", side_effect=load):
            loader = audit._loaded_images(rows, 2)
            row, image = next(loader)
            self.assertEqual(row["image_id"], "0")
            self.assertTrue(filled.wait(timeout=5))
            loader.close()
            image.close()
        self.assertEqual(len(images), 4)
        for image in images:
            with self.assertRaises(ValueError):
                image.getpixel((0, 0))
        with mock.patch.object(audit, "open_raw_image", side_effect=lambda p: Image.new("RGB", (10, 20))):
            seen = []
            for row, image in audit._loaded_images(rows, 2):
                seen.append(row["image_id"])
                image.close()
            self.assertEqual(seen, [str(i) for i in range(100)])


if __name__ == "__main__":
    unittest.main()
