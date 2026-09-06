"""CPU regressions: parallel preparation, legacy shards and GPU-only handoff."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from PIL import Image
import ui14_cache_prepare as metadata
import prepare_ui14_detector_crops as pipeline
import prepare_ui5_eval_detector_crops as detector
import prepare_ui14_sft as prepare
from ui14_common import read_json, read_jsonl, write_json, write_jsonl, file_digest, paths_for, UI9_TASKS, SCAN_NAME
from tests.test_ui14_pipeline import source_fixture


class ImagePreparationTests(unittest.TestCase):
    def test_parallel_scan_is_durable_and_reuse_never_opens_images(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            images = [root / f"{i}.png" for i in range(4)]
            for i, path in enumerate(images):
                exif = Image.Exif(); exif[274] = 0
                Image.new("RGB", (11 + i, 23), "red").save(path, exif=exif)
            barrier = threading.Barrier(2)
            worker_ids = set()
            original = metadata.inspect_image

            def scan(path, signature):
                worker_ids.add(threading.get_ident())
                barrier.wait(timeout=5)
                return original(path, signature)

            journal_path = root / "info.jsonl"
            journal = metadata.ImageInfoJournal(journal_path, workers=2)
            with mock.patch.object(metadata, "inspect_image", side_effect=scan):
                first = journal.load(images)
            self.assertEqual(len(worker_ids), 2)
            self.assertEqual(journal.totals, {"reused_images": 0, "scanned_images": 4})
            # Simulate a killed process leaving a partial final journal row.
            with journal_path.open("ab") as handle: handle.write(b'{"partial":')
            resumed = metadata.ImageInfoJournal(journal_path, workers=2)
            with mock.patch.object(Image, "open", side_effect=AssertionError("reused image opened")), \
                 mock.patch.object(metadata, "inspect_image", side_effect=AssertionError("reused image hashed")):
                self.assertEqual(resumed.load(images), first)
            self.assertEqual(resumed.totals, {"reused_images": 4, "scanned_images": 0})
            Image.new("RGB", (12, 25), "blue").save(images[0])
            with mock.patch.object(metadata, "inspect_image", wraps=original) as scan:
                changed = resumed.load(images)
            self.assertEqual(scan.call_count, 1)
            self.assertNotEqual(changed[str(images[0])], first[str(images[0])])

    def test_failure_keeps_completed_image_journal_entries(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            image = root / "ok.png"; Image.new("RGB", (10, 20)).save(image)
            journal = metadata.ImageInfoJournal(root / "info.jsonl", workers=1)
            def load(path, previous):
                if path.endswith("bad.png"):
                    raise RuntimeError("fixture interruption")
                return metadata.inspect_image(path, metadata.stat_signature(path)), False
            # Consume a durable image, then fail a subsequent split before its marker.
            journal.load([image])
            with mock.patch.object(journal, "_load_one", side_effect=load):
                with self.assertRaisesRegex(RuntimeError, "fixture interruption"):
                    journal.load([root / "bad.png"])
            with mock.patch.object(Image, "open", side_effect=AssertionError("image reopened")):
                metadata.ImageInfoJournal(root / "info.jsonl").load([image])

    def test_parallel_preparation_keeps_legacy_manifests_and_gpu_shards_identical(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            records = []
            for index, orientation in enumerate((0, 1, 6)):
                path = root / f"image{index}.png"
                exif = Image.Exif(); exif[274] = orientation
                Image.new("RGB", (20 + index, 40), "blue").save(path, exif=exif)
                records.append({"images": [str(path)], "ignored_gt": [1, 2, 3, 4]})
            write_jsonl(root / "input/source.jsonl", records)
            write_json(root / "input/tasks.json", {"synth_cropping": str(root / "input/source.jsonl")})
            args = detector.parse_args(["--stage", "prepare", "--input-dir", str(root / "input"),
                "--task-input-manifest", str(root / "input/tasks.json"), "--output-dir", str(root / "cache"),
                "--parser-root", str(root), "--resume", "--shard-size", "2", "--no-skip-figma"])
            legacy = detector.prepare_manifest(args)
            paths = detector.AuditPaths(args.output_dir)
            for stage in ("text", "icon"):
                for shard in paths.shards.glob("*.jsonl"):
                    rows = list(read_jsonl(shard))
                    write_jsonl(paths.stage_dir(stage) / shard.name, [{"image_id": row["image_id"]} for row in rows])
                    write_json(paths.stage_dir(stage) / (shard.stem + ".done.json"), {
                        "stage": stage, "count": len(rows), "image_id_digest": detector.digest_ids(r["image_id"] for r in rows)})
            before = {p: file_digest(p) for p in args.output_dir.rglob("*.json*") if p.name != "run_status.json"}
            journal = metadata.ImageInfoJournal(root / "info.jsonl", workers=3)
            self.assertEqual(detector.prepare_manifest(args, image_info_loader=journal.load), legacy)
            self.assertEqual(before, {p: file_digest(p) for p in before})
            from run_ui5_crop_audit import completed_shard_valid
            for stage in ("text", "icon"):
                for shard in paths.shards.glob("*.jsonl"):
                    self.assertTrue(completed_shard_valid(shard, paths.stage_dir(stage) / shard.name,
                        paths.stage_dir(stage) / (shard.stem + ".done.json"), stage))
            with mock.patch.object(Image, "open", side_effect=AssertionError("image reopened")):
                self.assertEqual(detector.prepare_manifest(args, image_info_loader=journal.load), legacy)
            # A kill after the last shard but before stage_summary must also resume.
            import run_ui5_crop_audit as audit
            with mock.patch.object(audit, "print_stage_preflight"), \
                 mock.patch.object(audit.subprocess, "run", side_effect=AssertionError("resume launched a GPU probe")):
                for stage in ("text", "icon"):
                    audit.run_detection_stage(args, stage)
                    summary = read_json(paths.stage_dir(stage) / "stage_summary.json")
                    self.assertEqual(summary["images"], len(legacy))
                    self.assertTrue(summary["restored_from_completed_shards"])


class StageSeparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__(); self.addCleanup(self.output.__exit__, None, None, None)
        self.root = Path(self.temp.name)
        source_fixture(self.root / "input")
        self.data = self.root / "data"
        prepare.normalize(SimpleNamespace(ui9_data_root=self.root / "input", output_dir=self.data,
                                          ui5_recipe="audited.json", ui5_test_dir=self.root / "old_test"))
        write_json(self.root / "old_cache/detections/detector_config.json", {
            "text": {"long_side": 960, "box_threshold": .3}, "icon": {"long_side": 640, "confidence": .1}})
        self.args = SimpleNamespace(stage="prepare", prepare_workers=4, data_root=self.data,
            ui5_cache=self.root / "old_cache", parser_root=str(self.root), text_python=sys.executable,
            icon_python=sys.executable, gpus="0,1,2,3", text_model_dir=None, icon_model=None,
            progress_interval_seconds=5)
        self.jobs = [(t, s) for t in UI9_TASKS if t.view_policy == "crops" for s in ("train", "test")]

    def prepare(self):
        self.args.stage = "prepare"
        with mock.patch.object(pipeline.subprocess, "run", side_effect=AssertionError("CPU started a subprocess")):
            pipeline.run(self.args)

    def test_prepare_needs_no_detector_environment_and_resumes_every_image(self):
        self.args.text_python = "unavailable-paddle-python"
        self.args.icon_python = "unavailable-icon-python"
        self.args.parser_root = str(self.root / "unavailable-parser-root")
        self.prepare()
        self.assertEqual(read_json(self.data / "cache_preparation/summary.json")["scanned_images"], 14)
        with mock.patch.object(Image, "open", side_effect=AssertionError("image reopened")), \
             mock.patch.object(metadata, "inspect_image", side_effect=AssertionError("image hashed")):
            self.prepare()
        report = read_json(self.data / "cache_preparation/summary.json")
        self.assertEqual((report["reused_images"], report["scanned_images"], report["splits"]), (14, 0, 14))
        self.assertFalse(read_json(self.data / "cpu_check_report.json")["ready"])

    def test_gpu_dispatch_only_text_icon_with_zero_original_image_scan(self):
        self.prepare()
        self.args.stage = "detect"
        with mock.patch.object(Image, "open", side_effect=AssertionError("driver decoded image")), \
             mock.patch.object(metadata, "stat_signature", side_effect=AssertionError("driver scanned image")), \
             mock.patch.object(detector, "prepare_manifest", side_effect=AssertionError("GPU stage prepared images")), \
             mock.patch.object(prepare, "crop_annotations", side_effect=AssertionError("GPU stage made crops")), \
             mock.patch.object(pipeline.subprocess, "run") as worker:
            pipeline.run(self.args)
        self.assertEqual(worker.call_count, 28)
        stages = [c.args[0][c.args[0].index("--stage") + 1] for c in worker.call_args_list]
        self.assertEqual(stages, ["text"] * 14 + ["icon"] * 14)
        for call in worker.call_args_list:
            self.assertIn("--resume", call.args[0]); self.assertTrue(call.kwargs["check"])
            self.assertEqual(call.args[0][call.args[0].index("--workers-per-gpu") + 1], "4")
            self.assertEqual(call.args[0][call.args[0].index("--image-loader-threads") + 1], "2")
            self.assertIn("--allow-multiple-processes-per-gpu", call.args[0])
        # Worker count is not part of the prepared detector configuration.
        self.args.detector_workers_per_gpu = 5
        with mock.patch.object(Image, "open", side_effect=AssertionError("driver decoded image")), \
             mock.patch.object(metadata, "stat_signature", side_effect=AssertionError("driver scanned image")), \
             mock.patch.object(pipeline.subprocess, "run") as worker:
            pipeline.run(self.args)
        self.assertEqual(worker.call_count, 28)
        for call in worker.call_args_list:
            self.assertEqual(call.args[0][call.args[0].index("--workers-per-gpu") + 1], "5")

    def test_missing_final_prepare_marker_recovers_without_rebuilding_manifests_or_images(self):
        self.prepare()
        task, split = self.jobs[4]
        paths = paths_for(self.data, task.task_key, split)
        (paths["cache"] / "manifest/ui14_prepare_ready.json").unlink()
        # An existing detector shard must remain untouched by CPU recovery.
        write_json(paths["cache"] / "detections/text/shard_00000.done.json", {"fixture": "preserve"})
        files = metadata.prepared_files(paths["cache"]) + [paths["cache"] / "detections/text/shard_00000.done.json"]
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in files}
        with mock.patch.object(Image, "open", side_effect=AssertionError("recovery decoded image")), \
             mock.patch.object(metadata, "inspect_image", side_effect=AssertionError("recovery hashed original")), \
             mock.patch.object(detector, "prepare_manifest", side_effect=AssertionError("recovery rebuilt manifests")), \
             mock.patch.object(pipeline, "recover_prepared", wraps=metadata.recover_prepared) as recovered:
            self.prepare()
        recovered.assert_called_once()
        self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in files})
        self.assertTrue((paths["cache"] / "manifest/ui14_prepare_ready.json").is_file())

    def test_recovery_rejects_truncated_shards_and_stale_selection(self):
        self.prepare()
        task, split = self.jobs[4]; paths = paths_for(self.data, task.task_key, split)
        marker = paths["cache"] / "manifest/ui14_prepare_ready.json"; marker.unlink()
        config = read_json(paths["cache"] / "detections/detector_config.json")
        normalization_id = read_json(self.data / "source_snapshot.json")["normalization_id"]
        journal = metadata.ImageInfoJournal(self.data / "cache_preparation/image_info.jsonl")
        shard = next((paths["cache"] / "manifest/shards").glob("*.jsonl"))
        payload = shard.read_bytes(); shard.write_bytes(b"")
        self.assertIsNone(metadata.recover_prepared(paths, normalization_id, config, journal))
        self.assertFalse(marker.exists())
        shard.write_bytes(payload)
        selection = paths["cache"] / "manifest/selection_config.json"
        value = read_json(selection); value["data_split"] = "wrong-split"; write_json(selection, value)
        self.assertIsNone(metadata.recover_prepared(paths, normalization_id, config, journal))
        self.assertFalse(marker.exists())

    def test_missing_or_tampered_last_split_fails_before_any_gpu_work(self):
        self.prepare(); self.args.stage = "detect"
        task, split = self.jobs[-1]
        paths = paths_for(self.data, task.task_key, split)
        marker = paths["cache"] / "manifest/ui14_prepare_ready.json"
        marker_bytes = marker.read_bytes()
        marker.unlink()
        with mock.patch.object(pipeline.subprocess, "run") as worker:
            with self.assertRaisesRegex(RuntimeError, "cache-prepare"):
                pipeline.run(self.args)
            worker.assert_not_called()
        marker.write_bytes(marker_bytes)
        shard = next((paths["cache"] / "manifest/shards").glob("*.jsonl"))
        with shard.open("a") as handle: handle.write("{}\n")
        with mock.patch.object(pipeline.subprocess, "run") as worker:
            with self.assertRaisesRegex(RuntimeError, "prepared manifest/shards changed"):
                pipeline.run(self.args)
            worker.assert_not_called()

    def test_old_five_task_registry_cannot_silently_skip_all_gpu_jobs(self):
        registry = pipeline.load_registry(self.data / "task_registry.json")
        self.args.stage = "detect"
        with mock.patch.object(pipeline, "load_registry", return_value=registry[:5]), \
             mock.patch.object(pipeline.subprocess, "run") as worker:
            with self.assertRaisesRegex(ValueError, "complete 14-task registry"):
                pipeline.run(self.args)
            worker.assert_not_called()

    def test_detector_failure_stops_and_never_starts_cpu_crops(self):
        self.prepare(); self.args.stage = "detect"
        with mock.patch.object(pipeline.subprocess, "run", side_effect=subprocess.CalledProcessError(3, "fixture")) as worker, \
             mock.patch.object(prepare, "crop_annotations") as labels:
            with self.assertRaises(subprocess.CalledProcessError): pipeline.run(self.args)
            self.assertEqual(worker.call_count, 1); labels.assert_not_called()

    def test_cpu_finish_only_merge_crop_and_all_split_labels(self):
        self.prepare(); self.args.stage = "crops"
        self.args.text_python = "unavailable-paddle-python"
        with mock.patch.object(pipeline.subprocess, "run") as worker, mock.patch.object(prepare, "crop_annotations") as labels:
            pipeline.run(self.args)
        self.assertEqual([c.args[0][c.args[0].index("--stage") + 1] for c in worker.call_args_list],
                         ["merge"] * 14 + ["crop"] * 14)
        self.assertTrue(all("--text-python" not in c.args[0] for c in worker.call_args_list))
        self.assertEqual([(c.args[1], c.args[2]) for c in labels.call_args_list], self.jobs)

    def test_real_cpu_merge_crop_and_labels_accept_prepared_handoff(self):
        # Exercise the actual CPU subprocess CLI and crop-label connection for
        # one train/test pair. Only GPU outputs are fixed CPU fixtures.
        self.prepare()
        task = next(t for t in UI9_TASKS if t.task_key == "synth_cropping")
        for split in ("train", "test"):
            paths = paths_for(self.data, task.task_key, split)
            audit = detector.AuditPaths(paths["cache"])
            for step in ("text", "icon"):
                for shard in audit.shards.glob("*.jsonl"):
                    rows = list(read_jsonl(shard))
                    write_jsonl(audit.stage_dir(step) / shard.name, [
                        {"image_id": r["image_id"], "image": r["image_path"], "width": r["width"], "height": r["height"],
                         f"{step}_detections": [{"bbox": [20, 80, 200, 160], "score": .9}], "inference_ms": 1}
                        for r in rows])
                    write_json(audit.stage_dir(step) / (shard.stem + ".done.json"), {
                        "stage": step, "count": len(rows), "image_id_digest": detector.digest_ids(r["image_id"] for r in rows)})
                write_json(audit.stage_dir(step) / "stage_summary.json", {"images": 1, "runtime": {"fixture_only": True}})
        self.args.stage = "crops"
        self.args.text_python = "unavailable-paddle-python"
        cpu_run, label_run = subprocess.run, prepare.crop_annotations

        def worker(command, **kwargs):
            output = Path(command[command.index("--output-dir") + 1])
            return cpu_run(command, **kwargs) if output.parent.name == task.task_key else subprocess.CompletedProcess(command, 0)

        def labels(root, current, split, rows, **kwargs):
            return label_run(root, current, split, rows) if current == task else []

        with mock.patch.object(pipeline.subprocess, "run", side_effect=worker), \
             mock.patch.object(prepare, "crop_annotations", side_effect=labels):
            pipeline.run(self.args)
        for split in ("train", "test"):
            paths = paths_for(self.data, task.task_key, split)
            self.assertTrue((paths["cache"] / "ui14_label_cache_ready.json").is_file())
            self.assertGreater(len(list(read_jsonl(paths["derived"]))), 0)


if __name__ == "__main__": unittest.main()
