"""CPU regressions for real PNG materialization, restart and verification evidence."""
import contextlib
import copy
import io
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from PIL import Image, ImageOps
import ui14_crop_materialization as crops
import ui14_cache_prepare as metadata
from ui14_common import UI_TASKS, SCAN_NAME, image_identity, paths_for, read_json, read_jsonl, write_json, write_jsonl
from ui14_annotations import training_record, crop_boxes
from ui14_verification import Journal, verification_session, preparation_lock
import prepare_ui14_sft as prepare


class ParallelCropTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.task = UI_TASKS[7]
        self.paths = paths_for(self.root, self.task.task_key, "train")
        self.rows, self.plans = [], []
        for i in range(3):
            source = self.root / f"source-{i}.png"
            image = Image.new("RGB", (120, 240))
            image.putdata([((x + i * 30) % 256, y % 256, (x*y) % 256) for y in range(240) for x in range(120)])
            exif = Image.Exif(); exif[274] = 0
            image.save(source, exif=exif); image.close()
            identity, w, h = image_identity(source)
            self.rows.append(dict(source_image=str(source), source_image_id=identity, width=w, height=h,
                source_record_id=f"row-{i}", source_dataset=self.task.source_dataset, source_version="fixture",
                task_key=self.task.task_key, task_id=self.task.task_id, split="train", normalization_id="fixture-id",
                boxes_px=[[10, 20, 60, 70], [10, 110, 40, 130]] if i != 1 else []))
            self.plans.append(dict(image_paths=[str(source)], width=w, height=h,
                                   tiles=[[0, 0, 120, 120], [0, 120, 120, 240]]))
        duplicate = copy.deepcopy(self.rows[0]); duplicate["source_record_id"] = "other-record-same-image"
        self.rows.insert(2, duplicate)
        write_json(self.root / "source_snapshot.json", {"normalization_id": "fixture-id", "repair_run_id": "fixture"})
        write_jsonl(self.paths["detector_input"], [{"fixture": True}])
        write_json(self.paths["detector_inputs"], {"fixture": True})
        write_json(self.paths["cache"] / SCAN_NAME / "eval_detector_cache_ready.json", {"fixture": True})
        self.publish()
        self.stack = contextlib.ExitStack(); self.addCleanup(self.stack.close)
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.stack.enter_context(mock.patch.dict("os.environ", {"UI14_CROP_WORKERS": "2", "UI14_PNG_COMPRESS_LEVEL": "1"}))
        self.stack.enter_context(mock.patch.object(prepare, "validate_task_cache", return_value={}))

    def publish(self):
        write_jsonl(self.paths["normalized"], self.rows)
        write_jsonl(self.paths["cache"] / SCAN_NAME / "detector_scan_crops.jsonl", self.plans)

    def run_crops(self):
        return prepare.crop_annotations(self.root, self.task, "train", self.rows)

    def test_parallel_pixels_order_labels_coverage_legacy_adoption_and_full_reuse(self):
        # Recreate an interrupted old serial run: PNGs exist, no split marker.
        self.paths["crop_images"].mkdir(parents=True)
        old_files = {}
        with Image.open(self.rows[0]["source_image"]) as image:
            for i, tile in enumerate(self.plans[0]["tiles"]):
                path = self.paths["crop_images"] / f"{self.rows[0]['source_image_id']}-scan-{i:03d}.png"
                image.convert("RGB").crop(tile).save(path)
                old_files[str(path)] = path.read_bytes()
        barrier = threading.Barrier(2); original = crops.materialize; threads = set()
        def run(job, *a, **kw):
            threads.add(threading.get_ident())
            # The first two submitted original-image jobs overlap in time.
            if job["rows"][0]["source_record_id"] in ("row-0", "row-1"):
                barrier.wait(timeout=5)
            return original(job, *a, **kw)
        with mock.patch.object(crops, "materialize", side_effect=run): result = self.run_crops()
        self.assertEqual(len(threads), 2)
        self.assertEqual(len(result), len(self.rows)*2)
        expected_coverage = []
        for index, row in enumerate(self.rows):
            covered = set()
            with Image.open(row["source_image"]) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                for tile_index, tile in enumerate(self.plans[0]["tiles"]):
                    record = result[index*2 + tile_index]
                    boxes, contained = crop_boxes(row["boxes_px"], tile); covered.update(contained)
                    expected = training_record(row, self.task, record["image"], boxes, 120, 120, f"scan-{tile_index:03d}")
                    self.assertEqual(record["conversations"], expected["conversations"])
                    self.assertEqual(record["crop_boxes_px"], boxes)
                    self.assertEqual(record["source_record_id"], row["source_record_id"])
                    with Image.open(record["image"]) as saved:
                        self.assertEqual(saved.convert("RGB").tobytes(), image.crop(tile).tobytes())
            expected_coverage.append(dict(source_record_id=row["source_record_id"], gt_count=len(row["boxes_px"]),
                fully_contained_gt=len(covered), uncontained_gt=sorted(set(range(len(row["boxes_px"]))) - covered)))
        self.assertEqual(read_json(self.paths["derived"].with_suffix(".coverage.json")), expected_coverage)
        for path, payload in old_files.items(): self.assertEqual(Path(path).read_bytes(), payload)
        report = read_json(self.root / "crop_performance/latest.json")
        self.assertEqual(report["counts"], {"built": 2, "migrated": 1, "reused": 0, "verified": 0})
        self.assertEqual(report["built_images_measured"], 2)  # Reuse must not inflate throughput.
        with mock.patch.object(Image, "open", side_effect=AssertionError("reused PNG/source decoded")), \
             mock.patch.object(crops, "atomic_png", side_effect=AssertionError("re-encoded valid PNG")), \
             mock.patch.dict("os.environ", {"UI14_PNG_COMPRESS_LEVEL": "9"}):
            self.assertEqual(self.run_crops(), result)
            self.assertEqual(crops.load_completed(self.root, self.task, "train", self.rows), result)
        self.assertEqual(read_json(self.root / "crop_performance/latest.json")["counts"]["reused"], 3)

    def test_interrupt_only_finishes_missing_images_and_corrupt_png_repaired(self):
        original = crops.materialize
        def interrupt(job, *a, **kw):
            if job["rows"][0]["source_record_id"] == "row-1": raise RuntimeError("interruption")
            return original(job, *a, **kw)
        with mock.patch.dict("os.environ", {"UI14_CROP_WORKERS": "1"}), \
             mock.patch.object(crops, "materialize", side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError, "interruption"): self.run_crops()
        self.assertFalse((self.paths["cache"] / "ui14_crop_complete.json").exists())
        with (self.paths["cache"] / "crop_index/images.jsonl").open("ab") as f: f.write(b'{"torn":')
        result = self.run_crops()
        self.assertGreaterEqual(read_json(self.root / "crop_performance/latest.json")["counts"]["reused"], 1)
        broken = Path(result[0]["image"]); broken.write_bytes(b"partial PNG")
        with mock.patch.object(crops, "atomic_png", wraps=crops.atomic_png) as save: repaired = self.run_crops()
        self.assertEqual(save.call_count, 1)
        with Image.open(repaired[0]["image"]) as restored: self.assertEqual(restored.size, (120, 120))
        self.assertEqual(read_json(self.root / "crop_performance/latest.json")["counts"]["reused"], 2)

    def test_label_plan_and_source_invalidation_are_local(self):
        first = self.run_crops()
        self.rows[1]["boxes_px"] = [[15, 25, 65, 75]]; self.publish()
        with mock.patch.object(Image, "open", side_effect=AssertionError("label-only change read pixels")):
            relabeled = self.run_crops()
        self.assertNotEqual(first[2]["conversations"], relabeled[2]["conversations"])
        self.assertEqual([r["image"] for r in first], [r["image"] for r in relabeled])
        self.plans[1]["tiles"] = [[0, 0, 120, 90], [0, 90, 120, 240]]; self.publish()
        self.run_crops()
        self.assertEqual(read_json(self.root / "crop_performance/latest.json")["counts"]["built"], 1)
        source = Path(self.rows[-1]["source_image"])
        Image.new("RGB", (120, 240), "blue").save(source)
        with self.assertRaisesRegex(ValueError, "Normalized screenshot content changed"): self.run_crops()
        # Simulate the matching normalize + detector/geometry refresh for that image.
        self.rows[-1]["source_image_id"] = image_identity(source)[0]; self.publish()
        self.run_crops()
        self.assertEqual(read_json(self.root / "crop_performance/latest.json")["counts"]["built"], 1)
        self.assertEqual(read_json(self.root / "crop_performance/latest.json")["counts"]["reused"], 2)

    def test_full_check_reads_pixels_and_rejects_tampered_crop(self):
        result = self.run_crops()
        with mock.patch.object(Image, "open", wraps=Image.open) as opened:
            crops.load_completed(self.root, self.task, "train", self.rows, full=True)
        self.assertGreater(opened.call_count, 0)
        Image.new("RGB", (120, 120), "pink").save(result[0]["image"])
        with self.assertRaisesRegex(RuntimeError, "PNG/geometry changed"):
            crops.load_completed(self.root, self.task, "train", self.rows, full=True)


class MetadataEvidenceTests(unittest.TestCase):
    def test_first_500_profile_excludes_migrations_and_keeps_all_split_eta(self):
        import time
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            meter = crops.CropMeter(tmp, 16, 1)
            meter.publish_interval = 1e9
            meter.pending = {"a/train": 600, "b/test": 100}
            sample = dict(built_crops=2, started=time.perf_counter() - 1, finished=time.perf_counter(),
                          read_seconds=.1, decode_seconds=.2, encode_seconds=.3, write_seconds=.4)
            for _ in range(100): meter.complete("a/train", {**sample, "status": "migrated"})
            for _ in range(499): meter.complete("a/train", {**sample, "status": "built"})
            path = Path(tmp) / "crop_performance/first_500_built.json"
            self.assertFalse(path.exists())
            meter.complete("a/train", {**sample, "status": "built"})
            report = read_json(path)
            self.assertEqual(report["built_images_measured"], 500)
            self.assertEqual(report["built_crops_measured"], 1000)
            self.assertEqual(report["pending_by_split"], {"a/train": 0, "b/test": 100})
            self.assertEqual(report["counts"]["migrated"], 100)
            self.assertGreater(report["eta_seconds_by_split"]["b/test"], 0)

    def test_stable_shards_only_invalidate_changed_image_shard(self):
        from run_ui5_crop_audit import AuditPaths, completed_shard_valid, digest_ids
        with tempfile.TemporaryDirectory() as tmp:
            paths = AuditPaths(Path(tmp)); paths.shards.mkdir(parents=True)
            rows = [dict(image_id=f"eval_{i}", image_path=f"{i}.png", image_paths=[f"{i}.png"],
                         content_id=str(i), width=20, height=40) for i in range(4)]
            metadata.refresh_stable_shards(paths, rows, 2)
            for shard in paths.shards.glob("*.jsonl"):
                members = list(read_jsonl(shard))
                write_jsonl(paths.stage_dir("text") / shard.name, members)
                write_json(paths.stage_dir("text") / (shard.stem + ".done.json"),
                    {"stage": "text", "count": len(members), "image_id_digest": digest_ids(r["image_id"] for r in members)})
            def valid():
                return [completed_shard_valid(p, paths.stage_dir("text") / p.name,
                    paths.stage_dir("text") / (p.stem + ".done.json"), "text") for p in sorted(paths.shards.glob("*.jsonl"))]
            before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths.shards.glob("*.jsonl")}
            metadata.refresh_stable_shards(paths, rows, 2)
            self.assertEqual(valid(), [True, True])
            self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths.shards.glob("*.jsonl")})
            rows[0] = {**rows[0], "image_id": "eval_z_changed", "content_id": "changed"}
            metadata.refresh_stable_shards(paths, rows, 2)
            self.assertEqual(valid(), [False, True])
            self.assertEqual(before["shard_00001.jsonl"], ((paths.shards / "shard_00001.jsonl").read_bytes(),
                (paths.shards / "shard_00001.jsonl").stat().st_mtime_ns))

    def test_dimensions_use_headers_and_orientation_swap_without_decode(self):
        with tempfile.TemporaryDirectory() as tmp:
            for orientation in (0, 1, 2, 3, 4, 5, 6, 7, 8):
                path = Path(tmp) / f"{orientation}.jpg"
                exif = Image.Exif(); exif[274] = orientation
                Image.new("RGB", (20, 40), "red").save(path, exif=exif)
                with mock.patch.object(ImageOps, "exif_transpose", side_effect=AssertionError("full transpose")), \
                     mock.patch("PIL.JpegImagePlugin.JpegImageFile.load", side_effect=AssertionError("full decode")):
                    row = metadata.inspect_image(str(path), metadata.stat_signature(path))
                self.assertEqual((row["width"], row["height"]), (40, 20) if orientation >= 5 else (20, 40))

    def test_verification_receipts_recheck_only_changes_and_full_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = root / "data.json"; path.write_text("first")
            calls = []
            def check(): calls.append(1); return {"sha": checks.sha256(path)}
            with verification_session(root) as checks: original = checks.validated_call({"fixture": 1}, check)
            with verification_session(root) as checks: self.assertEqual(checks.validated_call({"fixture": 1}, check), original)
            self.assertEqual(len(calls), 1)
            path.write_text("updated")
            with verification_session(root) as checks: self.assertNotEqual(checks.validated_call({"fixture": 1}, check), original)
            self.assertEqual(len(calls), 2)
            with verification_session(root, full=True) as checks: checks.validated_call({"fixture": 1}, check)
            self.assertEqual(len(calls), 3)

    def test_same_root_cannot_have_two_preparation_coordinators(self):
        with tempfile.TemporaryDirectory() as tmp, preparation_lock(tmp):
            with self.assertRaisesRegex(RuntimeError, "Another UI14"):
                with preparation_lock(tmp): pass


if __name__ == "__main__": unittest.main()
