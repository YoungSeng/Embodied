"""Bounded parallel crop materialization with per-image durable completion."""
from __future__ import annotations
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from collections import OrderedDict
import hashlib
import io
import os
from pathlib import Path
import threading
import time
from datetime import datetime, timezone

from PIL import Image, ImageOps
from ui14_common import SCAN_NAME, digest, file_digest, paths_for, read_json, read_jsonl, write_json, write_jsonl
from ui14_annotations import training_record, crop_boxes
from ui14_repair import cache_label_binding
from ui14_progress import phase, duration
from ui14_verification import Journal, current_checks, signature, rgb_identity, verification_session

VERSION = 1


def crop_settings():
    workers = int(os.environ.get("UI14_CROP_WORKERS", "16"))
    compression = int(os.environ.get("UI14_PNG_COMPRESS_LEVEL", "1"))
    if workers < 1 or not 0 <= compression <= 9:
        raise ValueError("UI14_CROP_WORKERS must be positive; PNG compression must be 0..9")
    return workers, compression


def image_jobs(root, task, split, records):
    paths = paths_for(root, task.task_key, split)
    plans = list(read_jsonl(paths["cache"] / SCAN_NAME / "detector_scan_crops.jsonl"))
    by_path = {str(Path(p).resolve()): plan for plan in plans for p in plan["image_paths"]}
    jobs = OrderedDict()
    for position, row in enumerate(records):
        plan = by_path[row["source_image"]]
        if (plan["width"], plan["height"]) != (row["width"], row["height"]):
            raise ValueError("Crop cache screenshot dimensions changed")
        geometry = {"width": plan["width"], "height": plan["height"], "tiles": plan["tiles"],
                    "colors": "ImageOps.exif_transpose->RGB", "gt_used": False}
        key = digest([row["source_image_id"], geometry])
        job = jobs.setdefault(key, {"key": key, "geometry": geometry, "rows": [], "positions": []})
        job["rows"].append(row); job["positions"].append(position)
    return paths, list(jobs.values()), len(plans)


def physical_binding(job):
    return {"version": VERSION, "source_image_id": job["rows"][0]["source_image_id"],
            "sources": {r["source_image"]: signature(r["source_image"]) for r in job["rows"]},
            "geometry": job["geometry"]}


def labels_binding(job, task, split, physical):
    return {"version": VERSION, "task_key": task.task_key, "task_id": task.task_id, "split": split,
            "prompt": task.prompt, "prompt_label": task.prompt_label,
            "records": digest([{k: v for k, v in r.items() if k != "source_metadata"} for r in job["rows"]]),
            "physical": digest(physical)}


def physical_valid(state, binding, checks):
    if not state or state.get("binding") != binding: return False
    files = state.get("files", [])
    if len(files) != len(binding["geometry"]["tiles"]): return False
    try:
        for item in files:
            if signature(item["path"]) != item["stat"]:
                if checks.sha256(item["path"]) != item["sha256"]: return False
        return True
    except (OSError, KeyError, ValueError): return False


def atomic_png(path, image, compression, timing):
    started = time.perf_counter()
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=compression, optimize=False)
    payload = buffer.getbuffer()
    checksum = hashlib.sha256(payload).hexdigest()
    timing["encode_seconds"] += time.perf_counter() - started
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}-{threading.get_ident()}")
    started = time.perf_counter()
    try:
        with temporary.open("wb") as f:
            f.write(payload); f.flush(); os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
        payload.release(); buffer.close()
    timing["write_seconds"] += time.perf_counter() - started
    return checksum


def materialize(job, previous, task, split, paths, checks, compression, *, readonly=False, legacy_hashes=None):
    started = time.perf_counter()
    timing = {k: 0.0 for k in ("read_seconds", "decode_seconds", "encode_seconds", "write_seconds")}
    binding = physical_binding(job)
    old_physical = previous.get("physical", {}) if previous else {}
    valid = physical_valid(old_physical, binding, checks)
    built, migrated = 0, 0
    if valid and not checks.full:
        physical = old_physical
        status = "reused"
    else:
        if readonly and not valid: raise RuntimeError("Crop completion changed; run cache-finalize")
        row = job["rows"][0]; source = row["source_image"]
        before = time.perf_counter(); encoded = Path(source).read_bytes()
        timing["read_seconds"] += time.perf_counter() - before
        before = time.perf_counter()
        with Image.open(io.BytesIO(encoded)) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
        actual_identity = rgb_identity(image)
        if actual_identity != (row["source_image_id"], row["width"], row["height"]):
            image.close()
            raise ValueError(f"Normalized screenshot content changed: {source}; update normalization/detector first")
        checks.remember(source, "rgb_identity", actual_identity, binding["sources"][source])
        timing["decode_seconds"] += time.perf_counter() - before
        del encoded
        files = []
        try:
            for index, tile in enumerate(job["geometry"]["tiles"]):
                expected = image.crop(tile)
                name = f"{row['source_image_id']}-{job['key'][:16]}-scan-{index:03d}.png"
                target = paths["crop_images"] / name
                old = old_physical.get("files", [])
                candidates = ([Path(old[index]["path"])] if index < len(old) else []) + [
                    paths["crop_images"] / f"{row['source_image_id']}-scan-{index:03d}.png", target]
                if readonly: candidates = [Path(old[index]["path"])]
                checksum = None
                for candidate in dict.fromkeys(candidates):
                    if not candidate.is_file(): continue
                    before = time.perf_counter()
                    payload = candidate.read_bytes()
                    timing["read_seconds"] += time.perf_counter() - before
                    actual_hash = hashlib.sha256(payload).hexdigest()
                    # An audited legacy split can bind its existing byte hash.
                    # Partial old directories have no such evidence: compare pixels.
                    trusted_hash = (legacy_hashes or {}).get(str(candidate))
                    matches = bool(trusted_hash and trusted_hash == actual_hash and not checks.full)
                    if not matches:
                        before = time.perf_counter()
                        try:
                            with Image.open(io.BytesIO(payload)) as saved:
                                matches = saved.size == expected.size and saved.convert("RGB").tobytes() == expected.tobytes()
                        except (OSError, ValueError): matches = False
                        timing["decode_seconds"] += time.perf_counter() - before
                    if matches:
                        target, checksum = candidate, actual_hash
                        migrated += int(not valid)
                        break
                if checksum is None:
                    if readonly: raise RuntimeError(f"Crop pixels differ from source and plan: {target}")
                    checksum = atomic_png(target, expected, compression, timing)
                    built += 1
                expected.close()
                checks.remember(target, "sha256", checksum)
                files.append({"path": str(target), "stat": signature(target), "sha256": checksum,
                              "width": tile[2] - tile[0], "height": tile[3] - tile[1]})
        finally: image.close()
        if physical_binding(job) != binding: raise ValueError("Source changed during crop materialization")
        physical = {"binding": binding, "files": files}
        status = "built" if built else "migrated" if not valid else "verified"
    label_binding = labels_binding(job, task, split, physical)
    if previous and previous.get("labels_binding") == label_binding:
        derived, coverage = previous["derived"], previous["coverage"]
        labels_reused = True
    else:
        derived, coverage = [], []
        for row in job["rows"]:
            covered, output = set(), []
            for index, (tile, item) in enumerate(zip(job["geometry"]["tiles"], physical["files"])):
                boxes, complete = crop_boxes(row["boxes_px"], tile)
                covered.update(complete)
                record = training_record(row, task, item["path"], boxes, item["width"], item["height"], f"scan-{index:03d}")
                record.update(crop_box=tile, crop_width=item["width"], crop_height=item["height"],
                              detector_plan_gt_used=False, crop_boxes_px=boxes, crop_image_sha256=item["sha256"])
                output.append(record)
            derived.append(output)
            coverage.append({"source_record_id": row["source_record_id"], "gt_count": len(row["boxes_px"]),
                "fully_contained_gt": len(covered), "uncontained_gt": sorted(set(range(len(row["boxes_px"]))) - covered)})
        labels_reused = False
    return {"physical": physical, "labels_binding": label_binding, "derived": derived, "coverage": coverage}, {
        "status": status, "built_crops": built, "reused_crops": len(physical["files"]) - built,
        "migrated_crops": migrated, "labels_reused": labels_reused, "started": started,
        "finished": time.perf_counter(), **timing}


def process_resources():
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if os.sys.platform == "darwin" else 1024)
    except ImportError: peak = None
    try:
        import psutil
        return {"rss_bytes": psutil.Process().memory_info().rss, "peak_rss_bytes": peak}
    except ImportError:
        try:
            rss = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, AttributeError): rss = None
        return {"rss_bytes": rss, "peak_rss_bytes": peak}


class CropMeter:
    def __init__(self, root, workers, compression):
        self.root, self.workers, self.compression = Path(root), workers, compression
        self.started, self.cpu_start = time.perf_counter(), time.process_time()
        self.built = []; self.counts = {"reused": 0, "built": 0, "migrated": 0, "verified": 0}
        self.pending, self.active, self.peak = {}, 0, 0
        self.lock = threading.Lock(); self.profile_written = False
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + f"-{os.getpid()}"
        self.split_counts = {}
        self.last_publish = time.perf_counter()
        self.publish_interval = float(os.environ.get("UI14_PROGRESS_INTERVAL_SECONDS", "10"))

    def run(self, fn, *args, **kwargs):
        with self.lock: self.active += 1; self.peak = max(self.peak, self.active)
        try: return fn(*args, **kwargs)
        finally:
            with self.lock: self.active -= 1

    def complete(self, split, result):
        self.counts[result["status"]] += 1
        counts = self.split_counts.setdefault(split, {k: 0 for k in self.counts})
        counts[result["status"]] += 1
        self.pending[split] = max(0, self.pending.get(split, 0) - 1)
        if result["status"] == "built": self.built.append(result)
        if len(self.built) >= 500 and not self.profile_written:
            self.profile_written = True
            self.publish("first_500_built.json", self.built[:500])
        if time.perf_counter() - self.last_publish >= self.publish_interval:
            self.publish()

    def report(self, samples=None):
        samples = self.built if samples is None else samples
        wall = max(r["finished"] for r in samples) - min(r["started"] for r in samples) if samples else None
        rate = len(samples) / wall if wall else None
        elapsed = time.perf_counter() - self.started
        return {"run_id": self.run_id, "measurement": "actual new PNG work; reused/migrated images excluded from built throughput",
                "built_images_measured": len(samples), "built_crops_measured": sum(r["built_crops"] for r in samples),
                "original_images_per_second": rate,
                "crops_per_second": sum(r["built_crops"] for r in samples) / wall if wall else None,
                "measurement_wall_seconds": wall, "counts": self.counts, "counts_by_split": self.split_counts,
                "pending_by_split": self.pending,
                "eta_seconds_by_split": {k: v / rate if rate else None for k, v in self.pending.items()},
                "all_pending_eta_seconds": sum(self.pending.values()) / rate if rate else None,
                "eta_basis": "crop-only approximation at measured new-image rate; unvisited splits still include potential reuse/migration; excludes merge/geometry/finalize",
                "stage_seconds_sum": {k: sum(r[k] for r in samples) for k in (
                    "read_seconds", "decode_seconds", "encode_seconds", "write_seconds")},
                "workers_configured": self.workers, "peak_active_workers": self.peak,
                "png_compress_level": self.compression,
                "process_cpu_percent": (time.process_time() - self.cpu_start) / elapsed * 100 if elapsed else None,
                "cpu_percent_basis": "coordinator plus worker threads; 100% is one core",
                "memory": process_resources(), "old_log_images_per_second": .79, "target_images_per_second": 8,
                "speedup_vs_old_log": rate / .79 if rate else None, "gpu_loaded": False}

    def publish(self, name="latest.json", samples=None):
        value = self.report(samples)
        write_json(self.root / "crop_performance" / name, value)
        write_json(self.root / "crop_performance" / self.run_id / name, value)
        self.last_publish = time.perf_counter()
        rate = value["original_images_per_second"]
        print(f"[crop performance] built={value['built_images_measured']} images; "
              f"new images/s={f'{rate:.2f}' if rate else '估算中'}; pending={sum(self.pending.values())}; "
              f"crop 剩余≈{duration(value['all_pending_eta_seconds'])}; "
              f"workers peak={self.peak}/{self.workers}; CPU={value['process_cpu_percent']:.1f}%; "
              f"RSS={value['memory']['rss_bytes']}; report={name}", flush=True)


def materialize_split(root, task, split, records, *, meter=None):
    from prepare_ui14_sft import validate_task_cache
    workers, compression = crop_settings()
    with verification_session(root) as checks:
        paths, jobs, count = image_jobs(root, task, split, records)
        validate_task_cache(root, task, split, count)
        paths["crop_images"].mkdir(parents=True, exist_ok=True)
        journal = Journal(paths["cache"] / "crop_index/images.jsonl")
        marker = paths["cache"] / "ui14_crop_complete.json"
        marker.unlink(missing_ok=True)
        legacy_hashes = {}
        legacy = paths["cache"] / "ui14_label_cache_ready.json"
        if legacy.exists():
            try:
                if read_json(legacy) == cache_label_binding(root, task, split):
                    legacy_hashes = {r["image"]: r["crop_image_sha256"] for r in read_jsonl(paths["derived"])}
            except (OSError, KeyError, ValueError): pass
        meter = meter or CropMeter(root, workers, compression)
        split_key = f"{task.task_key}/{split}"
        meter.pending[split_key] = len(jobs)
        ordered, ordered_coverage = [None] * len(records), [None] * len(records)
        counts = {"reused": 0, "built": 0, "migrated": 0, "verified": 0}
        iterator = iter(jobs)
        try:
            with phase(f"{split_key} 并行 crop（{workers} workers，完成数含复用）", len(jobs), "原图", estimate=False) as progress, \
                 ThreadPoolExecutor(max_workers=workers) as executor:
                pending = {}
                def enqueue():
                    job = next(iterator, None)
                    if job is not None:
                        future = executor.submit(meter.run, materialize, job, journal.rows.get(job["key"]),
                            task, split, paths, checks, compression, legacy_hashes=legacy_hashes)
                        pending[future] = job
                for _ in range(min(len(jobs), workers * 2)): enqueue()
                try:
                    while pending:
                        finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                        # A failure may finish in the same batch as successful
                        # jobs. Commit those successes before propagating it.
                        for future in sorted(finished, key=lambda f: f.exception() is not None):
                            job = pending.pop(future); state, result = future.result()
                            if journal.rows.get(job["key"]) != state: journal.put(job["key"], state)
                            result["finished"] = time.perf_counter()  # Includes coordinator publication work.
                            for index, position in enumerate(job["positions"]):
                                ordered[position] = state["derived"][index]
                                ordered_coverage[position] = state["coverage"][index]
                            counts[result["status"]] += 1; meter.complete(split_key, result)
                            progress.advance(detail=f"reused={counts['reused']}, built={counts['built']}, "
                                f"migrated={counts['migrated']}, pending={len(jobs) - sum(counts.values())}")
                            enqueue()
                finally:
                    for future in pending: future.cancel()
            derived = [r for rows in ordered for r in rows]
            write_jsonl(paths["derived"], derived)
            write_json(paths["derived"].with_suffix(".coverage.json"), ordered_coverage)
            journal.flush(); checks.journal.flush()
            write_json(legacy, cache_label_binding(root, task, split))
            completed = {"version": VERSION, "binding": cache_label_binding(root, task, split),
                         "index_sha256": file_digest(journal.path),
                         "unique_source_plans": len(jobs)}
            write_json(marker, completed)
            meter.publish()
            print(f"[crop complete] {split_key}: {counts}", flush=True)
            return derived
        finally: journal.close()


def load_completed(root, task, split, records, *, full=False):
    """Read-only consumer: never silently start materialization from finalize."""
    with verification_session(root, full=full) as checks:
        paths, jobs, _ = image_jobs(root, task, split, records)
        marker = paths["cache"] / "ui14_crop_complete.json"
        if not marker.exists():
            raise RuntimeError(f"Missing per-image crop completion: {task.task_key}/{split}; run cache-finalize first")
        completed = read_json(marker)
        journal = Journal(paths["cache"] / "crop_index/images.jsonl")
        try:
            if (completed.get("version") != VERSION or completed["binding"] != cache_label_binding(root, task, split)
                    or completed["index_sha256"] != file_digest(journal.path)):
                raise RuntimeError("Crop completion input/label/index changed; run cache-finalize")
            ordered, coverage = [None] * len(records), [None] * len(records)
            for job in jobs:
                state = journal.rows.get(job["key"])
                binding = physical_binding(job)
                if not state or not physical_valid(state["physical"], binding, checks):
                    raise RuntimeError("Crop source/PNG/geometry changed; run cache-finalize")
                if state["labels_binding"] != labels_binding(job, task, split, state["physical"]):
                    raise RuntimeError("Crop annotation binding changed; run cache-finalize")
                if full:
                    materialize(job, state, task, split, paths, checks, 1, readonly=True)
                for i, position in enumerate(job["positions"]):
                    ordered[position], coverage[position] = state["derived"][i], state["coverage"][i]
            derived = list(read_jsonl(paths["derived"]))
            if derived != [r for rows in ordered for r in rows] or coverage != read_json(paths["derived"].with_suffix(".coverage.json")):
                raise RuntimeError("Derived labels/coverage differ from per-image completion index")
            return derived
        finally: journal.close()
