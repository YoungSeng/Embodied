"""CPU image preparation and digest-bound handoff to UI14 GPU detection."""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import json
import os
from pathlib import Path
import time

from PIL import Image

from ui14_common import digest, file_digest, read_json, write_json
from ui14_progress import phase

IMAGE_INFO_VERSION = 1
PREPARE_VERSION = 1


def stat_signature(path):
    info = Path(path).stat()
    return [info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def inspect_image(path, signature):
    # Preserve the exact legacy detector ID: BLAKE2b of file bytes, not the
    # normalized dataset's decoded-RGB SHA256. Existing GPU shards stay valid.
    value = hashlib.blake2b(digest_size=20)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    with Image.open(path) as opened:
        width, height = opened.size
        if opened.getexif().get(274) in (5, 6, 7, 8):
            width, height = height, width
    if stat_signature(path) != signature:
        raise ValueError(f"Image changed during CPU preparation: {path}")
    return {"version": IMAGE_INFO_VERSION, "path": str(path), "stat": signature,
            "content_id": value.hexdigest(), "width": width, "height": height}


class ImageInfoJournal:
    """One writer, bounded readers; durable checkpoints every 32 new images/5s.

    Metadata-only stat checks run in the thread pool, too. Completed entries
    survive interruption independently of split publication. A torn final line
    is removed on reopen; every complete entry has its own digest.
    """
    def __init__(self, path, workers=16):
        if workers < 1:
            raise ValueError("CPU prepare workers must be positive")
        self.path, self.workers = Path(path), workers
        self.entries = {}
        self.invocation_cache, self.expected_dimensions = {}, {}
        self.totals = {"reused_images": 0, "scanned_images": 0}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_file():
            with self.path.open("rb+") as handle:
                while True:
                    offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    if not line.endswith(b"\n"):
                        handle.truncate(offset)
                        break
                    try:
                        row = json.loads(line)
                        checksum = row.pop("digest")
                        if row["version"] == IMAGE_INFO_VERSION and digest(row) == checksum:
                            self.entries[row["path"]] = row
                    except (ValueError, KeyError, TypeError):
                        continue  # Never reuse an invalid entry; inspect it again.

    def _load_one(self, path, previous):
        signature = stat_signature(path)
        if path in self.invocation_cache and self.invocation_cache[path]["stat"] == signature:
            previous = self.invocation_cache[path]
        if previous is not None and previous["stat"] == signature:
            row, reused = previous, True
        else:
            row, reused = inspect_image(path, signature), False
        expected = self.expected_dimensions.get(path)
        if expected and (row["width"], row["height"]) != tuple(expected):
            raise ValueError(f"Normalized dimensions changed: {path}; normalize this source before cache preparation")
        self.invocation_cache[path] = row
        return row, reused

    def load(self, paths):
        paths = list(dict.fromkeys(str(p) for p in paths))
        result, reused, scanned = {}, 0, 0
        iterator = iter(paths)
        with phase(f"CPU 图片指纹/尺寸（{self.workers} 线程，可续跑）", len(paths), "图片") as counter:
            with self.path.open("ab") as journal, ThreadPoolExecutor(max_workers=self.workers) as executor:
                pending = set()

                def enqueue():
                    path = next(iterator, None)
                    if path is not None:
                        pending.add(executor.submit(self._load_one, path, self.entries.get(path)))

                for _ in range(min(len(paths), self.workers * 2)):
                    enqueue()
                uncommitted, last_sync = 0, time.monotonic()
                try:
                    while pending:
                        finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for future in finished:
                            row, was_reused = future.result()
                            if was_reused:
                                reused += 1
                            else:
                                journal.write((json.dumps({**row, "digest": digest(row)}, ensure_ascii=False) + "\n").encode())
                                self.entries[row["path"]] = row
                                scanned += 1
                                uncommitted += 1
                            result[row["path"]] = (row["content_id"], row["width"], row["height"])
                            counter.advance(detail=f"reused={reused}, scanned={scanned}")
                            if uncommitted >= 32 or time.monotonic() - last_sync >= 5:
                                journal.flush()
                                os.fsync(journal.fileno())
                                uncommitted, last_sync = 0, time.monotonic()
                            enqueue()
                finally:
                    for future in pending:
                        future.cancel()
                    journal.flush()
                    os.fsync(journal.fileno())
        self.totals["reused_images"] += reused
        self.totals["scanned_images"] += scanned
        print(f"[cache-prepare images] reused={reused}, scanned={scanned}, workers={self.workers}", flush=True)
        return result


def prepared_files(cache):
    cache = Path(cache)
    return [cache / "manifest" / name for name in (
        "selection_config.json", "unique_images.jsonl", "task_samples.jsonl")
    ] + sorted((cache / "manifest/shards").glob("shard_*.jsonl")) + [cache / "detections/detector_config.json"]


def refresh_stable_shards(paths, unique, shard_size):
    """Keep old shard slots; replace changed paths locally, append new images.

    Existing GPU completion checks compare each shard's ordered IDs. Unchanged
    shards retain their bytes; a changed image does not shift all later shards.
    """
    from run_ui5_crop_audit import atomic_write_jsonl
    from ui14_common import read_jsonl
    by_id = {r["image_id"]: r for r in unique}
    by_path = {p: r for r in unique for p in r["image_paths"]}
    claimed, last = set(), -1
    for path in sorted(paths.shards.glob("shard_*.jsonl")):
        last = max(last, int(path.stem.split("_")[-1]))
        original, updated = list(read_jsonl(path)), []
        for row in original:
            replacement = by_id.get(row["image_id"])
            if replacement is None:
                replacement = next((by_path[p] for p in row["image_paths"] if p in by_path), None)
            if replacement and replacement["image_id"] not in claimed:
                claimed.add(replacement["image_id"]); updated.append(replacement)
        if updated:
            if updated != original: atomic_write_jsonl(path, updated)
        else:
            path.unlink()
            # Removed shards must not be consumed by merge/publication.
            for stage in ("text", "icon"):
                for old in (paths.stage_dir(stage) / path.name,
                            paths.stage_dir(stage) / (path.stem + ".done.json")):
                    old.unlink(missing_ok=True)
    pending = [r for r in unique if r["image_id"] not in claimed]
    for start in range(0, len(pending), shard_size):
        last += 1
        atomic_write_jsonl(paths.shards / f"shard_{last:05d}.jsonl", pending[start:start + shard_size])


def preparation_binding(paths, normalization_id, config):
    return {"version": PREPARE_VERSION, "normalization_id": normalization_id,
            "inputs": {key: {"path": str(paths[key]), "sha256": file_digest(paths[key])}
                       for key in ("normalized", "detector_input", "detector_inputs")},
            "detector_config": config}


def publish_prepared(paths, normalization_id, config, unique_images):
    payload = {**preparation_binding(paths, normalization_id, config),
               "unique_images": unique_images,
               "artifacts": {str(p.relative_to(paths["cache"])): file_digest(p)
                             for p in prepared_files(paths["cache"])}}
    write_json(paths["cache"] / "manifest/ui14_prepare_ready.json", {**payload, "digest": digest(payload)})


def recover_prepared(paths, normalization_id, config, journal):
    """CPU-only recovery after manifests/shards finished but ready publication failed.

    Check source selection, every task/path/ID/dimension and shard member before
    accepting old manifests. Image journal entries still require live stat;
    unchanged originals are not read or decoded. Never rewrite manifests here.
    """
    from ui14_common import read_jsonl
    from prepare_ui5_eval_detector_crops import DETECTOR_MANIFEST_FORMAT_VERSION, digest_ids
    cache = paths["cache"]
    required = prepared_files(cache)
    if not all(p.is_file() for p in required) or not list((cache / "manifest/shards").glob("shard_*.jsonl")):
        return None
    try:
        before = {str(p): file_digest(p) for p in required}
        input_binding = preparation_binding(paths, normalization_id, config)
        task_files = read_json(paths["detector_inputs"])
        if len(task_files) != 1: raise ValueError("expected one isolated task")
        task, input_name = next(iter(task_files.items()))
        if Path(input_name).resolve() != paths["detector_input"].resolve():
            raise ValueError("task input path differs")
        selected = list(dict.fromkeys(r["image"] for r in read_jsonl(paths["detector_input"])))
        if not selected or not all(isinstance(p, str) and Path(p).is_absolute() for p in selected):
            raise ValueError("recovery requires canonical normalized image paths")
        rows = list(read_jsonl(cache / "manifest/unique_images.jsonl"))
        by_id, by_path = {}, {}
        for row in rows:
            image_id = row["image_id"]
            if (image_id in by_id or image_id != "eval_" + row["content_id"][:20]
                    or row["tasks"] != [task] or not row["image_paths"]
                    or row["image_path"] != row["image_paths"][0]):
                raise ValueError("unique image identity/role differs")
            by_id[image_id] = row
            for path in row["image_paths"]:
                if path in by_path: raise ValueError("duplicate unique-manifest image path")
                by_path[path] = row
        if set(by_path) != set(selected): raise ValueError("manifest image paths differ from current inputs")
        expected_selection = {"format_version": DETECTOR_MANIFEST_FORMAT_VERSION,
            "input_dir": str(paths["detector_input"].parent.resolve()),
            "task_files": {task: str(paths["detector_input"].resolve())},
            "task_file_digests": {task: file_digest(paths["detector_input"])},
            "max_images_per_task": 0, "skip_figma": False, "unique_images": len(rows),
            "image_id_digest": digest_ids(row["image_id"] for row in rows),
            "content_id_digest": digest_ids(row["content_id"] for row in rows),
            "data_split": paths["normalized"].stem}
        if read_json(cache / "manifest/selection_config.json") != expected_selection:
            raise ValueError("selection/source digest differs")
        samples = list(read_jsonl(cache / "manifest/task_samples.jsonl"))
        if len(samples) != len(selected): raise ValueError("incomplete task samples")
        for index, (sample, path) in enumerate(zip(samples, selected)):
            row = by_path[path]
            if sample != {"task": task, "task_index": index, "image_id": row["image_id"],
                          "content_id": row["content_id"], "image_path": path}:
                raise ValueError("task sample order or image identity differs")
        seen = set()
        for shard in sorted((cache / "manifest/shards").glob("shard_*.jsonl")):
            members = list(read_jsonl(shard))
            if not members: raise ValueError("empty shard")
            for row in members:
                if row["image_id"] in seen or by_id.get(row["image_id"]) != row:
                    raise ValueError("stale/duplicate shard member")
                seen.add(row["image_id"])
        if seen != set(by_id): raise ValueError("incomplete shard membership")
        if read_json(cache / "detections/detector_config.json") != config:
            raise ValueError("detector config differs")
        info = journal.load(selected)
        if any(info[p] != (row["content_id"], row["width"], row["height"]) for p, row in by_path.items()):
            raise ValueError("image bytes/dimensions changed since manifest generation")
        after = {str(p): file_digest(p) for p in prepared_files(cache)}
        if before != after or input_binding != preparation_binding(paths, normalization_id, config):
            raise ValueError("manifest/source changed during recovery")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        # Do not turn a persistently unstable storage read into a giant rebuild.
        from ui14_verification import FileChangedDuringVerification
        if isinstance(exc, FileChangedDuringVerification): raise
        print(f"[cache-prepare recovery unavailable] {task_files if 'task_files' in locals() else cache}: {exc}", flush=True)
        return None
    publish_prepared(paths, normalization_id, config, len(rows))
    print(f"[cache-prepare recovered] {task}/{paths['normalized'].stem}: "
          f"{len(selected)} task-images, {len(rows)} unique images; manifests/shards unchanged; ready marker published", flush=True)
    return len(rows)


def validate_prepared(paths, normalization_id, config):
    """GPU handoff: only JSON/JSONL hashes, never image stat/read/decode/hash."""
    marker = paths["cache"] / "manifest/ui14_prepare_ready.json"
    try:
        payload = read_json(marker)
        checksum = payload.pop("digest")
        if checksum != digest(payload):
            raise ValueError("marker digest mismatch")
        binding = preparation_binding(paths, normalization_id, config)
        if any(payload.get(key) != value for key, value in binding.items()):
            raise ValueError("normalization, input or detector configuration changed")
        actual = {str(p.relative_to(paths["cache"])): file_digest(p) for p in prepared_files(paths["cache"])}
        if not list((paths["cache"] / "manifest/shards").glob("shard_*.jsonl")) or payload["artifacts"] != actual:
            raise ValueError("prepared manifest/shards changed")
        count = payload["unique_images"]
        if not isinstance(count, int) or count <= 0:
            raise ValueError("invalid unique image count")
        return count
    except (OSError, ValueError, KeyError, TypeError) as exc:
        from ui14_verification import FileChangedDuringVerification
        if isinstance(exc, FileChangedDuringVerification): raise
        raise RuntimeError(f"CPU preparation missing/stale at {marker}: {exc}. "
                           "Run bash shell/ui14_cpt9000_a800.sh cache-prepare on CPU first.") from exc
