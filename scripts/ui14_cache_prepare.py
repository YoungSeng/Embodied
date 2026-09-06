"""CPU image preparation and digest-bound handoff to UI14 GPU detection."""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import json
import os
from pathlib import Path
import time

from PIL import Image, ImageOps

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
        oriented = ImageOps.exif_transpose(opened)
        width, height = oriented.size
        oriented.close()
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
        if previous is not None and previous["stat"] == signature:
            return previous, True
        return inspect_image(path, signature), False

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
        raise RuntimeError(f"CPU preparation missing/stale at {marker}: {exc}. "
                           "Run bash shell/ui14_cpt9000_a800.sh cache-prepare on CPU first.") from exc
