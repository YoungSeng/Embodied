"""Persistent CPU verification evidence; unchanged files need only stat checks."""
from __future__ import annotations
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import os
from pathlib import Path
import threading
import time

_current = ContextVar("ui14_verification", default=None)


def checksum(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def signature(path):
    s = Path(path).stat()
    return stat_values(s)


def stat_values(s):
    return [s.st_size, s.st_mtime_ns, s.st_ctime_ns]


class FileChangedDuringVerification(ValueError):
    """Only this transient condition permits a bounded content-read retry."""
    def __init__(self, path, before, after):
        super().__init__(f"File changed during verification: {path}; "
                         f"before={before}; after={after}; stat=[size,mtime_ns,ctime_ns]")


def stable_sha256(path):
    """Hash one opened file; reject writes or atomic replacement while reading."""
    before = signature(path)
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        from ui14_progress import file_activity
        opened = os.fstat(stream.fileno())
        count = 0
        with file_activity(path, opened.st_size) as progress:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block); count += len(block); progress.advance(len(block))
        finished = os.fstat(stream.fileno())
        current = Path(path).stat()
        # Compare ctime across reads of the same API. Some Windows runtimes
        # expose different ctime semantics through stat and fstat; cross-API
        # equality is required only for size/mtime, plus the device/inode pair.
        if (before != stat_values(current) or stat_values(opened) != stat_values(finished)
                or before[:2] != stat_values(opened)[:2] or count != before[0]
                or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)):
            raise FileChangedDuringVerification(path, before, {
                "opened": stat_values(opened), "finished": stat_values(finished),
                "current": stat_values(current), "bytes_read": count,
                "opened_file_id": [opened.st_dev, opened.st_ino],
                "current_file_id": [current.st_dev, current.st_ino]})
    return h.hexdigest(), before


class Journal:
    """Single coordinator, thread-safe append; discard only torn final lines."""
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows, self.lock = {}, threading.RLock()
        if self.path.exists():
            from ui14_progress import file_activity
            with self.path.open("rb+") as f, file_activity(self.path, self.path.stat().st_size) as progress:
                while True:
                    offset = f.tell(); line = f.readline()
                    if not line: break
                    progress.advance(len(line))
                    if not line.endswith(b"\n"):
                        f.truncate(offset); break
                    try:
                        item = json.loads(line); digest = item.pop("digest")
                        if checksum(item) == digest: self.rows[item["key"]] = item["value"]
                    except (ValueError, KeyError, TypeError):
                        continue
        self.stream = self.path.open("ab")
        self.pending, self.synced = 0, time.monotonic()

    def put(self, key, value):
        with self.lock:
            row = {"key": key, "value": value}
            self.stream.write((json.dumps({**row, "digest": checksum(row)}, ensure_ascii=False) + "\n").encode())
            self.rows[key] = value
            self.pending += 1
            if self.pending >= 32 or time.monotonic() - self.synced >= 5: self.flush()

    def flush(self):
        with self.lock:
            self.stream.flush(); os.fsync(self.stream.fileno())
            self.pending, self.synced = 0, time.monotonic()

    def close(self):
        self.flush(); self.stream.close()


class Verification:
    VERSION = 1

    def __init__(self, root, full=False):
        self.root, self.full = Path(root), full
        self.journal = Journal(self.root / "verification/files.jsonl")
        self.checked_this_run, self.collectors = {}, []
        self.image_keys = set()
        self.counts = {"reused": 0, "checked": 0}

    def get(self, path, kind):
        path = str(Path(path).absolute()); state = signature(path)
        key = kind + ":" + path
        if kind == "rgb_identity" or Path(path).suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"):
            self.image_keys.add(key)
        row = self.journal.rows.get(key)
        if (row and row.get("version") == self.VERSION and row.get("stat") == state
                and (not self.full or self.checked_this_run.get(key) == state)):
            self.counts["reused"] += 1
            return row["value"]
        return None

    def remember(self, path, kind, value, measured_stat=None):
        path = str(Path(path).absolute()); state = signature(path)
        if measured_stat is not None and measured_stat != state:
            raise FileChangedDuringVerification(path, measured_stat, state)
        key = kind + ":" + path
        if kind == "rgb_identity" or Path(path).suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"):
            self.image_keys.add(key)
        self.journal.put(key, {"version": self.VERSION, "stat": state, "value": value})
        self.checked_this_run[key] = state
        self.counts["checked"] += 1
        return value

    def sha256(self, path):
        value = self.get(path, "sha256")
        if value is None:
            # Newly published files can briefly expose different attributes
            # between stat/open/read. Re-read only this file; never cache the
            # digest from an unstable attempt or waive the metadata comparison.
            for attempt in range(4):
                try:
                    measured, state = stable_sha256(path)
                    value = self.remember(path, "sha256", measured, state)
                    break
                except FileChangedDuringVerification as exc:
                    if attempt == 3:
                        raise FileChangedDuringVerification(path, "4 unstable read attempts",
                            f"{exc}; check other writers/storage; no digest accepted") from exc
                    delay = (.5, 1, 2)[attempt]
                    print(f"[verification retry {attempt + 1}/3] {exc}; re-read this file after {delay}s", flush=True)
                    time.sleep(delay)
        for collector in self.collectors: collector[str(Path(path).absolute())] = value
        return value

    def rgb_identity(self, path):
        value = self.get(path, "rgb_identity")
        if value is None:
            from PIL import Image, ImageOps
            state = signature(path)
            with Image.open(path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
                value = rgb_identity(image)
                image.close()
            self.remember(path, "rgb_identity", value, state)
        return tuple(value)

    def validated_call(self, key, call):
        """Reuse a successful structural check only if every input still matches."""
        path = self.root / "verification/receipts" / (checksum(key) + ".json")
        if path.exists() and not self.full:
            try:
                previous = json.loads(path.read_text(encoding="utf-8")); claimed = previous.pop("digest")
                if (claimed == checksum(previous) and previous["key"] == key and previous["files"]
                        and all(self.sha256(p) == h for p, h in previous["files"].items())):
                    return previous["result"]
            except (OSError, KeyError, ValueError, TypeError): pass
        files = {}; self.collectors.append(files)
        try: result = call()
        finally: self.collectors.pop()
        if files:
            from ui14_common import write_json
            payload = {"key": key, "files": files, "result": result}
            write_json(path, {**payload, "digest": checksum(payload)})
        return result

    def export_images(self, path):
        from ui14_common import write_jsonl
        write_jsonl(path, ({"key": key, **self.journal.rows[key]} for key in sorted(self.image_keys) if key in self.journal.rows))

    def prefetch_images(self, paths, kind="rgb_identity"):
        """First-time/changed image checks use bounded CPU workers as well."""
        from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
        from ui14_progress import phase
        paths = list(dict.fromkeys(str(p) for p in paths))
        workers = max(1, int(os.environ.get("UI14_PREPARE_WORKERS", "16")))
        operation = self.rgb_identity if kind == "rgb_identity" else self.sha256
        iterator = iter(paths)
        with phase(f"CPU 图片验证记录（{workers} workers，未变项只 stat）", len(paths), "图片") as counter, \
             ThreadPoolExecutor(max_workers=workers) as executor:
            pending = set()
            def enqueue():
                path = next(iterator, None)
                if path is not None: pending.add(executor.submit(operation, path))
            for _ in range(min(len(paths), workers * 2)): enqueue()
            try:
                while pending:
                    finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in finished:
                        future.result(); counter.advance(); enqueue()
            finally:
                for future in pending: future.cancel()

    def validate_images(self, path):
        from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
        from ui14_common import read_jsonl
        from ui14_progress import phase
        with phase("读取已验证图片清单（不打开图片）"):
            by_path = {}
            for row in read_jsonl(path):
                kind, name = row["key"].split(":", 1)
                by_path.setdefault(name, []).append((kind, row))
        workers = int(os.environ.get("UI14_SUBMIT_CHECK_WORKERS", os.environ.get("UI14_PREPARE_WORKERS", "16")))
        if not 1 <= workers <= 64: raise ValueError("UI14_SUBMIT_CHECK_WORKERS must be in 1..64")
        def validate(item):
            name, rows = item
            state = signature(name)
            refreshed = False
            for kind, row in rows:
                if state == row["stat"] and not self.full:
                    continue
                cached = self.journal.rows.get(row["key"], {})
                if not self.full and cached.get("version") == self.VERSION and cached.get("stat") == state:
                    actual = cached["value"]
                else:
                    actual = self.rgb_identity(name) if kind == "rgb_identity" else self.sha256(name)
                    refreshed = True
                if json.loads(json.dumps(actual)) != row["value"]:
                    raise RuntimeError(f"Prepared image content changed: {name}; run CPU preparation/check again")
            return name, refreshed
        reused = refreshed = 0
        iterator = iter(by_path.items())
        with phase(f"图片属性与内容检查（{workers} workers，未变项只 stat）", len(by_path), "图片") as counter, \
             ThreadPoolExecutor(max_workers=workers) as executor:
            pending = set()
            def enqueue():
                item = next(iterator, None)
                if item is not None: pending.add(executor.submit(validate, item))
            for _ in range(min(len(by_path), workers * 2)): enqueue()
            try:
                while pending:
                    finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in finished:
                        name, changed = future.result()
                        refreshed += int(changed); reused += int(not changed)
                        counter.advance(detail=f"reused={reused}, refreshed={refreshed}, pending={len(by_path) - counter.completed - 1}; {name}")
                        enqueue()
            finally:
                for future in pending: future.cancel()
        return {"images": len(by_path), "reused": reused, "refreshed": refreshed, "workers": workers}


def rgb_identity(image):
    width, height = image.size
    return (hashlib.sha256(f"{width}x{height}:RGB:".encode() + image.tobytes()).hexdigest(), width, height)


def current_checks():
    return _current.get()


@contextmanager
def verification_session(root, full=False):
    existing = current_checks()
    if existing is not None:
        yield existing
        return
    checks = Verification(root, full)
    token = _current.set(checks)
    try: yield checks
    finally:
        checks.journal.close(); _current.reset(token)


@contextmanager
def preparation_lock(root, *, filename=".ui14-preparation.lock"):
    """OS releases the coordinator lock even after SIGKILL/node process exit."""
    path = Path(root) / filename
    busy = f"Another UI14 preparation coordinator is running (lock: {path}); keep it running and use submit-status --watch for submission progress"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as f:
        if os.name == "nt":
            import msvcrt
            if f.tell() == 0: f.write(b"0"); f.flush()
            f.seek(0)
            try: msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc: raise RuntimeError(busy) from exc
        else:
            import fcntl
            try: fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc: raise RuntimeError(busy) from exc
        try: yield
        finally:
            if os.name == "nt":
                f.seek(0); msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else: fcntl.flock(f, fcntl.LOCK_UN)
