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
    return [s.st_size, s.st_mtime_ns, s.st_ctime_ns]


class Journal:
    """Single coordinator, thread-safe append; discard only torn final lines."""
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows, self.lock = {}, threading.RLock()
        if self.path.exists():
            with self.path.open("rb+") as f:
                while True:
                    offset = f.tell(); line = f.readline()
                    if not line: break
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
            raise ValueError(f"File changed during verification: {path}")
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
            state = signature(path); h = hashlib.sha256()
            with Path(path).open("rb") as f:
                for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
            value = self.remember(path, "sha256", h.hexdigest(), state)
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
        from ui14_common import read_jsonl
        for row in read_jsonl(path):
            kind, name = row["key"].split(":", 1)
            if signature(name) == row["stat"] and not self.full:
                continue
            actual = self.rgb_identity(name) if kind == "rgb_identity" else self.sha256(name)
            if json.loads(json.dumps(actual)) != row["value"]:
                raise RuntimeError(f"Prepared image content changed: {name}; run CPU preparation/check again")


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
def preparation_lock(root):
    """OS releases the coordinator lock even after SIGKILL/node process exit."""
    path = Path(root) / ".ui14-preparation.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as f:
        if os.name == "nt":
            import msvcrt
            if f.tell() == 0: f.write(b"0"); f.flush()
            f.seek(0)
            try: msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc: raise RuntimeError("Another UI14 preparation coordinator is running") from exc
        else:
            import fcntl
            try: fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc: raise RuntimeError("Another UI14 preparation coordinator is running") from exc
        try: yield
        finally:
            if os.name == "nt":
                f.seek(0); msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else: fcntl.flock(f, fcntl.LOCK_UN)
