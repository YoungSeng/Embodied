"""Stdlib-only preparation progress. Display state never enters data/cache digests."""
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
import json
import math
import os
from pathlib import Path
import threading
import time

_current = None


def duration(seconds):
    if seconds is None:
        return "估算中"
    seconds = max(0, round(seconds))
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


class Counter:
    def __init__(self, label, total=None, unit="项", estimate=True):
        self.label, self.total, self.unit = label, total, unit
        self.estimate = estimate
        self.completed, self.detail, self.status = 0, "", "running"
        self.reused = self.built = self.failed = self.gap = None
        self.started = time.monotonic()

    def advance(self, amount=1, detail=None):
        self.completed += amount
        if detail is not None:
            self.detail = detail

    def snapshot(self, now):
        elapsed = max(0, now - self.started)
        rate = self.completed / elapsed if elapsed else 0
        remaining = max(0, self.total - self.completed) if self.total is not None else None
        eta = (0 if remaining == 0 else remaining / rate if rate else None) if remaining is not None else None
        if not self.estimate or self.status in ("failed", "interrupted"):
            eta = None
        return {"label": self.label, "completed": self.completed, "total": self.total, "unit": self.unit,
                "percent": min(100, self.completed * 100 / self.total) if self.total else None,
                "elapsed_seconds": elapsed, "rate_per_second": rate, "eta_seconds": eta,
                "status": self.status, "detail": self.detail, "estimate": self.estimate,
                "reused":self.reused,"built":self.built,"failed":self.failed,"gap":self.gap,
                "new_rate_per_second": self.built/elapsed if self.built is not None and elapsed else None}


class ProgressSession:
    def __init__(self, stage, root, interval=10):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("progress interval must be a finite positive number")
        self.stage, self.root, self.interval = stage, Path(root), interval
        self.started = time.monotonic()
        self.frames, self.activity, self.status = [], None, "running"
        self.activities = []
        self.external = None
        self.delegated = False
        self.lock, self.stop = threading.RLock(), threading.Event()
        self.thread = None

    def __enter__(self):
        global _current
        self.root.joinpath("progress").mkdir(parents=True, exist_ok=True)
        self.previous, _current = _current, self
        self.emit()
        self.thread = threading.Thread(target=self._heartbeat, daemon=True, name="ui14-progress")
        self.thread.start()
        return self

    def __exit__(self, kind, error, traceback):
        global _current
        self.stop.set()
        if self.thread is not None:
            self.thread.join()
        self.status = "completed" if kind is None else "interrupted" if issubclass(kind, (KeyboardInterrupt, GeneratorExit)) else "failed"
        try:
            self.emit()
        finally:
            _current = self.previous

    def _heartbeat(self):
        while not self.stop.wait(self.interval):
            self.emit()

    def _detector_snapshot(self):
        if self.external is None:
            return None
        path, since = self.external
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            updated = datetime.fromisoformat(value["updated_at"]).timestamp()
            if updated < since:
                return None  # Never display a prior invocation's 100% as current work.
            for field in ("completed", "total", "percent", "elapsed_seconds", "rate_per_second"):
                if not isinstance(value.get(field), (int, float)) or not math.isfinite(value[field]) or value[field] < 0:
                    return None
            eta = value.get("eta_seconds")
            if eta is not None and (not isinstance(eta, (int, float)) or not math.isfinite(eta) or eta < 0):
                return None
            value["update_age_seconds"] = max(0, time.time() - updated)
            return value
        except (OSError, ValueError, TypeError, KeyError):
            return None

    def emit(self):
        with self.lock:
            if self.delegated:
                return  # Child stage owns the live progress.json until it exits.
            now = time.monotonic()
            frames = [frame.snapshot(now) for frame in self.frames]
            payload = {"stage": self.stage, "status": self.status, "pid": os.getpid(),
                       "output_path": str(self.root),
                       "updated_at": datetime.now(timezone.utc).isoformat(),
                       "elapsed_seconds": now - self.started, "phases": frames,
                       "activity": self.activity.snapshot(now) if self.activity else None,
                       "detector": self._detector_snapshot(),
                       "eta_scope": "current phase only; different stages are not equally timed"}
            parts = [f"[UI14 {self.stage}] {self.status} | 总已用 {duration(payload['elapsed_seconds'])}"]
            for frame in frames:
                if frame["total"] is None and not frame["completed"]:
                    parts.append(f"{frame['label']}: {frame['status']} | 已用 {duration(frame['elapsed_seconds'])}"
                                 + (" | 本阶段剩余≈估算中" if frame["estimate"] else " | 总剩余时间不可估算"))
                    continue
                total = f"/{frame['total']:,}" if frame["total"] is not None else ""
                percent = f" ({frame['percent']:.1f}%)" if frame["percent"] is not None else ""
                eta_text = (f"本阶段剩余≈{duration(frame['eta_seconds'])}" if frame["estimate"]
                            else "剩余时间见当前子阶段")
                parts.append(f"{frame['label']}: {frame['completed']:,}{total} {frame['unit']}{percent}"
                             f" | 已用 {duration(frame['elapsed_seconds'])} | {frame['rate_per_second']:.2f} {frame['unit']}/s"
                             f" | {eta_text}"
                             + (f" | {frame['detail']}" if frame["detail"] else ""))
                if frame["built"] is not None or frame["reused"] is not None:
                    parts[-1] += (f" | reused={frame['reused']} built={frame['built']}"
                                  f" failed={frame['failed']} gap={frame['gap']}"
                                  f" new/s={frame['new_rate_per_second'] or 0:.2f}")
            if self.activity:
                activity = payload["activity"]
                amounts = (f"{activity['completed']/1048576:.1f}/{activity['total']/1048576:.1f} MiB"
                           if activity["unit"] == "bytes" else
                           f"{activity['completed']:,}/{activity['total'] if activity['total'] is not None else '?'} {activity['unit']}")
                parts.append(f"当前文件: {activity['label']} | {amounts} | 文件剩余≈{duration(activity['eta_seconds'])}")
            detector = payload["detector"]
            if detector:
                parts.append(f"detector {detector.get('stage')}: {detector.get('status')}"
                             f" | {detector.get('completed')}/{detector.get('total')} {detector.get('unit')}"
                             f" | {100 * detector.get('percent', 0):.1f}%"
                             f" | 已用 {duration(detector.get('elapsed_seconds'))}"
                             f" | {detector.get('rate_per_second', 0):.2f}/s"
                             f" | 本阶段剩余≈{duration(detector.get('eta_seconds'))}"
                             f" | 状态更新于 {duration(detector['update_age_seconds'])} 前")
            line = "\n  ".join(parts)
            if os.environ.get("UI14_NEG11") == "1":
                line += f"\n  output={self.root}"
            # Progress is optional observability: a log failure must not change labels,
            # consume iterator items, or hide the original preparation exception.
            try:
                print(line, flush=True)
                folder = self.root / "progress"
                with (folder / f"{self.stage}.log").open("a", encoding="utf-8") as handle:
                    handle.write(payload["updated_at"] + " " + line + "\n")
                path = folder / f"{self.stage}.json"
                temp = path.with_name(path.name + f".tmp-{os.getpid()}")
                temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                os.replace(temp, path)
                if os.environ.get("UI14_NEG11") == "1":
                    latest=self.root/"progress.json"
                    tmp=latest.with_name(latest.name+f".tmp-{os.getpid()}")
                    tmp.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
                    os.replace(tmp,latest)
            except (OSError, UnicodeError):
                pass


@contextmanager
def phase(label, total=None, unit="项", estimate=True):
    session = _current
    counter = Counter(label, total, unit, estimate)
    if session is None:
        yield counter
        return
    with session.lock:
        session.frames.append(counter)
        session.emit()
    try:
        yield counter
    except BaseException as exc:
        counter.status = "interrupted" if isinstance(exc, (KeyboardInterrupt, GeneratorExit)) else "failed"
        raise
    else:
        counter.status = "completed"
    finally:
        with session.lock:
            session.emit()
            session.frames.remove(counter)


def phase_function(label):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with phase(label):
                return function(*args, **kwargs)
        return wrapped
    return decorate


def track(iterable, label, total=None, unit="项", detail=None, estimate=True):
    if _current is None:
        yield from iterable
        return
    if total is None and hasattr(iterable, "__len__"):
        total = len(iterable)
    with phase(label, total, unit, estimate) as counter:
        for item in iterable:
            if detail:
                counter.detail = str(detail(item))
            yield item
            counter.advance()


@contextmanager
def file_activity(path, total, unit="bytes"):
    session = _current
    counter = Counter(str(path), total, unit)
    if session is None:
        yield counter
        return
    with session.lock:
        session.activities.append(counter)
        session.activity = counter
    try:
        yield counter
    finally:
        with session.lock:
            # Parallel hash reads can finish in a different order from entry.
            # Never restore an already finished worker's stale file counter.
            session.activities.remove(counter)
            session.activity = session.activities[-1] if session.activities else None


@contextmanager
def detector_status(path):
    """Mirror the existing worker coordinator's actual progress without running GPU work."""
    session = _current
    if session is None:
        yield
        return
    with session.lock:
        previous, session.external = session.external, (Path(path), time.time())
    try:
        yield
    finally:
        with session.lock:
            session.emit()
            session.external = previous
