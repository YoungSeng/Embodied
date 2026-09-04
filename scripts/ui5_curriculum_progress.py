"""CPU-only curriculum build progress, stage ETA, and durable heartbeats.

The files in ``progress/`` are operational logs, not curriculum content.  Never
include them in a recipe identity or completion inventory.  ETA describes only
the current stage; stages can have very different costs and no full-build ETA
can be inferred from one stage's rate.
"""
from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TextIO


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0, int(math.ceil(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class _Stage:
    def __init__(self, owner: "BuildProgress", name: str, total: int | None, unit: str):
        if total is not None and (isinstance(total, bool) or not isinstance(total, int) or total < 0):
            raise ValueError("stage total must be a nonnegative integer or None")
        self.owner = owner
        self.name = name
        self.total = total
        self.unit = unit
        self.completed = 0
        self.detail = ""
        self.bytes_processed: int | None = None
        self.started_at = 0.0
        self.last_progress_at = 0.0
        self.status = "pending"

    def __enter__(self) -> "_Stage":
        with self.owner._lock:
            self.started_at = self.last_progress_at = self.owner.clock()
            self.status = "running"
            self.owner._stages.append(self)
            self.owner._emit("stage_started", force=True)
        return self

    def set_detail(self, detail: str) -> None:
        """Identify the next potentially slow operation before starting it."""
        with self.owner._lock:
            self.detail = str(detail)
            self.owner._emit("progress")

    def update(
        self, completed: int, detail: str | None = None,
        bytes_processed: int | None = None, *, force: bool = False,
    ) -> None:
        """Set cumulative completed work; this is not an increment operation."""
        with self.owner._lock:
            if isinstance(completed, bool) or not isinstance(completed, int) or completed < self.completed:
                raise ValueError("completed work must be an integer and cannot decrease")
            if self.total is not None and completed > self.total:
                raise ValueError("completed work cannot exceed stage total")
            if completed > self.completed:
                self.last_progress_at = self.owner.clock()
            self.completed = completed
            if detail is not None:
                self.detail = str(detail)
            if bytes_processed is not None:
                self.bytes_processed = bytes_processed
            self.owner._emit("progress", force=force)

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> bool:
        with self.owner._lock:
            self.status = "failed" if exc is not None else "complete"
            self.owner._emit(
                "stage_failed" if exc is not None else "stage_completed",
                force=True, error=exc,
            )
            self.owner._last_stage = self
            self.owner._stages.remove(self)
        return False


class BuildProgress:
    """Report progress without importing torch, Pillow, or the training stack.

    ``stage.update`` accepts cumulative counts.  A daemon heartbeat also reports
    the active item during blocking I/O; it is not evidence that the item itself
    has progressed.  Nested stages restore their parent after they complete.
    Progress I/O errors are printed but cannot mask a recipe exception.
    """
    def __init__(
        self, output_dir: Path, interval_seconds: float = 10.0,
        stream: TextIO | None = None, clock: Callable[[], float] = time.monotonic,
    ):
        if not math.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError("progress interval must be positive and finite")
        self.directory = Path(output_dir) / "progress"
        self.latest_path = self.directory / "build_progress.json"
        self.history_path = self.directory / "build_progress.jsonl"
        self.interval_seconds = interval_seconds
        self.stream = sys.stderr if stream is None else stream
        self.clock = clock
        self.run_id = uuid.uuid4().hex
        self.pid = os.getpid()
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._stages: list[_Stage] = []
        self._last_stage: _Stage | None = None
        self._started_at = 0.0
        self._last_emit = float("-inf")
        self._status = "pending"
        self._io_warning_printed = False

    def __enter__(self) -> "BuildProgress":
        with self._lock:
            self._started_at = self.clock()
            self._status = "running"
            self._emit("build_started", force=True)
        self._thread = threading.Thread(
            target=self._heartbeat, name="ui5-curriculum-progress", daemon=True,
        )
        self._thread.start()
        return self

    def stage(self, name: str, total: int | None = None, unit: str = "items") -> _Stage:
        return _Stage(self, name, total, unit)

    def _heartbeat(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            with self._lock:
                if self._status == "running":
                    self._emit("heartbeat")

    def _emit(self, event: str, *, force: bool = False, error: BaseException | None = None) -> None:
        now = self.clock()
        if not force and now - self._last_emit < self.interval_seconds:
            return
        self._last_emit = now
        stage = self._stages[-1] if self._stages else self._last_stage
        elapsed = max(0.0, now - stage.started_at) if stage else 0.0
        rate = stage.completed / elapsed if stage and stage.completed and elapsed else None
        eta = None
        percent = None
        if stage and stage.total is not None:
            percent = 100.0 * stage.completed / stage.total if stage.total else 100.0
            if stage.completed == stage.total:
                eta = 0.0
            elif rate:
                eta = max(0, stage.total - stage.completed) / rate
        record = {
            "schema_version": 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "pid": self.pid,
            "event": event,
            "status": self._status,
            "stage": stage.name if stage else None,
            "stage_status": stage.status if stage else None,
            "completed": stage.completed if stage else None,
            "total": stage.total if stage else None,
            "unit": stage.unit if stage else None,
            "percent": percent,
            "elapsed_seconds": elapsed,
            "build_elapsed_seconds": max(0.0, now - self._started_at),
            "speed_per_second": rate,
            "eta_seconds": eta,
            "eta_scope": "current_stage",
            "last_progress_age_seconds": max(0.0, now - stage.last_progress_at) if stage else None,
            "detail": stage.detail if stage else "",
            "bytes_processed": stage.bytes_processed if stage else None,
            "error": f"{type(error).__name__}: {error}" if error is not None else None,
        }
        total = str(record["total"]) if record["total"] is not None else "?"
        completed = record["completed"] if record["completed"] is not None else 0
        speed = f"{rate:.2f}" if rate is not None else "unknown"
        percentage = f"{percent:.1f}%" if percent is not None else "unknown"
        line = (
            f"[CURRICULUM PROGRESS] event={event} pid={self.pid} "
            f"stage={json.dumps(record['stage'] or 'initializing')} "
            f"completed={completed}/{total} {record['unit'] or 'items'} "
            f"percent={percentage} elapsed={_duration(elapsed)} "
            f"build_elapsed={_duration(record['build_elapsed_seconds'])} "
            f"speed={speed} {record['unit'] or 'items'}/s stage_eta={_duration(eta)} "
            f"last_progress_age={_duration(record['last_progress_age_seconds'])} "
            f"detail={json.dumps(record['detail'], ensure_ascii=False)}"
        )
        if error is not None:
            line += f" error={json.dumps(record['error'], ensure_ascii=False)}"
        try:
            print(line, file=self.stream, flush=True)
        except (OSError, ValueError):
            pass
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            temporary = self.directory / f".build_progress.{self.run_id}.tmp"
            temporary.write_text(payload, encoding="utf-8")
            os.replace(temporary, self.latest_path)
            with self.history_path.open("a", encoding="utf-8") as handle:
                handle.write(payload)
        except OSError as exc:
            if not self._io_warning_printed:
                self._io_warning_printed = True
                try:
                    print(f"[CURRICULUM PROGRESS WARNING] cannot save progress: {exc}", file=self.stream, flush=True)
                except (OSError, ValueError):
                    pass

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> bool:
        self._stop.set()
        # Normally the daemon is asleep on the event and exits immediately.
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        with self._lock:
            self._status = "failed" if exc is not None else "complete"
            self._emit("build_failed" if exc is not None else "build_completed", force=True, error=exc)
        return False
