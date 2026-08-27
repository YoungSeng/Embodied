"""Locked, resumable queue operations for independent CPT evaluation jobs."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping


def _queue_id(row: Mapping[str, Any]) -> str:
    existing = row.get("queue_id")
    if existing:
        return str(existing)
    value = f"{int(row.get('step') or 0)}\0{row.get('checkpoint', '')}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_eval_queue(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: queue row is not an object")
            value.setdefault("queue_id", _queue_id(value))
            value.setdefault("status", "pending")
            rows.append(value)
    return rows


def _atomic_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", delete=False
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@contextmanager
def eval_queue_lock(path: Path) -> Iterator[None]:
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            import fcntl
        except ImportError:  # Windows unit-test fallback.
            fcntl = None
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def enqueue_pending_eval(path: Path, row: Mapping[str, Any]) -> bool:
    """Append one pending checkpoint unless its step is already queued."""
    path = path.expanduser().resolve()
    candidate = dict(row)
    candidate["queue_id"] = _queue_id(candidate)
    candidate["status"] = "pending"
    with eval_queue_lock(path):
        rows = read_eval_queue(path)
        if any(int(value.get("step") or -1) == int(candidate["step"]) for value in rows):
            return False
        rows.append(candidate)
        _atomic_write(path, rows)
    return True


def claim_next_eval(
    path: Path,
    *,
    retry_failed: bool = False,
    worker: str | None = None,
) -> dict[str, Any] | None:
    path = path.expanduser().resolve()
    allowed = {"pending", "failed"} if retry_failed else {"pending"}
    with eval_queue_lock(path):
        rows = read_eval_queue(path)
        candidates = [row for row in rows if str(row.get("status")) in allowed]
        if not candidates:
            return None
        selected = min(candidates, key=lambda row: (int(row.get("step") or -1), row["queue_id"]))
        selected["status"] = "running"
        selected["attempt"] = int(selected.get("attempt") or 0) + 1
        selected["claimed_by"] = worker or socket.gethostname()
        selected["started_at_unix"] = time.time()
        selected.pop("error", None)
        _atomic_write(path, rows)
        return dict(selected)


def finish_eval(
    path: Path,
    queue_id: str,
    *,
    status: str,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if status not in {"completed", "failed"}:
        raise ValueError(f"invalid terminal eval status: {status}")
    path = path.expanduser().resolve()
    with eval_queue_lock(path):
        rows = read_eval_queue(path)
        selected = next((row for row in rows if row["queue_id"] == queue_id), None)
        if selected is None:
            raise KeyError(f"queue_id not found: {queue_id}")
        selected["status"] = status
        selected["finished_at_unix"] = time.time()
        if details:
            selected.update(dict(details))
        _atomic_write(path, rows)
        return dict(selected)
