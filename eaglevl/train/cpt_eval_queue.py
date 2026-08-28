"""Locked, resumable queue operations for independent CPT evaluation jobs."""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import os
import socket
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping


logger = logging.getLogger(__name__)
_UNSUPPORTED_FLOCK_ERRNOS = {
    errno.ENOSYS,
    errno.EINVAL,
    getattr(errno, "ENOTSUP", errno.EINVAL),
    getattr(errno, "EOPNOTSUPP", errno.EINVAL),
}
_DIRECTORY_LOCK_OWNER = "owner.json"


def _pid_is_alive(pid: int) -> bool:
    """Best-effort liveness check for an owner on the current host."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        return exc.errno != errno.ESRCH
    return True


def _read_directory_lock_owner(directory: Path) -> dict[str, Any] | None:
    owner_path = directory / _DIRECTORY_LOCK_OWNER
    try:
        value = json.loads(owner_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def _write_directory_lock_owner(directory: Path, owner: Mapping[str, Any]) -> None:
    owner_path = directory / _DIRECTORY_LOCK_OWNER
    temporary = directory / f".{_DIRECTORY_LOCK_OWNER}.{owner['token']}.tmp"
    temporary.write_text(
        json.dumps(dict(owner), ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, owner_path)


def _directory_lock_age_seconds(directory: Path) -> float:
    owner_path = directory / _DIRECTORY_LOCK_OWNER
    try:
        modified = owner_path.stat().st_mtime
    except FileNotFoundError:
        modified = directory.stat().st_mtime
    return max(0.0, time.time() - modified)


def _directory_lock_is_stale(
    directory: Path,
    *,
    stale_seconds: float,
    legacy_stale_seconds: float,
) -> tuple[bool, str, float]:
    """Return staleness without stealing a live same-host process lock."""
    age_seconds = _directory_lock_age_seconds(directory)
    owner = _read_directory_lock_owner(directory)
    if owner is None:
        return (
            age_seconds > legacy_stale_seconds,
            "legacy lock has no owner metadata",
            age_seconds,
        )
    hostname = str(owner.get("hostname") or "")
    try:
        pid = int(owner.get("pid") or 0)
    except (TypeError, ValueError):
        pid = 0
    if hostname == socket.gethostname() and pid > 0:
        alive = _pid_is_alive(pid)
        return (not alive, f"same-host owner pid={pid} alive={alive}", age_seconds)
    return (
        age_seconds > stale_seconds,
        f"remote/unknown owner host={hostname!r} pid={pid}",
        age_seconds,
    )


def _reclaim_directory_lock(directory: Path, *, reason: str, age_seconds: float) -> bool:
    """Atomically move a stale lock aside, then remove only known metadata files."""
    quarantine = directory.with_name(
        f"{directory.name}.stale-{os.getpid()}-{uuid.uuid4().hex}"
    )
    try:
        directory.rename(quarantine)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    try:
        for child in quarantine.iterdir():
            if child.is_file() and (
                child.name == _DIRECTORY_LOCK_OWNER
                or child.name.startswith(f".{_DIRECTORY_LOCK_OWNER}.")
            ):
                child.unlink(missing_ok=True)
        quarantine.rmdir()
    except OSError as exc:
        logger.warning(
            "stale eval lock was quarantined but cleanup was incomplete: path=%s error=%s",
            quarantine,
            exc,
        )
    logger.warning(
        "reclaimed stale eval directory lock: path=%s age=%.1fs reason=%s",
        directory,
        age_seconds,
        reason,
    )
    return True


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


def fsync_if_supported(handle: Any, *, path: Path) -> None:
    try:
        os.fsync(handle.fileno())
    except OSError as exc:
        if exc.errno not in _UNSUPPORTED_FLOCK_ERRNOS:
            raise
        logger.warning(
            "eval queue filesystem does not support fsync for %s (%s); "
            "continuing with close + atomic replace",
            path,
            exc,
        )


def _atomic_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", delete=False
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        fsync_if_supported(handle, path=temporary)
    os.replace(temporary, path)


def _acquire_flock(handle: Any) -> Any | None:
    """Return the fcntl module when flock works, otherwise request fallback."""

    try:
        import fcntl
    except ImportError:  # Windows unit-test fallback.
        return None
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    except OSError as exc:
        if exc.errno not in _UNSUPPORTED_FLOCK_ERRNOS:
            raise
        logger.warning(
            "eval queue filesystem does not support flock (%s); "
            "using atomic directory lock",
            exc,
        )
        return None
    return fcntl


@contextmanager
def _directory_queue_lock(lock_path: Path) -> Iterator[None]:
    """Portable NAS lock with owner liveness and heartbeat-based recovery."""

    directory = Path(str(lock_path) + ".mkdir")
    timeout_seconds = float(os.environ.get("CPT_EVAL_QUEUE_LOCK_TIMEOUT_SECONDS", "120"))
    stale_seconds = float(os.environ.get("CPT_EVAL_QUEUE_LOCK_STALE_SECONDS", "600"))
    legacy_stale_seconds = float(
        os.environ.get("CPT_EVAL_QUEUE_LEGACY_LOCK_STALE_SECONDS", "600")
    )
    heartbeat_seconds = max(
        0.1,
        float(os.environ.get("CPT_EVAL_QUEUE_LOCK_HEARTBEAT_SECONDS", "30")),
    )
    owner = {
        "schema_version": 1,
        "token": uuid.uuid4().hex,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "acquired_at_unix": time.time(),
        "heartbeat_at_unix": time.time(),
    }
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            directory.mkdir()
            try:
                _write_directory_lock_owner(directory, owner)
            except Exception:
                try:
                    (directory / _DIRECTORY_LOCK_OWNER).unlink(missing_ok=True)
                    for temporary in directory.glob(
                        f".{_DIRECTORY_LOCK_OWNER}.*.tmp"
                    ):
                        temporary.unlink(missing_ok=True)
                    directory.rmdir()
                except OSError:
                    pass
                raise
            break
        except FileExistsError:
            try:
                stale, reason, age_seconds = _directory_lock_is_stale(
                    directory,
                    stale_seconds=stale_seconds,
                    legacy_stale_seconds=legacy_stale_seconds,
                )
            except FileNotFoundError:
                continue
            if stale:
                if _reclaim_directory_lock(
                    directory,
                    reason=reason,
                    age_seconds=age_seconds,
                ):
                    continue
            if time.monotonic() >= deadline:
                owner_info = _read_directory_lock_owner(directory)
                raise TimeoutError(
                    f"timed out waiting for eval queue lock {directory} after "
                    f"{timeout_seconds:.1f}s; owner={owner_info}"
                )
            time.sleep(0.1)

    stop_heartbeat = threading.Event()

    def heartbeat() -> None:
        while not stop_heartbeat.wait(heartbeat_seconds):
            current = _read_directory_lock_owner(directory)
            if current is None or current.get("token") != owner["token"]:
                return
            owner["heartbeat_at_unix"] = time.time()
            try:
                _write_directory_lock_owner(directory, owner)
            except (FileNotFoundError, OSError):
                return

    heartbeat_thread = threading.Thread(
        target=heartbeat,
        name="cpt-eval-lock-heartbeat",
        daemon=True,
    )
    heartbeat_thread.start()
    try:
        yield
    finally:
        stop_heartbeat.set()
        heartbeat_thread.join(timeout=max(1.0, heartbeat_seconds + 1.0))
        try:
            current = _read_directory_lock_owner(directory)
            if current is not None and current.get("token") == owner["token"]:
                (directory / _DIRECTORY_LOCK_OWNER).unlink(missing_ok=True)
                for temporary in directory.glob(f".{_DIRECTORY_LOCK_OWNER}.*.tmp"):
                    temporary.unlink(missing_ok=True)
                directory.rmdir()
        except FileNotFoundError:
            pass


@contextmanager
def exclusive_file_lock(lock_path: Path) -> Iterator[None]:
    """Cross-process lock with a ByteNAS-safe fallback when flock is absent."""

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl_module = _acquire_flock(handle)
        if fcntl_module is None:
            with _directory_queue_lock(lock_path):
                yield
            return
        try:
            yield
        finally:
            fcntl_module.flock(handle.fileno(), fcntl_module.LOCK_UN)


@contextmanager
def eval_queue_lock(path: Path) -> Iterator[None]:
    lock_path = path.with_suffix(path.suffix + ".lock")
    with exclusive_file_lock(lock_path):
        yield


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
