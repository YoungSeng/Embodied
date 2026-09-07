#!/usr/bin/env python3
"""Read-only Linux observer, including submit processes started before progress support."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

from ui14_common import DATA_ROOT
from ui14_progress import duration


def words_for(pid, proc=Path("/proc")):
    try:
        return [v.decode("utf-8", errors="replace") for v in
                (proc / str(pid) / "cmdline").read_bytes().split(b"\0") if v]
    except OSError:
        return []


def option(words, name, default=None):
    for index, value in enumerate(words):
        if value.startswith(name + "="):
            return value[len(name) + 1:]
        if value == name and index + 1 < len(words):
            return words[index + 1]
    return default


def is_submit(words, root):
    return (any(Path(v).name == "submit_locany_ui5.py" for v in words)
            and option(words, "--profile") == "m32-cpt9000-ui14-v1"
            and os.path.normpath(option(words, "--ui14-data-root", DATA_ROOT)) == os.path.normpath(str(root)))


def process_info(pid, proc=Path("/proc")):
    try:
        value = (proc / str(pid) / "stat").read_text()
        fields = value[value.rfind(")") + 2:].split()
        if fields[0] in ("Z", "X"):
            return None
        ticks = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
        elapsed = float((proc / "uptime").read_text().split()[0]) - int(fields[19]) / ticks
        return {"pid": pid, "start_ticks": int(fields[19]), "elapsed_seconds": max(0, elapsed),
                "state": fields[0]}
    except (OSError, ValueError, IndexError):
        return None


def find_processes(root, proc=Path("/proc")):
    return [int(p.name) for p in proc.iterdir() if p.name.isdigit()
            and is_submit(words_for(int(p.name), proc), root) and process_info(int(p.name), proc)]


def open_readers(pid, proc=Path("/proc")):
    """Only inspect fd metadata; never open an image or read a manifest's contents."""
    readers = []
    folder = proc / str(pid)
    try:
        descriptors = list((folder / "fd").iterdir())
    except OSError:
        return readers
    for descriptor in descriptors:
        try:
            path = Path(os.readlink(descriptor))
            info = dict(line.split(":", 1) for line in
                        (folder / "fdinfo" / descriptor.name).read_text().splitlines() if ":" in line)
            flags = int(info["flags"].strip(), 8)
            # The append journal stays open after loading. Do not mistake its
            # permanent EOF offset for an in-progress verification read.
            if flags & os.O_APPEND or (flags & getattr(os, "O_ACCMODE", 3)) == os.O_WRONLY:
                continue
            if not path.is_absolute() or path.suffix not in (".json", ".jsonl", ".yaml"):
                continue
            readers.append({"path": str(path), "fd": descriptor.name,
                            "position": int(info["pos"]), "total": path.stat().st_size})
        except (OSError, ValueError, KeyError):
            continue
    return readers


def snapshot(pid, root, proc=Path("/proc")):
    info = process_info(pid, proc)
    if info is None:
        return None
    try:
        progress = json.loads((Path(root) / "progress/submit.json").read_text(encoding="utf-8"))
        from datetime import datetime
        updated = datetime.fromisoformat(progress["updated_at"]).timestamp()
        if (progress.get("pid") == pid and progress.get("stage") == "submit"
                and updated >= time.time() - info["elapsed_seconds"] - 2):
            info["progress"] = progress
    except (OSError, ValueError, TypeError, KeyError):
        pass
    try:
        children = (proc / str(pid) / "task" / str(pid) / "children").read_text().split()
    except OSError:
        children = []
    info["mlx_pids"] = [int(child) for child in children if
        any(Path(w).name in ("mlx", "mlx.py") for w in words_for(child, proc))
        and "submitv2" in words_for(child, proc)]
    info["readers"] = open_readers(pid, proc)
    after = process_info(pid, proc)
    return info if after and after["start_ticks"] == info["start_ticks"] else None


class Observer:
    def __init__(self):
        self.samples = {}

    def describe(self, value, now=None):
        now = time.monotonic() if now is None else now
        lines = [f"[UI14 submit-status] PID={value['pid']} | 原进程已用 {duration(value['elapsed_seconds'])}"
                 " | 只读观察，原进程继续运行"]
        if value["mlx_pids"]:
            lines.append(f"已调用 mlx（PID={value['mlx_pids']}），等待集群响应；服务端剩余时间不可估算。")
        elif value.get("progress"):
            progress = value["progress"]
            lines.append(f"提交入口状态: {progress['status']} | 更新于 {progress['updated_at']}")
            for frame in progress.get("phases", []):
                total = frame.get("total")
                lines.append(f"{frame['label']}: {frame['completed']}/{total if total is not None else '?'} {frame['unit']}"
                             f" | 本阶段剩余≈{duration(frame.get('eta_seconds'))} | {frame.get('detail', '')}")
            activity = progress.get("activity")
            if activity:
                lines.append(f"当前文件: {activity['label']} | {activity['completed']}/{activity['total']} bytes"
                             f" | 文件剩余≈{duration(activity.get('eta_seconds'))}")
        elif value["readers"]:
            priority = {"image_evidence.jsonl": 0, "files.jsonl": 1}
            reader = min(value["readers"], key=lambda r: priority.get(Path(r["path"]).name, 2))
            key = (value["pid"], value["start_ticks"], reader["fd"], reader["path"], reader["total"])
            position, total = min(reader["position"], reader["total"]), reader["total"]
            first = self.samples.get(key)
            if first is None or position < first[1]:
                first = self.samples[key] = (now, position)
            rate = (position - first[1]) / (now - first[0]) if now > first[0] else 0
            eta = (total - position) / rate if rate > 0 and position < total else None
            label = {"files.jsonl": "加载已保存的验证记录", "image_evidence.jsonl": "图片清单读取/属性检查"}.get(
                Path(reader["path"]).name, "产物/来源文件检查")
            percent = 100 * position / total if total else 0
            lines.append(f"{label}: {position / 1048576:.1f}/{total / 1048576:.1f} MiB ({percent:.1f}%)"
                         f" | 本文件剩余≈{duration(eta)} | 按读取字节估算，非整次提交 ETA")
            lines.append(reader["path"])
            if position == total:
                lines.append("文件已读入缓冲区，CPU 处理可能尚未完成；不据此判定检查完成。")
        else:
            lines.append(f"原进程仍存在（状态 {value['state']}）；此采样未捕获可量化文件读取或 mlx 子进程。"
                         "旧版未记录该阶段计数，剩余时间暂不可估算；请结合原窗口输出。")
        return "\n  ".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(DATA_ROOT))
    parser.add_argument("--pid", type=int)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=10)
    args = parser.parse_args(argv)
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error("--interval must be finite and positive")
    if os.name != "posix" or not Path("/proc/uptime").exists():
        parser.error("Run on the same Linux host and as the same user as the existing submit process")
    if args.pid is not None:
        candidates = [args.pid] if is_submit(words_for(args.pid), args.data_root) and process_info(args.pid) else []
    else:
        candidates = find_processes(args.data_root)
    if not candidates:
        print("当前主机/用户下未找到匹配的 UI14 submit 进程；请在原提交主机运行，并查看原窗口的 mlx 返回。\n"
              "本命令未发起提交，不能据此判断集群任务是否已创建。", flush=True)
        return 1
    if len(candidates) != 1:
        parser.error(f"Multiple matching submit processes: {candidates}; select --pid PID")
    pid, identity, observer = candidates[0], None, Observer()
    try:
        while True:
            value = snapshot(pid, args.data_root)
            if value is None or identity is not None and identity != value["start_ticks"]:
                print("原 submit 进程已退出；请查看原窗口 mlx 返回与任务 ID。本观察器未提交或重试。", flush=True)
                return 0
            identity = value["start_ticks"]
            print(observer.describe(value), flush=True)
            if not args.watch:
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("仅停止观察；原 submit 进程继续运行。", flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
