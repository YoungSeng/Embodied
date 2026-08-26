#!/usr/bin/env python3
"""Record and compare the binary Python environment used by UI5 train segments."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


PACKAGES = (
    "torch",
    "torchvision",
    "transformers",
    "deepspeed",
    "accelerate",
    "numpy",
    "safetensors",
    "magi_attention",
    "triton",
)
MODULES = ("torch", "transformers", "deepspeed", "magi_attention")


def _file_stat(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    resolved = Path(path).expanduser().resolve()
    try:
        stat = resolved.stat()
    except OSError as exc:
        return {"path": str(resolved), "error": f"{type(exc).__name__}: {exc}"}
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
    }


def collect_environment() -> dict[str, Any]:
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    module_files = {}
    for name in MODULES:
        try:
            spec = importlib.util.find_spec(name)
            module_files[name] = _file_stat(spec.origin if spec else None)
        except Exception as exc:
            module_files[name] = {"error": f"{type(exc).__name__}: {exc}"}

    try:
        import torch

        torch_runtime = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "git_version": getattr(torch.version, "git_version", None),
            "debug_build": bool(getattr(torch.version, "debug", False)),
            "torch_file": _file_stat(torch.__file__),
            "torch_c_file": _file_stat(getattr(torch._C, "__file__", None)),
        }
        torch_root = Path(torch.__file__).resolve().parent
        torch_libraries = [
            _file_stat(path)
            for path in sorted((torch_root / "lib").glob("*.so*"))
            if path.is_file()
        ]
    except Exception as exc:
        raise RuntimeError(
            f"PyTorch import failed during environment audit: {type(exc).__name__}: {exc}"
        ) from exc

    stable = {
        "python": {
            "executable": str(Path(sys.executable).resolve()),
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "packages": packages,
        "module_files": module_files,
        "torch_runtime": torch_runtime,
        "torch_libraries": torch_libraries,
    }
    canonical = json.dumps(stable, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": 1,
        "fingerprint_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "stable": stable,
    }


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("pre", "post"), required=True)
    parser.add_argument("--allow-change", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    environment_dir = output_dir / "environment"
    baseline_path = environment_dir / "runtime_environment_baseline.json"
    current = collect_environment()
    current.update(
        {
            "captured_at_unix": time.time(),
            "phase": args.phase,
            "hostname": platform.node(),
            "pid": os.getpid(),
        }
    )
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    snapshot_path = environment_dir / f"runtime_environment_{args.phase}_{stamp}_{os.getpid()}.json"
    atomic_write_json(snapshot_path, current)

    if not baseline_path.is_file():
        atomic_write_json(baseline_path, current)
        status = "BASELINE_CREATED"
        changed = False
    else:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        changed = baseline.get("fingerprint_sha256") != current["fingerprint_sha256"]
        status = "CHANGED" if changed else "MATCH"

    print("===== LocateAnything runtime environment audit =====")
    print(f"phase                  : {args.phase}")
    print(f"status                 : {status}")
    print(f"fingerprint_sha256     : {current['fingerprint_sha256']}")
    print(f"baseline               : {baseline_path}")
    print(f"snapshot               : {snapshot_path}")
    print(f"torch                   : {current['stable']['packages']['torch']}")
    print(f"transformers            : {current['stable']['packages']['transformers']}")
    print(f"deepspeed               : {current['stable']['packages']['deepspeed']}")
    print("=====================================================")
    if changed and not args.allow_change:
        print(
            "[ENVIRONMENT ERROR] Python/PyTorch binary environment changed during this run. "
            "Do not pip/conda install into the shared ENV_DIR while training is active.",
            file=sys.stderr,
        )
        return 46
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
