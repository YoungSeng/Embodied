"""DeepSpeed API validation and audited runtime-only updates of an existing run."""
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import subprocess

from eaglevl.train.ui5_grpo_core import digest


# Model, decoding, reward, sampling, data and optimizer configuration changes
# require a new run. These files are the execution/telemetry repair surface.
RUNTIME_UPDATE_FILES = frozenset({
    "README_UI5_CROP_GRPO_MIXED_V1.md",
    "eaglevl/train/ui5_grpo_runtime.py",
    "eaglevl/train/ui5_grpo_trainer.py",
    "eaglevl/train/ui5_grpo_checkpoint.py",
    "scripts/submit_ui5_grpo_mixed.py",
    "scripts/run_ui5_grpo_pipeline.py",
    "scripts/ui5_grpo_artifacts.py",
    "tests/test_ui5_grpo_runtime.py",
    "tests/test_ui5_grpo_distributed.py",
    "tests/test_ui5_grpo_pipeline.py",
})


def backward_api_info(backward):
    """NVTX wrappers expose (*args, **kwargs), without functools.wraps.

    Accept their keyword forwarding; the actual backward call still explicitly
    sends scale_wrt_gas=False. Never retry backward or compensate loss scaling
    on a TypeError, since a backward call can already have side effects.
    """
    signature = inspect.signature(backward)
    parameters = signature.parameters
    forwarded = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    named = parameters.get("scale_wrt_gas")
    if not forwarded and (named is None or named.kind == inspect.Parameter.POSITIONAL_ONLY):
        raise RuntimeError(f"DeepSpeed backward cannot accept scale_wrt_gas=False: {signature}")
    return dict(backward_signature=str(signature), keyword_forwarding=forwarded,
                scale_wrt_gas=False)


def _git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True, encoding="utf-8").strip()


def _code_state(run):
    payload = dict(run)
    if payload.pop("identity") != digest(payload):
        raise ValueError("immutable run.json identity changed")
    root = Path(run["project_root"])
    if _git(root, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("formal source has uncommitted modifications")
    return root, _git(root, "rev-parse", "HEAD")


def _runtime_changes(root, original, current):
    subprocess.run(["git", "merge-base", "--is-ancestor", original, current], cwd=root, check=True)
    names = _git(root, "diff", "--name-only", "--no-renames", original, current, "--").splitlines()
    unsupported = set(names) - RUNTIME_UPDATE_FILES
    if unsupported:
        raise ValueError(f"code update changes the experiment contract; use a new run: {sorted(unsupported)}")
    result = {}
    for name in names:
        before = _git(root, "ls-tree", original, "--", name)
        after = _git(root, "ls-tree", current, "--", name)
        if not after:
            raise ValueError(f"runtime update deleted a required source file: {name}")
        result[name] = dict(before=before.split()[2] if before else None, after=after.split()[2])
    return result


def _verify_revision(run, root, current, record):
    payload = dict(record)
    if payload.pop("identity") != digest(payload):
        raise ValueError("runtime code revision record changed")
    if (record["run_identity"], record["run_code_sha"], record["execution_code_sha"]) != (
            run["identity"], run["code_sha"], current):
        raise ValueError("runtime code revision is not bound to this run/checkout")
    if record["changed_files"] != _runtime_changes(root, run["code_sha"], current):
        raise ValueError("runtime code revision file inventory changed")


def verify_execution_code(run):
    root, current = _code_state(run)
    path = Path(run["output_dir"]) / "runtime_code_revision.json"
    if current == run["code_sha"] and not path.exists():
        return current
    if not path.is_file():
        raise ValueError("checkout SHA differs from run.json; submit with --resume-code-update")
    _verify_revision(run, root, current, json.loads(path.read_text(encoding="utf-8")))
    return current


def bind_execution_code(run, *, allow_update=False):
    """Caller holds the pipeline lock; run.json and all training state stay intact."""
    from scripts.build_ui5_grpo_mixed_manifest import write_json
    root, current = _code_state(run)
    path = Path(run["output_dir"]) / "runtime_code_revision.json"
    if current == run["code_sha"] and not path.exists():
        return current
    if path.is_file():
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("execution_code_sha") == current:
            _verify_revision(run, root, current, record)
            return current
    if not allow_update:
        raise ValueError("existing run uses another code SHA; use --resume-code-update for a runtime-only repair")
    changes = _runtime_changes(root, run["code_sha"], current)
    record = dict(schema_version=1, run_identity=run["identity"], run_code_sha=run["code_sha"],
                  execution_code_sha=current, changed_files=changes,
                  created_at=datetime.now(timezone.utc).isoformat(),
                  reason="runtime compatibility repair; preserve fixed reference, manifest and completed evaluation")
    record["identity"] = digest(record)
    write_json(Path(run["output_dir"]) / "diagnostics/code_revisions" / f"{record['identity']}.json", record)
    write_json(path, record)
    return current
