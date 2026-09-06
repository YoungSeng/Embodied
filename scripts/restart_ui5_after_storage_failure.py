#!/usr/bin/env python3
"""Explicit new-run restart after the FIRST checkpoint save runs out of storage.

Never repairs an incomplete checkpoint, deletes old data, or resets a valid
resume checkpoint. The caller must confirm the old platform job has stopped
and storage has been reclaimed, and explicitly opt into repeating steps 0-200.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eaglevl.train.ui5_checkpoint_utils import validate_checkpoint
from scripts import prepare_ui5_curriculum_snapshot as preparation

GIB = 1024 ** 3


def verify_first_save_failure(output: Path) -> dict:
    """Read only. A failed save is NOT proof that its tensor files are valid."""
    logs = []
    for pattern in ("train-*.log", "logs/curriculum-*.log"):
        candidates = sorted(output.glob(pattern))
        if candidates:
            logs.append(candidates[-1])
    if not logs:
        raise RuntimeError("failed run logs are missing; cannot identify a storage failure")
    texts = [path.read_text(encoding="utf-8", errors="replace") for path in logs]
    for text in texts:
        statuses = re.findall(r"^TRAIN_STATUS:\s*(\w+)\s*$", text, re.M)
        if statuses and statuses[-1] != "FAILED":
            raise RuntimeError("latest log does not end with failed training; inspect the platform job")
    # Require the storage error and the exact first-save path on the SAME line.
    # A quota-full tee may lose the terminal footer. The explicit stopped-job
    # confirmation remains mandatory even when the footer is present.
    failure_lines = [line for text in texts for line in text.splitlines()
                     if re.search(r"Disk quota exceeded|No space left on device|\[Errno (?:122|28)\]", line)
                     and re.search(r"checkpoint-200[/\\]", line)]
    if not failure_lines:
        raise RuntimeError("latest logs do not identify a storage failure saving checkpoint-200")
    if (output / "pipeline_complete.json").exists():
        raise RuntimeError("run is complete; refusing a failed-run restart")
    if list((output / "resume").glob("*")) or list((output / "checkpoints").glob("*")):
        raise RuntimeError("rolling/permanent checkpoint state exists; use exact resume instead")
    for checkpoint in output.glob("checkpoint-*"):
        if checkpoint.name != "checkpoint-200" or checkpoint.is_symlink():
            raise RuntimeError(f"unexpected checkpoint state; preserve it and inspect exact resume: {checkpoint}")
        if (checkpoint / "checkpoint_complete.json").exists():
            raise RuntimeError("checkpoint-200 has a completion marker; inspect exact resume instead")
    ledger = output / "checkpoints.json"
    if ledger.exists():
        state = json.loads(ledger.read_text(encoding="utf-8"))
        if any(int(row["step"]) != 0 for row in state.get("evaluations", [])):
            raise RuntimeError("nonzero evaluations exist; use exact resume instead")
    trainer_state = output / "trainer_state.json"
    if trainer_state.exists():
        raise RuntimeError("root trainer state exists; inspect completed training before restarting")
    return {"kind": "first_checkpoint_storage_failure", "failed_save_step": 200,
            "restart_global_step": 0, "incomplete_checkpoint_trusted": False,
            "logs": [str(path) for path in logs], "storage_error": failure_lines[-1],
            "old_job_stopped": "operator_confirmed"}


def tree_bytes(directory: Path) -> int:
    # Only checkpoint metadata/stat calls; never traverse the crop/image bundle.
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def check_storage(output_parent: Path, logs_parent: Path, checkpoint: Path, model: Path) -> dict:
    # Conservative estimate, not a quota API: fp32 master/moments, sharded state,
    # model copies and serialization overhead depend on the training backend.
    observed = tree_bytes(checkpoint) if checkpoint.is_dir() else 0
    model_bytes = sum(path.stat().st_size for path in model.rglob("*")
                      if path.is_file() and (path.suffix == ".safetensors" or
                                            path.name.startswith("pytorch_model") and path.suffix == ".bin"))
    estimate = max(observed, model_bytes * 8)
    next_save = int(estimate * 1.2) + 256 * 1024 ** 2
    free = shutil.disk_usage(output_parent).free
    preparation.log(f"[STORAGE CHECK] filesystem_free_gib={free / GIB:.2f} "
                    f"observed_partial_checkpoint_gib={observed / GIB:.2f} "
                    f"estimated_checkpoint_gib={estimate / GIB:.2f} "
                    f"next_save_headroom_gib={next_save / GIB:.2f} "
                    f"all_nodes_peak_estimate_gib={estimate * 8 / GIB:.2f} "
                    "user_quota_available=UNKNOWN")
    if free < next_save:
        raise RuntimeError("filesystem free space is below the next-save estimate; reclaim storage before submitting")
    for directory in dict.fromkeys((output_parent, logs_parent)):
        # Test both the model and job-log locations, including fsync. This probe
        # owns only its unique temporary file, removed by the context manager.
        with tempfile.TemporaryFile(prefix=".ui5-storage-probe-", dir=directory) as handle:
            handle.write(b"\0" * 1024 ** 2)
            handle.flush()
            os.fsync(handle.fileno())
    preparation.log("[STORAGE PASS] 1 MiB write+fsync per location; this does NOT prove remaining user quota. "
                    "Check platform byte/inode quota; permanent best checkpoints can accumulate at all six nodes.")
    return {"filesystem_free_bytes": free, "observed_partial_checkpoint_bytes": observed,
            "estimated_checkpoint_bytes": estimate, "next_save_headroom_bytes": next_save,
            "all_nodes_peak_estimate_bytes": estimate * 8, "user_quota_available_bytes": None,
            "storage_reclaimed": "operator_confirmed", "probe_bytes_per_location": 1024 ** 2}


def restart(failed_submission_dir: Path, *, restart_from_base: bool = False,
            confirm_job_stopped: bool = False, confirm_storage_reclaimed: bool = False,
            check_only: bool = False, mlx_bin: str = "mlx") -> Path:
    if not (restart_from_base and confirm_job_stopped and confirm_storage_reclaimed):
        raise RuntimeError("require --restart-from-base --confirm-job-stopped --confirm-storage-reclaimed; "
                           "this repeats steps 0-200, not an exact checkpoint resume")
    old_state_path = failed_submission_dir.resolve(strict=True) / "snapshot-switch.json"
    old_state = json.loads(old_state_path.read_text(encoding="utf-8"))
    # Directly select the actual failed run, NOT an ancestor's old restart lock.
    lock = old_state_path.parent / "storage-restart.started"
    for name in ("storage-restart.started", "detail-audit-restart.started", "caption-retry.started"):
        marker = old_state_path.parent / name
        if marker.exists():
            raise RuntimeError(f"this run already has a successor reservation: {marker}; inspect its target, do not repeat")
    job_path = Path(old_state["job_yaml"]).resolve(strict=True)
    if job_path.parent != old_state_path.parent:
        raise RuntimeError("saved job YAML is outside its submission directory")
    old_job = preparation.yaml.safe_load(job_path.read_text(encoding="utf-8"))
    env = dict(old_job["jobRunParams"]["envsList"])
    if any(env.get(key) != value for key, value in old_state["runtime"].items()):
        raise RuntimeError("saved YAML/runtime mismatch")
    if any(env.get(key) != value for key, value in preparation.FORMAL_ENV.items()):
        raise RuntimeError("saved job is not the formal H20x2 curriculum profile")
    if Path(env["PROJECT_ROOT"]).resolve() != preparation.PROJECT_ROOT.resolve():
        raise RuntimeError("run recovery from the original project checkout")
    output = Path(env["OUTPUT_DIR"]).resolve(strict=True)
    if output.name != env["RUN_NAME"] or old_state_path.parent.name != env["RUN_NAME"]:
        raise RuntimeError("failed submission/RUN_NAME/OUTPUT_DIR identities disagree")
    preparation.log(f"[STORAGE RECOVERY] failed_state={old_state_path} failed_output={output}")
    evidence = verify_first_save_failure(output)
    preparation.log("[RECOVERY DATA] published metadata only; no freeze, build, PNG scan or relinking")
    manifest = preparation.verify_prepared_curriculum(env)
    model = Path(env["MODEL_PATH"]).resolve(strict=True)
    if model == output or model.is_relative_to(output):
        raise RuntimeError("MODEL_PATH must be the original crop model, not this failed run")
    report = validate_checkpoint(model, mode="eval")
    if not report["valid"]:
        raise RuntimeError(f"original crop checkpoint is missing/incomplete (was it deleted?): {report['errors']}")
    processor = Path(env["PROCESSOR_PATH"])
    if not (processor / "tokenizer_config.json").is_file() or not any(
            (processor / name).is_file() for name in ("processor_config.json", "preprocessor_config.json")):
        raise RuntimeError("processor/tokenizer assets are missing; restore them before submitting")
    storage = check_storage(output.parent, old_state_path.parent.parent, output / "checkpoint-200", model)
    if check_only:
        preparation.log("[CHECK ONLY PASS] no job submitted; no restart reservation created")
        return job_path
    mlx = shutil.which(mlx_bin)
    if not mlx:
        raise RuntimeError("mlx is missing; run on the authenticated development host")
    snapshot = Path(old_state["snapshot"]).name
    if not re.fullmatch(r"hour_\d{3}_\d{8}T\d{6}Z", snapshot):
        raise RuntimeError("saved snapshot identity is invalid")
    hour = snapshot.split("_")[1]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    name = f"locany-ui5-crop-rollout4-curriculum-hour{hour}-h20x2-sdpa7268-{stamp}"
    submission = old_state_path.parent.parent / name
    new_output = output.parent / name
    if new_output.exists():
        raise RuntimeError(f"new OUTPUT_DIR already exists: {new_output}")
    for key in ("RESUME_FROM_CHECKPOINT", "CURRICULUM_START_STEP", "LOCANY_STOP_AFTER_STEP", "LOCANY_SEGMENT_MODE"):
        env.pop(key, None)
    env.update({"RUN_NAME": name, "OUTPUT_DIR": str(new_output),
                "CODE_REVISION": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                         cwd=preparation.PROJECT_ROOT, text=True).strip()})
    job = preparation.render_job(old_job, env, f"ui5-curriculum-hour{hour}-{stamp}")
    submission.mkdir(exist_ok=False)
    state_path = submission / "snapshot-switch.json"
    new_job_path = submission / "formal.yaml"
    with new_job_path.open("x", encoding="utf-8") as handle:
        preparation.yaml.safe_dump(job, handle, sort_keys=False)
    state = {"status": "prepared_storage_restart", "retry_of": str(old_state_path),
             "recovery_mode": "explicit_restart_from_original_crop", "snapshot": old_state["snapshot"],
             "source": old_state.get("source"), "runtime": env, "job_yaml": str(new_job_path),
             "failure_evidence": evidence, "storage_check": storage,
             "curriculum_identity": manifest["identity_digest"],
             "reused_crop_assets": len(manifest["crop_assets"]), "generated_crop_assets": 0}
    preparation.write_state(state_path, state)
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(str(state_path) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    preparation.log(f"[RECOVERY SUBMIT] start_step=0 total_steps=1200 hard_groups={manifest['hard_groups']} "
                    f"reused={len(manifest['crop_assets'])} generated=0 old_checkpoint=UNTOUCHED "
                    f"new_output={new_output}")
    preparation.submit_job(mlx, new_job_path, state_path, state)
    return new_job_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failed-submission-dir", type=Path, required=True)
    parser.add_argument("--restart-from-base", action="store_true", help="explicitly repeat the lost first 200 steps")
    parser.add_argument("--confirm-job-stopped", action="store_true", help="operator confirms old platform job is terminal")
    parser.add_argument("--confirm-storage-reclaimed", action="store_true", help="operator has reclaimed/checks platform quota")
    parser.add_argument("--check-only", action="store_true", help="CPU checks only, no reservation/submission")
    parser.add_argument("--mlx-bin", default="mlx")
    args = parser.parse_args(argv)
    try:
        restart(**vars(args))
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"[STORAGE RECOVERY STOPPED] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
