#!/usr/bin/env python3
"""Restart only a confirmed step-zero Detail Pyramid audit failure, without rebuilding data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import prepare_ui5_curriculum_snapshot as preparation


def resolve_submitted_state(previous: Path) -> Path:
    previous = previous.resolve(strict=True)
    root = previous.parent

    def read_pointer(marker):
        path = Path(marker.read_text(encoding="utf-8").strip()).resolve(strict=True)
        if path.name != "snapshot-switch.json" or path.parent.parent != root:
            raise RuntimeError(f"submission pointer leaves the known run-log root: {marker}")
        return path

    current = read_pointer(previous / "snapshot-switch-submit.started")
    seen = set()
    while True:
        if current in seen:
            raise RuntimeError("submission pointer cycle; refusing to guess a run")
        seen.add(current)
        marker = current.parent / "caption-retry.started"
        if not marker.exists():
            return current
        following = read_pointer(marker)
        state = json.loads(following.read_text(encoding="utf-8"))
        if Path(state.get("retry_of", "")).resolve() != current:
            raise RuntimeError("caption retry does not point back to the previous run")
        current = following


def verify_step_zero_failure(output: Path) -> dict:
    train_logs = sorted(output.glob("train-*.log"))
    pipeline_logs = sorted((output / "logs").glob("curriculum-*.log"))
    if not train_logs or not pipeline_logs:
        raise RuntimeError("training/pipeline logs are missing; cannot confirm the failed run")
    train_log, pipeline_log = train_logs[-1], pipeline_logs[-1]
    training = train_log.read_text(encoding="utf-8", errors="replace")
    pipeline = pipeline_log.read_text(encoding="utf-8", errors="replace")
    if "Initial Detail Pyramid scale weights are not thirds:" not in training:
        raise RuntimeError("latest training log is not the known Detail Pyramid step-zero failure")
    statuses = re.findall(r"^TRAIN_STATUS:\s*(\w+)\s*$", training, re.M)
    exits = re.findall(r"^TRAIN_EXIT_CODE:\s*(\d+)\s*$", training, re.M)
    if not statuses or statuses[-1] != "FAILED" or not exits or int(exits[-1]) == 0:
        raise RuntimeError("training has not recorded a terminal failure; do not submit another job yet")
    if not re.search(r"\[LOCANY FATAL\] script=[^\n]*run_locany_ui5_crop_rollout4_curriculum_h20x2\.sh "
                     r"line=\d+ exit_code=[1-9]\d*", pipeline):
        raise RuntimeError("curriculum controller has not recorded its fatal exit; wait for it to stop")
    # This recovery creates a NEW run only when there is no optimizer progress
    # to discard. Any rolling/transient state needs the normal exact-resume path.
    if list((output / "resume").glob("*")) or list(output.glob("checkpoint-*")):
        raise RuntimeError("checkpoint state exists; preserve it and use exact resume, not a step-zero restart")
    for path in (output / "trainer_state.json", output / "checkpoints.json"):
        if not path.exists():
            continue
        state = json.loads(path.read_text(encoding="utf-8"))
        steps = [int(state.get("global_step", 0))]
        steps.extend(int(row["step"]) for row in state.get("evaluations", []))
        if any(step != 0 for step in steps):
            raise RuntimeError("nonzero training/evaluation progress exists; exact resume is required")
    return {"training_log": str(train_log), "pipeline_log": str(pipeline_log),
            "failure": "Initial Detail Pyramid scale weights are not thirds", "completed_optimizer_steps": 0}


def restart(previous: Path, mlx_bin: str = "mlx") -> Path:
    old_state_path = resolve_submitted_state(previous)
    lock = old_state_path.parent / "detail-audit-restart.started"
    if lock.exists():
        raise RuntimeError(f"this failed run already has a restart reservation; inspect {lock}, do not repeat")
    old_state = json.loads(old_state_path.read_text(encoding="utf-8"))
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
    old_output = Path(env["OUTPUT_DIR"]).resolve(strict=True)
    preparation.log(f"[RECOVERY RUN] state={old_state_path} failed_output={old_output}")
    failure = verify_step_zero_failure(old_output)
    preparation.log("[RECOVERY DATA] checking published metadata only; no freeze/build/PNG relinking")
    manifest = preparation.verify_prepared_curriculum(env)
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
    submission.mkdir(exist_ok=False)
    state_path = submission / "snapshot-switch.json"
    env.update({"RUN_NAME": name, "OUTPUT_DIR": str(old_output.parent / name),
                "CODE_REVISION": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                         cwd=preparation.PROJECT_ROOT, text=True).strip()})
    job = preparation.render_job(old_job, env, f"ui5-curriculum-hour{hour}-{stamp}")
    new_job_path = submission / "formal.yaml"
    with new_job_path.open("x", encoding="utf-8") as handle:
        preparation.yaml.safe_dump(job, handle, sort_keys=False)
    state = {"status": "prepared_training_restart", "retry_of": str(old_state_path),
             "snapshot": old_state["snapshot"], "source": old_state.get("source"),
             "runtime": env, "job_yaml": str(new_job_path), "failure_evidence": failure,
             "curriculum_identity": manifest["identity_digest"],
             "reused_crop_assets": len(manifest["crop_assets"]), "generated_crop_assets": 0}
    preparation.write_state(state_path, state)
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(str(state_path) + "\n")
        handle.flush()
        preparation.os.fsync(handle.fileno())
    preparation.log(f"[RECOVERY SUBMIT] hard_groups={manifest['hard_groups']} reused={len(manifest['crop_assets'])} "
                    f"generated=0 baseline=reevaluate_step_0 new_output={env['OUTPUT_DIR']}")
    preparation.submit_job(mlx, new_job_path, state_path, state)
    return new_job_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-submission-dir", type=Path, required=True)
    parser.add_argument("--mlx-bin", default="mlx")
    args = parser.parse_args(argv)
    try:
        restart(args.previous_submission_dir, args.mlx_bin)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"[RECOVERY STOPPED] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
