#!/usr/bin/env python3
"""Foreground: finish existing crops -> freeze a new snapshot -> reuse -> submit H20x2.

An explicitly selected legacy foreground submitter can be held with SIGSTOP;
its crop-building child is NEVER signalled. The submitter is terminated only
after that child exits. Failures retain the hold and never submit or recrop.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = Path("/mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace")
FORMAL_ENV = {
    "CUDA_VISIBLE_DEVICES": "0,1", "CURRICULUM_MODE": "scheduled",
    "TOTAL_STEPS": "1200", "EVAL_INTERVAL_STEPS": "200",
    "ROLLING_CHECKPOINT_DIR": "resume/latest", "CHECKPOINT_SAVE_POLICY": "best_only",
    "UI5_GPU0_WORKERS": "2", "UI5_GPU1_WORKERS": "3",
    "HARD_RATIOS": "0.60,0.45,0.30", "ANCHOR_RATIOS": "0.25,0.35,0.30",
    "GLOBAL_REPLAY_RATIOS": "0.15,0.20,0.40", "LLM_LRS": "1e-6,7e-7,5e-7",
    "ATTN_IMPLEMENTATION": "sdpa", "MAX_SEQ_LENGTH": "7268",
    "MAX_NUM_TOKENS_PER_SAMPLE": "7268", "MAX_NUM_TOKENS": "7268",
    "SEED": "42", "NNODES": "1", "NODE_RANK": "0",
    "CURRICULUM_PROGRESS_INTERVAL_SECONDS": "10", "UI5_EVAL_HEARTBEAT_SECONDS": "30",
}


def log(message: str) -> None:
    print(message, flush=True)


def write_state(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_process(pid: int) -> dict[str, Any] | None:
    """PID plus kernel start ticks prevents signalling a recycled PID."""
    root = Path(f"/proc/{pid}")
    try:
        stat = (root / "stat").read_text()
        fields = stat[stat.rfind(")") + 2:].split()
        argv = [os.fsdecode(arg) for arg in (root / "cmdline").read_bytes().split(b"\0") if arg]
        return {"pid": pid, "ppid": int(fields[1]), "state": fields[0],
                "start_ticks": fields[19], "argv": argv}
    except FileNotFoundError:
        return None


def process_active(identity: dict[str, Any]) -> bool:
    current = read_process(identity["pid"])
    return bool(current and current["start_ticks"] == identity["start_ticks"]
                and current["state"] not in {"Z", "X"})


def signal_process(identity: dict[str, Any], sig: int) -> bool:
    if not process_active(identity):
        return False
    # pidfd closes the remaining check-to-signal race when available (Linux).
    if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
        try:
            descriptor = os.pidfd_open(identity["pid"])
        except ProcessLookupError:
            return False
        try:
            if not process_active(identity):
                return False
            signal.pidfd_send_signal(descriptor, sig)
        finally:
            os.close(descriptor)
    else:
        raise RuntimeError("safe takeover requires Linux Python pidfd signal support")
    return True


def assert_no_previous_submission(directory: Path | None) -> None:
    if directory is None:
        return
    if not directory.is_dir():
        raise RuntimeError(f"previous submission directory is missing: {directory}")
    for name in ("foreground-submit.started", "submission-attempt.started", "snapshot-switch-submit.started"):
        if (directory / name).exists():
            raise RuntimeError("previous workflow already attempted GPU submission; reconcile its job first")
    old_log = directory / "prepare-and-submit.log"
    if old_log.exists():
        with old_log.open(encoding="utf-8", errors="replace") as handle:
            if any("[STAGE 3/3]" in line or "[SUBMIT FINISHED]" in line for line in handle):
                raise RuntimeError("previous workflow already attempted GPU submission; reconcile its job first")


def require_takeover_platform() -> None:
    if os.name != "posix" or not Path("/proc").is_dir():
        raise RuntimeError("takeover must run on the same Linux development host as the old builder")
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise RuntimeError("safe takeover requires Linux Python pidfd signal support")


def hold_legacy_submitter(builder_pid: int, source: Path, previous: Path) -> tuple[dict | None, dict | None]:
    require_takeover_platform()
    assert_no_previous_submission(previous)
    builder = read_process(builder_pid)
    if not builder or builder["state"] in {"Z", "X"}:
        if (source / "_SUCCESS.json").is_file():
            return None, None
        raise RuntimeError("specified builder has exited but the source has no _SUCCESS.json")
    argv = builder["argv"]
    if not any(Path(value).name == "build_ui5_curriculum_recipe.py" for value in argv):
        raise RuntimeError("specified PID is not the curriculum builder; refusing takeover")
    if "--output-dir" not in argv or argv.index("--output-dir") + 1 >= len(argv):
        raise RuntimeError("builder lacks an explicit output directory; refusing takeover")
    if Path(argv[argv.index("--output-dir") + 1]).resolve() != source:
        raise RuntimeError("builder writes a different curriculum directory; refusing takeover")
    parent = read_process(builder["ppid"])
    if (not parent or parent["pid"] <= 1 or not parent["argv"]
            or not Path(parent["argv"][0]).name.startswith("python")
            or parent["argv"][1:] != ["-u", "-"]):
        raise RuntimeError("builder parent is not the known foreground Python submitter; refusing to signal it")
    # Both identities and their relationship must remain stable before the hold.
    latest_builder = read_process(builder_pid)
    if (not process_active(builder) or latest_builder is None
            or latest_builder["ppid"] != parent["pid"]):
        raise RuntimeError("builder changed while inspecting takeover")
    if not signal_process(parent, signal.SIGSTOP):
        raise RuntimeError("old submitter exited before it could be held")
    log(f"[OLD SUBMIT HELD] parent_pid={parent['pid']} builder_pid={builder_pid}; builder continues")
    return builder, parent


def retire_held_submitter(parent: dict | None) -> None:
    if parent is None or not process_active(parent):
        return
    signal_process(parent, signal.SIGTERM)
    signal_process(parent, signal.SIGCONT)
    for _ in range(50):
        if not process_active(parent):
            log(f"[OLD SUBMIT RETIRED] parent_pid={parent['pid']}; no hour009 submission")
            return
        time.sleep(0.2)
    raise RuntimeError("old submitter has not exited; refusing to prepare another submission")


def wait_for_source(source: Path, builder: dict | None, parent: dict | None) -> None:
    while True:
        live = builder is not None and process_active(builder)
        complete = (source / "_SUCCESS.json").is_file()
        if not live:
            if not complete:
                raise RuntimeError("source build is incomplete; refusing to regenerate its PNGs")
            retire_held_submitter(parent)
            return
        latest = source / "progress" / "build_progress.json"
        state = json.loads(latest.read_text()) if latest.is_file() else {}
        log(
            f"[WAIT SOURCE] builder_pid={builder['pid']} stage={state.get('stage', 'unknown')} "
            f"completed={state.get('completed')}/{state.get('total')} percent={state.get('percent')} "
            f"stage_eta_seconds={state.get('eta_seconds')} old_submit=held"
        )
        time.sleep(10)


def render_job(template: dict, env: dict[str, str], name: str) -> dict:
    job = copy.deepcopy(template)
    roles = job["jobDefVersion"]["resource"]["arnoldConfig"]["roles"]
    if (len(roles) != 1 or roles[0].get("num") != 1
            or roles[0].get("gpu") != 2 or roles[0].get("gpuv") != "NVIDIA_H20"):
        raise RuntimeError("submission template is not one H20x2 worker")
    job["caption"] = "UI5 Crop Rollout4 Curriculum - reused PNGs - " + name
    job["jobDefVersion"]["name"] = name
    job["jobDefVersion"]["gitRepo"] = {"mnt": env["PROJECT_ROOT"]}
    job["jobRunParams"] = {
        "entrypointFullScript": (
            "set -Eeuo pipefail\n"
            f"cd {shlex.quote(env['PROJECT_ROOT'])}\n"
            f"test \"$(git rev-parse HEAD)\" = {shlex.quote(env['CODE_REVISION'])}\n"
            "exec bash shell/run_locany_ui5_crop_rollout4_curriculum_h20x2.sh\n"
        ),
        "envsList": {**FORMAL_ENV, **env},
    }
    return job


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--reuse-crops-from", type=Path, required=True)
    parser.add_argument("--previous-submission-dir", type=Path)
    parser.add_argument("--take-over-builder-pid", type=int)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--mlx-bin", default="mlx")
    return parser.parse_args(argv)


def prepare(args) -> Path:
    snapshot = args.snapshot.resolve(strict=True)
    source = args.reuse_crops_from.resolve(strict=True)
    workspace = args.workspace.resolve(strict=True)
    previous = args.previous_submission_dir.resolve(strict=True) if args.previous_submission_dir else None
    if not re.fullmatch(r"hour_\d{3}_\d{8}T\d{6}Z", snapshot.name):
        raise ValueError("--snapshot must explicitly name an hourly snapshot, not a mutable latest alias")
    if not (snapshot / "_SUCCESS").is_file() or not (snapshot / "manifest.json").is_file():
        raise RuntimeError("snapshot is not atomically published")
    if args.take_over_builder_pid is not None and previous is None:
        raise ValueError("takeover requires --previous-submission-dir to guard against duplicate jobs")
    mlx = shutil.which(args.mlx_bin)
    if not mlx:
        raise RuntimeError("mlx is missing; run on the authenticated CPU development host")
    assert_no_previous_submission(previous)
    hour = snapshot.name.split("_")[1]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    run_name = f"locany-ui5-crop-rollout4-curriculum-hour{hour}-h20x2-sdpa7268-{stamp}"
    submission = workspace / "gui_logs" / "ui5_curriculum" / run_name
    submission.mkdir(parents=True, exist_ok=False)
    state_path = submission / "snapshot-switch.json"
    state = {"status": "preparing", "source": str(source), "snapshot": str(snapshot)}
    parent = None
    try:
        builder = None
        if args.take_over_builder_pid is not None:
            builder, parent = hold_legacy_submitter(args.take_over_builder_pid, source, previous)
            state.update({"held_submitter": parent, "source_builder": builder})
            write_state(state_path, state)
            assert_no_previous_submission(previous)
        log(f"[RUN] name={run_name} state={state_path}")
        wait_for_source(source, builder, parent)
        assert_no_previous_submission(previous)
        frozen = snapshot.parent.parent / "frozen" / f"{snapshot.name}-curriculum-{stamp}"
        data_dir = workspace / "gui_data" / "ui5_curriculum" / f"hour{hour}-s42-reuse-{stamp}"
        output = workspace / "gui_models" / "Embodied-ui5-det-crop" / run_name
        env_dir = workspace / "conda_envs" / "LocateAnything"
        model = workspace / "gui_models/Embodied-ui5-det-crop/locany-ui5-v5-croponly-sourcebalanced-a800x4-20260830/checkpoint-12000"
        hf_snapshot = "hub/models--nvidia--LocateAnything-3B/snapshots/c32291ca5e996f5a7a485845b4f57a233936bba0"
        processor = next((p for p in (workspace / "hf_home" / hf_snapshot,
                                     workspace / "cache/huggingface" / hf_snapshot) if p.is_dir()), None)
        eval_manifest = workspace / "gui_models/ui5_eval_detector_cache_horizontal_v5/detector_scan_crops.h20.jsonl"
        if not model.is_dir() or processor is None or not eval_manifest.is_file():
            raise RuntimeError("model, processor or relocated H20 evaluation manifest is missing")
        env = {
            "WORKSPACE": str(workspace), "PROJECT_ROOT": str(PROJECT_ROOT),
            "ENV_DIR": str(env_dir), "PYTHON_BIN": str(Path(sys.executable).absolute()),
            "FROZEN_SELECTION": str(frozen), "CURRICULUM_DATA_DIR": str(data_dir),
            "CURRICULUM_REUSE_CROPS_FROM": str(source),
            "ROLLOUT_BUNDLE_ROOT": str(workspace / "gui_data/ui5_train_rollout_bundle_v1"),
            "EVAL_INPUT_DIR": str(workspace / "data"), "EVAL_DETECTOR_MANIFEST": str(eval_manifest),
            "MODEL_PATH": str(model), "PROCESSOR_PATH": str(processor),
            "RUN_NAME": run_name, "OUTPUT_DIR": str(output),
            "CODE_REVISION": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
        }
        template = yaml.safe_load((PROJECT_ROOT / "jobs/ui5_train_rollouts_h20x2_merlin.yaml").read_text())
        job = render_job(template, env, f"ui5-curriculum-hour{hour}-{stamp}")
        job_path = submission / "formal.yaml"
        with job_path.open("x", encoding="utf-8") as handle:
            yaml.safe_dump(job, handle, sort_keys=False)
        cpu_env = {**os.environ, **env, "CUDA_VISIBLE_DEVICES": "", "PYTHONUNBUFFERED": "1"}

        def run(script: str, *parameters: str):
            subprocess.run([sys.executable, "-u", str(PROJECT_ROOT / "scripts" / script), *parameters],
                           cwd=PROJECT_ROOT, env=cpu_env, check=True)

        log(f"[1/4 FREEZE] {snapshot} -> {frozen}")
        run("merge_ui5_rollout_selections.py", "--input", str(snapshot), "--output-dir", str(frozen))
        run("ui5_frozen_selection.py", "--frozen-selection", str(frozen))
        log("[2/4 EVAL INPUT] validating evaluation bytes and geometry on CPU")
        run("relocate_ui5_eval_detector_manifest.py", "--manifest", str(eval_manifest),
            "--input-dir", env["EVAL_INPUT_DIR"])
        log(f"[3/4 REUSE] source={source} target={data_dir}; strict all-PNG reuse, zero recropping")
        run("build_ui5_curriculum_recipe.py", "--rollout-difficulty", str(frozen / "complete8.jsonl"),
            "--rollout-bundle-root", env["ROLLOUT_BUNDLE_ROOT"], "--output-dir", str(data_dir),
            "--reuse-crops-from", str(source), "--seed", "42", "--progress-interval-seconds", "10")
        manifest = json.loads((data_dir / "curriculum_manifest.json").read_text())
        success = json.loads((data_dir / "_SUCCESS.json").read_text())
        audit = manifest.get("crop_asset_reuse", {})
        if (success.get("complete") is not True or success.get("identity_digest") != manifest.get("identity_digest")
                or audit.get("generated_crop_assets") != 0
                or audit.get("reused_crop_assets") != len(manifest["crop_assets"])):
            raise RuntimeError("new curriculum was not completely published with all crops reused")
        assert_no_previous_submission(previous)
        if previous is not None:
            # This shared, exclusive marker also blocks a second invocation
            # with a fresh RUN_NAME from submitting the same switch twice.
            with (previous / "snapshot-switch-submit.started").open("x", encoding="utf-8") as handle:
                handle.write(str(state_path) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        marker = submission / "submission-attempt.started"
        marker.touch(exist_ok=False)
        state.update({"status": "submission_attempted", "job_yaml": str(job_path), "runtime": env,
                      "reused_crop_assets": audit["reused_crop_assets"], "generated_crop_assets": 0})
        write_state(state_path, state)
        log(f"[4/4 SUBMIT] frozen={frozen} hard_groups={manifest['hard_groups']} "
            f"reused={audit['reused_crop_assets']} generated=0")
        subprocess.run([mlx, "job", "submitv2", "--path", str(job_path)], cwd=PROJECT_ROOT, check=True)
        state["status"] = "submitted"
        write_state(state_path, state)
        log(f"[SUBMITTED] output={output} state={state_path}")
        return job_path
    except BaseException as exc:
        state.update({"status": "failed_or_interrupted", "error": f"{type(exc).__name__}: {exc}"})
        write_state(state_path, state)
        if parent and process_active(parent):
            log(f"[HOLD RETAINED] old submitter PID={parent['pid']} remains held; source builder was NOT stopped")
        log(f"[STOPPED] no automatic retry; inspect {state_path}")
        raise


def main(argv=None) -> int:
    try:
        prepare(parse_args(argv))
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"[SNAPSHOT SWITCH ERROR] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
