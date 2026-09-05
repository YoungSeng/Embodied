#!/usr/bin/env python3
"""Foreground: finish existing crops -> freeze a new snapshot -> reuse -> submit H20x2.

An explicitly selected legacy foreground submitter can be held with SIGSTOP;
its crop-building child is NEVER signalled. The submitter is terminated only
after that child exits. Failures retain the hold and never submit or recrop.
"""
from __future__ import annotations

import argparse
import copy
import ctypes
import hashlib
import json
import os
import platform
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
CAPTION_REJECTION = "JobRunCaptionExceedMaxLen"
MAX_CAPTION_LENGTH = 90


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


def open_process_descriptor(pid: int) -> int:
    # An open /proc/PID directory is a supported pidfd_send_signal handle.
    # It pins the process identity even if its numeric PID is later recycled.
    # https://man7.org/linux/man-pages/man2/pidfd_send_signal.2.html
    return os.open(f"/proc/{pid}", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)


def descriptor_matches_identity(descriptor: int, identity: dict[str, Any]) -> bool:
    # Inspect the pinned process, not /proc/PID again after opening the handle.
    stat_fd = os.open("stat", os.O_RDONLY | os.O_CLOEXEC, dir_fd=descriptor)
    with os.fdopen(stat_fd) as handle:
        stat = handle.read()
    fields = stat[stat.rfind(")") + 2:].split()
    return fields[19] == identity["start_ticks"] and fields[0] not in {"Z", "X"}


def send_process_descriptor(descriptor: int, sig: int) -> str:
    native = getattr(signal, "pidfd_send_signal", None)
    if callable(native):
        native(descriptor, sig)
        return "python_pidfd_send_signal"
    # Older Python builds can lack the wrapper despite a supporting kernel.
    # Invoke the same syscall, never a numeric-PID os.kill fallback. Syscall 424
    # is shared by Linux x86-64 and the asm-generic aarch64 ABI only here.
    if (sys.platform != "linux" or ctypes.sizeof(ctypes.c_void_p) != 8
            or platform.machine().lower() not in {"x86_64", "amd64", "aarch64", "arm64"}):
        raise RuntimeError("pidfd syscall compatibility requires 64-bit Linux x86_64/aarch64")
    libc = ctypes.CDLL(None, use_errno=True)
    syscall = libc.syscall
    syscall.restype = ctypes.c_long
    result = syscall(ctypes.c_long(424), ctypes.c_int(descriptor), ctypes.c_int(sig),
                     ctypes.c_void_p(None), ctypes.c_uint(0))
    if result < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return "libc_syscall_pidfd_send_signal"


def signal_process(identity: dict[str, Any], sig: int) -> bool:
    if not process_active(identity):
        return False
    try:
        descriptor = open_process_descriptor(identity["pid"])
    except (FileNotFoundError, ProcessLookupError):
        return False
    try:
        if not descriptor_matches_identity(descriptor, identity):
            return False
        send_process_descriptor(descriptor, sig)
        return True
    except (FileNotFoundError, ProcessLookupError):
        return False
    finally:
        os.close(descriptor)


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
    if sys.platform != "linux" or not Path("/proc").is_dir():
        raise RuntimeError("takeover must run on the same Linux development host as the old builder")
    # Signal 0 checks kernel/permission support without delivering a signal.
    # Probe ourselves BEFORE holding any old workflow; never weaken safety if
    # seccomp, permissions or the kernel reject process-descriptor signalling.
    descriptor = open_process_descriptor(os.getpid())
    try:
        backend = send_process_descriptor(descriptor, 0)
    except OSError as exc:
        raise RuntimeError(
            f"safe takeover signal-0 check failed: {exc}; Linux pidfd_send_signal "
            "must be supported and permitted; no old process was stopped"
        ) from exc
    finally:
        os.close(descriptor)
    log(f"[TAKEOVER SIGNAL] backend={backend} handle=proc_directory probe=signal_0_pass")


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
    job["caption"] = "UI5 H20x2 " + name
    if len(job["caption"].encode("utf-8")) > MAX_CAPTION_LENGTH:
        raise ValueError(f"job caption exceeds {MAX_CAPTION_LENGTH} bytes")
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


def submission_result(returncode: int, output: str) -> dict[str, Any]:
    """MLX can exit 0 on API errors. Absence of errors is not a receipt."""
    clean = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", output)
    errors, job_ids = [], []

    def inspect(value):
        if isinstance(value, dict):
            for key, item in value.items():
                normalized = key.replace("_", "").lower()
                if normalized in {"errcode", "errorcode"} and item not in (None, "", 0, "0"):
                    errors.append(str(item))
                if normalized in {"code", "statuscode"} and isinstance(item, (str, int)):
                    if str(item).isdigit() and int(item) not in (0, 200):
                        errors.append(f"{key}={item}")
                if normalized == "error" and item not in (None, "", 0, False, [], {}):
                    errors.append(str(item))
                if normalized in {"jobrunid", "jobid"} and isinstance(item, (str, int)) and not isinstance(item, bool):
                    if str(item).strip() not in {"", "0"}:
                        job_ids.append(str(item).strip())
                if normalized == "success" and item is False:
                    errors.append("success=false")
                inspect(item)
        elif isinstance(value, list):
            for item in value:
                inspect(item)

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", clean):
        try:
            value, _ = decoder.raw_decode(clean[match.start():])
        except ValueError:
            continue
        inspect(value)
    if CAPTION_REJECTION in clean:
        errors.append(CAPTION_REJECTION)
    # Accept labelled job receipts as well as structured JSON. A bare URL or
    # an exit code alone cannot prove that the server accepted a job.
    job_ids.extend(re.findall(r"\bjob[_ ]?(?:run[_ ]?)?id\s*[:=]\s*([A-Za-z0-9_-]+)", clean, re.I))
    job_ids = [value for value in job_ids if value.lower() not in {"0", "-1", "none", "null", "false"}]
    failed_text = bool(re.search(
        r"提交任务失败|任务提交失败|提交失败|提交不成功|failed\s+to\s+submit|submit[^\n]*\bfailed\b|"
        r"\b(?:not|never)\s+(?:successfully\s+)?submitted\b|"
        r"Traceback \(most recent call last\)|(?:^|\n)[^\n{]*\b(?:ERROR|FATAL)\b", clean, re.I,
    ))
    accepted_text = bool(re.search(
        r"提交任务成功|任务提交成功|提交成功|\bsubmitted\s+successfully\b|"
        r"\bsuccessfully\s+submitted\b|\bsubmit(?:ted)?(?:\s+\w+){0,3}\s+success(?:ful(?:ly)?)?\b",
        clean, re.I,
    ))
    if errors or failed_text or returncode != 0:
        status = "submission_rejected" if CAPTION_REJECTION in errors and not job_ids else "submission_failed"
    elif job_ids or accepted_text:
        status = "submitted"
    else:
        status = "submission_unconfirmed"
    return {"status": status, "returncode": returncode, "error_codes": sorted(set(errors)),
            "job_ids": sorted(set(job_ids)), "explicit_success_message": accepted_text}


def submit_job(mlx: str, job_path: Path, state_path: Path, state: dict) -> None:
    """One attempt, with durable CLI output and positive acknowledgement checks."""
    job = yaml.safe_load(job_path.read_text(encoding="utf-8"))
    if not job.get("caption") or len(job["caption"].encode("utf-8")) > MAX_CAPTION_LENGTH:
        raise ValueError("refusing to submit an empty/overlength job caption")
    marker = state_path.parent / "submission-attempt.started"
    marker.touch(exist_ok=False)
    transcript = state_path.parent / "mlx-submit.log"
    state.update({"status": "submission_attempted", "job_yaml": str(job_path),
                  "submission_log": str(transcript)})
    write_state(state_path, state)
    log(f"[SUBMIT START] caption_length={len(job['caption'])} log={transcript}; one attempt only")
    try:
        # The file retains the CLI output even if this Python process is
        # interrupted; do not trust MLX's exit status as an API success flag.
        with transcript.open("x", encoding="utf-8") as handle:
            completed = subprocess.run([mlx, "job", "submitv2", "--path", str(job_path)],
                                       cwd=PROJECT_ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
            handle.flush()
            os.fsync(handle.fileno())
        output = transcript.read_text(encoding="utf-8", errors="replace")
        if output:
            log(output.rstrip())
        result = submission_result(completed.returncode, output)
        result["log_sha256"] = hashlib.sha256(transcript.read_bytes()).hexdigest()
        state.update({"status": result["status"], "submission_result": result})
        write_state(state_path.parent / "submission-result.json", result)
        write_state(state_path, state)
    except BaseException as exc:
        state.update({"status": "submission_unconfirmed", "error": f"{type(exc).__name__}: {exc}"})
        write_state(state_path, state)
        raise
    if result["status"] != "submitted":
        log(f"[SUBMIT NOT CONFIRMED] status={result['status']} errors={result['error_codes']} "
            f"log={transcript}; do not automatically retry")
        raise RuntimeError(f"MLX {result['status']}; inspect {transcript} and the platform before retrying")
    log(f"[SUBMITTED] job_ids={result['job_ids']} output={state['runtime']['OUTPUT_DIR']} state={state_path}")


def verify_prepared_curriculum(env: dict[str, str]) -> dict:
    """Metadata-only retry check. Do not walk, link, decode or rebuild PNGs."""
    data_dir = Path(env["CURRICULUM_DATA_DIR"]).resolve(strict=True)
    manifest = json.loads((data_dir / "curriculum_manifest.json").read_text(encoding="utf-8"))
    success = json.loads((data_dir / "_SUCCESS.json").read_text(encoding="utf-8"))
    payload = dict(manifest)
    identity = payload.pop("identity_digest", None)
    digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                      separators=(",", ":")).encode("utf-8")).hexdigest()
    audit = manifest.get("crop_asset_reuse", {})
    if (not identity or identity != digest or success.get("identity_digest") != identity
            or success.get("complete") is not True or not manifest.get("crop_assets")
            or audit.get("generated_crop_assets") != 0
            or audit.get("reused_crop_assets") != len(manifest["crop_assets"])):
        raise RuntimeError("prepared curriculum identity/publication/reuse check failed; refusing rebuild")
    frozen = Path(env["FROZEN_SELECTION"]).resolve(strict=True)
    frozen_state = manifest["inputs"]["frozen_selection_summary"]
    summary_path = frozen / "summary.json"
    if (Path(frozen_state["path"]).resolve() != summary_path
            or hashlib.sha256(summary_path.read_bytes()).hexdigest() != frozen_state["sha256"]):
        raise RuntimeError("prepared curriculum is not bound to the selected frozen summary")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("formal_crop_hard_groups") != manifest.get("hard_groups"):
        raise RuntimeError("prepared hard-group count differs from frozen summary")
    if not (frozen / "_SUCCESS").is_file():
        raise RuntimeError("frozen selection is not published")
    return manifest


def retry_caption_rejected(args) -> Path:
    """Explicit recovery for the known 91-character legacy caption rejection."""
    old_state_path = args.retry_caption_rejected_state.resolve(strict=True)
    old_state = json.loads(old_state_path.read_text(encoding="utf-8"))
    old_job_path = Path(old_state["job_yaml"]).resolve(strict=True)
    if old_job_path.parent != old_state_path.parent:
        raise RuntimeError("old job YAML is outside its submission directory")
    old_job = yaml.safe_load(old_job_path.read_text(encoding="utf-8"))
    receipt = old_state.get("submission_result")
    receipt_path = old_state_path.parent / "submission-result.json"
    if receipt_path.is_file():
        saved_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt is not None and receipt != saved_receipt:
            raise RuntimeError("saved submission receipts disagree; reconcile the platform job first")
        receipt = saved_receipt
    transcript = old_state_path.parent / "mlx-submit.log"
    if receipt is None and transcript.is_file():
        receipt = submission_result(0, transcript.read_text(encoding="utf-8", errors="replace"))
    if receipt is not None and (receipt.get("status") != "submission_rejected"
                               or CAPTION_REJECTION not in receipt.get("error_codes", [])
                               or receipt.get("job_ids")):
        raise RuntimeError("previous submission is successful/uncertain, not a confirmed caption rejection")
    if len(str(old_job.get("caption", "")).encode("utf-8")) <= MAX_CAPTION_LENGTH:
        raise RuntimeError("old caption is not overlength; refusing caption-rejection retry")
    if receipt is None and old_state.get("status") != "submitted":
        raise RuntimeError("not the legacy false-success case; reconcile the platform job first")
    # The flag is the user's explicit confirmation of the reported API error
    # when the old version did not retain its CLI output. Preserve old evidence.
    env = dict(old_job["jobRunParams"]["envsList"])
    if any(env.get(key) != value for key, value in old_state["runtime"].items()):
        raise RuntimeError("old YAML runtime differs from its saved submission state")
    if any(env.get(key) != value for key, value in FORMAL_ENV.items()):
        raise RuntimeError("old job does not match the formal H20x2 curriculum profile")
    if Path(env["PROJECT_ROOT"]).resolve() != PROJECT_ROOT.resolve():
        raise RuntimeError("run retry from the same project checkout as the prepared job")
    old_output = Path(env["OUTPUT_DIR"])
    if old_output.exists() and (not old_output.is_dir() or next(old_output.iterdir(), None) is not None):
        raise RuntimeError("old output is nonempty; a GPU job may have started, reconcile it first")
    lock = old_state_path.parent / "caption-retry.started"
    if lock.exists():
        raise RuntimeError(f"a caption retry was already reserved; inspect {lock}, do not repeat")
    mlx = shutil.which(args.mlx_bin)
    if not mlx:
        raise RuntimeError("mlx is missing; run on the authenticated CPU development host")
    log("[RETRY CHECK] validating published curriculum metadata only; no freeze/build/PNG scan")
    manifest = verify_prepared_curriculum(env)
    snapshot_name = Path(old_state["snapshot"]).name
    if not re.fullmatch(r"hour_\d{3}_\d{8}T\d{6}Z", snapshot_name):
        raise RuntimeError("saved snapshot identity is invalid")
    hour = snapshot_name.split("_")[1]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    name = f"locany-ui5-crop-rollout4-curriculum-hour{hour}-h20x2-sdpa7268-{stamp}"
    submission = old_state_path.parent.parent / name
    submission.mkdir(exist_ok=False)
    state_path = submission / "snapshot-switch.json"
    env.update({"RUN_NAME": name, "OUTPUT_DIR": str(old_output.parent / name),
                "CODE_REVISION": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                         cwd=PROJECT_ROOT, text=True).strip()})
    job = render_job(old_job, env, f"ui5-curriculum-hour{hour}-{stamp}")
    job_path = submission / "formal.yaml"
    with job_path.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(job, handle, sort_keys=False)
    state = {"status": "prepared_submission_retry", "retry_of": str(old_state_path),
             "confirmed_rejection": CAPTION_REJECTION, "legacy_user_confirmation": receipt is None,
             "snapshot": old_state["snapshot"], "source": old_state.get("source"),
             "runtime": env, "job_yaml": str(job_path), "curriculum_identity": manifest["identity_digest"],
             "reused_crop_assets": len(manifest["crop_assets"]), "generated_crop_assets": 0}
    write_state(state_path, state)
    # Keep all old attempt markers; one exclusive additional marker guards
    # concurrent/manual retries, including cases with an uncertain CLI result.
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(str(state_path) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    log(f"[RETRY SUBMIT ONLY] hard_groups={manifest['hard_groups']} "
        f"reused={len(manifest['crop_assets'])} generated=0 curriculum={env['CURRICULUM_DATA_DIR']}")
    submit_job(mlx, job_path, state_path, state)
    return job_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--reuse-crops-from", type=Path)
    parser.add_argument("--retry-caption-rejected-state", type=Path,
                        help="submit prepared data only; explicitly confirms the old job was rejected "
                             "with JobRunCaptionExceedMaxLen, never use for an uncertain/successful job")
    parser.add_argument("--previous-submission-dir", type=Path)
    parser.add_argument("--take-over-builder-pid", type=int)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--mlx-bin", default="mlx")
    args = parser.parse_args(argv)
    if args.retry_caption_rejected_state:
        if args.snapshot or args.reuse_crops_from or args.previous_submission_dir or args.take_over_builder_pid:
            parser.error("caption-rejection retry cannot be combined with snapshot preparation/takeover")
    elif not args.snapshot or not args.reuse_crops_from:
        parser.error("preparation requires --snapshot and --reuse-crops-from")
    return args


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
        state.update({"status": "prepared", "job_yaml": str(job_path), "runtime": env,
                      "reused_crop_assets": audit["reused_crop_assets"], "generated_crop_assets": 0})
        write_state(state_path, state)
        log(f"[4/4 SUBMIT] frozen={frozen} hard_groups={manifest['hard_groups']} "
            f"reused={audit['reused_crop_assets']} generated=0")
        submit_job(mlx, job_path, state_path, state)
        return job_path
    except BaseException as exc:
        if not state.get("status", "").startswith("submission_"):
            state["status"] = "failed_or_interrupted"
        state["error"] = f"{type(exc).__name__}: {exc}"
        write_state(state_path, state)
        if parent and process_active(parent):
            log(f"[HOLD RETAINED] old submitter PID={parent['pid']} remains held; source builder was NOT stopped")
        log(f"[STOPPED] no automatic retry; inspect {state_path}")
        raise


def main(argv=None) -> int:
    try:
        args = parse_args(argv)
        if args.retry_caption_rejected_state:
            retry_caption_rejected(args)
        else:
            prepare(args)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"[SNAPSHOT SWITCH ERROR] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
