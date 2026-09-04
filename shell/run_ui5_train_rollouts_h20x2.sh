#!/usr/bin/env bash
set -Eeuo pipefail

WORKSPACE=${WORKSPACE:-/mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace}
ORCHESTRATOR_REPO=${ORCHESTRATOR_REPO:-${WORKSPACE}/code/Eagle/Embodied-rollout8-h20x2-v6}
M31_REPO=${M31_REPO:-${WORKSPACE}/code/Eagle/Embodied-ui5-rollout8-m31}
CROP_REPO=${CROP_REPO:-${ORCHESTRATOR_REPO}}
BUNDLE_ROOT=${BUNDLE_ROOT:-${WORKSPACE}/gui_data/ui5_train_rollout_bundle_v1}
M31_CHECKPOINT=${M31_CHECKPOINT:-${WORKSPACE}/gui_models/Embodied/locany-ui5-m31-taskmoe-setdecoder-a800x4-sft-20260830-r2/checkpoint-12000}
CROP_CHECKPOINT=${CROP_CHECKPOINT:-${WORKSPACE}/gui_models/Embodied-ui5-det-crop/locany-ui5-v5-croponly-sourcebalanced-a800x4-20260830/checkpoint-12000}
OUTPUT_ROOT=${OUTPUT_ROOT:-${WORKSPACE}/gui_rollouts/ui5-train-rollout8-h20x2-v6-20260904}
ENV_DIR=${ENV_DIR:-${WORKSPACE}/conda_envs/LocateAnything}
PYTHON_BIN=${PYTHON_BIN:-${ENV_DIR}/bin/python}
SEEDS=(20260903 20260917 20260931 20260947)

PROCESSOR_CANDIDATES=(
  "${WORKSPACE}/hf_home/hub/models--nvidia--LocateAnything-3B/snapshots/c32291ca5e996f5a7a485845b4f57a233936bba0"
  "${WORKSPACE}/cache/huggingface/hub/models--nvidia--LocateAnything-3B/snapshots/c32291ca5e996f5a7a485845b4f57a233936bba0"
)
PROCESSOR_PATH=""
for candidate in "${PROCESSOR_CANDIDATES[@]}"; do
  if [[ -d "${candidate}" ]]; then
    PROCESSOR_PATH=${candidate}
    break
  fi
done
if [[ -z "${PROCESSOR_PATH}" ]]; then
  echo "ERROR: neither fixed processor/tokenizer snapshot exists" >&2
  exit 20
fi

test -x "${PYTHON_BIN}" || { echo "ERROR: Python missing: ${PYTHON_BIN}" >&2; exit 21; }
test -d "${BUNDLE_ROOT}" || { echo "ERROR: bundle missing: ${BUNDLE_ROOT}" >&2; exit 22; }
test -d "${M31_CHECKPOINT}" || { echo "ERROR: M31 checkpoint missing" >&2; exit 23; }
test -d "${CROP_CHECKPOINT}" || { echo "ERROR: crop checkpoint missing" >&2; exit 24; }
test -f "${M31_REPO}/scripts/inference_ui_defect_locany.py" || { echo "ERROR: M31 inference entrypoint missing" >&2; exit 25; }
test -f "${CROP_REPO}/scripts/inference_ui_defect_locany.py" || { echo "ERROR: crop inference entrypoint missing" >&2; exit 26; }
test -f "${ORCHESTRATOR_REPO}/scripts/run_ui5_train_rollout_worker.py" || { echo "ERROR: v6 rollout worker missing" >&2; exit 27; }
test -f "${ORCHESTRATOR_REPO}/scripts/aggregate_ui5_train_rollouts.py" || { echo "ERROR: v6 aggregator missing" >&2; exit 28; }
test -f "${ORCHESTRATOR_REPO}/scripts/snapshot_ui5_train_rollouts.py" || { echo "ERROR: v6 snapshotter missing" >&2; exit 29; }

export PATH="${ENV_DIR}/bin:${PATH}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=2
mkdir -p \
  "${OUTPUT_ROOT}/logs" \
  "${OUTPUT_ROOT}/diagnostics" \
  "${OUTPUT_ROOT}/runtime_cache/hf_modules" \
  "${OUTPUT_ROOT}/runtime_cache/pycache"
# A resumed run keeps raw/progress data, but its formal-validity markers must be
# earned again by this launch's eight PIDs.  Logs below are also truncated per
# physical worker before Python starts, so an old MODEL_LOAD_OK cannot match.
rm -f -- \
  "${OUTPUT_ROOT}/diagnostics/_MODEL_LOADS_OK" \
  "${OUTPUT_ROOT}/diagnostics/formal_run_valid.json" \
  "${OUTPUT_ROOT}/diagnostics/formal_run_invalid.json"
CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" "${ORCHESTRATOR_REPO}/scripts/preflight_ui5_train_rollouts.py" \
  --bundle-root "${BUNDLE_ROOT}" \
  --diagnostics-dir "${OUTPUT_ROOT}/diagnostics" \
  --m31-checkpoint "${M31_CHECKPOINT}" \
  --crop-checkpoint "${CROP_CHECKPOINT}" \
  --processor-candidate "${PROCESSOR_CANDIDATES[0]}" \
  --processor-candidate "${PROCESSOR_CANDIDATES[1]}" \
  --m31-repo "${M31_REPO}" \
  --crop-repo "${CROP_REPO}" \
  --output-root "${OUTPUT_ROOT}" \
  --require-runtime

"${PYTHON_BIN}" -c 'import openpyxl, scipy, PIL; assert tuple(map(int, openpyxl.__version__.split(".")[:2])) >= (3, 1)'
nvidia-smi >"${OUTPUT_ROOT}/diagnostics/nvidia_smi_before_model_load.txt"
cat "${OUTPUT_ROOT}/diagnostics/nvidia_smi_before_model_load.txt"

RUN_STARTED_EPOCH=$(date +%s)
NEXT_SNAPSHOT_HOUR=3
SNAPSHOT_INTERVAL_SECONDS=10800
NEXT_SNAPSHOT_ELAPSED_SECONDS=${SNAPSHOT_INTERVAL_SECONDS}
snapshot_failures=0
echo "[ROLLOUT_START] epoch=${RUN_STARTED_EPOCH} first_snapshot_hour=${NEXT_SNAPSHOT_HOUR} interval_seconds=${SNAPSHOT_INTERVAL_SECONDS}"

PIDS=()
NAMES=()
GPUS=()
MODELS=()
ROLLOUT_IDS=()
LOG_PATHS=()
launch_worker() {
  local model_id=$1
  local gpu=$2
  local rollout_ids=$3
  local seeds=$4
  local repo checkpoint log_path hf_modules_cache python_pycache
  if [[ "${model_id}" == "m31" ]]; then
    repo=${M31_REPO}
    checkpoint=${M31_CHECKPOINT}
  else
    repo=${CROP_REPO}
    checkpoint=${CROP_CHECKPOINT}
  fi
  log_path=${OUTPUT_ROOT}/logs/${model_id}_rollout_${rollout_ids}.log
  local worker_key=${model_id}_rollout_${rollout_ids}
  hf_modules_cache=${OUTPUT_ROOT}/runtime_cache/hf_modules/${worker_key}
  python_pycache=${OUTPUT_ROOT}/runtime_cache/pycache/${worker_key}
  mkdir -p "${hf_modules_cache}" "${python_pycache}"
  (
    cd "${ORCHESTRATOR_REPO}"
    export PYTHONPATH="${repo}${PYTHONPATH:+:${PYTHONPATH}}"
    exec env \
      CUDA_VISIBLE_DEVICES="${gpu}" \
      HF_MODULES_CACHE="${hf_modules_cache}" \
      PYTHONPYCACHEPREFIX="${python_pycache}" \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "${PYTHON_BIN}" \
      "${ORCHESTRATOR_REPO}/scripts/run_ui5_train_rollout_worker.py" \
      --mode run \
      --model-id "${model_id}" \
      --checkpoint "${checkpoint}" \
      --processor-path "${PROCESSOR_PATH}" \
      --bundle-root "${BUNDLE_ROOT}" \
      --output-root "${OUTPUT_ROOT}" \
      --repo-root "${repo}" \
      --rollout-ids "${rollout_ids}" \
      --seeds "${seeds}" \
      --physical-gpu "${gpu}" \
      --gpu-model-processes 4 \
      --dtype bf16 \
      --attn-implementation sdpa \
      --vision-attn-implementation flash_attention_2 \
      --generation-mode hybrid \
      --max-seq-length 7268 \
      --max-num-tokens-per-sample 7268 \
      --training-max-num-tokens 12800 \
      --processor-in-token-limit 25600 \
      --max-new-tokens 512 \
      --n-future-tokens 6 \
      --temperature 0.7 \
      --top-p 0.9 \
      --top-k 0 \
      --repetition-penalty 1.1
  ) >"${log_path}" 2>&1 &
  PIDS+=("$!")
  NAMES+=("${model_id}/rollout_${rollout_ids}/gpu_${gpu}")
  GPUS+=("${gpu}")
  MODELS+=("${model_id}")
  ROLLOUT_IDS+=("${rollout_ids}")
  LOG_PATHS+=("${log_path}")
  echo "[WORKER_LAUNCH] model=${model_id} gpu=${gpu} pid=$! rollout_ids=${rollout_ids} seed=${seeds} log=${log_path} hf_modules_cache=${hf_modules_cache} python_pycache=${python_pycache}"
}

echo "[FORMAL_ARCHITECTURE] physical_processes=8 logical_rollouts=8 gpu_model_processes=4 gpu0=m31:0+1+2+3 gpu1=crop:0+1+2+3 text_attention=sdpa vision_attention=flash_attention_2 sample_order=fixed_shared"
for rollout_id in 0 1 2 3; do
  launch_worker m31 0 "${rollout_id}" "${SEEDS[rollout_id]}"
done
for rollout_id in 0 1 2 3; do
  launch_worker crop 1 "${rollout_id}" "${SEEDS[rollout_id]}"
done

terminate_workers() {
  for pid in "${PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
}
trap 'terminate_workers; exit 130' INT
trap 'terminate_workers; exit 143' TERM

PROCESS_MAP=${OUTPUT_ROOT}/diagnostics/physical_processes.tsv
printf 'pid\tgpu\tmodel\trollout\tseed\tlog\n' >"${PROCESS_MAP}"
for index in "${!PIDS[@]}"; do
  rollout_id=${ROLLOUT_IDS[index]}
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${PIDS[index]}" "${GPUS[index]}" "${MODELS[index]}" "${rollout_id}" \
    "${SEEDS[rollout_id]}" "${LOG_PATHS[index]}" \
    >>"${PROCESS_MAP}"
done
cat "${PROCESS_MAP}"

unique_pid_count=$(printf '%s\n' "${PIDS[@]}" | sort -u | wc -l)
if [[ ${#PIDS[@]} -ne 8 || ${unique_pid_count} -ne 8 ]]; then
  echo "ERROR: launcher requires eight distinct physical worker PIDs; launched=${#PIDS[@]} unique=${unique_pid_count}" >&2
  terminate_workers
  wait || true
  exit 30
fi
echo "[PHYSICAL_PID_VALIDATION_OK] count=8 unique=8 pids=${PIDS[*]}"

for gpu in 0 1; do
  gpu_pids=()
  for index in "${!PIDS[@]}"; do
    if [[ "${GPUS[index]}" == "${gpu}" ]]; then
      gpu_pids+=("${PIDS[index]}")
    fi
  done
  echo "[GPU_MODEL_PROCESS_COUNT] gpu=${gpu} processes=${#gpu_pids[@]} pids=${gpu_pids[*]}"
done

load_status_count=0
while [[ ${load_status_count} -lt 8 ]]; do
  load_status_count=0
  premature_exit=0
  for index in "${!LOG_PATHS[@]}"; do
    log_path=${LOG_PATHS[index]}
    if grep -q -m1 -E '^\[MODEL_LOAD_(OK|FAIL)\]' "${log_path}" 2>/dev/null; then
      load_status_count=$((load_status_count + 1))
    elif ! kill -0 "${PIDS[index]}" 2>/dev/null; then
      premature_exit=1
    fi
  done
  if [[ ${premature_exit} -ne 0 ]]; then
    break
  fi
  if [[ ${load_status_count} -lt 8 ]]; then
    sleep 5
  fi
done
MODEL_LOAD_STATUS=${OUTPUT_ROOT}/diagnostics/model_load_status.txt
: >"${MODEL_LOAD_STATUS}"
for log_path in "${LOG_PATHS[@]}"; do
  if status_line=$(grep -m1 -E '^\[MODEL_LOAD_(OK|FAIL)\]' "${log_path}"); then
    echo "${status_line}"
    echo "${status_line}" >>"${MODEL_LOAD_STATUS}"
  else
    status_line="[MODEL_LOAD_STATUS_MISSING] log=${log_path}"
    echo "${status_line}"
    echo "${status_line}" >>"${MODEL_LOAD_STATUS}"
  fi
done
if ! "${PYTHON_BIN}" - "${OUTPUT_ROOT}" \
  "${PIDS[0]}" "m31" "0" "${LOG_PATHS[0]}" \
  "${PIDS[1]}" "m31" "1" "${LOG_PATHS[1]}" \
  "${PIDS[2]}" "m31" "2" "${LOG_PATHS[2]}" \
  "${PIDS[3]}" "m31" "3" "${LOG_PATHS[3]}" \
  "${PIDS[4]}" "crop" "0" "${LOG_PATHS[4]}" \
  "${PIDS[5]}" "crop" "1" "${LOG_PATHS[5]}" \
  "${PIDS[6]}" "crop" "2" "${LOG_PATHS[6]}" \
  "${PIDS[7]}" "crop" "3" "${LOG_PATHS[7]}" <<'PY'
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
workers = []
arguments = sys.argv[2:]
if len(arguments) != 32:
    raise SystemExit(f"expected eight pid/model/rollout/log groups, got {arguments!r}")
for offset in range(0, len(arguments), 4):
    pid_text, model, rollout_text, log_text = arguments[offset : offset + 4]
    pid = int(pid_text)
    log_path = Path(log_text)
    status_line = None
    if log_path.is_file():
        with log_path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.rstrip("\r\n")
                if re.match(r"^\[MODEL_LOAD_(?:OK|FAIL)\]", line):
                    status_line = line
                    break
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        alive = False
    else:
        alive = True
    expected_tokens = (
        "text_config=sdpa",
        "vision_config=flash_attention_2",
        "vision_first_layer=flash_attention_2",
        "vision_blocks=27/27",
    )
    expected_gpu = {"m31": 0, "crop": 1}.get(model)
    ok = bool(
        alive
        and status_line
        and status_line.startswith("[MODEL_LOAD_OK]")
        and f"model={model}" in status_line
        and f"gpu={expected_gpu}" in status_line
        and f"pid={pid}" in status_line
        and f"rollouts={rollout_text}" in status_line
        and all(token in status_line for token in expected_tokens)
    )
    workers.append(
        {
            "model_id": model,
            "physical_gpu": expected_gpu,
            "pid": pid,
            "rollout_ids": [int(value) for value in rollout_text.split(",")],
            "log_path": str(log_path),
            "alive": alive,
            "model_load_status_line": status_line,
            "validated": ok,
        }
    )
expected_ownership = {
    (model, rollout_id)
    for model in ("m31", "crop")
    for rollout_id in range(4)
}
actual_ownership = {
    (str(worker["model_id"]), int(worker["rollout_ids"][0]))
    for worker in workers
    if len(worker["rollout_ids"]) == 1
}
pids = [int(worker["pid"]) for worker in workers]
valid = bool(
    len(workers) == 8
    and len(set(pids)) == 8
    and actual_ownership == expected_ownership
    and all(worker["validated"] for worker in workers)
)
payload = {
    "schema_version": 3,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "valid": valid,
    "required_physical_workers": 8,
    "required_models": {"m31": 4, "crop": 4},
    "required_rollouts_per_worker": 1,
    "unique_pid_count": len(set(pids)),
    "workers": workers,
}
name = "formal_run_valid.json" if valid else "formal_run_invalid.json"
destination = root / "diagnostics" / name
temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, destination)
if valid:
    marker = root / "diagnostics" / "_MODEL_LOADS_OK"
    marker_tmp = marker.with_name(f".{marker.name}.tmp-{os.getpid()}")
    marker_tmp.write_text(payload["created_at"] + "\n", encoding="utf-8")
    os.replace(marker_tmp, marker)
else:
    raise SystemExit(
        "formal run invalid: all eight distinct live physical workers must report "
        "MODEL_LOAD_OK with text SDPA and 27/27 vision FlashAttention2 blocks"
    )
PY
then
  echo "ERROR: formal run invalid because all eight validated model loads did not succeed" >&2
  terminate_workers
  wait || true
  exit 30
fi
echo "[FORMAL_RUN_VALID] marker=${OUTPUT_ROOT}/diagnostics/_MODEL_LOADS_OK"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader \
  >"${OUTPUT_ROOT}/diagnostics/gpu_memory_after_model_load.csv"
cat "${OUTPUT_ROOT}/diagnostics/gpu_memory_after_model_load.csv"
if ! nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader \
  >"${OUTPUT_ROOT}/diagnostics/gpu_process_memory_after_model_load.csv"; then
  :
fi
cat "${OUTPUT_ROOT}/diagnostics/gpu_process_memory_after_model_load.csv"

run_incremental_snapshot() {
  local kind=$1
  local scheduled_hour=${2:-}
  local snapshot_args=(
    --output-root "${OUTPUT_ROOT}"
    --bundle-root "${BUNDLE_ROOT}"
    --kind "${kind}"
    --started-at-epoch "${RUN_STARTED_EPOCH}"
  )
  if [[ "${kind}" == "hourly" ]]; then
    snapshot_args+=(--scheduled-hour "${scheduled_hour}")
  else
    snapshot_args+=(--export-selection-dir "${OUTPUT_ROOT}/selection")
  fi
  echo "[INCREMENTAL_SNAPSHOT_START] kind=${kind} scheduled_hour=${scheduled_hour:-none}"
  if CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" \
    "${ORCHESTRATOR_REPO}/scripts/snapshot_ui5_train_rollouts.py" \
    "${snapshot_args[@]}"; then
    echo "[INCREMENTAL_SNAPSHOT_OK] kind=${kind} scheduled_hour=${scheduled_hour:-none}"
  else
    snapshot_failures=$((snapshot_failures + 1))
    echo "[INCREMENTAL_SNAPSHOT_FAIL] kind=${kind} scheduled_hour=${scheduled_hour:-none}" >&2
    return 1
  fi
}

while true; do
  running=0
  for pid in "${PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      running=$((running + 1))
    fi
  done
  CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" "${ORCHESTRATOR_REPO}/scripts/run_ui5_train_rollout_worker.py" \
    --mode progress-snapshot \
    --output-root "${OUTPUT_ROOT}" \
    --expected-workers 8 \
    --physical-worker "m31,0,${PIDS[0]},0" \
    --physical-worker "m31,0,${PIDS[1]},1" \
    --physical-worker "m31,0,${PIDS[2]},2" \
    --physical-worker "m31,0,${PIDS[3]},3" \
    --physical-worker "crop,1,${PIDS[4]},0" \
    --physical-worker "crop,1,${PIDS[5]},1" \
    --physical-worker "crop,1,${PIDS[6]},2" \
    --physical-worker "crop,1,${PIDS[7]},3" || true
  current_epoch=$(date +%s)
  elapsed_seconds=$((current_epoch - RUN_STARTED_EPOCH))
  while (( elapsed_seconds >= NEXT_SNAPSHOT_ELAPSED_SECONDS )); do
    if run_incremental_snapshot hourly "${NEXT_SNAPSHOT_HOUR}"; then
      NEXT_SNAPSHOT_HOUR=$((NEXT_SNAPSHOT_HOUR + 3))
      NEXT_SNAPSHOT_ELAPSED_SECONDS=$((NEXT_SNAPSHOT_ELAPSED_SECONDS + SNAPSHOT_INTERVAL_SECONDS))
    else
      break
    fi
  done
  if [[ ${running} -eq 0 ]]; then
    break
  fi
  sleep 60
done

worker_failures=0
set +e
for index in "${!PIDS[@]}"; do
  wait "${PIDS[index]}"
  status=$?
  echo "worker ${NAMES[index]} exit=${status}"
  if [[ ${status} -ne 0 ]]; then
    worker_failures=$((worker_failures + 1))
  fi
done
set -e

oom_summary_status=0
set +e
CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" \
  "${ORCHESTRATOR_REPO}/scripts/summarize_ui5_rollout_oom.py" \
  --output-root "${OUTPUT_ROOT}"
oom_summary_status=$?
set -e

run_incremental_snapshot final || true

aggregate_status=0
gallery_status=0
set +e
CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" "${ORCHESTRATOR_REPO}/scripts/aggregate_ui5_train_rollouts.py" \
  --output-root "${OUTPUT_ROOT}" \
  --bundle-root "${BUNDLE_ROOT}" \
  --repo-root "${CROP_REPO}"
aggregate_status=$?
if [[ ${aggregate_status} -eq 0 ]]; then
  CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" "${ORCHESTRATOR_REPO}/scripts/render_ui5_train_rollout_gallery.py" \
    --output-root "${OUTPUT_ROOT}" \
    --bundle-root "${BUNDLE_ROOT}"
  gallery_status=$?
fi
set -e

if [[ ${worker_failures} -ne 0 || ${snapshot_failures} -ne 0 || ${oom_summary_status} -ne 0 || ${aggregate_status} -ne 0 || ${gallery_status} -ne 0 ]]; then
  echo "ERROR: rollout job failed workers=${worker_failures} snapshots=${snapshot_failures} oom_summary=${oom_summary_status} aggregate=${aggregate_status} gallery=${gallery_status}" >&2
  exit 1
fi
echo "UI5 train rollout8 v6 completed with eight physical workers: ${OUTPUT_ROOT}"
