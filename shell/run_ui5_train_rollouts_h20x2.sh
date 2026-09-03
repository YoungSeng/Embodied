#!/usr/bin/env bash
set -Eeuo pipefail

WORKSPACE=${WORKSPACE:-/mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace}
ORCHESTRATOR_REPO=${ORCHESTRATOR_REPO:-${WORKSPACE}/code/Eagle/Embodied-rollout8-h20x2-v2}
M31_REPO=${M31_REPO:-${WORKSPACE}/code/Eagle/Embodied-ui5-rollout8-m31}
CROP_REPO=${CROP_REPO:-${ORCHESTRATOR_REPO}}
BUNDLE_ROOT=${BUNDLE_ROOT:-${WORKSPACE}/gui_data/ui5_train_rollout_bundle_v1}
M31_CHECKPOINT=${M31_CHECKPOINT:-${WORKSPACE}/gui_models/Embodied/locany-ui5-m31-taskmoe-setdecoder-a800x4-sft-20260830-r2/checkpoint-12000}
CROP_CHECKPOINT=${CROP_CHECKPOINT:-${WORKSPACE}/gui_models/Embodied-ui5-det-crop/locany-ui5-v5-croponly-sourcebalanced-a800x4-20260830/checkpoint-12000}
OUTPUT_ROOT=${OUTPUT_ROOT:-${WORKSPACE}/gui_rollouts/ui5-train-rollout8-h20x2-v2-20260903}
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
test -f "${ORCHESTRATOR_REPO}/scripts/run_ui5_train_rollout_worker.py" || { echo "ERROR: v2 rollout worker missing" >&2; exit 27; }
test -f "${ORCHESTRATOR_REPO}/scripts/aggregate_ui5_train_rollouts.py" || { echo "ERROR: v2 aggregator missing" >&2; exit 28; }
test -f "${ORCHESTRATOR_REPO}/scripts/snapshot_ui5_train_rollouts.py" || { echo "ERROR: v2 incremental snapshotter missing" >&2; exit 29; }

export PATH="${ENV_DIR}/bin:${PATH}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=2
mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/diagnostics"

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
nvidia-smi

RUN_STARTED_EPOCH=$(date +%s)
NEXT_SNAPSHOT_HOUR=3
SNAPSHOT_INTERVAL_SECONDS=10800
NEXT_SNAPSHOT_ELAPSED_SECONDS=${SNAPSHOT_INTERVAL_SECONDS}
snapshot_failures=0
echo "[ROLLOUT_START] epoch=${RUN_STARTED_EPOCH} first_snapshot_hour=${NEXT_SNAPSHOT_HOUR} interval_seconds=${SNAPSHOT_INTERVAL_SECONDS}"

PIDS=()
NAMES=()
GPUS=()
LOG_PATHS=()
launch_worker() {
  local model_id=$1
  local gpu=$2
  local rollout_ids=$3
  local seeds=$4
  local repo checkpoint log_path
  if [[ "${model_id}" == "m31" ]]; then
    repo=${M31_REPO}
    checkpoint=${M31_CHECKPOINT}
  else
    repo=${CROP_REPO}
    checkpoint=${CROP_CHECKPOINT}
  fi
  log_path=${OUTPUT_ROOT}/logs/${model_id}_rollouts_${rollout_ids//,/_}.log
  (
    cd "${ORCHESTRATOR_REPO}"
    export PYTHONPATH="${repo}${PYTHONPATH:+:${PYTHONPATH}}"
    exec env CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" \
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
      --gpu-model-processes 2 \
      --dtype bf16 \
      --attn-implementation sdpa \
      --vision-attn-implementation sdpa \
      --generation-mode hybrid
  ) >"${log_path}" 2>&1 &
  PIDS+=("$!")
  NAMES+=("${model_id}/rollouts_${rollout_ids}/gpu_${gpu}")
  GPUS+=("${gpu}")
  LOG_PATHS+=("${log_path}")
  echo "[WORKER_LAUNCH] model=${model_id} gpu=${gpu} pid=$! rollout_ids=${rollout_ids} log=${log_path}"
}

launch_worker m31 0 "0,1" "${SEEDS[0]},${SEEDS[1]}"
launch_worker m31 0 "2,3" "${SEEDS[2]},${SEEDS[3]}"
launch_worker crop 1 "0,1" "${SEEDS[0]},${SEEDS[1]}"
launch_worker crop 1 "2,3" "${SEEDS[2]},${SEEDS[3]}"

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
load_deadline=$((SECONDS + 900))
while [[ ${load_status_count} -lt 4 && ${SECONDS} -lt ${load_deadline} ]]; do
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
  if [[ ${load_status_count} -lt 4 ]]; then
    sleep 5
  fi
done
for log_path in "${LOG_PATHS[@]}"; do
  grep -m1 -E '^\[MODEL_LOAD_(OK|FAIL)\]' "${log_path}" || {
    echo "[MODEL_LOAD_STATUS_MISSING] log=${log_path}"
  }
done
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader || true

terminate_workers() {
  for pid in "${PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
}
trap 'terminate_workers; exit 130' INT
trap 'terminate_workers; exit 143' TERM

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
    --expected-workers 8 || true
  current_epoch=$(date +%s)
  elapsed_seconds=$((current_epoch - RUN_STARTED_EPOCH))
  while (( elapsed_seconds >= NEXT_SNAPSHOT_ELAPSED_SECONDS )); do
    run_incremental_snapshot hourly "${NEXT_SNAPSHOT_HOUR}"
    NEXT_SNAPSHOT_HOUR=$((NEXT_SNAPSHOT_HOUR + 3))
    NEXT_SNAPSHOT_ELAPSED_SECONDS=$((NEXT_SNAPSHOT_ELAPSED_SECONDS + SNAPSHOT_INTERVAL_SECONDS))
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

run_incremental_snapshot final

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

if [[ ${worker_failures} -ne 0 || ${snapshot_failures} -ne 0 || ${aggregate_status} -ne 0 || ${gallery_status} -ne 0 ]]; then
  echo "ERROR: rollout job failed workers=${worker_failures} snapshots=${snapshot_failures} aggregate=${aggregate_status} gallery=${gallery_status}" >&2
  exit 1
fi
echo "UI5 train rollout 4+4 completed: ${OUTPUT_ROOT}"
