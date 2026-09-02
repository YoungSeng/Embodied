#!/usr/bin/env bash
set -Eeuo pipefail

WORKSPACE=${WORKSPACE:-/mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace}
M31_REPO=${M31_REPO:-${WORKSPACE}/code/Eagle/Embodied-ui5-rollout8-m31}
CROP_REPO=${CROP_REPO:-${WORKSPACE}/code/Eagle/Embodied-ui5-rollout8-crop}
BUNDLE_ROOT=${BUNDLE_ROOT:-${WORKSPACE}/gui_data/ui5_train_rollout_bundle_v1}
M31_CHECKPOINT=${M31_CHECKPOINT:-${WORKSPACE}/gui_models/Embodied/locany-ui5-m31-taskmoe-setdecoder-a800x4-sft-20260830-r2/checkpoint-12000}
CROP_CHECKPOINT=${CROP_CHECKPOINT:-${WORKSPACE}/gui_models/Embodied-ui5-det-crop/locany-ui5-v5-croponly-sourcebalanced-a800x4-20260830/checkpoint-12000}
OUTPUT_ROOT=${OUTPUT_ROOT:-${WORKSPACE}/gui_rollouts/ui5-train-rollout8-20260903}
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
test -f "${M31_REPO}/scripts/run_ui5_train_rollout_worker.py" || { echo "ERROR: M31 rollout worker missing" >&2; exit 27; }
test -f "${CROP_REPO}/scripts/run_ui5_train_rollout_worker.py" || { echo "ERROR: crop rollout worker missing" >&2; exit 28; }

export PATH="${ENV_DIR}/bin:${PATH}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=2
mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/diagnostics"

CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" "${CROP_REPO}/scripts/preflight_ui5_train_rollouts.py" \
  --bundle-root "${BUNDLE_ROOT}" \
  --diagnostics-dir "${OUTPUT_ROOT}/diagnostics" \
  --m31-checkpoint "${M31_CHECKPOINT}" \
  --crop-checkpoint "${CROP_CHECKPOINT}" \
  --processor-candidate "${PROCESSOR_CANDIDATES[0]}" \
  --processor-candidate "${PROCESSOR_CANDIDATES[1]}" \
  --m31-repo "${M31_REPO}" \
  --crop-repo "${CROP_REPO}" \
  --require-runtime

"${PYTHON_BIN}" -c 'import openpyxl, scipy, PIL; assert tuple(map(int, openpyxl.__version__.split(".")[:2])) >= (3, 1)'
nvidia-smi

PIDS=()
NAMES=()
launch_worker() {
  local model_id=$1
  local gpu=$2
  local rollout_id=$3
  local seed=$4
  local repo checkpoint log_path
  if [[ "${model_id}" == "m31" ]]; then
    repo=${M31_REPO}
    checkpoint=${M31_CHECKPOINT}
  else
    repo=${CROP_REPO}
    checkpoint=${CROP_CHECKPOINT}
  fi
  log_path=${OUTPUT_ROOT}/logs/${model_id}_rollout_${rollout_id}.log
  (
    cd "${repo}"
    export PYTHONPATH="${repo}${PYTHONPATH:+:${PYTHONPATH}}"
    exec env CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" \
      scripts/run_ui5_train_rollout_worker.py \
      --mode run \
      --model-id "${model_id}" \
      --checkpoint "${checkpoint}" \
      --processor-path "${PROCESSOR_PATH}" \
      --bundle-root "${BUNDLE_ROOT}" \
      --output-root "${OUTPUT_ROOT}" \
      --repo-root "${repo}" \
      --rollout-id "${rollout_id}" \
      --seed "${seed}" \
      --dtype bf16 \
      --attn-implementation sdpa \
      --vision-attn-implementation sdpa \
      --generation-mode hybrid
  ) >"${log_path}" 2>&1 &
  PIDS+=("$!")
  NAMES+=("${model_id}/rollout_${rollout_id}/gpu_${gpu}")
}

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

while true; do
  running=0
  for pid in "${PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      running=$((running + 1))
    fi
  done
  CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" "${CROP_REPO}/scripts/run_ui5_train_rollout_worker.py" \
    --mode progress-snapshot \
    --output-root "${OUTPUT_ROOT}" \
    --expected-workers 8 || true
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

aggregate_status=0
gallery_status=0
set +e
CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" "${CROP_REPO}/scripts/aggregate_ui5_train_rollouts.py" \
  --output-root "${OUTPUT_ROOT}" \
  --bundle-root "${BUNDLE_ROOT}" \
  --repo-root "${CROP_REPO}"
aggregate_status=$?
if [[ ${aggregate_status} -eq 0 ]]; then
  CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" "${CROP_REPO}/scripts/render_ui5_train_rollout_gallery.py" \
    --output-root "${OUTPUT_ROOT}" \
    --bundle-root "${BUNDLE_ROOT}"
  gallery_status=$?
fi
set -e

if [[ ${worker_failures} -ne 0 || ${aggregate_status} -ne 0 || ${gallery_status} -ne 0 ]]; then
  echo "ERROR: rollout job failed workers=${worker_failures} aggregate=${aggregate_status} gallery=${gallery_status}" >&2
  exit 1
fi
echo "UI5 train rollout 4+4 completed: ${OUTPUT_ROOT}"
