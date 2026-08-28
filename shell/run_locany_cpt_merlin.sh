#!/usr/bin/env bash
set -Eeuo pipefail

# Common in-container launcher used by the CPT Merlin job definitions.
# Usage: bash shell/run_locany_cpt_merlin.sh <a100|h20> <smoke|formal>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
MACHINE_TYPE="${1:-}"
CPT_MODE="${2:-}"

if [[ "${MACHINE_TYPE}" != "a100" && "${MACHINE_TYPE}" != "h20" ]]; then
  echo "Usage: bash shell/run_locany_cpt_merlin.sh <a100|h20> <smoke|formal>" >&2
  exit 2
fi
if [[ "${CPT_MODE}" != "smoke" && "${CPT_MODE}" != "formal" ]]; then
  echo "CPT mode must be smoke or formal, got: ${CPT_MODE}" >&2
  exit 2
fi
case "${MACHINE_TYPE}" in
  a100)
    WORKSPACE="${WORKSPACE:-/mnt/bn/intelligent-service-yg/logging/sicheng_workspace}"
    FILESYSTEM_ROOT=/mnt/bn/intelligent-service-yg
    ;;
  h20)
    WORKSPACE="${WORKSPACE:-/mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace}"
    FILESYSTEM_ROOT=/mnt/bn/intelligent-service-arnold-hl
    ;;
esac

ENV_DIR="${ENV_DIR:-${WORKSPACE}/conda_envs/LocateAnything}"
RAW_JOB_ID="${ARNOLD_TRIAL_ID:-${ARNOLD_JOB_ID:-manual-$$}}"
JOB_ID="${RAW_JOB_ID//[^a-zA-Z0-9._-]/_}"
LOCAL_RUNTIME_DIR="${LOCAL_RUNTIME_DIR:-/tmp/locany-cpt-${JOB_ID}}"
SHARED_RUNTIME_DIR="${SHARED_RUNTIME_DIR:-${WORKSPACE}/runtime/locany-cpt/${JOB_ID}}"
CACHE_ROOT="${CACHE_ROOT:-${LOCAL_RUNTIME_DIR}/cache}"
GPU_COUNT="${GPU_COUNT:-4}"
export GPU_COUNT

mkdir -p \
  "${SHARED_RUNTIME_DIR}" \
  "${CACHE_ROOT}/tmp" \
  "${CACHE_ROOT}/pycache" \
  "${CACHE_ROOT}/hf_datasets" \
  "${CACHE_ROOT}/triton" \
  "${CACHE_ROOT}/torch_extensions" \
  "${CACHE_ROOT}/torchinductor" \
  "${CACHE_ROOT}/cuda" \
  "${CACHE_ROOT}/xdg" \
  "${CACHE_ROOT}/wandb" \
  "${WORKSPACE}/cache/huggingface/hub" \
  "${WORKSPACE}/cache/torch"

export PROJECT_ROOT WORKSPACE ENV_DIR CACHE_ROOT
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${CUDA_DEVICES:-0,1,2,3}}"
export PATH="${ENV_DIR}/bin:${PATH}"
export CONDA_PREFIX="${ENV_DIR}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Lock-heavy caches stay on the worker-local disk. Reusable model downloads
# remain on ByteNAS, but formal jobs run offline against MODEL_PATH.
export TMPDIR="${CACHE_ROOT}/tmp"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"
export PYTHONPYCACHEPREFIX="${CACHE_ROOT}/pycache"
export HF_HOME="${HF_HOME:-${WORKSPACE}/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HUB_CACHE}}"
export HF_DATASETS_CACHE="${CACHE_ROOT}/hf_datasets"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export TORCH_EXTENSIONS_DIR="${CACHE_ROOT}/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/torchinductor"
export CUDA_CACHE_PATH="${CACHE_ROOT}/cuda"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export WANDB_DIR="${CACHE_ROOT}/wandb"
export TORCH_HOME="${WORKSPACE}/cache/torch"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DISABLE_TELEMETRY=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

if [[ "${CPT_MODE}" == "smoke" ]]; then
  export RUN_NAME="${RUN_NAME:-locany-3b-ui-cpt-v4-v2-${MACHINE_TYPE}x${GPU_COUNT}-smoke-${JOB_ID}}"
  export REPORT_TO="${REPORT_TO:-none}"
else
  export RUN_NAME="${RUN_NAME:-locany-3b-ui-cpt-v4-v2-${MACHINE_TYPE}x${GPU_COUNT}-formal}"
  export REPORT_TO="${REPORT_TO:-tensorboard}"
fi

test -d "${PROJECT_ROOT}" || {
  echo "ERROR: Project directory not found: ${PROJECT_ROOT}" >&2
  exit 20
}
test -x "${ENV_DIR}/bin/python" || {
  echo "ERROR: Python environment not found: ${ENV_DIR}" >&2
  exit 21
}
test -f "${PROJECT_ROOT}/shell/run_locany_cpt.sh" || {
  echo "ERROR: CPT launcher not found under ${PROJECT_ROOT}" >&2
  exit 22
}
test -d "${FILESYSTEM_ROOT}" || {
  echo "ERROR: ByteNAS mount not found: ${FILESYSTEM_ROOT}" >&2
  exit 23
}

cd "${PROJECT_ROOT}"
hash -r
ulimit -n 65535 || true

echo "===== LocateAnything CPT Merlin preflight ====="
echo "job_id           : ${JOB_ID}"
echo "machine          : ${MACHINE_TYPE}"
echo "mode             : ${CPT_MODE}"
echo "project_root     : ${PROJECT_ROOT}"
echo "workspace        : ${WORKSPACE}"
echo "environment      : ${ENV_DIR}"
echo "run_name         : ${RUN_NAME}"
echo "visible_gpus     : ${CUDA_VISIBLE_DEVICES}"
echo "local_cache      : ${CACHE_ROOT}"
echo "launcher_log     : ${SHARED_RUNTIME_DIR}/launcher.log"
echo "integrated_eval  : ${CPT_INTEGRATED_EVAL:-0}"
echo "eval_at_start    : ${CPT_EVAL_AT_START:-1}"
git rev-parse --short HEAD 2>/dev/null || true
bash -n "${PROJECT_ROOT}/shell/run_locany_cpt.sh"
bash -n "${PROJECT_ROOT}/shell/train_locany_ui_defect.sh"

"${ENV_DIR}/bin/python" - "${MACHINE_TYPE}" <<'PY'
import importlib.util
import sys
import torch
import os
expected_gpu_count = int(os.environ.get("GPU_COUNT", "4"))

machine = sys.argv[1]
print("python:", sys.executable)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    print(f"gpu {index}:", torch.cuda.get_device_name(index))
if not torch.cuda.is_available() or torch.cuda.device_count() != expected_gpu_count:
    raise SystemExit(
        f"Merlin CPT job requires {expected_gpu_count} visible GPUs; "
        f"found {torch.cuda.device_count()}"
    )
if (
    machine == "h20"
    and os.environ.get("ATTN_IMPLEMENTATION", "sdpa").lower() == "magi"
    and importlib.util.find_spec("magi_attention") is None
):
    raise SystemExit("H20 CPT profile requires magi_attention")
PY

df -h /tmp "${FILESYSTEM_ROOT}" || true
nvidia-smi

LAUNCH_LOG="${SHARED_RUNTIME_DIR}/launcher.log"

run_training_phase() {
  local phase_name="$1"
  local phase_exit_code
  echo "===== Start ${MACHINE_TYPE} ${CPT_MODE} phase=${phase_name} ====="
  set +e
  bash "${PROJECT_ROOT}/shell/run_locany_cpt.sh" "${MACHINE_TYPE}" "${CPT_MODE}" \
    2>&1 | tee -a "${LAUNCH_LOG}"
  phase_exit_code="${PIPESTATUS[0]}"
  set -e
  echo "TRAIN_PHASE=${phase_name} EXIT_CODE=${phase_exit_code}"
  return "${phase_exit_code}"
}

run_integrated_eval_phase() {
  local phase_name="$1"
  local formal_output_dir="$2"
  local eval_exit_code
  echo "===== Start ${MACHINE_TYPE} ${CPT_MODE} integrated eval phase=${phase_name} ====="
  set +e
  CUDA_VISIBLE_DEVICES=0 \
  RUN_DIR="${formal_output_dir}" \
  RUN_NAME="${RUN_NAME}" \
  DATA_DIR="${DATA_DIR}" \
  EVAL_MAX_PENDING="${EVAL_MAX_PENDING:-20}" \
  EVAL_SAMPLES_PER_TASK="${EVAL_SAMPLES_PER_TASK:-10}" \
  EVAL_IOU_THRESHOLD="${EVAL_IOU_THRESHOLD:-0.1}" \
  EVAL_RETRY_FAILED="${EVAL_RETRY_FAILED:-1}" \
  bash "${PROJECT_ROOT}/shell/run_locany_cpt_eval_merlin.sh" "${MACHINE_TYPE}" \
    2>&1 | tee -a "${LAUNCH_LOG}"
  eval_exit_code="${PIPESTATUS[0]}"
  set -e
  echo "EVAL_PHASE=${phase_name} EXIT_CODE=${eval_exit_code}"
  return "${eval_exit_code}"
}

run_initial_eval_phase() {
  local formal_output_dir="$1"
  local base_model="${MODEL_PATH:-${WORKSPACE}/hf_home/hub/models--nvidia--LocateAnything-3B/snapshots/c32291ca5e996f5a7a485845b4f57a233936bba0}"
  local eval_exit_code
  echo "===== Start ${MACHINE_TYPE} ${CPT_MODE} integrated step-0 held-out + external UI5 eval ====="
  set +e
  CUDA_VISIBLE_DEVICES=0 \
  "${ENV_DIR}/bin/python" "${PROJECT_ROOT}/scripts/run_locany_cpt_initial_eval.py" \
    --run-dir "${formal_output_dir}" \
    --data-dir "${DATA_DIR}" \
    --eval-recipe-name "${EVAL_RECIPE_NAME:-locany_cpt_val_fast.json}" \
    --base-model "${base_model}" \
    --python "${ENV_DIR}/bin/python" \
    --samples-per-task "${EVAL_SAMPLES_PER_TASK:-10}" \
    --device cuda:0 \
    --dtype "${EVAL_DTYPE:-bf16}" \
    --attn-implementation "${EVAL_ATTN_IMPLEMENTATION:-sdpa}" \
    --vision-attn-implementation "${EVAL_VISION_ATTN_IMPLEMENTATION:-flash_attention_2}" \
    --max-new-tokens "${EVAL_MAX_NEW_TOKENS:-1024}" \
    --iou-threshold "${EVAL_IOU_THRESHOLD:-0.1}" \
    --external-ui5-data-dir "${CPT_EXTERNAL_UI5_DATA_DIR:-${WORKSPACE}/data}" \
    --external-max-new-tokens "${EVAL_EXTERNAL_MAX_NEW_TOKENS:-4096}" \
    --external-max-images-per-task "${EVAL_EXTERNAL_MAX_IMAGES_PER_TASK:-0}" \
    --external-iou-thresholds "${EVAL_IOU_THRESHOLD:-0.1}" \
    --seed "${CPT_EVAL_SEED:-20260826}" \
    2>&1 | tee -a "${LAUNCH_LOG}"
  eval_exit_code="${PIPESTATUS[0]}"
  set -e
  echo "INITIAL_EVAL_PHASE=step-0 EXIT_CODE=${eval_exit_code}"
  return "${eval_exit_code}"
}

SMOKE_RESUME_STEP="${CPT_SMOKE_RESUME_STEP:-0}"
if [[ "${CPT_MODE}" == "formal" && "${CPT_INTEGRATED_EVAL:-0}" == "1" ]]; then
  export LOCANY_SEGMENT_MODE=1
  export LOCANY_STOP_AFTER_PERIODIC_SAVE=1
  export DATA_DIR="${DATA_DIR:-${WORKSPACE}/data/locany_cpt_v4_split_v2}"
  FORMAL_OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_BASE:-${WORKSPACE}/gui_models}/${RUN_NAME}}"
  FORMAL_MAX_STEPS="${MAX_STEPS:-20000}"
  SEGMENT_INDEX=0
  if [[ "${CPT_EVAL_AT_START:-1}" == "1" ]]; then
    if run_initial_eval_phase "${FORMAL_OUTPUT_DIR}"; then
      :
    else
      EVAL_EXIT_CODE=$?
      echo "INITIAL_EVAL_EXIT_CODE=${EVAL_EXIT_CODE}"
      echo "Initial held-out evaluation failed; formal training has not started."
      echo "The same step-0 gate also requires the external UI5 evaluation to pass."
      echo "Fix evaluation and resubmit the same formal job."
      echo "LAUNCH_LOG=${LAUNCH_LOG}"
      exit "${EVAL_EXIT_CODE}"
    fi
  fi
  if [[ -f "${FORMAL_OUTPUT_DIR}/diagnostics/cpt_eval_queue.jsonl" ]]; then
    # A resubmitted job repairs/evaluates an already completed checkpoint
    # before spending another six-hour training segment.
    run_integrated_eval_phase "pre-resume-backlog" "${FORMAL_OUTPUT_DIR}"
  fi
  while true; do
    SEGMENT_INDEX=$((SEGMENT_INDEX + 1))
    if run_training_phase "formal-segment-${SEGMENT_INDEX}"; then
      :
    else
      TRAIN_EXIT_CODE=$?
      echo "TRAIN_EXIT_CODE=${TRAIN_EXIT_CODE}"
      echo "LAUNCH_LOG=${LAUNCH_LOG}"
      exit "${TRAIN_EXIT_CODE}"
    fi

    LATEST_STEP="$(
      "${ENV_DIR}/bin/python" "${PROJECT_ROOT}/scripts/locany_ui5_checkpoint.py" \
        latest \
        --output-dir "${FORMAL_OUTPUT_DIR}" \
        --require-resume \
        --expected-ranks "${GPU_COUNT}" \
        --field step
    )"
    if [[ ! "${LATEST_STEP}" =~ ^[0-9]+$ || "${LATEST_STEP}" -le 0 ]]; then
      echo "ERROR: integrated eval found no resumable checkpoint after segment ${SEGMENT_INDEX}: ${LATEST_STEP}" >&2
      exit 31
    fi
    echo "INTEGRATED_EVAL_CHECKPOINT_STEP=${LATEST_STEP}"

    if run_integrated_eval_phase "checkpoint-${LATEST_STEP}" "${FORMAL_OUTPUT_DIR}"; then
      :
    else
      EVAL_EXIT_CODE=$?
      echo "EVAL_EXIT_CODE=${EVAL_EXIT_CODE}"
      echo "Training checkpoint is resumable; fix eval and resubmit the same formal job."
      echo "LAUNCH_LOG=${LAUNCH_LOG}"
      exit "${EVAL_EXIT_CODE}"
    fi

    if (( LATEST_STEP >= FORMAL_MAX_STEPS )); then
      echo "INTEGRATED_FORMAL_COMPLETE_STEP=${LATEST_STEP}"
      break
    fi
    echo "===== Resume training after integrated held-out + external UI5 eval at step ${LATEST_STEP} ====="
  done
elif [[ "${CPT_MODE}" == "smoke" && "${SMOKE_RESUME_STEP}" -gt 0 ]]; then
  SMOKE_OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_BASE:-${WORKSPACE}/gui_models}/${RUN_NAME}}"
  SMOKE_RESUME_CHECKPOINT="${SMOKE_OUTPUT_DIR}/checkpoint-${SMOKE_RESUME_STEP}"
  export LOCANY_SEGMENT_MODE=1
  if "${ENV_DIR}/bin/python" "${PROJECT_ROOT}/scripts/locany_ui5_checkpoint.py" \
    validate \
    --checkpoint "${SMOKE_RESUME_CHECKPOINT}" \
    --mode resume \
    --expected-ranks "${GPU_COUNT}" >/dev/null 2>&1; then
    echo "SMOKE_PRE_RESUME=SKIPPED_EXISTING_RESUMABLE_CHECKPOINT"
    echo "SMOKE_RESUME_CHECKPOINT=${SMOKE_RESUME_CHECKPOINT}"
  else
    export LOCANY_STOP_AFTER_STEP="${SMOKE_RESUME_STEP}"
    if run_training_phase "pre-resume-${SMOKE_RESUME_STEP}"; then
      :
    else
      TRAIN_EXIT_CODE=$?
      echo "TRAIN_EXIT_CODE=${TRAIN_EXIT_CODE}"
      echo "LAUNCH_LOG=${LAUNCH_LOG}"
      exit "${TRAIN_EXIT_CODE}"
    fi
  fi
  unset LOCANY_STOP_AFTER_STEP
  if run_training_phase "post-resume"; then
    :
  else
    TRAIN_EXIT_CODE=$?
    echo "TRAIN_EXIT_CODE=${TRAIN_EXIT_CODE}"
    echo "LAUNCH_LOG=${LAUNCH_LOG}"
    exit "${TRAIN_EXIT_CODE}"
  fi
else
  if run_training_phase "single"; then
    :
  else
    TRAIN_EXIT_CODE=$?
    echo "TRAIN_EXIT_CODE=${TRAIN_EXIT_CODE}"
    echo "LAUNCH_LOG=${LAUNCH_LOG}"
    exit "${TRAIN_EXIT_CODE}"
  fi
fi

echo "TRAIN_EXIT_CODE=0"
echo "LAUNCH_LOG=${LAUNCH_LOG}"
exit 0
