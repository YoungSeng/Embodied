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
  export RUN_NAME="${RUN_NAME:-locany-3b-ui-cpt-v4-${MACHINE_TYPE}x${GPU_COUNT}-smoke-${JOB_ID}}"
  export REPORT_TO="${REPORT_TO:-none}"
else
  export RUN_NAME="${RUN_NAME:-locany-3b-ui-cpt-v4-${MACHINE_TYPE}x${GPU_COUNT}-formal}"
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
if machine == "h20" and importlib.util.find_spec("magi_attention") is None:
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

SMOKE_RESUME_STEP="${CPT_SMOKE_RESUME_STEP:-0}"
if [[ "${CPT_MODE}" == "smoke" && "${SMOKE_RESUME_STEP}" -gt 0 ]]; then
  export LOCANY_SEGMENT_MODE=1
  export LOCANY_STOP_AFTER_STEP="${SMOKE_RESUME_STEP}"
  if run_training_phase "pre-resume-${SMOKE_RESUME_STEP}"; then
    :
  else
    TRAIN_EXIT_CODE=$?
    echo "TRAIN_EXIT_CODE=${TRAIN_EXIT_CODE}"
    echo "LAUNCH_LOG=${LAUNCH_LOG}"
    exit "${TRAIN_EXIT_CODE}"
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
