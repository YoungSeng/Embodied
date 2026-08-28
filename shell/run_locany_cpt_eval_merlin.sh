#!/usr/bin/env bash
set -Eeuo pipefail

# Consume checkpoint rows from diagnostics/cpt_eval_queue.jsonl on one GPU.
# Usage: RUN_DIR=/path/to/run bash shell/run_locany_cpt_eval_merlin.sh <a100|h20>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
MACHINE_TYPE="${1:-h20}"
if [[ "${MACHINE_TYPE}" != "a100" && "${MACHINE_TYPE}" != "h20" ]]; then
  echo "Usage: bash shell/run_locany_cpt_eval_merlin.sh <a100|h20>" >&2
  exit 2
fi

case "${MACHINE_TYPE}" in
  a100)
    WORKSPACE="${WORKSPACE:-/mnt/bn/intelligent-service-yg/logging/sicheng_workspace}"
    FILESYSTEM_ROOT=/mnt/bn/intelligent-service-yg
    DEFAULT_RUN_NAME=locany-3b-ui-cpt-v4-v2-a100x4-formal
    ;;
  h20)
    WORKSPACE="${WORKSPACE:-/mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace}"
    FILESYSTEM_ROOT=/mnt/bn/intelligent-service-arnold-hl
    DEFAULT_RUN_NAME=locany-3b-ui-cpt-v4-v2-h20x4-formal
    ;;
esac

ENV_DIR="${ENV_DIR:-${WORKSPACE}/conda_envs/LocateAnything}"
DATA_DIR="${DATA_DIR:-${WORKSPACE}/data/locany_cpt_v4_split_v2}"
RUN_NAME="${RUN_NAME:-${DEFAULT_RUN_NAME}}"
RUN_DIR="${RUN_DIR:-${WORKSPACE}/gui_models/${RUN_NAME}}"
QUEUE_PATH="${QUEUE_PATH:-${RUN_DIR}/diagnostics/cpt_eval_queue.jsonl}"
BASE_MODEL="${BASE_MODEL:-${WORKSPACE}/hf_home/hub/models--nvidia--LocateAnything-3B/snapshots/c32291ca5e996f5a7a485845b4f57a233936bba0}"
EVAL_SAMPLES_PER_TASK="${EVAL_SAMPLES_PER_TASK:-10}"
EVAL_MAX_PENDING="${EVAL_MAX_PENDING:-1}"

test -d "${PROJECT_ROOT}" || { echo "ERROR: missing project: ${PROJECT_ROOT}" >&2; exit 20; }
test -x "${ENV_DIR}/bin/python" || { echo "ERROR: missing environment: ${ENV_DIR}" >&2; exit 21; }
test -d "${FILESYSTEM_ROOT}" || { echo "ERROR: missing filesystem: ${FILESYSTEM_ROOT}" >&2; exit 22; }
test -d "${BASE_MODEL}" || { echo "ERROR: missing base model: ${BASE_MODEL}" >&2; exit 23; }
test -f "${QUEUE_PATH}" || { echo "ERROR: missing eval queue: ${QUEUE_PATH}" >&2; exit 24; }

RAW_JOB_ID="${ARNOLD_TRIAL_ID:-${ARNOLD_JOB_ID:-manual-$$}}"
JOB_ID="${RAW_JOB_ID//[^a-zA-Z0-9._-]/_}"
CACHE_ROOT="${CACHE_ROOT:-/tmp/locany-cpt-eval-${JOB_ID}}"
mkdir -p "${CACHE_ROOT}/tmp" "${CACHE_ROOT}/pycache" "${CACHE_ROOT}/hf" "${CACHE_ROOT}/torch"

export PATH="${ENV_DIR}/bin:${PATH}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${CUDA_DEVICES:-0}}"
export TMPDIR="${CACHE_ROOT}/tmp" TEMP="${CACHE_ROOT}/tmp" TMP="${CACHE_ROOT}/tmp"
export PYTHONPYCACHEPREFIX="${CACHE_ROOT}/pycache"
export HF_HOME="${HF_HOME:-${WORKSPACE}/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export TORCH_HOME="${WORKSPACE}/cache/torch"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export INSTALL_SYSTEM_RUNTIME_DEPS="${INSTALL_SYSTEM_RUNTIME_DEPS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PROJECT_ROOT ENV_DIR

cd "${PROJECT_ROOT}"
bash -n shell/run_locany_cpt_eval_merlin.sh
bash shell/ensure_locany_cpt_runtime.sh
"${ENV_DIR}/bin/python" - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit(f"CPT eval job requires exactly one visible GPU; found {torch.cuda.device_count()}")
print("eval gpu:", torch.cuda.get_device_name(0))
PY

"${ENV_DIR}/bin/python" scripts/validate_locany_cpt.py \
  --recipe "${DATA_DIR}/recipe/locany_cpt_val_fast.json" \
  --records-per-dataset 0 \
  --minimum-records-per-dataset "${EVAL_SAMPLES_PER_TASK}" \
  --split-manifest "${DATA_DIR}/diagnostics/split_manifest.jsonl" \
  --require-split heldout \
  --allow-manifest-subset

EVAL_ARGS=(
  --queue "${QUEUE_PATH}"
  --run-dir "${RUN_DIR}"
  --data-dir "${DATA_DIR}"
  --base-model "${BASE_MODEL}"
  --python "${ENV_DIR}/bin/python"
  --samples-per-task "${EVAL_SAMPLES_PER_TASK}"
  --max-pending "${EVAL_MAX_PENDING}"
  --device cuda:0
  --dtype "${EVAL_DTYPE:-bf16}"
  --attn-implementation "${EVAL_ATTN_IMPLEMENTATION:-sdpa}"
  --vision-attn-implementation "${EVAL_VISION_ATTN_IMPLEMENTATION:-flash_attention_2}"
  --max-new-tokens "${EVAL_MAX_NEW_TOKENS:-1024}"
  --seed "${CPT_EVAL_SEED:-20260826}"
  --require-zero-inference-errors
)
if [[ "${EVAL_RETRY_FAILED:-0}" == "1" ]]; then
  EVAL_ARGS+=(--retry-failed)
fi

exec "${ENV_DIR}/bin/python" scripts/run_locany_cpt_eval_queue.py "${EVAL_ARGS[@]}"
