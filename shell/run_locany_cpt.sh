#!/usr/bin/env bash
set -Eeuo pipefail

# Usage:
#   bash shell/run_locany_cpt.sh a100 smoke
#   bash shell/run_locany_cpt.sh a100 formal
#   bash shell/run_locany_cpt.sh h20 formal

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
MACHINE_TYPE="${1:-${MACHINE_TYPE:-}}"
CPT_MODE="${2:-${CPT_MODE:-formal}}"

if [[ "${MACHINE_TYPE}" != "a100" && "${MACHINE_TYPE}" != "h20" ]]; then
  echo "Usage: bash shell/run_locany_cpt.sh <a100|h20> <smoke|formal>" >&2
  exit 2
fi
if [[ "${CPT_MODE}" != "smoke" && "${CPT_MODE}" != "formal" ]]; then
  echo "CPT mode must be smoke or formal, got: ${CPT_MODE}" >&2
  exit 2
fi

case "${MACHINE_TYPE}" in
  a100)
    WORKSPACE="${WORKSPACE:-/mnt/bn/intelligent-service-yg/logging/sicheng_workspace}"
    ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
    MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-7268}"
    MAX_NUM_TOKENS_PER_SAMPLE="${MAX_NUM_TOKENS_PER_SAMPLE:-7268}"
    MAX_NUM_TOKENS="${MAX_NUM_TOKENS:-12800}"
    PACKING_BUFFER_SIZE="${PACKING_BUFFER_SIZE:-16}"
    ;;
  h20)
    WORKSPACE="${WORKSPACE:-/mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace}"
    ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-magi}"
    MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-8192}"
    MAX_NUM_TOKENS_PER_SAMPLE="${MAX_NUM_TOKENS_PER_SAMPLE:-8192}"
    MAX_NUM_TOKENS="${MAX_NUM_TOKENS:-25600}"
    PACKING_BUFFER_SIZE="${PACKING_BUFFER_SIZE:-32}"
    ;;
esac

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a _LOCANY_CPT_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
if (( ${#_LOCANY_CPT_GPUS[@]} != 4 )); then
  echo "LocateAnything CPT currently requires exactly four visible GPUs; got ${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi

if [[ "${CPT_MODE}" == "smoke" ]]; then
  DATA_DIR="${DATA_DIR:-${WORKSPACE}/data/locany_cpt_v4_smoke}"
  RECIPE_NAME="${RECIPE_NAME:-locany_cpt_smoke.json}"
  MAX_STEPS="${MAX_STEPS:-2}"
  SAVE_STEPS="${SAVE_STEPS:-2}"
  SAVE_EVERY_N_HOURS="${SAVE_EVERY_N_HOURS:-0}"
  SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
  DEFAULT_WARMUP_STEPS=0
else
  DATA_DIR="${DATA_DIR:-${WORKSPACE}/data/locany_cpt_v4}"
  RECIPE_NAME="${RECIPE_NAME:-locany_cpt_train.json}"
  MAX_STEPS="${MAX_STEPS:-20000}"
  # Time-based callback performs periodic saves; keep the ordinary step trigger out of the way.
  SAVE_STEPS="${SAVE_STEPS:-1000000000}"
  SAVE_EVERY_N_HOURS="${SAVE_EVERY_N_HOURS:-12}"
  SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-20}"
  DEFAULT_WARMUP_STEPS=500
fi

META_PATH="${META_PATH:-${DATA_DIR}/recipe/${RECIPE_NAME}}"
MODEL_PATH="${MODEL_PATH:-${WORKSPACE}/hf_home/hub/models--nvidia--LocateAnything-3B/snapshots/c32291ca5e996f5a7a485845b4f57a233936bba0}"
OUTPUT_BASE="${OUTPUT_BASE:-${WORKSPACE}/gui_models}"
RUN_NAME="${RUN_NAME:-locany-3b-ui-cpt-v4-${MACHINE_TYPE}x4-${CPT_MODE}}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_BASE}/${RUN_NAME}}"
ENV_DIR="${ENV_DIR:-${WORKSPACE}/conda_envs/LocateAnything}"

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  echo "Python environment not found: ${ENV_DIR}" >&2
  exit 20
fi
if [[ ! -f "${META_PATH}" ]]; then
  echo "CPT recipe not found: ${META_PATH}" >&2
  echo "Prepare it first with scripts/prepare_locany_cpt.py; see README_LOCANY_CPT.md" >&2
  exit 21
fi

export PATH="${ENV_DIR}/bin:${PATH}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
"${ENV_DIR}/bin/python" "${PROJECT_ROOT}/scripts/validate_locany_cpt.py" \
  --recipe "${META_PATH}" \
  --records-per-dataset "${VALIDATE_RECORDS_PER_DATASET:-8}"

export PROJECT_ROOT WORKSPACE ENV_DIR MODEL_PATH DATA_DIR META_PATH OUTPUT_BASE RUN_NAME OUTPUT_DIR
export ATTN_IMPLEMENTATION MAX_SEQ_LENGTH MAX_NUM_TOKENS_PER_SAMPLE MAX_NUM_TOKENS
export PACKING_BUFFER_SIZE MAX_STEPS SAVE_STEPS SAVE_EVERY_N_HOURS SAVE_TOTAL_LIMIT
export GPUS=4 GPU_COUNT=4 GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
export LEARNING_RATE="${LEARNING_RATE:-5e-6}"
export WARMUP_STEPS="${WARMUP_STEPS:-${DEFAULT_WARMUP_STEPS}}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
export LOGGING_STEPS="${LOGGING_STEPS:-5}"
export SAMPLE_LOG_INTERVAL="${SAMPLE_LOG_INTERVAL:-20}"
export SAVE_STRATEGY=steps
export REPORT_TO="${REPORT_TO:-tensorboard}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-deepspeed_configs/zero_stage2_config.json}"
export FREEZE_LLM="${FREEZE_LLM:-False}"
export FREEZE_BACKBONE="${FREEZE_BACKBONE:-False}"
export FREEZE_MLP="${FREEZE_MLP:-False}"
export ENABLE_UI_RELATION=False
export BALANCE_UI_DEFECTS=False
export LOCANY_ENABLE_MILESTONE_COPIES=0
export WANDB_PROJECT="${WANDB_PROJECT:-locateanything-ui-cpt}"
export CACHE_ROOT="${CACHE_ROOT:-/tmp/${USER:-$(id -un)}_locany_cpt_cache}"

echo "===== LocateAnything UI CPT ====="
echo "machine                     : ${MACHINE_TYPE}"
echo "mode                        : ${CPT_MODE}"
echo "model                       : ${MODEL_PATH}"
echo "recipe                      : ${META_PATH}"
echo "output                      : ${OUTPUT_DIR}"
echo "attention                   : ${ATTN_IMPLEMENTATION}"
echo "max_seq_length              : ${MAX_SEQ_LENGTH}"
echo "max_num_tokens_per_sample   : ${MAX_NUM_TOKENS_PER_SAMPLE}"
echo "max_num_tokens_per_rank     : ${MAX_NUM_TOKENS}"
echo "gradient_accumulation_steps : ${GRADIENT_ACCUMULATION_STEPS}"
echo "max_steps                   : ${MAX_STEPS}"
echo "save_every_hours            : ${SAVE_EVERY_N_HOURS}"
echo "================================="

cd "${PROJECT_ROOT}"
exec bash "${PROJECT_ROOT}/shell/train_locany_ui_defect.sh"
