#!/usr/bin/env bash
set -Eeuo pipefail

# One switch controls the machine-specific defaults:
#   bash shell/run_locany_ui_defect.sh 4090
#   bash shell/run_locany_ui_defect.sh h20
#   bash shell/run_locany_ui_defect.sh a800
# Every value remains overrideable through an environment variable.

PROFILE="${1:-${LOCANY_PROFILE:-4090}}"
if (( $# > 1 )); then
  echo "Usage: $0 [4090|h20|a800]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DATA_VERSION="${DATA_VERSION:-v3}"

count_visible_gpus() {
  local value="$1"
  local -a ids
  IFS=',' read -r -a ids <<< "${value}"
  echo "${#ids[@]}"
}

case "${PROFILE}" in
  4090|local_4090|smoke)
    PROFILE="4090"
    VERSION="${VERSION:-smoke}"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    GPUS="${GPUS:-$(count_visible_gpus "${CUDA_VISIBLE_DEVICES}")}"
    OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_ROOT}/work_dirs}"
    HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}"
    MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/LocateAnything-3B}"
    META_PATH="${META_PATH:-${PROJECT_ROOT}/samples/ui_defect_locany_smoke_real/recipe/ui_defect_5class_train.json}"
    RUN_NAME="${RUN_NAME:-locany-3b-ui5-4090-smoke-$(date +%Y%m%d-%H%M%S)}"

    ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
    MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"
    MAX_NUM_TOKENS_PER_SAMPLE="${MAX_NUM_TOKENS_PER_SAMPLE:-4096}"
    MAX_NUM_TOKENS="${MAX_NUM_TOKENS:-4096}"
    PACKING_BUFFER_SIZE="${PACKING_BUFFER_SIZE:-4}"
    DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
    GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"

    MAX_STEPS="${MAX_STEPS:-2}"
    WARMUP_STEPS="${WARMUP_STEPS:-0}"
    LEARNING_RATE="${LEARNING_RATE:-2e-5}"
    LOGGING_STEPS="${LOGGING_STEPS:-1}"
    SAVE_STEPS="${SAVE_STEPS:-1000}"
    SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"
    SAMPLE_LOG_INTERVAL="${SAMPLE_LOG_INTERVAL:-1}"
    REPORT_TO="${REPORT_TO:-none}"
    UI_RECORDS_PER_CLASS="${UI_RECORDS_PER_CLASS:-2}"

    # Keep the exact full-SFT entrypoint, but make a 24 GB smoke run practical.
    FREEZE_LLM="${FREEZE_LLM:-True}"
    FREEZE_BACKBONE="${FREEZE_BACKBONE:-True}"
    FREEZE_MLP="${FREEZE_MLP:-False}"
    DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-deepspeed_configs/zero_stage2_config.json}"
    CHECK_MAGI_IMPORT="${CHECK_MAGI_IMPORT:-0}"
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
    ;;

  h20|internal_h20)
    PROFILE="h20"
    VERSION="${VERSION:-v3_h20x4}"
    ROOT_PATH="${ROOT_PATH:-/mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace}"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
    GPUS="${GPUS:-$(count_visible_gpus "${CUDA_VISIBLE_DEVICES}")}"
    OUTPUT_BASE="${OUTPUT_BASE:-${ROOT_PATH}/gui_models}"
    HF_HOME="${HF_HOME:-${ROOT_PATH}/cache/huggingface}"
    MODEL_PATH="${MODEL_PATH:-${ROOT_PATH}/models/LocateAnything-3B}"
    META_PATH="${META_PATH:-${PROJECT_ROOT}/data/ui_defect_locany_${DATA_VERSION}/recipe/ui_defect_5class_train.json}"
    RUN_NAME="${RUN_NAME:-locany-3b-ui5-h20-full-${VERSION}-en}"

    ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-magi}"
    MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-8192}"
    MAX_NUM_TOKENS_PER_SAMPLE="${MAX_NUM_TOKENS_PER_SAMPLE:-8192}"
    MAX_NUM_TOKENS="${MAX_NUM_TOKENS:-25600}"
    PACKING_BUFFER_SIZE="${PACKING_BUFFER_SIZE:-32}"
    DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
    TARGET_GLOBAL_RANK_BATCH="${TARGET_GLOBAL_RANK_BATCH:-8}"

    MAX_STEPS="${MAX_STEPS:-25000}"
    WARMUP_STEPS="${WARMUP_STEPS:-500}"
    LEARNING_RATE="${LEARNING_RATE:-2e-5}"
    LOGGING_STEPS="${LOGGING_STEPS:-5}"
    SAVE_STEPS="${SAVE_STEPS:-2000}"
    SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1000}"
    SAMPLE_LOG_INTERVAL="${SAMPLE_LOG_INTERVAL:-5}"
    REPORT_TO="${REPORT_TO:-tensorboard}"
    UI_RECORDS_PER_CLASS="${UI_RECORDS_PER_CLASS:-17604}"

    FREEZE_LLM="${FREEZE_LLM:-False}"
    FREEZE_BACKBONE="${FREEZE_BACKBONE:-False}"
    FREEZE_MLP="${FREEZE_MLP:-False}"
    DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-deepspeed_configs/zero_stage2_config.json}"
    CHECK_MAGI_IMPORT="${CHECK_MAGI_IMPORT:-1}"
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
    ;;

  a800|internal_a800)
    PROFILE="a800"
    VERSION="${VERSION:-v3}"
    ROOT_PATH="${ROOT_PATH:-/mnt/bn/intelligent-service-yg/logging/sicheng_workspace}"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
    GPUS="${GPUS:-$(count_visible_gpus "${CUDA_VISIBLE_DEVICES}")}"
    OUTPUT_BASE="${OUTPUT_BASE:-${ROOT_PATH}/gui_models}"
    HF_HOME="${HF_HOME:-${ROOT_PATH}/cache/huggingface}"
    MODEL_PATH="${MODEL_PATH:-${ROOT_PATH}/models/LocateAnything-3B}"
    META_PATH="${META_PATH:-${PROJECT_ROOT}/data/ui_defect_locany_${DATA_VERSION}/recipe/ui_defect_5class_train.json}"
    RUN_NAME="${RUN_NAME:-locany-3b-ui5-a800-full-${VERSION}-en}"

    ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
    # Keep packed batches at 25.6K while allowing each raw sample 8K context.
    MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-8192}"
    MAX_NUM_TOKENS_PER_SAMPLE="${MAX_NUM_TOKENS_PER_SAMPLE:-8192}"
    MAX_NUM_TOKENS="${MAX_NUM_TOKENS:-25600}"
    PACKING_BUFFER_SIZE="${PACKING_BUFFER_SIZE:-32}"
    DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
    TARGET_GLOBAL_RANK_BATCH="${TARGET_GLOBAL_RANK_BATCH:-8}"

    MAX_STEPS="${MAX_STEPS:-25000}"
    WARMUP_STEPS="${WARMUP_STEPS:-500}"
    LEARNING_RATE="${LEARNING_RATE:-2e-5}"
    LOGGING_STEPS="${LOGGING_STEPS:-1}"
    SAVE_STEPS="${SAVE_STEPS:-2000}"
    SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1000}"
    SAMPLE_LOG_INTERVAL="${SAMPLE_LOG_INTERVAL:-5}"
    REPORT_TO="${REPORT_TO:-tensorboard}"
    UI_RECORDS_PER_CLASS="${UI_RECORDS_PER_CLASS:-17604}"

    FREEZE_LLM="${FREEZE_LLM:-False}"
    FREEZE_BACKBONE="${FREEZE_BACKBONE:-False}"
    FREEZE_MLP="${FREEZE_MLP:-False}"
    DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-deepspeed_configs/zero_stage2_config.json}"
    CHECK_MAGI_IMPORT="${CHECK_MAGI_IMPORT:-0}"
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
    ;;

  *)
    echo "[ERROR] Unknown profile: ${PROFILE}" >&2
    echo "Choose one of: 4090, h20, a800" >&2
    exit 2
    ;;
esac

BALANCE_UI_DEFECTS="${BALANCE_UI_DEFECTS:-True}"
UI_NEGATIVE_TO_POSITIVE_RATIO="${UI_NEGATIVE_TO_POSITIVE_RATIO:-2.0}"
ENABLE_UI_RELATION="${ENABLE_UI_RELATION:-True}"
RELATION_DETAIL_HIDDEN_SIZE="${RELATION_DETAIL_HIDDEN_SIZE:-256}"
RELATION_NUM_SLOTS="${RELATION_NUM_SLOTS:-8}"
RELATION_ADAPTER_BOTTLENECK="${RELATION_ADAPTER_BOTTLENECK:-64}"
RELATION_GATE_LOSS_WEIGHT="${RELATION_GATE_LOSS_WEIGHT:-1.0}"
RELATION_ATTENTION_LOSS_WEIGHT="${RELATION_ATTENTION_LOSS_WEIGHT:-0.1}"

export \
  PROFILE PROJECT_ROOT DATA_VERSION VERSION \
  CUDA_VISIBLE_DEVICES GPUS OUTPUT_BASE HF_HOME MODEL_PATH META_PATH RUN_NAME \
  ATTN_IMPLEMENTATION MAX_SEQ_LENGTH MAX_NUM_TOKENS_PER_SAMPLE MAX_NUM_TOKENS \
  PACKING_BUFFER_SIZE DATALOADER_NUM_WORKERS \
  MAX_STEPS WARMUP_STEPS LEARNING_RATE LOGGING_STEPS SAVE_STEPS \
  SAVE_TOTAL_LIMIT SAMPLE_LOG_INTERVAL REPORT_TO \
  BALANCE_UI_DEFECTS UI_RECORDS_PER_CLASS UI_NEGATIVE_TO_POSITIVE_RATIO \
  ENABLE_UI_RELATION RELATION_DETAIL_HIDDEN_SIZE RELATION_NUM_SLOTS \
  RELATION_ADAPTER_BOTTLENECK RELATION_GATE_LOSS_WEIGHT RELATION_ATTENTION_LOSS_WEIGHT \
  FREEZE_LLM FREEZE_BACKBONE FREEZE_MLP DEEPSPEED_CONFIG CHECK_MAGI_IMPORT \
  HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

if [[ -n "${TARGET_GLOBAL_RANK_BATCH:-}" ]]; then
  export TARGET_GLOBAL_RANK_BATCH
fi
if [[ -n "${GRADIENT_ACCUMULATION_STEPS:-}" ]]; then
  export GRADIENT_ACCUMULATION_STEPS
fi

echo "============================================================"
echo "LOCATEANYTHING UI DEFECT PROFILE"
echo "============================================================"
echo "PROFILE                       : ${PROFILE}"
echo "PROJECT_ROOT                  : ${PROJECT_ROOT}"
echo "MODEL_PATH                    : ${MODEL_PATH}"
echo "META_PATH                     : ${META_PATH}"
echo "OUTPUT_BASE                   : ${OUTPUT_BASE}"
echo "RUN_NAME                      : ${RUN_NAME}"
echo "CUDA_VISIBLE_DEVICES          : ${CUDA_VISIBLE_DEVICES}"
echo "GPUS                          : ${GPUS}"
echo "ATTN_IMPLEMENTATION           : ${ATTN_IMPLEMENTATION}"
echo "MAX_SEQ_LENGTH                : ${MAX_SEQ_LENGTH}"
echo "MAX_NUM_TOKENS_PER_SAMPLE     : ${MAX_NUM_TOKENS_PER_SAMPLE}"
echo "MAX_NUM_TOKENS                : ${MAX_NUM_TOKENS}"
echo "UI_RECORDS_PER_CLASS           : ${UI_RECORDS_PER_CLASS}"
echo "UI_NEGATIVE:POSITIVE           : ${UI_NEGATIVE_TO_POSITIVE_RATIO}:1"
echo "RELATION_NUM_SLOTS             : ${RELATION_NUM_SLOTS}"
echo "PACKING_BUFFER_SIZE           : ${PACKING_BUFFER_SIZE}"
echo "MAX_STEPS                     : ${MAX_STEPS}"
echo "FREEZE_LLM                    : ${FREEZE_LLM}"
echo "FREEZE_BACKBONE               : ${FREEZE_BACKBONE}"
echo "FREEZE_MLP                    : ${FREEZE_MLP}"
echo "DEEPSPEED_CONFIG              : ${DEEPSPEED_CONFIG}"
echo "============================================================"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1: configuration resolved; training was not started."
  exit 0
fi

if [[ "${PROFILE}" == "4090" ]]; then
  python "${PROJECT_ROOT}/scripts/validate_ui_defect_locany_sample.py" \
    --project-root "${PROJECT_ROOT}" \
    --recipe "${META_PATH}"
fi

exec bash "${PROJECT_ROOT}/shell/train_locany_ui_defect.sh"
