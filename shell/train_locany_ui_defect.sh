#!/usr/bin/env bash
set -Eeuo pipefail

# Generic full-SFT entrypoint. Machine-specific defaults live in:
#   shell/run_locany_ui_defect.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
# shellcheck source=shell/bash_error_report.sh
source "${SCRIPT_DIR}/bash_error_report.sh"
MODEL_PATH="${MODEL_PATH:-nvidia/LocateAnything-3B}"
DATA_VERSION="${DATA_VERSION:-v3}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/ui_defect_locany_${DATA_VERSION}}"
META_PATH="${META_PATH:-${DATA_DIR}/recipe/ui_defect_5class_train.json}"
UI5_USE_DETECTION_CROPS="${UI5_USE_DETECTION_CROPS:-0}"
UI5_CROP_AUDIT_DIR="${UI5_CROP_AUDIT_DIR:-}"
UI5_CROP_TRAIN_MODE="${UI5_CROP_TRAIN_MODE:-}"
UI5_CROP_META_PATH="${UI5_CROP_META_PATH:-}"
UI5_UI_SAMPLING_MODE="${UI5_UI_SAMPLING_MODE:-}"
CURRICULUM_MODE="${CURRICULUM_MODE:-none}"

OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_ROOT}/work_dirs}"
RUN_NAME="${RUN_NAME:-locateanything-3b-ui-defect-5class-full}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_BASE}/${RUN_NAME}}"
SAMPLE_LOG_INTERVAL="${SAMPLE_LOG_INTERVAL:-20}"

cd "${PROJECT_ROOT}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

if [[ -d "${MODEL_PATH}" ]]; then
  if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "[ERROR] Missing model config: ${MODEL_PATH}/config.json" >&2
    exit 1
  fi

  if [[ ! -f "${MODEL_PATH}/model.safetensors.index.json" ]] && \
     [[ ! -f "${MODEL_PATH}/model.safetensors" ]]; then
    echo "[ERROR] No model weights found under: ${MODEL_PATH}" >&2
    exit 1
  fi
elif [[ "${HF_HUB_OFFLINE}" == "1" || "${TRANSFORMERS_OFFLINE}" == "1" ]]; then
  echo "[ERROR] Local MODEL_PATH does not exist while offline mode is enabled:" >&2
  echo "        ${MODEL_PATH}" >&2
  echo "Download it first with:" >&2
  echo "  hf download nvidia/LocateAnything-3B --local-dir models/LocateAnything-3B" >&2
  exit 1
else
  echo "[INFO] MODEL_PATH is not a local directory; treating it as a Hub model id: ${MODEL_PATH}"
fi

# Resolve crop mode and the final meta before any environment preflight or
# torchrun.  Crop runs never silently fall back to the default full-image meta.
if [[ "${UI5_USE_DETECTION_CROPS}" != "0" && "${UI5_USE_DETECTION_CROPS}" != "1" ]]; then
  echo "[ERROR] UI5_USE_DETECTION_CROPS must be 0 or 1." >&2
  exit 1
fi
if [[ -z "${UI5_CROP_TRAIN_MODE}" ]]; then
  if [[ "${UI5_USE_DETECTION_CROPS}" == "1" ]]; then
    UI5_CROP_TRAIN_MODE="full_plus_crop"
  else
    UI5_CROP_TRAIN_MODE="full_only"
  fi
fi
if [[ "${UI5_CROP_TRAIN_MODE}" != "full_only" && "${UI5_CROP_TRAIN_MODE}" != "full_plus_crop" && "${UI5_CROP_TRAIN_MODE}" != "crop_only" ]]; then
  echo "[ERROR] UI5_CROP_TRAIN_MODE must be full_only, full_plus_crop, or crop_only." >&2
  exit 1
fi
if [[ "${UI5_USE_DETECTION_CROPS}" == "1" && "${UI5_CROP_TRAIN_MODE}" == "full_only" ]]; then
  echo "[ERROR] UI5_USE_DETECTION_CROPS=1 requires a crop-bearing train mode." >&2
  exit 1
fi
if [[ "${CURRICULUM_MODE}" == "scheduled" ]]; then
  if [[ -n "${UI5_UI_SAMPLING_MODE}" && "${UI5_UI_SAMPLING_MODE}" != "fixed_ratio" ]]; then
    echo "[ERROR] CURRICULUM_MODE=scheduled requires UI5_UI_SAMPLING_MODE=fixed_ratio." >&2
    exit 1
  fi
  UI5_UI_SAMPLING_MODE="fixed_ratio"
  BALANCE_UI_DEFECTS="False"
elif [[ "${CURRICULUM_MODE}" != "none" && "${CURRICULUM_MODE}" != "off" && "${CURRICULUM_MODE}" != "disabled" ]]; then
  echo "[ERROR] CURRICULUM_MODE must be scheduled or none." >&2
  exit 1
elif [[ -z "${UI5_UI_SAMPLING_MODE}" ]]; then
  if [[ "${UI5_CROP_TRAIN_MODE}" == "crop_only" ]]; then
    UI5_UI_SAMPLING_MODE="task_balanced_all_records"
  else
    UI5_UI_SAMPLING_MODE="fixed_ratio"
  fi
fi
if [[ "${UI5_UI_SAMPLING_MODE}" != "fixed_ratio" && \
      "${UI5_UI_SAMPLING_MODE}" != "task_balanced_all_records" && \
      "${UI5_UI_SAMPLING_MODE}" != "task_source_balanced_rotating" ]]; then
  echo "[ERROR] UI5_UI_SAMPLING_MODE must be fixed_ratio, task_balanced_all_records, or task_source_balanced_rotating." >&2
  exit 1
fi
if [[ "${CURRICULUM_MODE}" != "scheduled" && \
      "${UI5_CROP_TRAIN_MODE}" == "crop_only" && \
      "${UI5_UI_SAMPLING_MODE}" != "task_balanced_all_records" && \
      "${UI5_UI_SAMPLING_MODE}" != "task_source_balanced_rotating" ]]; then
  echo "[ERROR] crop_only requires an all-record sampling mode." >&2
  exit 1
fi
if [[ "${CURRICULUM_MODE}" == "scheduled" ]]; then
  # The curriculum builder emits a new three-entry recipe.  Its entries retain
  # their per-pool ui5_crop_recipe metadata, but the combined meta is not the
  # digest-bound single-recipe path produced by the older crop audit.
  :
elif [[ "${UI5_USE_DETECTION_CROPS}" == "1" || -n "${UI5_CROP_AUDIT_DIR}" ]]; then
  if [[ -z "${UI5_CROP_AUDIT_DIR}" ]]; then
    echo "[ERROR] UI5_USE_DETECTION_CROPS=1 requires UI5_CROP_AUDIT_DIR." >&2
    exit 1
  fi
  if [[ -z "${UI5_CROP_META_PATH}" ]]; then
    UI5_CROP_META_PATH="${UI5_CROP_AUDIT_DIR}/training_recipes/ui_defect_5class_train_${UI5_CROP_TRAIN_MODE}.json"
  fi
  if [[ ! -s "${UI5_CROP_META_PATH}" ]]; then
    echo "[ERROR] Audited crop recipe is missing or empty: ${UI5_CROP_META_PATH}" >&2
    exit 1
  fi
  UI5_AUDIT_PYTHON="${ENV_DIR:-}/bin/python"
  if [[ ! -x "${UI5_AUDIT_PYTHON}" ]]; then
    UI5_AUDIT_PYTHON="$(command -v python || true)"
  fi
  if [[ -z "${UI5_AUDIT_PYTHON}" ]]; then
    echo "[ERROR] Cannot find Python for crop marker validation." >&2
    exit 1
  fi
  "${UI5_AUDIT_PYTHON}" "${PROJECT_ROOT}/scripts/validate_ui5_crop_training_ready.py" \
    --audit-dir "${UI5_CROP_AUDIT_DIR}" \
    --recipe "${UI5_CROP_META_PATH}" || {
      echo "[ERROR] Crop training-ready validation failed; refusing to start training." >&2
      exit 1
    }
  META_PATH="${UI5_CROP_META_PATH}"
elif [[ -n "${UI5_CROP_META_PATH}" ]]; then
  echo "[ERROR] UI5_CROP_META_PATH requires UI5_CROP_AUDIT_DIR." >&2
  exit 1
fi
if [[ ! -s "${META_PATH}" ]]; then
  echo "[ERROR] Final META_PATH does not exist or is empty: ${META_PATH}" >&2
  exit 1
fi
export UI5_USE_DETECTION_CROPS UI5_CROP_AUDIT_DIR UI5_CROP_TRAIN_MODE UI5_UI_SAMPLING_MODE
export UI5_CROP_META_PATH META_PATH
export CURRICULUM_MODE BALANCE_UI_DEFECTS

DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-deepspeed_configs/zero_stage2_config.json}"
if [[ ! -f "${DEEPSPEED_CONFIG}" ]]; then
  echo "[ERROR] DeepSpeed config does not exist: ${DEEPSPEED_CONFIG}" >&2
  exit 1
fi

# Keep reusable downloads under HF_HOME; put lock-heavy compiler caches on local disk.
CACHE_ROOT="${CACHE_ROOT:-/tmp/${USER:-$(id -un)}_locany_cache}"
export TMPDIR="${TMPDIR:-${CACHE_ROOT}/tmp}"
export TEMP="${TEMP:-${TMPDIR}}"
export TMP="${TMP:-${TMPDIR}}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HUB_CACHE}}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${CACHE_ROOT}/hf_datasets}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${CACHE_ROOT}/torch_extensions}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${CACHE_ROOT}/torchinductor}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}/xdg}"
export WANDB_DIR="${WANDB_DIR:-${CACHE_ROOT}/wandb}"

mkdir -p \
  "${TMPDIR}" "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" \
  "${TRITON_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}" \
  "${TORCHINDUCTOR_CACHE_DIR}" "${XDG_CACHE_HOME}" \
  "${WANDB_DIR}" "${OUTPUT_DIR}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[ERROR] nvidia-smi was not found." >&2
  exit 1
fi

# CUDA_VISIBLE_DEVICES controls both torchrun rank count and the selected GPUs.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a _VISIBLE_GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
  GPUS="${GPUS:-${#_VISIBLE_GPU_ARRAY[@]}}"
else
  DETECTED_GPUS="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
  GPUS="${GPUS:-${DETECTED_GPUS}}"
  if (( GPUS < 1 )); then
    echo "[ERROR] No GPU detected." >&2
    exit 1
  fi
  export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((GPUS - 1)))"
fi

FIRST_GPU_ID="${CUDA_VISIBLE_DEVICES%%,*}"
GPU_NAME="$(nvidia-smi -i "${FIRST_GPU_ID}" --query-gpu=name --format=csv,noheader | sed -n '1p')"

# Safe fallbacks. The profile wrapper normally sets these explicitly.
if [[ -z "${ATTN_IMPLEMENTATION:-}" ]]; then
  case "${GPU_NAME}" in
    *H20*|*H100*|*H200*|*H800*|*B100*|*B200*)
      ATTN_IMPLEMENTATION="magi"
      ;;
    *)
      ATTN_IMPLEMENTATION="sdpa"
      ;;
  esac
fi

if [[ "${ATTN_IMPLEMENTATION}" == "magi" ]]; then
  MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-16384}"
  MAX_NUM_TOKENS_PER_SAMPLE="${MAX_NUM_TOKENS_PER_SAMPLE:-16384}"
  MAX_NUM_TOKENS="${MAX_NUM_TOKENS:-25600}"
  PACKING_BUFFER_SIZE="${PACKING_BUFFER_SIZE:-32}"
  DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
else
  MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"
  MAX_NUM_TOKENS_PER_SAMPLE="${MAX_NUM_TOKENS_PER_SAMPLE:-4096}"
  MAX_NUM_TOKENS="${MAX_NUM_TOKENS:-4096}"
  PACKING_BUFFER_SIZE="${PACKING_BUFFER_SIZE:-32}"
  DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
fi

if [[ "${ATTN_IMPLEMENTATION}" == "magi" && "${CHECK_MAGI_IMPORT:-1}" == "1" ]]; then
  python - <<'PY'
import importlib.util

if importlib.util.find_spec("magi_attention") is None:
    raise SystemExit(
        "[ERROR] ATTN_IMPLEMENTATION=magi, but magi_attention is not installed. "
        "Install MagiAttention 1.0.5 or select the SDPA profile."
    )
PY
fi

TARGET_GLOBAL_RANK_BATCH="${TARGET_GLOBAL_RANK_BATCH:-8}"
if [[ -z "${GRADIENT_ACCUMULATION_STEPS:-}" ]]; then
  if [[ "${GPU_COUNT:-8}" == "4" ]]; then
    GRADIENT_ACCUMULATION_STEPS=2
  else
    GRADIENT_ACCUMULATION_STEPS=1
  fi
fi

MAX_STEPS="${MAX_STEPS:-${TOTAL_STEPS:-25000}}"
TOTAL_STEPS="${TOTAL_STEPS:-${MAX_STEPS}}"
LLM_LRS="${LLM_LRS:-1e-6,7e-7,5e-7}"
if [[ "${CURRICULUM_MODE}" == "scheduled" ]]; then
  SCHEDULE_INITIAL_LR="${LLM_LRS%%,*}"
  if [[ -n "${LEARNING_RATE:-}" && "${LEARNING_RATE}" != "${SCHEDULE_INITIAL_LR}" ]]; then
    echo "[WARN] Ignoring LEARNING_RATE=${LEARNING_RATE}; scheduled LLM_LRS starts at ${SCHEDULE_INITIAL_LR}."
  fi
  LEARNING_RATE="${SCHEDULE_INITIAL_LR}"
  WARMUP_STEPS=0
  LR_SCHEDULER_TYPE="constant"
else
  LEARNING_RATE="${LEARNING_RATE:-2e-5}"
  WARMUP_STEPS="${WARMUP_STEPS:-500}"
  LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
fi
SEED="${SEED:-42}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
SAVE_STEPS="${SAVE_STEPS:-500}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
REPORT_TO="${REPORT_TO:-tensorboard}"
OVERWRITE_OUTPUT_DIR="${OVERWRITE_OUTPUT_DIR:-False}"
BF16="${BF16:-True}"
GRAD_CHECKPOINT="${GRAD_CHECKPOINT:-True}"
FREEZE_LLM="${FREEZE_LLM:-False}"
FREEZE_MLP="${FREEZE_MLP:-False}"
FREEZE_BACKBONE="${FREEZE_BACKBONE:-False}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-$(python - <<'PY'
import socket

with socket.socket() as sock:
    sock.bind(("", 0))
    print(sock.getsockname()[1])
PY
)}"
export TOTAL_STEPS LLM_LRS SEED

RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
ROLLING_CHECKPOINT_PATH=""
if [[ -n "${ROLLING_CHECKPOINT_DIR:-}" ]]; then
  if [[ "${ROLLING_CHECKPOINT_DIR}" = /* ]]; then
    ROLLING_CHECKPOINT_PATH="${ROLLING_CHECKPOINT_DIR}"
  else
    ROLLING_CHECKPOINT_PATH="${OUTPUT_DIR}/${ROLLING_CHECKPOINT_DIR}"
  fi
fi
if [[ -z "${RESUME_FROM_CHECKPOINT}" && -n "${ROLLING_CHECKPOINT_PATH}" && -d "${ROLLING_CHECKPOINT_PATH}" ]]; then
  RESUME_FROM_CHECKPOINT="${ROLLING_CHECKPOINT_PATH}"
fi
TRAIN_RESUME_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  if [[ ! -d "${RESUME_FROM_CHECKPOINT}" ]]; then
    echo "[ERROR] RESUME_FROM_CHECKPOINT does not exist: ${RESUME_FROM_CHECKPOINT}" >&2
    exit 1
  fi
  TRAIN_RESUME_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

export WANDB_PROJECT="${WANDB_PROJECT:-locateanything-ui-defect}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_NAME}}"

CURRENT_TIME="$(date +"%Y%m%d-%H%M%S")"
LOG_FILE="${OUTPUT_DIR}/train-${CURRENT_TIME}.log"
GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-2}"
GPU_MONITOR_FILE="${OUTPUT_DIR}/gpu-memory-${CURRENT_TIME}.csv"
GPU_MONITOR_PID=""

start_gpu_monitor() {
  if [[ "${ENABLE_GPU_MONITOR:-1}" != "1" ]]; then
    return
  fi

  echo "timestamp,gpu_index,gpu_name,memory_used_mib,memory_total_mib,gpu_util_percent" \
    > "${GPU_MONITOR_FILE}"

  (
    while true; do
      timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
      nvidia-smi \
        --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits |
      awk -F',' -v ts="${timestamp}" '
        {
          for (i = 1; i <= NF; i++) {
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", $i)
          }
          printf "%s,%s,%s,%s,%s,%s\n", ts, $1, $2, $3, $4, $5
        }
      ' >> "${GPU_MONITOR_FILE}"
      sleep "${GPU_MONITOR_INTERVAL}"
    done
  ) &

  GPU_MONITOR_PID=$!
  echo "[GPU Monitor] Started: PID=${GPU_MONITOR_PID}, interval=${GPU_MONITOR_INTERVAL}s"
  echo "[GPU Monitor] Output: ${GPU_MONITOR_FILE}"
}

stop_gpu_monitor() {
  if [[ -n "${GPU_MONITOR_PID:-}" ]]; then
    kill "${GPU_MONITOR_PID}" 2>/dev/null || true
    wait "${GPU_MONITOR_PID}" 2>/dev/null || true
    GPU_MONITOR_PID=""
  fi
}

print_gpu_memory_summary() {
  echo
  echo "============================================================"
  echo "GPU MEMORY SUMMARY"
  echo "============================================================"

  if [[ ! -s "${GPU_MONITOR_FILE}" ]]; then
    echo "[WARN] No GPU monitoring data found: ${GPU_MONITOR_FILE}"
    echo "============================================================"
    return
  fi

  python - "${GPU_MONITOR_FILE}" <<'PY'
import csv
import sys
from collections import defaultdict

csv_path = sys.argv[1]
records = defaultdict(list)

with open(csv_path, "r", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        try:
            gpu = int(row["gpu_index"])
            records[gpu].append(
                {
                    "timestamp": row["timestamp"],
                    "name": row["gpu_name"],
                    "used": int(float(row["memory_used_mib"])),
                    "total": int(float(row["memory_total_mib"])),
                    "util": int(float(row["gpu_util_percent"])),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue

if not records:
    print("[WARN] GPU monitoring CSV contains no valid samples.")
    raise SystemExit(0)

for gpu in sorted(records):
    values = records[gpu]
    peak = max(values, key=lambda item: item["used"])
    last = values[-1]
    avg_used = sum(item["used"] for item in values) / len(values)
    avg_util = sum(item["util"] for item in values) / len(values)
    print(f"GPU {gpu}: {peak['name']}")
    print(
        f"  Peak memory : {peak['used']:,} / {peak['total']:,} MiB "
        f"({100.0 * peak['used'] / peak['total']:.1f}%)"
    )
    print(f"  Peak time   : {peak['timestamp']}")
    print(
        f"  Last sample : {last['used']:,} / {last['total']:,} MiB "
        f"({100.0 * last['used'] / last['total']:.1f}%)"
    )
    print(f"  Average mem : {avg_used:,.0f} MiB")
    print(f"  Average util: {avg_util:.1f}%")
    print(f"  Samples     : {len(values):,}")

print(f"Monitoring file: {csv_path}")
PY

  echo "============================================================"
}

echo "============================================================"
echo "PROJECT_ROOT                  : ${PROJECT_ROOT}"
echo "MODEL_PATH                    : ${MODEL_PATH}"
echo "META_PATH                     : ${META_PATH}"
echo "UI5_USE_DETECTION_CROPS       : ${UI5_USE_DETECTION_CROPS}"
echo "UI5_CROP_AUDIT_DIR            : ${UI5_CROP_AUDIT_DIR:-<none>}"
echo "UI5_CROP_TRAIN_MODE           : ${UI5_CROP_TRAIN_MODE}"
echo "UI5_UI_SAMPLING_MODE          : ${UI5_UI_SAMPLING_MODE}"
echo "UI5_CROP_META_PATH            : ${UI5_CROP_META_PATH:-<none>}"
echo "OUTPUT_DIR                    : ${OUTPUT_DIR}"
echo "GPU_NAME                      : ${GPU_NAME}"
echo "CUDA_VISIBLE_DEVICES          : ${CUDA_VISIBLE_DEVICES}"
echo "GPUS                          : ${GPUS}"
echo "ATTN_IMPLEMENTATION           : ${ATTN_IMPLEMENTATION}"
echo "MAX_SEQ_LENGTH                : ${MAX_SEQ_LENGTH}"
echo "MAX_NUM_TOKENS_PER_SAMPLE     : ${MAX_NUM_TOKENS_PER_SAMPLE}"
echo "MAX_NUM_TOKENS                : ${MAX_NUM_TOKENS}"
echo "BALANCE_UI_DEFECTS             : ${BALANCE_UI_DEFECTS:-True}"
if [[ "${UI5_UI_SAMPLING_MODE}" == "task_balanced_all_records" ]]; then
  echo "UI_RECORDS_PER_CLASS           : inactive (all legal records retained)"
  echo "UI_NEGATIVE:POSITIVE           : natural recipe distribution"
elif [[ "${UI5_UI_SAMPLING_MODE}" == "task_source_balanced_rotating" ]]; then
  echo "UI_RECORDS_PER_CLASS           : inactive (all legal records retained in active pool)"
  echo "UI_NEGATIVE:POSITIVE           : ${UI_NEGATIVE_TO_POSITIVE_RATIO:-2.0}:1 effective rotating draws"
  echo "UI_SOURCE_GROUP_WEIGHTING      : uniform within task/polarity; crops rotate before repeat"
else
  echo "UI_RECORDS_PER_CLASS           : ${UI_RECORDS_PER_CLASS:-17604}"
  echo "UI_NEGATIVE:POSITIVE           : ${UI_NEGATIVE_TO_POSITIVE_RATIO:-2.0}:1"
fi
echo "PACKING_BUFFER_SIZE           : ${PACKING_BUFFER_SIZE}"
echo "GRADIENT_ACCUMULATION_STEPS   : ${GRADIENT_ACCUMULATION_STEPS}"
echo "RELATION_GATE_LOSS_WEIGHT     : ${RELATION_GATE_LOSS_WEIGHT:-1.0}"
echo "RELATION_SLOT_GATE_LOSS_WEIGHT: ${RELATION_SLOT_GATE_LOSS_WEIGHT:-0.1}"
echo "RELATION_ATTENTION_LOSS_WEIGHT: ${RELATION_ATTENTION_LOSS_WEIGHT:-0.1}"
echo "LEARNING_RATE                 : ${LEARNING_RATE}"
echo "LLM_LRS                      : ${LLM_LRS}"
echo "CURRICULUM_MODE               : ${CURRICULUM_MODE}"
echo "TOTAL_STEPS                  : ${TOTAL_STEPS}"
echo "SEED                         : ${SEED}"
echo "RESUME_FROM_CHECKPOINT       : ${RESUME_FROM_CHECKPOINT:-<none>}"
echo "MAX_STEPS                     : ${MAX_STEPS}"
echo "SAVE_EVERY_N_HOURS            : ${SAVE_EVERY_N_HOURS:-0}"
echo "FREEZE_LLM                    : ${FREEZE_LLM}"
echo "FREEZE_BACKBONE               : ${FREEZE_BACKBONE}"
echo "FREEZE_MLP                    : ${FREEZE_MLP}"
echo "DEEPSPEED_CONFIG              : ${DEEPSPEED_CONFIG}"
echo "REPORT_TO                     : ${REPORT_TO}"
echo "MASTER_PORT                   : ${MASTER_PORT}"
echo "LOG_FILE                      : ${LOG_FILE}"
echo "============================================================"

ENVIRONMENT_AUDIT_COMMAND=(
  "${ENV_DIR}/bin/python" "${PROJECT_ROOT}/scripts/check_locany_environment.py"
  --output-dir "${OUTPUT_DIR}"
)
if [[ "${ALLOW_RUNTIME_ENVIRONMENT_CHANGE:-0}" == "1" ]]; then
  ENVIRONMENT_AUDIT_COMMAND+=(--allow-change)
fi
"${ENVIRONMENT_AUDIT_COMMAND[@]}" --phase pre

cleanup_gpu_monitor() {
  stop_gpu_monitor
}
trap cleanup_gpu_monitor EXIT INT TERM

start_gpu_monitor

export LAUNCHER="${LAUNCHER:-pytorch}"

TRAIN_PIPESTATUS=()
if torchrun \
  --nnodes="${NNODES:-1}" \
  --node_rank="${NODE_RANK:-0}" \
  --master_addr="${MASTER_ADDR}" \
  --nproc_per_node="${GPUS}" \
  --master_port="${MASTER_PORT}" \
  eaglevl/train/locany_finetune_magi_stream.py \
  --model_name_or_path "${MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --meta_path "${META_PATH}" \
  --overwrite_output_dir "${OVERWRITE_OUTPUT_DIR}" \
  --block_size 6 \
  --attn_implementation "${ATTN_IMPLEMENTATION}" \
  --causal_attn False \
  --freeze_llm "${FREEZE_LLM}" \
  --freeze_mlp "${FREEZE_MLP}" \
  --freeze_backbone "${FREEZE_BACKBONE}" \
  --vision_select_layer -1 \
  --mlp_connector_layers 2 \
  --enable_ui_relation "${ENABLE_UI_RELATION:-True}" \
  --relation_detail_hidden_size "${RELATION_DETAIL_HIDDEN_SIZE:-256}" \
  --relation_num_slots "${RELATION_NUM_SLOTS:-8}" \
  --relation_adapter_bottleneck "${RELATION_ADAPTER_BOTTLENECK:-64}" \
  --relation_gate_loss_weight "${RELATION_GATE_LOSS_WEIGHT:-1.0}" \
  --relation_slot_gate_loss_weight "${RELATION_SLOT_GATE_LOSS_WEIGHT:-0.1}" \
  --relation_attention_loss_weight "${RELATION_ATTENTION_LOSS_WEIGHT:-0.1}" \
  --relation_gate_threshold "${RELATION_GATE_THRESHOLD:-0.5}" \
  --relation_focal_beta "${RELATION_FOCAL_BETA:-0.999}" \
  --relation_focal_gamma "${RELATION_FOCAL_GAMMA:-2.0}" \
  --balance_ui_defects "${BALANCE_UI_DEFECTS:-True}" \
  --ui_records_per_class "${UI_RECORDS_PER_CLASS:-17604}" \
  --ui_negative_to_positive_ratio "${UI_NEGATIVE_TO_POSITIVE_RATIO:-2.0}" \
  --ui_sampling_mode "${UI5_UI_SAMPLING_MODE}" \
  --bf16 "${BF16}" \
  --max_steps "${MAX_STEPS}" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --learning_rate "${LEARNING_RATE}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
  --max_grad_norm 1.0 \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --packing_buffer_size "${PACKING_BUFFER_SIZE}" \
  --max_seq_length "${MAX_SEQ_LENGTH}" \
  --max_num_tokens_per_sample "${MAX_NUM_TOKENS_PER_SAMPLE}" \
  --max_num_tokens "${MAX_NUM_TOKENS}" \
  --sample_log_interval "${SAMPLE_LOG_INTERVAL}" \
  --grad_checkpoint "${GRAD_CHECKPOINT}" \
  --group_by_length False \
  --save_strategy "${SAVE_STRATEGY}" \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --logging_steps "${LOGGING_STEPS}" \
  --do_train True \
  --deepspeed "${DEEPSPEED_CONFIG}" \
  --report_to "${REPORT_TO}" \
  --run_name "${RUN_NAME}" \
  --save_every_n_hours "${SAVE_EVERY_N_HOURS:-0}" \
  --seed "${SEED}" \
  "${TRAIN_RESUME_ARGS[@]}" \
  2>&1 | tee -a "${LOG_FILE}"; then
  TRAIN_PIPESTATUS=("${PIPESTATUS[@]}")
else
  TRAIN_PIPESTATUS=("${PIPESTATUS[@]}")
fi
TRAIN_EXIT_CODE="${TRAIN_PIPESTATUS[0]}"
stop_gpu_monitor

ENVIRONMENT_AUDIT_EXIT_CODE=0
if "${ENVIRONMENT_AUDIT_COMMAND[@]}" --phase post; then
  :
else
  ENVIRONMENT_AUDIT_EXIT_CODE=$?
  echo "[LOCANY ENVIRONMENT ERROR] post-training environment audit failed with exit_code=${ENVIRONMENT_AUDIT_EXIT_CODE}" >&2
  if (( TRAIN_EXIT_CODE == 0 )); then
    TRAIN_EXIT_CODE="${ENVIRONMENT_AUDIT_EXIT_CODE}"
  fi
fi

{
  print_gpu_memory_summary
  echo
  echo "TRAIN_EXIT_CODE: ${TRAIN_EXIT_CODE}"
  echo "ENVIRONMENT_AUDIT_EXIT_CODE: ${ENVIRONMENT_AUDIT_EXIT_CODE}"
  if (( TRAIN_EXIT_CODE == 0 )); then
    echo "TRAIN_STATUS: SUCCESS"
  else
    echo "TRAIN_STATUS: FAILED"
  fi
} 2>&1 | tee -a "${LOG_FILE}"

if (( TRAIN_EXIT_CODE != 0 )); then
  echo "[LOCANY FATAL] torchrun failed with exit_code=${TRAIN_EXIT_CODE}; full_log=${LOG_FILE}" >&2
  echo "[LOCANY FATAL] Last ${ERROR_TAIL_LINES:-200} training log lines:" >&2
  tail -n "${ERROR_TAIL_LINES:-200}" "${LOG_FILE}" >&2
fi
exit "${TRAIN_EXIT_CODE}"
