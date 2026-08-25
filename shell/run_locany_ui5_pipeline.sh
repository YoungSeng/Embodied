#!/usr/bin/env bash
set -Eeuo pipefail

# LocateAnything UI5 v4 training/evaluation state machine.
# Machine paths and defaults are resolved exclusively from configs/locany_ui5_machines.json.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
# shellcheck source=shell/bash_error_report.sh
source "${SCRIPT_DIR}/bash_error_report.sh"
PIPELINE_PYTHON="${ENV_DIR:-}/bin/python"
if [[ ! -x "${PIPELINE_PYTHON}" ]]; then
  PIPELINE_PYTHON="$(command -v python || true)"
fi
if [[ -z "${PIPELINE_PYTHON}" || ! -x "${PIPELINE_PYTHON}" ]]; then
  echo "[ERROR] Cannot find Python. ENV_DIR=${ENV_DIR:-<unset>}" >&2
  exit 20
fi

RESOLVED_SHELL="$(mktemp "${TMPDIR:-/tmp}/locany-ui5-config.XXXXXX")"
cleanup_resolved_config() {
  rm -f -- "${RESOLVED_SHELL}"
}
trap cleanup_resolved_config EXIT
"${PIPELINE_PYTHON}" "${PROJECT_ROOT}/scripts/resolve_locany_ui5_config.py" \
  --format shell > "${RESOLVED_SHELL}"
# shellcheck disable=SC1090
source "${RESOLVED_SHELL}"

export PATH="${ENV_DIR}/bin:${PATH}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export GPUS="${GPU_COUNT}"
export MODEL_PATH BASE_MODEL META_PATH OUTPUT_BASE OUTPUT_DIR RUN_NAME
export ATTN_IMPLEMENTATION MAX_SEQ_LENGTH MAX_NUM_TOKENS_PER_SAMPLE MAX_NUM_TOKENS
export MAX_STEPS WARMUP_STEPS LEARNING_RATE SAVE_STEPS

JOB_ID="${ARNOLD_TRIAL_ID:-${ARNOLD_JOB_ID:-manual-$$}}"
LOCAL_RUNTIME_DIR="/tmp/locany-ui5-${JOB_ID}"
SHARED_RUNTIME_DIR="${WORKSPACE}/runtime/${JOB_ID}"
mkdir -p \
  "${OUTPUT_DIR}" \
  "${OUTPUT_DIR}/evaluation" \
  "${HF_HOME}/hub" \
  "${WORKSPACE}/cache/huggingface/datasets" \
  "${WORKSPACE}/cache/pip" \
  "${WORKSPACE}/cache/torch" \
  "${WORKSPACE}/cache/xdg" \
  "${SHARED_RUNTIME_DIR}/torch_extensions" \
  "${LOCAL_RUNTIME_DIR}/tmp" \
  "${LOCAL_RUNTIME_DIR}/pycache" \
  "${LOCAL_RUNTIME_DIR}/triton" \
  "${LOCAL_RUNTIME_DIR}/torchinductor" \
  "${LOCAL_RUNTIME_DIR}/cuda_cache"

# Preserve everything needed for post-mortem debugging on the shared volume. Normal
# stdout/stderr goes both to Merlin and the combined log; xtrace is intentionally kept
# in a separate file so the main job log remains readable.
PIPELINE_RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
PIPELINE_LOG_DIR="${OUTPUT_DIR}/logs"
PIPELINE_LOG="${PIPELINE_LOG_DIR}/pipeline-${PIPELINE_RUN_STAMP}-${BASHPID}.log"
PIPELINE_TRACE_LOG="${PIPELINE_LOG_DIR}/pipeline-${PIPELINE_RUN_STAMP}-${BASHPID}.trace.log"
mkdir -p "${PIPELINE_LOG_DIR}"
touch "${PIPELINE_LOG}" "${PIPELINE_TRACE_LOG}"
export PIPELINE_LOG PIPELINE_TRACE_LOG
exec > >(tee -a "${PIPELINE_LOG}") 2>&1
exec 19>>"${PIPELINE_TRACE_LOG}"
export BASH_XTRACEFD=19
PS4='+ ${EPOCHREALTIME:-?} ${BASH_SOURCE[0]:-$0}:${LINENO}:${FUNCNAME[0]:-main}: '
export PS4
if [[ "${PIPELINE_TRACE:-1}" == "1" ]]; then
  set -x
fi

export TMPDIR="${LOCAL_RUNTIME_DIR}/tmp"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"
export PYTHONPYCACHEPREFIX="${LOCAL_RUNTIME_DIR}/pycache"
export TRITON_CACHE_DIR="${LOCAL_RUNTIME_DIR}/triton"
export TORCHINDUCTOR_CACHE_DIR="${LOCAL_RUNTIME_DIR}/torchinductor"
export CUDA_CACHE_PATH="${LOCAL_RUNTIME_DIR}/cuda_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"
export HF_DATASETS_CACHE="${WORKSPACE}/cache/huggingface/datasets"
export PIP_CACHE_DIR="${WORKSPACE}/cache/pip"
export TORCH_HOME="${WORKSPACE}/cache/torch"
export XDG_CACHE_HOME="${WORKSPACE}/cache/xdg"
export TORCH_EXTENSIONS_DIR="${SHARED_RUNTIME_DIR}/torch_extensions"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_TELEMETRY=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# Training defaults shared by direct and segmented modes. Explicit user values win.
export PACKING_BUFFER_SIZE="${PACKING_BUFFER_SIZE:-32}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
export TARGET_GLOBAL_RANK_BATCH="${TARGET_GLOBAL_RANK_BATCH:-8}"
if [[ -z "${GRADIENT_ACCUMULATION_STEPS:-}" ]]; then
  if [[ "${GPU_COUNT}" == "4" ]]; then
    export GRADIENT_ACCUMULATION_STEPS=2
  else
    export GRADIENT_ACCUMULATION_STEPS=1
  fi
fi
export LOGGING_STEPS="${LOGGING_STEPS:-1}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1000}"
export SAMPLE_LOG_INTERVAL="${SAMPLE_LOG_INTERVAL:-100}"
export REPORT_TO="${REPORT_TO:-tensorboard}"
export SAVE_EVERY_N_HOURS="${SAVE_EVERY_N_HOURS:-0}"
export FREEZE_LLM="${FREEZE_LLM:-False}"
export FREEZE_BACKBONE="${FREEZE_BACKBONE:-False}"
export FREEZE_MLP="${FREEZE_MLP:-False}"
export BALANCE_UI_DEFECTS="${BALANCE_UI_DEFECTS:-True}"
export UI_RECORDS_PER_CLASS="${UI_RECORDS_PER_CLASS:-17604}"
export UI_NEGATIVE_TO_POSITIVE_RATIO="${UI_NEGATIVE_TO_POSITIVE_RATIO:-2.0}"
export ENABLE_UI_RELATION="${ENABLE_UI_RELATION:-True}"
export RELATION_DETAIL_HIDDEN_SIZE="${RELATION_DETAIL_HIDDEN_SIZE:-256}"
export RELATION_NUM_SLOTS="${RELATION_NUM_SLOTS:-8}"
export RELATION_ADAPTER_BOTTLENECK="${RELATION_ADAPTER_BOTTLENECK:-64}"
export RELATION_GATE_LOSS_WEIGHT="${RELATION_GATE_LOSS_WEIGHT:-1.0}"
export RELATION_SLOT_GATE_LOSS_WEIGHT="${RELATION_SLOT_GATE_LOSS_WEIGHT:-0.1}"
export RELATION_ATTENTION_LOSS_WEIGHT="${RELATION_ATTENTION_LOSS_WEIGHT:-0.1}"
export RELATION_GATE_THRESHOLD="${RELATION_GATE_THRESHOLD:-0.5}"
export RELATION_GATE_MODE="${RELATION_GATE_MODE:-observe}"
export RELATION_FOCAL_BETA="${RELATION_FOCAL_BETA:-0.999}"
export RELATION_FOCAL_GAMMA="${RELATION_FOCAL_GAMMA:-2.0}"
export CHECK_MAGI_IMPORT="${CHECK_MAGI_IMPORT:-$([[ "${ATTN_IMPLEMENTATION}" == "magi" ]] && echo 1 || echo 0)}"
export LOCANY_ENABLE_MILESTONE_COPIES=0
export INSTALL_SYSTEM_RUNTIME_DEPS="${INSTALL_SYSTEM_RUNTIME_DEPS:-0}"

echo "===== LocateAnything UI5 Configuration ====="
printf '%-28s: %s\n' \
  "MACHINE_TYPE" "${MACHINE_TYPE}" \
  "RESOURCE_GROUP" "${RESOURCE_GROUP}" \
  "GPU_COUNT" "${GPU_COUNT}" \
  "CUDA_DEVICES" "${CUDA_DEVICES}" \
  "EVAL_GPU_DEVICES" "${EVAL_GPU_DEVICES}" \
  "ATTN_IMPLEMENTATION" "${ATTN_IMPLEMENTATION}" \
  "WORKSPACE" "${WORKSPACE}" \
  "PROJECT_ROOT" "${PROJECT_ROOT}" \
  "ENV_DIR" "${ENV_DIR}" \
  "BASE_MODEL" "${BASE_MODEL}" \
  "TRAINING_DATA_DIR" "${TRAINING_DATA_DIR}" \
  "TRAINING_DATA_SOURCE_DIR" "${TRAINING_DATA_SOURCE_DIR}" \
  "META_PATH" "${META_PATH}" \
  "EVAL_INPUT_DIR" "${EVAL_INPUT_DIR}" \
  "OUTPUT_DIR" "${OUTPUT_DIR}" \
  "SCORER_ROOT" "${SCORER_ROOT}" \
  "CPU_COUNT (nproc)" "$(nproc)" \
  "MAX_SEQ_LENGTH" "${MAX_SEQ_LENGTH}" \
  "MAX_NUM_TOKENS_PER_SAMPLE" "${MAX_NUM_TOKENS_PER_SAMPLE}" \
  "MAX_NUM_TOKENS" "${MAX_NUM_TOKENS}" \
  "MAX_NUM_TOKENS_SCOPE" "${MAX_NUM_TOKENS_SCOPE}" \
  "MAX_STEPS" "${MAX_STEPS}" \
  "SAVE_STEPS" "${SAVE_STEPS}" \
  "GRADIENT_ACCUMULATION_STEPS" "${GRADIENT_ACCUMULATION_STEPS}" \
  "RELATION_GATE_LOSS_WEIGHT" "${RELATION_GATE_LOSS_WEIGHT}" \
  "RELATION_SLOT_GATE_LOSS_WEIGHT" "${RELATION_SLOT_GATE_LOSS_WEIGHT}" \
  "RELATION_ATTENTION_LOSS_WEIGHT" "${RELATION_ATTENTION_LOSS_WEIGHT}" \
  "RELATION_GATE_THRESHOLD" "${RELATION_GATE_THRESHOLD}" \
  "RELATION_GATE_MODE" "${RELATION_GATE_MODE}" \
  "RELATION_FOCAL_BETA" "${RELATION_FOCAL_BETA}" \
  "RELATION_FOCAL_GAMMA" "${RELATION_FOCAL_GAMMA}" \
  "RELATION_NUM_SLOTS" "${RELATION_NUM_SLOTS}" \
  "ENABLE_EVAL" "${ENABLE_EVAL}" \
  "EVAL_AT_START" "${EVAL_AT_START}" \
  "EVAL_INTERVAL_STEPS" "${EVAL_INTERVAL_STEPS}" \
  "EVAL_FAIL_POLICY" "${EVAL_FAIL_POLICY}" \
  "INSTALL_SYSTEM_RUNTIME_DEPS" "${INSTALL_SYSTEM_RUNTIME_DEPS}" \
  "PIPELINE_MODE" "${PIPELINE_MODE}" \
  "PIPELINE_LOG" "${PIPELINE_LOG}" \
  "PIPELINE_TRACE_LOG" "${PIPELINE_TRACE_LOG}"
echo "============================================="

"${PIPELINE_PYTHON}" "${PROJECT_ROOT}/scripts/resolve_locany_ui5_config.py" \
  --format json > "${OUTPUT_DIR}/effective_config.json"

[[ -d "${PROJECT_ROOT}" ]] || locany_die 21 "Project root missing: ${PROJECT_ROOT}"
[[ -d "${BASE_MODEL}" ]] || locany_die 22 "Base model missing: ${BASE_MODEL}"
[[ -x "${ENV_DIR}/bin/python" ]] || locany_die 23 "Python environment missing: ${ENV_DIR}"
if ! "${PIPELINE_PYTHON}" -c 'import openpyxl; assert tuple(map(int, openpyxl.__version__.split(".")[:2])) >= (3, 1)' >/dev/null 2>&1; then
  locany_die 30 \
    "openpyxl>=3.1 is required for diagnostics/ui5_training_evaluation.xlsx in ${ENV_DIR}"
fi
[[ -f "${PROJECT_ROOT}/shell/train_locany_ui_defect.sh" ]] || \
  locany_die 24 "Training entrypoint missing: ${PROJECT_ROOT}/shell/train_locany_ui_defect.sh"
if [[ "${ENABLE_EVAL}" == "1" || "${PIPELINE_MODE}" == "eval" ]]; then
  [[ -f "${SCORER_ROOT}/qwen3vl_merge_and_score_fixed_5tasks.py" ]] || \
    locany_die 25 "Scorer missing: ${SCORER_ROOT}/qwen3vl_merge_and_score_fixed_5tasks.py"
fi

install_ui5_system_runtime_deps() {
  if ! command -v apt-get >/dev/null 2>&1; then
    locany_die 32 \
      "cv2 needs libGL.so.1, but apt-get is unavailable in this task container"
  fi
  local -a command_prefix=()
  if (( EUID != 0 )); then
    if ! command -v sudo >/dev/null 2>&1; then
      locany_die 32 \
        "cv2 needs libGL.so.1; task user is not root and sudo is unavailable"
    fi
    command_prefix=(sudo -n)
  fi
  echo "[RUNTIME DEPS] Installing libgl1 and libglib2.0-0 in the current task container"
  "${command_prefix[@]}" env DEBIAN_FRONTEND=noninteractive apt-get update
  "${command_prefix[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y \
    --no-install-recommends libgl1 libglib2.0-0
  if command -v ldconfig >/dev/null 2>&1; then
    "${command_prefix[@]}" ldconfig
  fi
}

ensure_ui5_inference_runtime() {
  local -a preflight=(
    "${PIPELINE_PYTHON}" "${PROJECT_ROOT}/scripts/preflight_locany_runtime.py"
    --processor-path "${BASE_MODEL}"
  )
  local preflight_code=0
  if "${preflight[@]}"; then
    return 0
  else
    preflight_code=$?
  fi
  if (( preflight_code != 42 )); then
    locany_die "${preflight_code}" \
      "LocateAnything inference runtime preflight failed before worker launch"
  fi
  if [[ "${INSTALL_SYSTEM_RUNTIME_DEPS}" != "1" ]]; then
    locany_die 32 \
      "libGL.so.1 is missing. Re-submit without --no-install-system-runtime-deps, or install libgl1 libglib2.0-0 in this container"
  fi
  install_ui5_system_runtime_deps
  if "${preflight[@]}"; then
    :
  else
    preflight_code=$?
    locany_die "${preflight_code}" \
      "Runtime dependency installation completed, but cv2/AutoProcessor preflight still failed"
  fi
}

if [[ "${ENABLE_EVAL}" == "1" || "${PIPELINE_MODE}" == "eval" ]]; then
  ensure_ui5_inference_runtime
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "===== GPU inventory ====="
  nvidia-smi --query-gpu=index,name,memory.total --format=csv
fi

run_evaluation() {
  local step="$1"
  local checkpoint="$2"
  local skip_patch="$3"
  local -a command=(
    "${PIPELINE_PYTHON}" "${PROJECT_ROOT}/scripts/run_ui5_eval.py"
    --checkpoint "${checkpoint}"
    --base-model "${BASE_MODEL}"
    --step "${step}"
    --machine-type "${MACHINE_TYPE}"
    --gpu-count "${GPU_COUNT}"
    --max-num-tokens "${MAX_NUM_TOKENS}"
    --eval-gpu-devices "${EVAL_GPU_DEVICES}"
    --attn-implementation "${ATTN_IMPLEMENTATION}"
    --input-dir "${EVAL_INPUT_DIR}"
    --output-dir "${OUTPUT_DIR}"
    --scorer-root "${SCORER_ROOT}"
    --project-root "${PROJECT_ROOT}"
    --relation-gate-mode "${RELATION_GATE_MODE}"
    --relation-gate-threshold "${RELATION_GATE_THRESHOLD}"
  )
  if [[ "${skip_patch}" == "1" ]]; then
    command+=(--skip-patch)
  fi
  if (( EVAL_MAX_IMAGES_PER_TASK > 0 )); then
    command+=(--max-images-per-task "${EVAL_MAX_IMAGES_PER_TASK}")
  fi

  echo "[PIPELINE] evaluation start: step=${step}, checkpoint=${checkpoint}"
  if "${command[@]}"; then
    echo "[PIPELINE] evaluation success: step=${step}"
    return 0
  else
    local code=$?
    echo "[PIPELINE] evaluation failed: step=${step}, checkpoint=${checkpoint}, exit_code=${code}" >&2
    if [[ "${EVAL_FAIL_POLICY}" == "stop" ]]; then
      return "${code}"
    fi
    echo "[WARN] EVAL_FAIL_POLICY=warn; training will continue" >&2
    return 100
  fi
}

has_successful_evaluation() {
  local step="$1"
  "${PIPELINE_PYTHON}" "${PROJECT_ROOT}/scripts/collect_ui5_metrics.py" \
    has-success --history-dir "${OUTPUT_DIR}/evaluation" --step "${step}" \
    --relation-gate-mode "${RELATION_GATE_MODE}"
}

if [[ "${PIPELINE_MODE}" == "eval" ]]; then
  : "${EVAL_CHECKPOINT:?PIPELINE_MODE=eval requires EVAL_CHECKPOINT}"
  : "${EVAL_STEP:?PIPELINE_MODE=eval requires EVAL_STEP}"
  run_evaluation "${EVAL_STEP}" "${EVAL_CHECKPOINT}" "${EVAL_SKIP_PATCH:-0}"
  exit $?
fi

if [[ ! -f "${META_PATH}" ]]; then
  source_meta_path="${TRAINING_DATA_SOURCE_DIR}/recipe/ui_defect_5class_train.json"
  echo "===== Training data bootstrap ====="
  echo "reason      : target metadata is missing"
  echo "source      : ${TRAINING_DATA_SOURCE_DIR}"
  echo "destination : ${TRAINING_DATA_DIR}"
  if [[ "${TRAINING_DATA_SOURCE_DIR}" == "${TRAINING_DATA_DIR}" ]]; then
    locany_die 26 \
      "Training data source and destination are identical, but metadata is missing: ${META_PATH}"
  fi
  if [[ ! -f "${source_meta_path}" ]]; then
    echo "source metadata is also missing: ${source_meta_path}" >&2
  else
    mkdir -p "${TRAINING_DATA_DIR}"
    copy_started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if command -v rsync >/dev/null 2>&1; then
      echo "copy_method : rsync -av --progress"
      if rsync -av --progress \
          "${TRAINING_DATA_SOURCE_DIR}/" "${TRAINING_DATA_DIR}/"; then
        :
      else
        code=$?
        locany_die "${code}" \
          "Failed to copy training data with rsync: source=${TRAINING_DATA_SOURCE_DIR}, destination=${TRAINING_DATA_DIR}"
      fi
    else
      echo "copy_method : cp -a (rsync is unavailable)"
      if cp -a "${TRAINING_DATA_SOURCE_DIR}/." "${TRAINING_DATA_DIR}/"; then
        :
      else
        code=$?
        locany_die "${code}" \
          "Failed to copy training data with cp: source=${TRAINING_DATA_SOURCE_DIR}, destination=${TRAINING_DATA_DIR}"
      fi
    fi
    echo "copy_started: ${copy_started}"
    echo "copy_ended  : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  fi
  echo "==================================="
fi

if [[ ! -f "${META_PATH}" ]]; then
  meta_parent="$(dirname "${META_PATH}")"
  echo "===== Training metadata diagnostic =====" >&2
  echo "expected_file : ${META_PATH}" >&2
  echo "parent_dir    : ${meta_parent}" >&2
  if [[ -d "${meta_parent}" ]]; then
    echo "parent_status : exists" >&2
    ls -la "${meta_parent}" >&2 || true
  else
    echo "parent_status : MISSING" >&2
  fi
  echo "matching recipes under project/workspace data roots:" >&2
  found_recipe=0
  for candidate_root in "${PROJECT_ROOT}/data" "${WORKSPACE}/data"; do
    if [[ -d "${candidate_root}" ]]; then
      while IFS= read -r candidate; do
        echo "  ${candidate}" >&2
        found_recipe=1
      done < <(
        find "${candidate_root}" -maxdepth 6 -type f \
          -name 'ui_defect_5class_train.json' -print 2>/dev/null || true
      )
    fi
  done
  if (( found_recipe == 0 )); then
    echo "  <none found>" >&2
  fi
  echo "========================================" >&2
  locany_die 26 "Training metadata missing: ${META_PATH}"
fi

if [[ "${ENABLE_EVAL}" == "0" ]]; then
  echo "[PIPELINE] ENABLE_EVAL=0: starting uninterrupted training; no step-0 or periodic evaluation"
  unset LOCANY_SEGMENT_MODE LOCANY_STOP_AFTER_STEP
  exec bash "${PROJECT_ROOT}/shell/train_locany_ui_defect.sh"
fi

CHECKPOINT_ZERO="${OUTPUT_DIR}/checkpoint-0"
if [[ "${EVAL_AT_START}" == "1" ]] && ! has_successful_evaluation 0; then
  echo "[PIPELINE] exporting deterministic full-model checkpoint-0"
  "${PIPELINE_PYTHON}" "${PROJECT_ROOT}/scripts/export_ui5_checkpoint0.py" \
    --base-model "${BASE_MODEL}" \
    --output "${CHECKPOINT_ZERO}" \
    --seed 42 \
    --block-size 6 \
    --attn-implementation "${ATTN_IMPLEMENTATION}" \
    --relation-detail-hidden-size "${RELATION_DETAIL_HIDDEN_SIZE}" \
    --relation-num-slots "${RELATION_NUM_SLOTS}" \
    --relation-adapter-bottleneck "${RELATION_ADAPTER_BOTTLENECK}" \
    --relation-gate-loss-weight "${RELATION_GATE_LOSS_WEIGHT}" \
    --relation-slot-gate-loss-weight "${RELATION_SLOT_GATE_LOSS_WEIGHT}" \
    --relation-attention-loss-weight "${RELATION_ATTENTION_LOSS_WEIGHT}" \
    --relation-gate-threshold "${RELATION_GATE_THRESHOLD}" \
    --relation-focal-beta "${RELATION_FOCAL_BETA}" \
    --relation-focal-gamma "${RELATION_FOCAL_GAMMA}"
  if run_evaluation 0 "${CHECKPOINT_ZERO}" 0; then
    :
  else
    code=$?
    if (( code != 100 )); then
      exit "${code}"
    fi
  fi
fi

current_step="$("${PIPELINE_PYTHON}" "${PROJECT_ROOT}/scripts/locany_ui5_checkpoint.py" \
  latest --output-dir "${OUTPUT_DIR}" --require-resume --expected-ranks "${GPU_COUNT}" --field step)"

if [[ ! "${current_step}" =~ ^[0-9]+$ ]]; then
  locany_die 27 "Could not resolve latest checkpoint step: ${current_step}"
fi

training_checkpoint_count="$("${PIPELINE_PYTHON}" "${PROJECT_ROOT}/scripts/locany_ui5_checkpoint.py" \
  training-candidates --output-dir "${OUTPUT_DIR}" --field count)"
if [[ ! "${training_checkpoint_count}" =~ ^[0-9]+$ ]]; then
  locany_die 27 "Could not count training checkpoint candidates: ${training_checkpoint_count}"
fi

# checkpoint-0 is a deterministic, full-model evaluation artifact.  It has no
# optimizer/scheduler/Trainer state by design, so it is neither corrupt nor a
# resume candidate.  Keep the fail-fast guard for every checkpoint-N, N > 0.
if (( current_step == 0 && training_checkpoint_count > 0 )); then
  invalid_training_checkpoints="$("${PIPELINE_PYTHON}" "${PROJECT_ROOT}/scripts/locany_ui5_checkpoint.py" \
    training-candidates --output-dir "${OUTPUT_DIR}" --field paths)"
  locany_die 28 \
    "Nonzero checkpoint directories exist, but none passed resume validation; refusing to restart from zero: ${invalid_training_checkpoints}"
fi

# Evaluation-only observe mode may deliberately load a legacy checkpoint whose
# image-level Gate did not exist yet.  Training resume must never do that: an
# old slot-Gate checkpoint would otherwise be accepted by Trainer and silently
# mix the repaired architecture with an invalid optimization history.
if (( current_step > 0 )); then
  resume_checkpoint="${OUTPUT_DIR}/checkpoint-${current_step}"
  echo "[PIPELINE] strict UI module audit before training resume: ${resume_checkpoint}"
  if ! "${PIPELINE_PYTHON}" "${PROJECT_ROOT}/scripts/patch_locany_checkpoint.py" \
      --base-model "${BASE_MODEL}" \
      --checkpoint "${resume_checkpoint}" \
      --project-root "${PROJECT_ROOT}" \
      --force \
      --validate-relation-weights; then
    locany_die 31 \
      "Training resume checkpoint is not a complete Image-Gate/Slot-Gate/Relation/PBD model. Use --eval-checkpoint with observe mode for legacy reproduction, or choose a fresh --run-name for training: ${resume_checkpoint}"
  fi
fi

if (( current_step > 0 )) && ! has_successful_evaluation "${current_step}"; then
  checkpoint="${OUTPUT_DIR}/checkpoint-${current_step}"
  if run_evaluation "${current_step}" "${checkpoint}" 0; then
    :
  else
    code=$?
    if (( code != 100 )); then
      exit "${code}"
    fi
  fi
fi

if (( current_step > 0 )) && has_successful_evaluation "${current_step}"; then
  "${PIPELINE_PYTHON}" "${PROJECT_ROOT}/scripts/locany_ui5_checkpoint.py" cleanup \
    --output-dir "${OUTPUT_DIR}" --formal-interval "${SAVE_STEPS}" --latest-step "${current_step}"
fi

while (( current_step < MAX_STEPS )); do
  next_step=$(( (current_step / EVAL_INTERVAL_STEPS + 1) * EVAL_INTERVAL_STEPS ))
  if (( next_step > MAX_STEPS )); then
    next_step="${MAX_STEPS}"
  fi
  echo "[PIPELINE] train segment: current=${current_step}, stop_after=${next_step}, total_max=${MAX_STEPS}"
  export LOCANY_SEGMENT_MODE=1
  export LOCANY_STOP_AFTER_STEP="${next_step}"

  if bash "${PROJECT_ROOT}/shell/train_locany_ui_defect.sh"; then
    :
  else
    code=$?
    locany_die "${code}" \
      "Training segment failed: from=${current_step}, target=${next_step}, exit_code=${code}"
  fi

  checkpoint="${OUTPUT_DIR}/checkpoint-${next_step}"
  if ! "${PIPELINE_PYTHON}" "${PROJECT_ROOT}/scripts/locany_ui5_checkpoint.py" validate \
      --checkpoint "${checkpoint}" --mode resume --expected-ranks "${GPU_COUNT}"; then
    locany_die 29 \
      "Segment checkpoint is incomplete: step=${next_step}, checkpoint=${checkpoint}"
  fi

  eval_succeeded=0
  if run_evaluation "${next_step}" "${checkpoint}" 0; then
    eval_succeeded=1
  else
    code=$?
    if (( code != 100 )); then
      exit "${code}"
    fi
  fi

  if (( eval_succeeded == 1 )); then
    "${PIPELINE_PYTHON}" "${PROJECT_ROOT}/scripts/locany_ui5_checkpoint.py" cleanup \
      --output-dir "${OUTPUT_DIR}" --formal-interval "${SAVE_STEPS}" --latest-step "${next_step}"
  else
    echo "[WARN] Evaluation failed under warn policy; temporary checkpoints were retained" >&2
  fi
  current_step="${next_step}"
done

echo "[PIPELINE COMPLETE] step=${current_step}, output=${OUTPUT_DIR}"
