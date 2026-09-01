#!/usr/bin/env bash
set -Eeuo pipefail

# Natural-exit CPT pipeline: train target step -> validate -> same-GPU eval -> resume.
# This script is entered after run_locany_cpt_merlin.sh has prepared local caches.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
MACHINE_TYPE="${1:-h20}"
CPT_MODE="${2:-formal}"

if [[ "${CPT_MODE}" != "formal" ]]; then
  echo "Segmented CPT evaluation is only supported for formal runs" >&2
  exit 2
fi

WORKSPACE="${WORKSPACE:-/mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace}"
ENV_DIR="${ENV_DIR:-${WORKSPACE}/conda_envs/LocateAnything}"
DATA_DIR="${DATA_DIR:-${WORKSPACE}/data/locany_cpt_v4_split_v2}"
OUTPUT_BASE="${OUTPUT_BASE:-${WORKSPACE}/gui_models}"
RUN_NAME="${RUN_NAME:-locany-3b-ui-cpt-v4-v3-h20x2-formal-segmented-eval}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_BASE}/${RUN_NAME}}"
MODEL_PATH="${MODEL_PATH:-${WORKSPACE}/hf_home/hub/models--nvidia--LocateAnything-3B/snapshots/c32291ca5e996f5a7a485845b4f57a233936bba0}"
GPU_COUNT="${GPU_COUNT:-2}"
EVAL_GPU_DEVICES="${EVAL_GPU_DEVICES:-0,1}"
PIPELINE_MAX_STEPS="${MAX_STEPS:-20000}"
EVAL_INTERVAL_STEPS="${EVAL_INTERVAL_STEPS:-1000}"
EVAL_SAMPLES_PER_TASK="${EVAL_SAMPLES_PER_TASK:-200}"
EVAL_FAIL_POLICY="${EVAL_FAIL_POLICY:-warn}"
EVAL_RECIPE="${DATA_DIR}/recipe/${EVAL_RECIPE_NAME:-locany_cpt_val_fast.json}"
SPLIT_MANIFEST="${DATA_DIR}/diagnostics/split_manifest.jsonl"
EXTERNAL_UI5_DIR="${CPT_EXTERNAL_UI5_DATA_DIR:-${WORKSPACE}/data}"
LAUNCH_LOG="${LAUNCH_LOG:-${SHARED_RUNTIME_DIR:-${OUTPUT_DIR}/diagnostics}/segmented_pipeline.log}"

if (( EVAL_INTERVAL_STEPS <= 0 || PIPELINE_MAX_STEPS <= 0 )); then
  echo "MAX_STEPS and EVAL_INTERVAL_STEPS must be positive" >&2
  exit 2
fi
if (( PIPELINE_MAX_STEPS % EVAL_INTERVAL_STEPS != 0 )); then
  echo "MAX_STEPS must be divisible by EVAL_INTERVAL_STEPS" >&2
  exit 2
fi
for path in "${ENV_DIR}/bin/python" "${EVAL_RECIPE}" "${SPLIT_MANIFEST}"; do
  test -e "${path}" || { echo "Required segmented-pipeline input missing: ${path}" >&2; exit 20; }
done
mkdir -p "${OUTPUT_DIR}/diagnostics" "$(dirname "${LAUNCH_LOG}")"

export OUTPUT_DIR OUTPUT_BASE RUN_NAME MODEL_PATH DATA_DIR
export LOCANY_SEGMENT_MODE=1
export LOCANY_PIPELINE_FINAL_STEP="${PIPELINE_MAX_STEPS}"
export LOCANY_LR_SCHEDULER_TOTAL_STEPS="${PIPELINE_MAX_STEPS}"
export CPT_INTEGRATED_EVAL=0
export ENABLE_UI_RELATION=False
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SAVE_EVERY_N_HOURS=0
export SAVE_STEPS="${EVAL_INTERVAL_STEPS}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-30}"
unset LOCANY_STOP_AFTER_STEP LOCANY_STOP_AFTER_PERIODIC_SAVE

checkpoint_status() {
  local step="$1"
  local checkpoint="${OUTPUT_DIR}/checkpoint-${step}"
  "${ENV_DIR}/bin/python" "${PROJECT_ROOT}/scripts/locany_ui5_checkpoint.py" \
    validate --checkpoint "${checkpoint}" --mode resume --expected-ranks "${GPU_COUNT}"
}

latest_step() {
  "${ENV_DIR}/bin/python" "${PROJECT_ROOT}/scripts/locany_ui5_checkpoint.py" \
    latest --output-dir "${OUTPUT_DIR}" --require-resume \
    --expected-ranks "${GPU_COUNT}" --field step
}

run_eval() {
  local step="$1"
  local checkpoint
  if (( step == 0 )); then checkpoint="${MODEL_PATH}"; else checkpoint="${OUTPUT_DIR}/checkpoint-${step}"; fi
  local command=(
    "${ENV_DIR}/bin/python" "${PROJECT_ROOT}/scripts/run_locany_cpt_segmented_eval.py"
    --checkpoint "${checkpoint}" --checkpoint-step "${step}"
    --base-model "${MODEL_PATH}" --processor-path "${MODEL_PATH}"
    --recipe "${EVAL_RECIPE}" --manifest "${SPLIT_MANIFEST}"
    --run-dir "${OUTPUT_DIR}" --external-input-dir "${EXTERNAL_UI5_DIR}"
    --gpu-devices "${EVAL_GPU_DEVICES}"
    --samples-per-task "${EVAL_SAMPLES_PER_TASK}"
    --python "${ENV_DIR}/bin/python" --dtype "${EVAL_DTYPE:-bf16}"
    --attn-implementation "${EVAL_ATTN_IMPLEMENTATION:-sdpa}"
    --vision-attn-implementation "${EVAL_VISION_ATTN_IMPLEMENTATION:-flash_attention_2}"
    --heldout-max-new-tokens "${EVAL_MAX_NEW_TOKENS:-1024}"
    --external-max-new-tokens "${EVAL_EXTERNAL_MAX_NEW_TOKENS:-4096}"
    --iou-thresholds 0.1 0.5 --seed "${CPT_EVAL_SEED:-20260826}"
    --fail-policy "${EVAL_FAIL_POLICY}"
  )
  if [[ "${CPT_EXTERNAL_UI5_EVAL:-1}" == "1" ]]; then
    command+=(--external-ui5)
  else
    command+=(--no-external-ui5)
  fi
  echo "===== CPT segmented eval step=${step} GPUs=${EVAL_GPU_DEVICES} =====" | tee -a "${LAUNCH_LOG}"
  "${command[@]}" 2>&1 | tee -a "${LAUNCH_LOG}"
}

run_train_segment() {
  local target="$1"
  export MAX_STEPS="${target}"
  echo "===== CPT natural-exit train segment target_max_steps=${target} =====" | tee -a "${LAUNCH_LOG}"
  set +e
  bash "${PROJECT_ROOT}/shell/run_locany_cpt.sh" "${MACHINE_TYPE}" formal \
    2>&1 | tee -a "${LAUNCH_LOG}"
  local code="${PIPESTATUS[0]}"
  set -e
  if (( code != 0 )); then
    echo "TRAIN_SEGMENT_FAILED target=${target} exit_code=${code}" | tee -a "${LAUNCH_LOG}"
    return "${code}"
  fi
  echo "TORCHRUN_ALL_RANKS_EXITED target=${target}" | tee -a "${LAUNCH_LOG}"
  checkpoint_status "${target}" 2>&1 | tee -a "${LAUNCH_LOG}"
}

eval_complete() {
  local step="$1"
  local status="${OUTPUT_DIR}/eval/checkpoint-${step}/segmented_eval_status.json"
  [[ -f "${status}" ]] || return 1
  "${ENV_DIR}/bin/python" - "${status}" <<'PY'
import json
import sys
from pathlib import Path
value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
raise SystemExit(0 if value.get("success") is True else 1)
PY
}

CURRENT_STEP="$(latest_step)"
if [[ ! "${CURRENT_STEP}" =~ ^[0-9]+$ ]]; then
  echo "Invalid latest checkpoint step: ${CURRENT_STEP}" >&2
  exit 30
fi
CHECKPOINT_CANDIDATE_COUNT="$(
  "${ENV_DIR}/bin/python" "${PROJECT_ROOT}/scripts/locany_ui5_checkpoint.py" \
    training-candidates --output-dir "${OUTPUT_DIR}" --field count
)"
if (( CURRENT_STEP == 0 && CHECKPOINT_CANDIDATE_COUNT > 0 )); then
  echo "Found checkpoint directories but the newest one is not resumable; refusing to skip it" >&2
  exit 31
fi
if [[ -f "${OUTPUT_DIR}/done.txt" && "${CURRENT_STEP}" -lt "${PIPELINE_MAX_STEPS}" ]]; then
  echo "done.txt exists before the configured final step; use a new RUN_NAME" >&2
  exit 32
fi

# Step 0 Base is evaluated once; cached fragments make this idempotent.
if ! eval_complete 0; then
  run_eval 0
fi

# A resubmission evaluates a complete-but-not-yet-evaluated checkpoint before training.
if (( CURRENT_STEP > 0 )); then
  checkpoint_status "${CURRENT_STEP}" >/dev/null
  if ! eval_complete "${CURRENT_STEP}"; then
    run_eval "${CURRENT_STEP}"
  fi
fi

while (( CURRENT_STEP < PIPELINE_MAX_STEPS )); do
  TARGET_STEP=$((CURRENT_STEP + EVAL_INTERVAL_STEPS))
  if (( TARGET_STEP > PIPELINE_MAX_STEPS )); then TARGET_STEP="${PIPELINE_MAX_STEPS}"; fi
  run_train_segment "${TARGET_STEP}"
  CURRENT_STEP="${TARGET_STEP}"
  # Training torchrun has returned and checkpoint resume validation passed.
  run_eval "${CURRENT_STEP}"
done

echo "CPT_SEGMENTED_PIPELINE_COMPLETE step=${CURRENT_STEP} workbook=${OUTPUT_DIR}/diagnostics/cpt_training_evaluation.xlsx" | tee -a "${LAUNCH_LOG}"
