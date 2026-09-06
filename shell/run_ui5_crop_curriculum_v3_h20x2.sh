#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
export UI5_CURRICULUM_PROFILE=global_replay_v3
export HARD_RATIOS=0.20,0.25,0.30
export ANCHOR_RATIOS=0.20,0.25,0.30
export GLOBAL_REPLAY_RATIOS=0.60,0.50,0.40
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
cd "${PROJECT_ROOT}"
: "${OUTPUT_DIR:?use the v3 submission renderer}"
: "${MODEL_PATH:?private original-crop model view required}"
: "${PYTHON_BIN:?Python runtime must be inherited from the working H20 profile}"
[[ -s "${OUTPUT_DIR}/diagnostics/v3_preparation.json" ]]
[[ "${MODEL_PATH}" == "${OUTPUT_DIR}/initial_model" ]]
# A v3 job starts fresh; only the in-job 200-step loop may resume its optimizer.
[[ ! -e "${OUTPUT_DIR}/resume/latest" && ! -e "${OUTPUT_DIR}/checkpoints.json" ]]
mkdir -p "${OUTPUT_DIR}/logs"
# An exclusive run reservation also protects against duplicate manual YAML submits.
(set -o noclobber; printf '%s\n' "${HOSTNAME:-unknown}:$$" > "${OUTPUT_DIR}/v3-job.started")
exec > >(tee -a "${OUTPUT_DIR}/logs/v3-start.log") 2>&1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset RESUME_FROM_CHECKPOINT CURRICULUM_START_STEP LOCANY_STOP_AFTER_STEP LOCANY_SEGMENT_MODE
"${PYTHON_BIN}" -u scripts/ui5_curriculum_v3.py --run-comparison
exec bash shell/run_locany_ui5_crop_rollout4_curriculum_h20x2.sh
