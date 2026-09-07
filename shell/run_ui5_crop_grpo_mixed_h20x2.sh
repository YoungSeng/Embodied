#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
source "${SCRIPT_DIR}/bash_error_report.sh"
: "${PYTHON_BIN:?the submitter binds the inherited conda interpreter}"
: "${ENV_DIR:?the submitter binds the inherited conda environment}"
: "${GRPO_RUN_CONFIG:?the submitter creates the immutable formal run config}"
: "${CODE_REVISION:?the submitter binds the code SHA}"
cd "${PROJECT_ROOT}"
test "$(git rev-parse HEAD)" = "${CODE_REVISION}"
export PATH="${ENV_DIR}/bin:${PATH}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec "${PYTHON_BIN}" -u scripts/run_ui5_grpo_pipeline.py --run-config "${GRPO_RUN_CONFIG}"
