#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON_BIN=${PYTHON_BIN:-python}
BUNDLE_ROOT=${BUNDLE_ROOT:-/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle_LocateUI5_v4/Embodied-ui5-det-crop/work_dirs/ui5_train_rollout_bundle_v1}
DIAGNOSTICS_DIR=${DIAGNOSTICS_DIR:-${BUNDLE_ROOT}/diagnostics}
M31_CHECKPOINT=${M31_CHECKPOINT:-/mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace/gui_models/Embodied/locany-ui5-m31-taskmoe-setdecoder-a800x4-sft-20260830-r2/checkpoint-12000}
CROP_CHECKPOINT=${CROP_CHECKPOINT:-/mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace/gui_models/Embodied-ui5-det-crop/locany-ui5-v5-croponly-sourcebalanced-a800x4-20260830/checkpoint-12000}
M31_REPO=${M31_REPO:-/mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace/code/Eagle/Embodied-ui5-rollout8-m31}
CROP_REPO=${CROP_REPO:-/mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace/code/Eagle/Embodied-ui5-rollout8-crop}

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "ERROR: CPU rollout preflight requires CUDA_VISIBLE_DEVICES to be empty" >&2
  exit 2
fi

ARGS=(
  --bundle-root "${BUNDLE_ROOT}"
  --diagnostics-dir "${DIAGNOSTICS_DIR}"
  --m31-checkpoint "${M31_CHECKPOINT}"
  --crop-checkpoint "${CROP_CHECKPOINT}"
  --m31-repo "${M31_REPO}"
  --crop-repo "${CROP_REPO}"
)
if [[ "${REQUIRE_RUNTIME:-0}" == "1" ]]; then
  ARGS+=(--require-runtime)
fi

cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" scripts/preflight_ui5_train_rollouts.py "${ARGS[@]}"
