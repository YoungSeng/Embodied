#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON_BIN=${PYTHON_BIN:-python}
FULL_DATA=${FULL_DATA:-/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle_LocateUI5_v4/Embodied/data/ui_defect_locany_v3}
AUDIT_ROOT=${AUDIT_ROOT:-/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle_LocateUI5_v4/Embodied-ui5-det-crop/work_dirs/ui5_crop_audit_20260825}
CROP_ROOT=${CROP_ROOT:-${AUDIT_ROOT}/crop_audit_v4_gt_repair/crop_only_horizontal_v5_train_repair_f04503b}
BUNDLE_ROOT=${BUNDLE_ROOT:-/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle_LocateUI5_v4/Embodied-ui5-det-crop/work_dirs/ui5_train_rollout_bundle_v1}

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "ERROR: CPU bundle preparation requires CUDA_VISIBLE_DEVICES to be empty" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" scripts/prepare_ui5_train_rollout_bundle.py \
  --full-data "${FULL_DATA}" \
  --audit-root "${AUDIT_ROOT}" \
  --crop-root "${CROP_ROOT}" \
  --output-dir "${BUNDLE_ROOT}"
