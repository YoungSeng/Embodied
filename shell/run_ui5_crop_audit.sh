#!/usr/bin/env bash
set -euo pipefail

# Every cluster path is passed through the CLI; do not hand-edit Python globals.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ $# -eq 0 ]]; then
  cat >&2 <<'USAGE'
Usage:
  bash shell/run_ui5_crop_audit.sh \
    --source-dir /mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace/data \
    --locany-data-dir /absolute/path/to/ui_defect_locany_v3 \
    --parser-root ../ui-region-parser \
    --output-dir work_dirs/ui5_crop_audit_20260825 \
    --gpus 0,1,2,3 \
    --workers-per-gpu 1 \
    --crop-workers 8 \
    --icon-python /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/conda_envs/LocateAnything/bin/python \
    --stage prepare \
    --resume

Stages: prepare, text, icon, merge, crop-audit, all.
USAGE
  exit 2
fi

cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" scripts/run_ui5_crop_audit.py "$@"
