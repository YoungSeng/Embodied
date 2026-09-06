#!/usr/bin/env bash
# Prepare and finalize caches on CPU; reserve four A800 GPUs only for cache.
set -Eeuo pipefail
export WORKSPACE="${WORKSPACE:-/mnt/bn/intelligent-service-yg/logging/sicheng_workspace}"
export UI9_DATA_ROOT="${UI9_DATA_ROOT:-/mnt/bn/intelligent-service-yg/dataset/gui/ui9_datasets_v1}"
export UI14_DATA_ROOT="${UI14_DATA_ROOT:-${WORKSPACE}/gui_data/ui14_cpt9000_repair_v2}"
export PYTHONUNBUFFERED=1
export UI14_PROGRESS_INTERVAL_SECONDS="${UI14_PROGRESS_INTERVAL_SECONDS:-10}"
export UI14_CROP_WORKERS="${UI14_CROP_WORKERS:-16}"
export UI14_PNG_COMPRESS_LEVEL="${UI14_PNG_COMPRESS_LEVEL:-1}"
UI14_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UI14_PYTHON="${UI14_PYTHON:-${WORKSPACE}/conda_envs/LocateAnything/bin/python}"
UI14_PROFILE="m32-cpt9000-ui14-v1"
cd "${UI14_PROJECT_ROOT}"
case "${1:-}" in
  normalize)
    exec "${UI14_PYTHON}" scripts/prepare_ui14_sft.py --stage normalize \
      --ui9-data-root "${UI9_DATA_ROOT}" --output-dir "${UI14_DATA_ROOT}"
    ;;
  cache)
    exec "${UI14_PYTHON}" scripts/prepare_ui14_detector_crops.py \
      --stage detect --data-root "${UI14_DATA_ROOT}" --gpus 0,1,2,3
    ;;
  cache-prepare)
    exec "${UI14_PYTHON}" scripts/prepare_ui14_detector_crops.py \
      --stage prepare --data-root "${UI14_DATA_ROOT}" --prepare-workers "${UI14_PREPARE_WORKERS:-16}"
    ;;
  cache-finalize)
    exec "${UI14_PYTHON}" scripts/prepare_ui14_detector_crops.py \
      --stage crops --data-root "${UI14_DATA_ROOT}"
    ;;
  finalize)
    exec "${UI14_PYTHON}" scripts/prepare_ui14_sft.py --stage finalize \
      --ui9-data-root "${UI9_DATA_ROOT}" --output-dir "${UI14_DATA_ROOT}"
    ;;
  submit)
    exec "${UI14_PYTHON}" scripts/submit_locany_ui5.py \
      --profile "${UI14_PROFILE}" --machine a800 --resource-group aiai_locate --gpus 4 \
      --ui14-data-root "${UI14_DATA_ROOT}" --output-yaml "${UI14_DATA_ROOT}/formal_job.yaml"
    ;;
  check-full)
    exec "${UI14_PYTHON}" scripts/prepare_ui14_sft.py --stage check --full-verify \
      --ui9-data-root "${UI9_DATA_ROOT}" --output-dir "${UI14_DATA_ROOT}"
    ;;
  *) printf 'Usage: bash shell/ui14_cpt9000_a800.sh {normalize|cache-prepare|cache|cache-finalize|finalize|check-full|submit}\n' >&2; exit 2 ;;
esac
