#!/usr/bin/env bash
# Four production stages. Run cache on an allocated A800 node with four GPUs.
set -Eeuo pipefail
export WORKSPACE="${WORKSPACE:-/mnt/bn/intelligent-service-yg/logging/sicheng_workspace}"
export UI9_DATA_ROOT="${UI9_DATA_ROOT:-/mnt/bn/intelligent-service-yg/dataset/gui/ui9_datasets_v1}"
export UI14_DATA_ROOT="${UI14_DATA_ROOT:-${WORKSPACE}/gui_data/ui14_cpt9000_v1}"
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
      --data-root "${UI14_DATA_ROOT}" --gpus 0,1,2,3
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
  *) printf 'Usage: bash shell/ui14_cpt9000_a800.sh {normalize|cache|finalize|submit}\n' >&2; exit 2 ;;
esac
