#!/usr/bin/env bash
set -Eeuo pipefail
export WORKSPACE="${WORKSPACE:-/mnt/bn/intelligent-service-yg/logging/sicheng_workspace}"
export UI14_ALIGNMENT_CROPS_DATA_ROOT="${UI14_ALIGNMENT_CROPS_DATA_ROOT:-${WORKSPACE}/gui_data/ui14_neg11_alignment_crops_v1}"
export UI14_ALIGNMENT_CROPS_PARENT_DATA_ROOT="${UI14_ALIGNMENT_CROPS_PARENT_DATA_ROOT:-${WORKSPACE}/gui_data/ui14_cpt9000_neg11_v1}"
export PYTHONUNBUFFERED=1
ALIGNMENT_CROPS_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ALIGNMENT_CROPS_PROJECT_ROOT}"
exec "${UI14_PYTHON:-${WORKSPACE}/conda_envs/LocateAnything/bin/python}" -u scripts/ui14_alignment_crops.py "$@"
