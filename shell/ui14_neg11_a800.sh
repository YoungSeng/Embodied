#!/usr/bin/env bash
# New experiment only. CPU preparation and GPU detection use separate allocations.
set -Eeuo pipefail
export WORKSPACE="${WORKSPACE:-/mnt/bn/intelligent-service-yg/logging/sicheng_workspace}"
export UI9_DATA_ROOT="${UI9_DATA_ROOT:-/mnt/bn/intelligent-service-yg/dataset/gui/ui9_datasets_v1}"
export UI14_PARENT_DATA_ROOT="${UI14_PARENT_DATA_ROOT:-${WORKSPACE}/gui_data/ui14_cpt9000_repair_v2}"
export UI14_DATA_ROOT="${UI14_DATA_ROOT:-${WORKSPACE}/gui_data/ui14_cpt9000_neg11_v1}"
export UI14_PREPARE_WORKERS="${UI14_PREPARE_WORKERS:-16}"
export UI14_CROP_WORKERS="${UI14_CROP_WORKERS:-16}"
export UI14_PNG_COMPRESS_LEVEL="${UI14_PNG_COMPRESS_LEVEL:-1}"
export UI14_PROGRESS_INTERVAL_SECONDS="${UI14_PROGRESS_INTERVAL_SECONDS:-10}"
export UI14_NEGATIVE_QUOTA_POLICY="${UI14_NEGATIVE_QUOTA_POLICY:-available}"
export PYTHONUNBUFFERED=1
export UI14_NEG11=1
NEG11_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$NEG11_PROJECT_ROOT"
exec "${UI14_PYTHON:-${WORKSPACE}/conda_envs/LocateAnything/bin/python}" -u scripts/ui14_neg11.py "$@"
