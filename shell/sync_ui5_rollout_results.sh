#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: bash shell/sync_ui5_rollout_results.sh <complete_OUTPUT_ROOT> <destination_directory>" >&2
  exit 2
fi

OUTPUT_ROOT=$1
DESTINATION=$2

if [[ "${OUTPUT_ROOT}" != /* || "${DESTINATION}" != /* ]]; then
  echo "ERROR: OUTPUT_ROOT and destination must both be absolute paths" >&2
  exit 2
fi
if [[ ! -d "${OUTPUT_ROOT}" ]]; then
  echo "ERROR: OUTPUT_ROOT does not exist: ${OUTPUT_ROOT}" >&2
  exit 3
fi

echo "[SYNC_UI5_ROLLOUT_RESULTS] source=${OUTPUT_ROOT} destination=${DESTINATION}"
nastk cp -c=32 "${OUTPUT_ROOT}" "${DESTINATION}"
