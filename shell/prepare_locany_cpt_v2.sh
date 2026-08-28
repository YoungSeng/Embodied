#!/usr/bin/env bash
set -Eeuo pipefail

# Build a fresh group-split CPT v2 dataset.  This command never touches the
# legacy locany_cpt_v4 directory used by checkpoint-1549/1860.
# Usage: bash shell/prepare_locany_cpt_v2.sh <a100|h20> <smoke|formal>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
MACHINE_TYPE="${1:-}"
CPT_MODE="${2:-}"

if [[ "${MACHINE_TYPE}" != "a100" && "${MACHINE_TYPE}" != "h20" ]]; then
  echo "Usage: bash shell/prepare_locany_cpt_v2.sh <a100|h20> <smoke|formal>" >&2
  exit 2
fi
if [[ "${CPT_MODE}" != "smoke" && "${CPT_MODE}" != "formal" ]]; then
  echo "CPT mode must be smoke or formal, got: ${CPT_MODE}" >&2
  exit 2
fi

case "${MACHINE_TYPE}" in
  a100)
    WORKSPACE="${WORKSPACE:-/mnt/bn/intelligent-service-yg/logging/sicheng_workspace}"
    FILESYSTEM_ROOT=/mnt/bn/intelligent-service-yg
    ;;
  h20)
    WORKSPACE="${WORKSPACE:-/mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace}"
    FILESYSTEM_ROOT=/mnt/bn/intelligent-service-arnold-hl
    ;;
esac

ENV_DIR="${ENV_DIR:-${WORKSPACE}/conda_envs/LocateAnything}"
SOURCE_ROOT="${SOURCE_ROOT:-${FILESYSTEM_ROOT}/dataset/gui/gui_base/sample/raw_data_v4.1_hl_norm1k/raw_data_v4.1_hl}"
if [[ "${CPT_MODE}" == "smoke" ]]; then
  DATA_DIR="${DATA_DIR:-${WORKSPACE}/data/locany_cpt_v4_split_v2_smoke}"
  RECIPE_NAME="${RECIPE_NAME:-locany_cpt_smoke.json}"
  MAX_RECORDS_PER_TASK="${MAX_RECORDS_PER_TASK:-2000}"
  VAL_FAST_PER_TASK="${VAL_FAST_PER_TASK:-10}"
  MIN_VAL_FAST_PER_TASK="${MIN_VAL_FAST_PER_TASK:-10}"
else
  DATA_DIR="${DATA_DIR:-${WORKSPACE}/data/locany_cpt_v4_split_v2}"
  RECIPE_NAME="${RECIPE_NAME:-locany_cpt_train.json}"
  MAX_RECORDS_PER_TASK="${MAX_RECORDS_PER_TASK:-0}"
  VAL_FAST_PER_TASK="${VAL_FAST_PER_TASK:-200}"
  MIN_VAL_FAST_PER_TASK="${MIN_VAL_FAST_PER_TASK:-1}"
fi

test -x "${ENV_DIR}/bin/python" || {
  echo "ERROR: Python environment not found: ${ENV_DIR}" >&2
  exit 20
}
test -d "${SOURCE_ROOT}" || {
  echo "ERROR: raw CPT source not found: ${SOURCE_ROOT}" >&2
  echo "Set SOURCE_ROOT to the raw_data_v4.1_hl directory on this filesystem." >&2
  exit 21
}

PREPARE_ARGS=(
  --source-root "${SOURCE_ROOT}"
  --output-dir "${DATA_DIR}"
  --recipe-name "${RECIPE_NAME}"
  --max-records-per-task "${MAX_RECORDS_PER_TASK}"
  --split-seed "${CPT_SPLIT_SEED:-20260826}"
  --val-fraction "${CPT_VAL_FRACTION:-0.02}"
  --val-fast-per-task "${VAL_FAST_PER_TASK}"
  --group-id-mode "${CPT_GROUP_ID_MODE:-sha256}"
  --split-progress-every "${CPT_SPLIT_PROGRESS_EVERY:-1000}"
)
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  PREPARE_ARGS+=(--overwrite)
fi

cd "${PROJECT_ROOT}"
if [[ "${SPLIT_ONLY:-0}" == "1" ]]; then
  COMBINED_RECIPE="${DATA_DIR}/recipe/locany_cpt_all.json"
  test -f "${COMBINED_RECIPE}" || {
    echo "ERROR: SPLIT_ONLY=1 requires existing normalized recipe: ${COMBINED_RECIPE}" >&2
    exit 22
  }
  echo "CPT_V2_SPLIT_ONLY=1"
  echo "CPT_V2_REUSE_HASH_CACHE=${DATA_DIR}/diagnostics/image_hash_cache.json"
  "${ENV_DIR}/bin/python" scripts/split_locany_cpt.py \
    --recipe "${COMBINED_RECIPE}" \
    --output-dir "${DATA_DIR}" \
    --seed "${CPT_SPLIT_SEED:-20260826}" \
    --val-fraction "${CPT_VAL_FRACTION:-0.02}" \
    --val-fast-per-task "${VAL_FAST_PER_TASK}" \
    --group-id-mode "${CPT_GROUP_ID_MODE:-sha256}" \
    --train-recipe-name "${RECIPE_NAME}" \
    --progress-every "${CPT_SPLIT_PROGRESS_EVERY:-1000}"
else
  "${ENV_DIR}/bin/python" scripts/prepare_locany_cpt.py "${PREPARE_ARGS[@]}"
fi

MANIFEST="${DATA_DIR}/diagnostics/split_manifest.jsonl"
"${ENV_DIR}/bin/python" scripts/validate_locany_cpt.py \
  --recipe "${DATA_DIR}/recipe/${RECIPE_NAME}" \
  --records-per-dataset 0 \
  --split-manifest "${MANIFEST}" \
  --require-split train \
  --require-equal-weights
"${ENV_DIR}/bin/python" scripts/validate_locany_cpt.py \
  --recipe "${DATA_DIR}/recipe/locany_cpt_val.json" \
  --records-per-dataset 0 \
  --split-manifest "${MANIFEST}" \
  --require-split heldout
"${ENV_DIR}/bin/python" scripts/validate_locany_cpt.py \
  --recipe "${DATA_DIR}/recipe/locany_cpt_val_fast.json" \
  --records-per-dataset 0 \
  --minimum-records-per-dataset "${MIN_VAL_FAST_PER_TASK}" \
  --split-manifest "${MANIFEST}" \
  --require-split heldout \
  --allow-manifest-subset

echo "CPT_V2_DATA_READY=${DATA_DIR}"
echo "TRAIN_RECIPE=${DATA_DIR}/recipe/${RECIPE_NAME}"
echo "VAL_FAST_RECIPE=${DATA_DIR}/recipe/locany_cpt_val_fast.json"
echo "SPLIT_MANIFEST=${MANIFEST}"
