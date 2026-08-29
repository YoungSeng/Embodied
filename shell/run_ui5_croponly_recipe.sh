#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
AUDIT_DIR=""
BASE_META=""
VALIDATION_DATA_DIR=""
TEST_DATA_DIR=""
OUTPUT_NAME="crop_only_horizontal_v5_train_repair"
MAX_CROPS=10
TARGET_HEIGHT=960
RESUME=0

while (( $# )); do
  case "$1" in
    --audit-dir) AUDIT_DIR="$2"; shift 2 ;;
    --base-meta) BASE_META="$2"; shift 2 ;;
    --validation-data-dir) VALIDATION_DATA_DIR="$2"; shift 2 ;;
    --test-data-dir) TEST_DATA_DIR="$2"; shift 2 ;;
    --output-name) OUTPUT_NAME="$2"; shift 2 ;;
    --max-crops) MAX_CROPS="$2"; shift 2 ;;
    --target-height) TARGET_HEIGHT="$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    *) echo "[ERROR] Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${AUDIT_DIR}" || -z "${BASE_META}" || -z "${VALIDATION_DATA_DIR}" || -z "${TEST_DATA_DIR}" ]]; then
  echo "Usage: $0 --audit-dir PATH --base-meta PATH --validation-data-dir PATH --test-data-dir PATH [--resume]" >&2
  exit 2
fi
AUDIT_DIR="$(cd "${AUDIT_DIR}" && pwd)"
BASE_META="$(cd "$(dirname "${BASE_META}")" && pwd)/$(basename "${BASE_META}")"
MANIFEST_DIR="${AUDIT_DIR}/${OUTPUT_NAME}"
RECIPE_DIR="${AUDIT_DIR}/training_recipes"
EXCLUDED="${AUDIT_DIR}/excluded_training_samples.jsonl"

if [[ ! -s "${AUDIT_DIR}/gt_repair_actions.jsonl" ]]; then
  echo "[ERROR] Missing GT repair actions: ${AUDIT_DIR}/gt_repair_actions.jsonl" >&2
  exit 1
fi
if [[ ! -s "${AUDIT_DIR}/../detections/merged/detections.jsonl" ]]; then
  echo "[ERROR] Missing immutable merged detections: ${AUDIT_DIR}/../detections/merged/detections.jsonl" >&2
  exit 1
fi
if [[ ! -s "${BASE_META}" || ! -s "${EXCLUDED}" ]]; then
  echo "[ERROR] Base meta or exclusion manifest is missing." >&2
  exit 1
fi

manifest_command=(
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/build_ui5_croponly_training_manifest.py"
  --audit-dir "${AUDIT_DIR}"
  --output-name "${OUTPUT_NAME}"
  --max-crops "${MAX_CROPS}"
  --target-height "${TARGET_HEIGHT}"
)
if (( RESUME )); then
  manifest_command+=(--resume)
fi
"${manifest_command[@]}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/check_ui5_train_eval_content_overlap.py" \
  --train-unique-manifest "${AUDIT_DIR}/../manifest/unique_images.jsonl" \
  --validation-data-dir "${VALIDATION_DATA_DIR}" \
  --test-data-dir "${TEST_DATA_DIR}" \
  --output "${MANIFEST_DIR}/data_split_overlap.json"

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/build_ui5_crop_training_recipe.py" \
  --audit-dir "${AUDIT_DIR}" \
  --base-meta "${BASE_META}" \
  --task-aware-manifest "${MANIFEST_DIR}/task_aware_manifest.jsonl" \
  --excluded-samples "${EXCLUDED}" \
  --mode crop_only \
  --output-dir "${RECIPE_DIR}" \
  --require-valid-gt-recall 1.0

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/validate_ui5_crop_training_ready.py" \
  --audit-dir "${AUDIT_DIR}" \
  --recipe "${RECIPE_DIR}/ui_defect_5class_train_crop_only.json"

for required in \
  "${MANIFEST_DIR}/task_aware_manifest.jsonl" \
  "${MANIFEST_DIR}/summary.json" \
  "${RECIPE_DIR}/ui_defect_5class_train_crop_only.json" \
  "${RECIPE_DIR}/ui_defect_5class_train_crop_only.jsonl" \
  "${RECIPE_DIR}/crop_only_recipe_summary.json" \
  "${AUDIT_DIR}/training_ready.json"; do
  if [[ ! -s "${required}" ]]; then
    echo "[ERROR] Required crop-only artifact is missing or empty: ${required}" >&2
    exit 1
  fi
done

echo "[crop-only ready] audit=${AUDIT_DIR}"
echo "[crop-only ready] meta=${RECIPE_DIR}/ui_defect_5class_train_crop_only.json"
