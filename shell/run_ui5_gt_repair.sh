#!/usr/bin/env bash
set -Eeuo pipefail

# Training-only UI5 v4 GT repair.  This wrapper never invokes prepare/text/icon/merge.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${UI5_PYTHON:-$(command -v python || command -v python3 || true)}"

OUTPUT_DIR=""
PARSER_ROOT=""
BASE_META=""
SOURCE_AUDIT_NAME="crop_audit_v3"
CROP_AUDIT_NAME="crop_audit_v4_gt_repair"
EXPECTED_UNIQUE_IMAGES=17281
RESUME=0

usage() {
  cat <<'EOF'
Usage: bash shell/run_ui5_gt_repair.sh \
  --output-dir work_dirs/ui5_crop_audit_20260825 \
  --parser-root ../ui-region-parser \
  --base-meta data/ui_defect_locany_v3/recipe/ui_defect_5class_train.json \
  [--source-audit-name crop_audit_v3] \
  [--crop-audit-name crop_audit_v4_gt_repair] [--resume]
EOF
}

while (( $# > 0 )); do
  case "$1" in
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --parser-root) PARSER_ROOT="$2"; shift 2 ;;
    --base-meta) BASE_META="$2"; shift 2 ;;
    --source-audit-name) SOURCE_AUDIT_NAME="$2"; shift 2 ;;
    --crop-audit-name) CROP_AUDIT_NAME="$2"; shift 2 ;;
    --expected-unique-images) EXPECTED_UNIQUE_IMAGES="$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "${PYTHON_BIN}" ]] || { echo "[ERROR] Python not found" >&2; exit 2; }
[[ -n "${OUTPUT_DIR}" ]] || { echo "[ERROR] --output-dir is required" >&2; exit 2; }
[[ -n "${PARSER_ROOT}" ]] || { echo "[ERROR] --parser-root is required" >&2; exit 2; }
[[ -n "${BASE_META}" ]] || { echo "[ERROR] --base-meta is required" >&2; exit 2; }
[[ -d "${OUTPUT_DIR}/${SOURCE_AUDIT_NAME}" ]] || {
  echo "[ERROR] Source audit is missing: ${OUTPUT_DIR}/${SOURCE_AUDIT_NAME}" >&2
  exit 2
}
[[ -f "${OUTPUT_DIR}/detections/merged/detections.jsonl" ]] || {
  echo "[ERROR] Immutable merged detections are missing" >&2
  exit 2
}
[[ -f "${BASE_META}" ]] || { echo "[ERROR] Base meta is missing: ${BASE_META}" >&2; exit 2; }

cd "${PROJECT_ROOT}"
echo "[UI5 v4] detector_stages_executed=[]; OCR/icon/merge disabled"
echo "[UI5 v4] source audit=${SOURCE_AUDIT_NAME}; target audit=${CROP_AUDIT_NAME}"

REPAIR_COMMAND=(
  "${PYTHON_BIN}" scripts/run_ui5_gt_repair.py
  --output-dir "${OUTPUT_DIR}"
  --parser-root "${PARSER_ROOT}"
  --source-audit-name "${SOURCE_AUDIT_NAME}"
  --crop-audit-name "${CROP_AUDIT_NAME}"
  --expected-unique-images "${EXPECTED_UNIQUE_IMAGES}"
)
if (( RESUME == 1 )); then
  REPAIR_COMMAND+=(--resume)
fi
"${REPAIR_COMMAND[@]}"

AUDIT_DIR="${OUTPUT_DIR}/${CROP_AUDIT_NAME}"
"${PYTHON_BIN}" scripts/build_ui5_crop_training_recipe.py \
  --audit-dir "${AUDIT_DIR}" \
  --base-meta "${BASE_META}" \
  --task-aware-manifest "${AUDIT_DIR}/task_aware_manifest.jsonl" \
  --excluded-samples "${AUDIT_DIR}/excluded_training_samples.jsonl" \
  --mode full_plus_crop \
  --output-dir "${AUDIT_DIR}/training_recipes" \
  --require-valid-gt-recall 1.0

for REQUIRED in \
  "${AUDIT_DIR}/summary.json" \
  "${AUDIT_DIR}/ui5_crop_audit.xlsx" \
  "${AUDIT_DIR}/training_recipes/ui_defect_5class_train_full_plus_crop.json" \
  "${AUDIT_DIR}/training_recipes/ui_defect_5class_train_full_plus_crop.jsonl" \
  "${AUDIT_DIR}/training_recipes/recipe_summary.json" \
  "${AUDIT_DIR}/training_ready.json"; do
  [[ -s "${REQUIRED}" ]] || { echo "[ERROR] Required v4 output is missing: ${REQUIRED}" >&2; exit 3; }
done

echo "[UI5 v4 COMPLETE] audit=${AUDIT_DIR}"
echo "[UI5 v4 COMPLETE] marker was written last; training has not been started"
