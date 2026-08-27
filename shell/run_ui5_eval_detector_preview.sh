#!/usr/bin/env bash
set -Eeuo pipefail

# One-command GT-free test-set detector/crop preview.  PP-OCR and icon models
# run sequentially; --resume reuses every complete shard.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ $# -eq 0 ]]; then
  cat <<'EOF'
用法：
bash shell/run_ui5_eval_detector_preview.sh \
  --input-dir /path/to/ui5_test_jsonls \
  --parser-root ../ui-region-parser \
  --output-dir work_dirs/ui5_eval_detector_preview \
  --gpus 0,1,2,3 \
  --max-images-per-task 20 \
  --visualization-samples 20 \
  --resume

输出重点：scan_crops/gallery/index.html、scan_crops/summary.json、
scan_crops/statistics.csv、scan_crops/preview_crops/。
EOF
  exit 2
fi

cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" scripts/prepare_ui5_eval_detector_crops.py --stage all "$@"
