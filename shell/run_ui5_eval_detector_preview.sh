#!/usr/bin/env bash
set -Eeuo pipefail

# One-command GT-free test-set detector/crop preview.  The main/icon process
# uses LocateAnything while PP-OCR workers use the explicitly supplied Paddle
# Python.  The two detector phases remain sequential and resumable.
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
  --text-python /path/to/UI5PaddleOCR/bin/python \
  --icon-python /path/to/LocateAnything/bin/python \
  --max-images-per-task 200 \
  --visualization-samples 60 \
  --scan-name horizontal_scan_v2 \
  --resume

原始检测固定保存在 detections/；修改几何时换 --scan-name 并只跑 --stage crop。
输出重点：horizontal_scan_v2/gallery/index.html、horizontal_scan_v2/summary.json、
horizontal_scan_v2/statistics.csv、horizontal_scan_v2/preview_crops/。
EOF
  exit 2
fi

cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" scripts/prepare_ui5_eval_detector_crops.py --stage all "$@"
