#!/usr/bin/env bash
set -Eeuo pipefail

WORKSPACE=${WORKSPACE:-/mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace}
OUTPUT_ROOT=${OUTPUT_ROOT:-${WORKSPACE}/gui_rollouts/ui5-train-rollout8-h20x2-v3-20260904}
PYTHON_BIN=${PYTHON_BIN:-${WORKSPACE}/conda_envs/LocateAnything/bin/python}
A800_DESTINATION=/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle_LocateUI5_v4/Embodied-ui5-det-crop/work_dirs/

test -x "${PYTHON_BIN}" || { echo "ERROR: Python missing: ${PYTHON_BIN}" >&2; exit 2; }
test -s "${OUTPUT_ROOT}/summary.json" || { echo "ERROR: final summary missing" >&2; exit 3; }

"${PYTHON_BIN}" - "${OUTPUT_ROOT}/summary.json" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if summary.get("status") != "completed":
    raise SystemExit("final summary status is not completed")
alignment = summary.get("raw_alignment") or []
if len(alignment) != 8 or not all(row.get("complete") for row in alignment):
    raise SystemExit("all-ready requires eight complete and aligned raw streams")
expected = int(summary["expected_total_per_rollout"])
if any(int(row.get("actual_total", -1)) != expected for row in alignment):
    raise SystemExit("all-ready raw stream counts do not equal expected_total")
print(f"all-ready verified: streams=8 expected_total={expected}")
PY

echo "[NASTK_SYNC_ALL_READY] source=${OUTPUT_ROOT} destination=${A800_DESTINATION}"
nastk cp -c=32 \
  "${OUTPUT_ROOT}" \
  "${A800_DESTINATION}"
