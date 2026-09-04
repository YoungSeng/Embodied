#!/usr/bin/env bash
set -Eeuo pipefail

WORKSPACE=${WORKSPACE:-/mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace}
OUTPUT_ROOT=${OUTPUT_ROOT:-${WORKSPACE}/gui_rollouts/ui5-train-rollout8-h20x2-v6-20260904}
PYTHON_BIN=${PYTHON_BIN:-${WORKSPACE}/conda_envs/LocateAnything/bin/python}
SNAPSHOT_ROOT=${OUTPUT_ROOT}/snapshots
A800_DESTINATION=/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/gui_rollouts/ui5-train-rollout8-h20x2-v6-20260904/snapshots/

test -x "${PYTHON_BIN}" || { echo "ERROR: Python missing: ${PYTHON_BIN}" >&2; exit 2; }
test -d "${SNAPSHOT_ROOT}" || { echo "ERROR: snapshot directory missing: ${SNAPSHOT_ROOT}" >&2; exit 3; }

LATEST_SNAPSHOT=$("${PYTHON_BIN}" - "${SNAPSHOT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
candidates = []
for path in root.iterdir():
    summary_path = path / "summary.json"
    if (
        not path.is_dir()
        or not summary_path.is_file()
        or not (path / "manifest.json").is_file()
        or not (path / "_SUCCESS").is_file()
    ):
        continue
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    candidates.append((float(summary["created_at_epoch"]), path))
if not candidates:
    raise SystemExit("no atomic snapshot with manifest.json and _SUCCESS is ready")
print(max(candidates, key=lambda item: item[0])[1])
PY
)

echo "[NASTK_SYNC_LATEST] source=${LATEST_SNAPSHOT} destination=${A800_DESTINATION}"
nastk cp -c=32 \
  "${LATEST_SNAPSHOT}" \
  "${A800_DESTINATION}"
