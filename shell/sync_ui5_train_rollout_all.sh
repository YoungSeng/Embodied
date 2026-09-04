#!/usr/bin/env bash
set -Eeuo pipefail

WORKSPACE=${WORKSPACE:-/mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace}
OUTPUT_ROOT=${OUTPUT_ROOT:-${WORKSPACE}/gui_rollouts/ui5-train-rollout8-h20x2-v4-20260904}
PYTHON_BIN=${PYTHON_BIN:-${WORKSPACE}/conda_envs/LocateAnything/bin/python}
SNAPSHOT_ROOT=${OUTPUT_ROOT}/snapshots
A800_DESTINATION=/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/gui_rollouts/ui5-train-rollout8-h20x2-v4-20260904/snapshots/

test -x "${PYTHON_BIN}" || { echo "ERROR: Python missing: ${PYTHON_BIN}" >&2; exit 2; }
test -d "${SNAPSHOT_ROOT}" || { echo "ERROR: snapshot directory missing: ${SNAPSHOT_ROOT}" >&2; exit 3; }

snapshot_count=0
while IFS= read -r snapshot; do
  [[ -n "${snapshot}" ]] || continue
  echo "[NASTK_SYNC_ALL] source=${snapshot} destination=${A800_DESTINATION}"
  nastk cp -c=32 \
    "${snapshot}" \
    "${A800_DESTINATION}"
  snapshot_count=$((snapshot_count + 1))
done < <("${PYTHON_BIN}" - "${SNAPSHOT_ROOT}" <<'PY'
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
for _, path in sorted(candidates):
    print(path)
PY
)

if [[ ${snapshot_count} -eq 0 ]]; then
  echo "ERROR: no atomic snapshot with manifest.json and _SUCCESS is ready" >&2
  exit 4
fi
echo "[NASTK_SYNC_ALL_OK] snapshots=${snapshot_count} destination=${A800_DESTINATION}"
